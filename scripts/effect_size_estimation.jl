using DrWatson
quickactivate(@__DIR__)
@show projectdir()
##
using StatsBase
using Plots
using DataFrames
using Underscores
using CSV
using Dates
## ============================================================================
#                    apple
#  ============================================================================

numericcols(df::DataFrame) =
    filter(x->nonmissingtype( eltype( df[!,x] ) ) <: Number, names(df))

function rollingmean(xs, window)
    n  = length(xs)-window+1
    ys = Vector{Float64}(undef, n)
    for i in 1:n
        x = xs[i:i+window-1]
        @assert sum( ismissing.(x) ) < 4 "nr of missing: $(sum( ismissing.(x) )) at i=$i"
        ys[i] = mean(skipmissing(x))
        @assert isfinite(ys[i]) "infinite at i=$i"
    end
    return ys
end

function rollingmean(df::DataFrame, window)
    @assert isodd(window)
    Δ = window ÷ 2
    smoothed = DataFrame( :date => df.date[1+Δ:end-Δ] )
    for col in numericcols(df)
        smoothed[!,col] = rollingmean(df[:,col], window)
    end
    smoothed
end

function normalize(df::DataFrame)
    for col in numericcols(df)
        df[!,col] ./= abs(df[1,col])
        df[!,col] .-= df[1,col]
        df[!,col] ./= StatsBase.std(df[!,col])
    end
    df
end

function normalize(df::DataFrame)
    for col in numericcols(df)
        df[!,col] ./= abs(df[1,col])
        df[!,col] .-= df[1,col]
        df[!,col] ./= StatsBase.std(df[!,col])
    end
    df
end

function normalize(df::DataFrame, μ, σ)
    for col in numericcols(df)
        df[!,col] ./= abs(df[1,col])
        df[!,col] .-= df[1,col]
        df[!,col] ./= StatsBase.std(df[!,col])
        df[!,col] .*= σ
        df[!,col] .+= μ
    end
    df
end
## ============================================================================
(i, region)  = (1, "capital")
rt = @_ "/home/and/dev/CovidSurvey/reports/regional_prediction_end_of_peak/2021-09-01/Rt-$(uppercase(region))_chains=10_model=hospit_observ=2021-03-01_regional=true_rwstep=1_semipar=true_std=false_steps=1000_warmup=1000.csv" |>
    CSV.File |> DataFrame |>
    select(__, "dates"=>"date", "mq" => "Rt", "llq"=>"lower95", "uuq" => "upper95")

surv = @_ projectdir("data/mean_contact_rates_region=$(i).csv") |>
    CSV.File |> DataFrame |>
    rename(__, "dates"=>"date") |>
    rollingmean(__, 7)

mobil = @_ projectdir("data/mobility/", "mobility_region=$region.csv") |>
    CSV.File |> DataFrame

dat = @_  leftjoin(surv, mobil, on=:date) |>
    rightjoin(__, rt[:, [:date, :Rt]], on=:date) |>
    filter( Date("2020-06-10") < _.date < Date("2021-03-01"), __) |> #Date("2021-02-01")
    normalize(__, rt.Rt[1], std(rt.Rt)*4)



pp = let
    fname = "/home/and/dev/CovidSurvey/reports/regional_prediction_end_of_peak/2021-09-01/CHAIN_chains=10_model=hospit_observ=2021-03-01_regional=true_rwstep=1_semipar=true_std=false_steps=1000_warmup=1000.jls"
    @_  Regional.parse_fname(fname) |>
        Regional.PostProcessing2(__..., [], fname) |>
        Regional.skip_warmup
end

gp = @_ "/home/and/dev/CovidSurvey/reports/regional_prediction_end_of_peak/2021-09-01/GENERATED-QUANTITIES_chains=10_model=hospit_observ=2021-03-01_regional=true_rwstep=1_semipar=true_std=false_steps=1000_warmup=1000.bson" |>
    BSON.load(__) |>
    NamedTuple


pgfplotsx()
let

    p1 = @_ Regional.ObservationPlottingRecipe1(pp.data, gp, region, "hospitalizations") |>
        plot!(plot(), __, plot_only_fit=true)

    p2 = @_ Regional.RtPlottingRecipe1(pp.data, gp, region, "") |>
        plot!(plot(), __, plot_only_fit=true)

    plot!(dat.date, dat.total, lab="contacts", lw=2)
    plot!(dat.date, dat.apple, lab="apple", lw=2)
    plot!(dat.date, dat.google, lab="google", lw=2)
    plot!(dat.date, dat.telco, lab="telco", lw=2)

    p = plot(p1, p2, layout=(2,1), size=(800, 2*250), sharex=true, link=:x)
    xlims!( Dates.value.([Date("2020-11-10"), Date("2021-02-01")])... )

    fname = "/home/and/.tmp/compare_hospit_rt_predictors_REGION=$(uppercase(region)).png"
    savefig(p, projectdir(fname))
    run(`firefox $fname`, wait=false)
end

plotlyjs()
let
    p = plot()

    lo = rt.Rt - rt.lower95
    up = rt.upper95 - rt.Rt
    p = plot()
    plot!(dat.date .- Day(5), dat.Rt, ribbon=(lo, up), lab="Rt", c="moccasin")
    # plot!(dat.date, dat.friends, lab="friends")
    # plot!(dat.date, dat.colleagues, lab="colleagues")
    # plot!(dat.date, dat.family, lab="family")
    # plot!(dat.date, dat.strangers, lab="strangers")

    plot!(dat.date, dat.total, lab="total")
    plot!(dat.date, dat.apple, lab="apple")
    plot!(dat.date, dat.google, lab="google")
    plot!(dat.date, dat.telco, lab="telco")

    #xlims!( Dates.value.([Date("2020-09-10"), Date("2021-02-01")])... )

    fname = "/home/and/.tmp/compare_rt_predictors_REGION=$(uppercase(region)).html"
    savefig(p, projectdir(fname))
    run(`firefox $fname`, wait=false)
end

lags = -10:10
c = crosscor(dat.Rt, dat.strangers, lags)
plot(lags, c)
lags[argmax(c)]
