using DrWatson
quickactivate(@__DIR__)
@show projectdir()
##
using CovidSurvey
using CSV
using DataFrames
using Dates
using StatsPlots
using Plots.PlotMeasures
using LaTeXStrings

using Random
using Turing
## ===========================================================================
# load data
# ============================================================================
fdir = projectdir("reports/parametric-model-on-2nd-wave-newdata/")
fname = "cases=2020-06-01_chains=10_hospit=true_model=parametric-cases_observ=2021-01-13_preds=CF,CC,CR,CS_sero=true_steps=2000_warmup=1000"
fname = "cases=2020-06-01_chains=10_hospit=true_model=parametric-cases_observ=2021-01-13_preds=CF,CC,CR,CS_sero=true_steps=2000_warmup=1000"
PREFIX = "RAW-"
# fname = "cases=2020-06-01_chains=10_hospit=true_model=parametric-cases_observ=2021-01-13_preds=CF,CC,CR,CS_sero=true_steps=4000_warmup=1000"
rt    = load(fdir*PREFIX*"Rt_"*fname*".csv") |> DataFrame
chain = read(fdir*PREFIX*"CHAIN_"*fname*".jls", Chains)
posterior = load(fdir*PREFIX*"GENERATED-QUANTITIES_"*fname*".bson") |> NamedTuple
data = let
    _, ps, _ = parse_savename(fname, parsetypes = (Int, Float64, Bool))
    !("preds" in keys(ps)) && (ps["preds"] = nothing)
    # ps = NamedTuple{Tuple(Symbol.(keys(ps)))}(values(ps))

    National.load_data(
          ps["observ"]
        , ps["preds"]|> CovidSurvey.parse_predictors
        , ps["cases"]
        , ps["hospit"]
        , ps["sero"]
        , iar_step = 1
        , covariates_kwargs = Dict(
            :fname => projectdir("data", "inferred_contact_rates.csv"),
            #:fname => projectdir("data", "smoothed_contact_rates.csv"), #normpath( homedir(), "data/covidsurvey/smoothed_contacts.csv" ),
            # :fname => normpath( homedir(), "data/covidsurvey/smoothed_contacts.csv" ),
            :shift => -1,
            :startdate => "2020-11-10",
            # :enddate => "2021-01-13"
        )
    )
end
# turing_data = data.turing_data;
## ===========================================================================
# plot prediction

plotlyjs()
pgfplotsx()
Plots.scalefontsizes()
Plots.scalefontsizes(2.)
p = let
    p1 = let
        i = findfirst(==(Date("2020-05-15")), data.dates)
        dates = data.dates[i:end]
        xlims   = (dates[1], dates[end])
        p = plot(; xaxis = true, xlims, ylabel="counts", legend = :outerright);
        # plot_hospit_posterior!(p, data, expected_daily_hospit)
        obs = hcat(posterior.expected_daily_cases...)[i:end,:]
        tNobs  = data.dates[data.num_observations]
        # title!(p, "posterior hospitalization counts")
        plot!(p, dates, data.cases[i:end], alpha=0.5, lc=:match, label="cases", c=:midnightblue, lw=2) #, label="Recorded daily hospitalizations (observed)"
        # vspan!(p, [DateTime("2020-11-10"), tNobs], label="fit contacts", α=0.2)
        vline!(p, [tNobs], label="end of observations", linewidth=2, lc=:black, hover="$(tNobs)")
        # vline!(p, [], label="end of observations", linewidth=2, lc=:black, ls=:dash)
        plot_confidence_timeseries!(p, dates, obs; label = "Expected daily hospitalizations")
    end

    p2 = let
        i = findfirst(==(Date("2020-05-15")), data.dates)
        dates = data.dates[i:end]
        xlims   = (dates[1], dates[end])
        p = plot(; xaxis = true, xlims, legend = :outerright, ylabel=L"R_t");
        Rt = hcat(posterior.Rt...)[i:end,:]
        tNobs  = data.dates[data.num_observations]
        # title!(p, "effective reproduction number")
        vspan!(p, [Date("2020-11-10"), tNobs], label="fit contacts", α=0.2, fc=:midnightblue)
        # vline!(p, [Date("2020-11-10")], label="fitted contacts", linewidth=2, lc=:black, ls=:dash)
        vline!(p, [tNobs], label="end of observations", linewidth=2, lc=:black, hover="$(tNobs)")
        # vline!(p, [], label="end of observations", linewidth=2, lc=:black, ls=:dash)
        plot_confidence_timeseries!(p, dates, Rt; label = "reproduction number")
    end

    plot(p1, p2, layout=(2,1), size=(800, 500), sharex=true, link=:x, legend = nothing)#xticks = tickdates
end
savefig(p, normpath(fdir,PREFIX*"FIG_hospit_"*fname*".png") )
# savefig(p, projectdir(fdir*"FIG_hospitalization_"*fname*".png") )

##

p = let#CF,CC,CR,CS
    xs = chain["α[1]"]|>vec
    p1 = histogram(xs, normalize=true, title="contacts to family", label=nothing, fc=:gray, lw=.0, xlabel="effect size", bottom_margin=5mm);
    plot!(Exponential(0.2), xlims=(0,2.0), lw=2, label="prior", legend=:topright);
    vline!([mean(xs)], lw=2, label="mean effect = $(round(mean(xs), sigdigits=2))", c=:black, ls=:dash)
    xs = chain["α[2]"]|>vec
    p2 = histogram(xs, normalize=true, title="contacts to colleagues", label=nothing, fc=:gray, lw=.0, xlabel="effect size", bottom_margin=5mm);
    plot!(Exponential(0.2), xlims=(0,2.0), lw=2, label="prior", legend=:topright);
    vline!([mean(xs)], lw=2, label="mean effect = $(round(mean(xs), sigdigits=2))", c=:black, ls=:dash)
    xs = chain["α[3]"]|>vec
    p3 = histogram(xs, normalize=true, title="contacts to friends", label=nothing, fc=:gray, lw=.0, xlabel="effect size", top_margin=5mm);
    plot!(Exponential(0.2), xlims=(0,2.0), lw=2, label="prior", legend=:topright);
    vline!([mean(xs)], lw=2, label="mean effect = $(round(mean(xs), sigdigits=2))", c=:black, ls=:dash)
    xs = chain["α[4]"]|>vec
    p4 = histogram(xs, normalize=true, title="contacts to strangers", label=nothing, fc=:gray, lw=.0, xlabel="effect size", top_margin=5mm);
    plot!(Exponential(0.2), xlims=(0,2.0), lw=2, label="prior", legend=:topright);
    vline!([mean(xs)], lw=2, label="mean effect = $(round(mean(xs), sigdigits=2))", c=:black, ls=:dash)
    plot(p1,p2,p3,p4, layout=(2,2), size=(1000, 700))
end
# savefig(p, projectdir(fdir*"FIG_hospitalization_effectsizes_"*fname*".png") )
savefig(p, normpath(fdir,PREFIX*"FIG-cases-effectsizes_"*fname*".png") )

## ============================================================================
X = data.turing_data.covariates[end,:]
i = findfirst(==(Date("2020-11-10")), rt.date)
Δrt = rt.mq[i] - rt.mq[end]
r0 = rt.mq[i]

α1 = chain["α[1]"]|>vec|>mean
α2 = chain["α[2]"]|>vec|>mean
α3 = chain["α[3]"]|>vec|>mean
α4 = chain["α[4]"]|>vec|>mean
α = [α1,α2,α3,α4]

f(r0 - α)

f = KLogistic(3.)
f⁻¹ = KLogit(3.)
f(f⁻¹(r0) + X'α) / r0
X .* α
## ==========================================================================
covariates_kwargs = Dict(
    :fname => normpath( homedir(), "data/covidsurvey/smoothed_contacts.csv" ),
    :shift => -1,
    :startdate => "2020-11-10", # >= 1000 cases / day
    # :enddate => "2021-01-13" # <= 1000 cases / day
)
covariates = National.readcovariates(; covariates_kwargs... )
