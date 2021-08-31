using DrWatson
quickactivate(@__DIR__)
@show projectdir()
##
using StatsBase
using StatsPlots
using DataFrames
using Dates
using CSV
plotlyjs()
##
fname = projectdir("data/smoothed_contact_rates.csv")
surv = CSV.File(fname) |> DataFrame
##
fname = projectdir("data/google/mobility_denmark.csv")
mobil = CSV.File(fname) |> DataFrame
##
fname = "/home/and/data/covidsurvey/contacts.csv"
rt = CSV.File(fname) |> DataFrame
fname = projectdir("reports/2021-07-21/Rt_cases=_chains=1_hospit=false_model=v1_sero=false_steps=1000_warmup=1000.csv")
fname = projectdir("reports/2021-07-21/Rt_cases=2020-06-01_chains=5_hospit=false_model=hospit_sero=false_steps=1000_warmup=1000.csv")
rt = CSV.File(fname) |> DataFrame
fname = projectdir("reports/2021-07-21/Rt_cases=_chains=1_hospit=false_model=hospit_sero=false_steps=1000_warmup=1000.csv")
rt_d = CSV.File(fname) |> DataFrame
fname = projectdir("reports/2021-07-21/Rt_cases=2020-06-01_chains=5_hospit=false_model=cases_sero=false_steps=1000_warmup=1000.csv")
fname = projectdir("reports/national_prediction_end_of_peak/Rt_chains=5_model=hospit_observ=2021-01-13_predstart=2020-11-10_regional=false_semipar=true_steps=200_warmup=1000.csv")
rt = CSV.File(fname) |> DataFrame
rename!(rt, :dates=>:date)
##
function StatsBase.zscore(df::DataFrame)
    y = similar(df)
    for n in names(df)
        y[!,n] = zscore(df[:,n])
    end
    return y
end

function conform(args...; normalize = false)
    s = maximum( first.( select.(args, Ref(:date) ) ) ).date
    e = minimum( last.( select.(args, Ref(:date) ) ) ).date
    x = s:Day(1):e
    y = filter.(Ref(:date => >=(s)), args)
    y = filter.(Ref(:date => <=(e)), y)
    normalize && ( y = zscore.( select.( y, Ref(Not(:date)) ) ) )
    (x, y...)
end

let
    x,yr,ys,ym = conform(rt[:,[:mq,:date]], surv, mobil[:,[:nonresidential, :date]], normalize=true)#rt_ssi[:,[:mq,:date]])

    p = plot(xticks=:native, size=(1300,800), lw=2)
    plot!(x, yr.mq, label="Rt (hospit)", lw=2)
    # plot!(x, yssi.mq, label="Rt (SSI)", lw=2)
    plot!(x, ys.strangers, lab="contacts to strangers", lw=2)
    plot!(x, ym.nonresidential, lab="nonresidential mobility", lw=2)
    # plot!(x, y2.friends, lab="friends", lw=2)
    # plot!(x, y2.family, lab="family", lw=2)
    # plot!(x, y2.colleagues, lab="colleague", lw=2)

    fname = "/home/and/.tmp/tmp.html"
    savefig(p, fname)
    run(`firefox $(fname)`, wait=false)
end

## ============================================================================
plotlyjs()
let
    obs = Dict()
    National.observables!(obs)

    p = plot(xticks=:native, size=(1000,800))
    x, y = eachcol(obs["cases"])
    plot!(x,y|>zscore,lab="cases")
    x, y = eachcol(obs["hospit"])
    plot!(x,y|>zscore,lab="hospit")
    x, y = eachcol(obs["deaths"])
    plot!(x,y|>zscore,lab="deaths")

    fname = "/home/and/.tmp/tmp.html"
    savefig(p, fname)
    run(`firefox $(fname)`, wait=false)
    p
end

ys = []
for n in ["cases", "hospit", "deaths"]
    x, y = eachcol(obs[n])
    i = findall(x-> Date("2020-11-30")<x<Date("2021-02-01"), x)
    push!(ys, y[i]|>zscore)
end

lags = 0:50
c = crosscor(ys[2], ys[3], lags)
plot(lags, c)
argmax(c)


## =============================================================================
plot(NegativeBinomial2(1., 5.41))
plot!(NegativeBinomial2(4.5, 5.41))
plot!(GammaMeanCv(5.1, 0.86))
mean(GammaMeanCv(5.1, 0.86))

## =============================================================================
# compare with SSI
p = let
    fname = projectdir("reports/2021-07-21/Rt_cases=2020-06-01_chains=5_hospit=false_model=cases_sero=false_steps=1000_warmup=1000.csv")
    rt = CSV.File(fname) |> DataFrame
    rt_ssi = projectdir("data/Rt_SSI.csv") |> load |> DataFrame
    date, yrt, yrt_ssi = conform(rt[:,[:mq,:date]], rt_ssi[:,[:mq,:date]]; normalize = false)
    p = plot(legend=:topleft, ylabel="R_t", size=(800, 500), xticks=:native)
    plot!(date, yrt_ssi.mq, lw=2, label="SSI", c=:sienna)
    plot!(date, yrt.mq, lw=2, label="model without predictors", c=:royalblue)
end

## =============================================================================
# sero
data = National.load_data()
plot!(data.cases.date  .- Day(16), cumsum(data.cases.country) / data.turing_data.population*2)
scatter!(data.seroprev.date , data.seroprev.country)
