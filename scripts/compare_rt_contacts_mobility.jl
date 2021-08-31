using DrWatson
quickactivate(@__DIR__)
@show projectdir()
##
using CovidSurvey
using StatsBase
using StatsPlots
using DataFrames
using Dates
using CSV

plotlyjs()
## ============================================================================
@info "National"

function load_national_data(;kwargs...)

    observables = Dict()
    National.observables!(observables)
    hospit = observables["hospit"]
    cases = observables["cases"]

    fname = projectdir("reports/variable-delays/Rt_cases=2020-06-01_chains=10_hospit=true_model=hospit_observ=2021-02-06_sero=true_steps=4000_warmup=1000.csv")
    rt = CSV.File(fname) |> DataFrame
    "dates" in names(rt) && rename!(rt, "dates"=>"date")

    fname = projectdir("data/raw_contact_rates.csv")
    surv = CSV.File(fname) |> DataFrame

    fname = projectdir("data/google/mobility_denmark.csv")
    mobil = CSV.File(fname) |> DataFrame

    return conform(cases, hospit, rt, surv, mobil; kwargs...)
end

function StatsBase.zscore(df::DataFrame)
    y = similar(df)
    for n in names(df)
        y[!,n] = zscore(df[:,n])
    end
    return y
end

function conform(args...; normalize = false, stopdate=nothing, startdate=nothing)
    s = isnothing(startdate) ? maximum( first.( select.(args, Ref(:date) ) ) ).date : Date(startdate)
    e = isnothing(stopdate) ? minimum( last.( select.(args, Ref(:date) ) ) ).date : Date(stopdate)
    x = s:Day(1):e
    y = filter.(Ref(:date => >=(s)), args)
    y = filter.(Ref(:date => <=(e)), y)
    normalize && ( y = zscore.( select.( y, Ref(Not(:date)) ) ) )
    (x, y...)
end

function plot_comparison(date, cases, hospit, rt, surv, mobil; column=:country)
    plots  = Vector{Plots.Plot}()

    p = plot(; xaxis = true, legend = :outertopright, title="observed cases and hospitalizations", xticks=:native)
    plot!(date, zscore(cases[:,column]), lab="cases", lw=2)
    plot!(date, zscore(hospit[:,column]), lab="hospitalizations", lw=2)
    push!(plots, p)

    p = plot(; xaxis = true, legend = :outertopright, title="reproduction number from hospitalizations", xticks=:native)
    plot!(rt.date, rt.mq, ribbon=(rt.mq - rt.lq, rt.uq - rt.mq), c=:peru, lab="reproduction number", lw=2)
    push!(plots, p)

    p = plot(; xaxis = true, legend = :outertopright, title="contact survey", xticks=:native)
    for c in names(surv, Not(:date))
        plot!(surv.date, surv[:,c], lab=c, lw=2)
    end
    push!(plots, p)

    p = plot(; xaxis = true, legend = :outertopright, title="google mobility", xticks=:native)
    for c in names(mobil, Not(:date))
        plot!(date, mobil[:,c], lab=c, lw=2)
    end
    push!(plots, p)

    return plot(plots..., layout=(4,1), size=(1500, 4*250), sharex=true, link=:x)
end

let
    data  = load_national_data(stopdate="2021-02-01")
    p     = plot_comparison( data...);
    fname = "figures/compare_rt_smothedsurvey_mobility_DENMARK.html"
    savefig(p, projectdir(fname))
    run(`firefox $fname`, wait=false)
end

## ===========================================================================
# regional

function load_regional_data(i; kwargs...)

    observables = Dict()
    Regional.observables!(observables)
    hospit = observables["hospit"]
    cases = observables["cases"]

    fname = projectdir("reports/2021-08-01/Rt-$(uppercase(Regional.regions[i]))_chains=10_epidemicstart=10_folder=2021-08-01_model=cases_numimpute=6_rwstep=1_steps=100_warmup=500.csv")
    rt = CSV.File(fname) |> DataFrame
    rename!(rt, "dates"=>"date")

    # fname = projectdir("data/raw_contact_rates_region=$(i)_aggregation=7.csv")
    fname = projectdir("data/mean_contact_rates_region=$i.csv")
    surv = CSV.File(fname) |> DataFrame
    "dates" in names(surv) && rename!(surv, "dates"=>"date")

    fname = projectdir("data/google/mobility_region=$(Regional.regions[i]).csv")
    mobil = CSV.File(fname) |> DataFrame

    return conform(cases, hospit, rt, surv, mobil; kwargs...)
end


for (i,r) in enumerate(Regional.regions)
    data  = load_regional_data(i; stopdate="2021-02-01")
    p     = plot_comparison( data...; column=Regional.regions[i])
    fname = "figures/compare_rt_meansurvey_mobility_REGION=$(uppercase(r)).html"
    savefig(p, projectdir(fname))
    run(`firefox $fname`, wait=false)
end

## ===========================================================================
# regional with aggregation

function load_aggregated_regional_data(i, aggregation; kwargs...)

    observables = Dict()
    Regional.observables!(observables)
    hospit = observables["hospit"]
    cases = observables["cases"]

    # fname = projectdir("reports/aggregated_random_walk/aggregation=7/")
    fname = projectdir("reports/aggregated_random_walk/aggregation=$(aggregation)/Rt-$(uppercase(Regional.regions[i]))_chains=5_epidemicstart=30_model=hospit_numimpute=6_observationsend=2021-01-13_regional=true_rwstep=$(aggregation)_steps=200_warmup=1000.csv")
    rt = CSV.File(fname) |> DataFrame
    rename!(rt, "dates"=>"date")

    # fname = projectdir("data/raw_contact_rates_region=$(i)_aggregation=7.csv")
    fname = projectdir("data/mean_contact_rates_region=$(i)_aggregation=$(aggregation).csv")
    surv = CSV.File(fname) |> DataFrame
    "dates" in names(surv) && rename!(surv, "dates"=>"date")

    fname = projectdir("data/google/mobility_region=$(Regional.regions[i]).csv")
    mobil = CSV.File(fname) |> DataFrame

    return conform(cases, hospit, rt, surv, mobil; kwargs...)
end

aggregation = 3
for (i,r) in enumerate(Regional.regions)
    data  = load_aggregated_regional_data(i, aggregation; stopdate="2021-02-01")
    p     = plot_comparison( data...; column=Regional.regions[i])
    fname = "figures/compare_rt_meansurvey_mobility_REGION=$(uppercase(r)).html"
    savefig(p, projectdir(fname))
    run(`firefox $fname`, wait=false)
end
