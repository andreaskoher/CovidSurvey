using OrderedCollections

abstract type PlottingRecipe end

Plots.plot(pr::PlottingRecipe, args...) = plot!(plot(), pr, args...)

# =============================================================================
# hospitalizations

struct ObservationsPlottingRecipe{Tsd, Ted, To, Te, Tl} <: PlottingRecipe
    startdate ::Tsd
    enddate   ::Ted
    observed  ::To
    expected  ::Te
    label     ::Tl
end

function expected(data, gp, label)
    @assert label in ["cases", "hospitalizations", "deaths"]
    dates  = data.dates
    if label == "cases"
        values = gp.expected_daily_cases
        return (; dates, values )
    elseif label == "hospitalizations"
        values = gp.expected_daily_hospit
        return (; dates, values )
    else
        values = gp.expected_daily_deaths
        return (; dates, values )
    end
end

function observed(data, label)
    @assert label in ["cases", "hospitalizations", "deaths"]
    dates  = data.hospit.date
    if label == "cases"
        values = data.cases.country
        return (; dates, values )
    elseif label == "hospitalizations"
        values = data.hospit.country
        return (; dates, values )
    else
        values = data.deaths.country
        return (; dates, values )
    end
end

function startdate(data::National.Data, label::String)
    @assert label in ["cases", "hospitalizations", "deaths"]
    s = if label == "cases"
        data.turing_data.casemodel.start
    elseif label == "hospitalizations"
        data.turing_data.hospitmodel.start
    else
        data.turing_data.deathmodel.start
    end
    return data.dates[s]
end

function enddate(data::National.Data, label::String)
    @assert label in ["cases", "hospitalizations", "deaths"]
    s = if label == "cases"
        data.turing_data.casemodel.stop
    elseif label == "hospitalizations"
        data.turing_data.hospitmodel.stop
    else
        data.turing_data.deathmodel.stop
    end
    return data.dates[s]
end

function ObservationsPlottingRecipe(data::National.Data, gp, label)
    e = expected(data, gp, label)
    o = observed(data, label)
    sd = startdate(data, label)
    ed = enddate(data, label)
    return ObservationsPlottingRecipe( sd, ed, o, e, label)
end

function Plots.plot!(p::Plots.Plot, r::ObservationsPlottingRecipe)
    o  = r.observed
    e  = r.expected
    ed = r.enddate
    sd = r.startdate

    plot!(p, o.dates, o.values, α=0.5, lc=:match, lab="observed $(r.label)", c=:midnightblue, lw=4, ylab="cases")
    vline!(p, [ed], lab="end observations", lw=2, lc=:black, hover="$ed")
    vline!(p, [sd], lab="start observations", lw=2, lc=:black, hover="$sd", ls=:dash)
    plot_confidence_timeseries!(p, e.dates, e.values; label = "expected $(r.label)") #Dict(hover=>strdates)
end

# ============================================================================
# reproduction number

struct RtPlottingRecipe{Tlo, Ted, Te, Tl} <: National.PlottingRecipe
    lockdown  ::Tlo
    enddates  ::Ted
    expected  ::Te
    label     ::Tl
end

function RtPlottingRecipe(data::National.Data, gp, label)
    dates      = data.dates
    values     = label == "reproduction number" ? gp.Rt : gp.effective_Rt
    expected   = (; dates, values )
    lockdown   = Date(data.lockdown)
    enddates   = Date(data.observations_end)
    RtPlottingRecipe( lockdown, enddates, expected, label)
end

function Plots.plot!(p::Plots.Plot, r::RtPlottingRecipe)
    e  = r.expected
    ed = r.enddates
    lo = r.lockdown

    vline!(p, [ed], lab="end observations", lw=2, lc=:black, hover="$ed")
    vline!(p, [lo], lab="lockdown", lw=2, lc=:black, hover="$lo", ls=:dash)
    plot_confidence_timeseries!(p, e.dates, e.values; r.label) #Dict(hover=>strdates)
end
# ============================================================================
# infection ascertainment rate

struct IARPlottingRecipe{Tso, Ted, Te} <: National.PlottingRecipe
    startdate ::Tso
    enddates  ::Ted
    expected  ::Te
end

function IARPlottingRecipe(data::National.Data, generated_posterior, args...)
    dates      = data.dates
    values     = generated_posterior.iar
    expected   = (; dates, values )
    startdate  = dates[ data.turing_data.iar_start_idx ]
    enddate    = Date(data.observations_end)
    IARPlottingRecipe( startdate, enddate, expected)
end

function Plots.plot!(p::Plots.Plot, r::IARPlottingRecipe)
    e  = r.expected
    ed = r.enddates
    sd = r.startdate

    vline!(p, [ed], lab="end observations", lw=2, lc=:black, hover="$ed")
    vline!(p, [sd], lab="start case observations", lw=2, lc=:black, hover="$sd", ls=:dash)
    plot_confidence_timeseries!(p, e.dates, e.values; label = "infections ascertainment rate") #Dict(hover=>strdates)
end
# ============================================================================
# region plot

struct OverviewPlottingRecipe{Tr,Tt} <: National.PlottingRecipe
    recipes::Tr
    titles ::Tt
end

posterior2recipe = OrderedDict(
    :expected_daily_cases  => National.ObservationsPlottingRecipe,
    :expected_daily_hospit => National.ObservationsPlottingRecipe,
    :expected_daily_deaths => National.ObservationsPlottingRecipe,
    :Rt                    => National.RtPlottingRecipe,
    :iar                   => National.IARPlottingRecipe,
    :effective_Rt          => National.RtPlottingRecipe,
)

posterior2label = OrderedDict(
    :expected_daily_cases  => "cases",
    :expected_daily_hospit => "hospitalizations",
    :expected_daily_deaths => "deaths",
    :Rt                    => "reproduction number",
    :iar                   => "infections ascertainment rate",
    :effective_Rt          => "effective reproduction number"
)

function OverviewPlottingRecipe(data::National.Data, generated_posterior)
    ks = keys(generated_posterior)
    recipes = Vector{National.PlottingRecipe}()
    labels  = Vector{String}()
    for (k,recipe) in posterior2recipe
        if k in ks
            label = posterior2label[k]
            r     = recipe(data, generated_posterior, label)
            push!(recipes, r)
            push!(labels, label)
        end
    end
    OverviewPlottingRecipe(recipes, labels)
end

function Plots.plot(r::OverviewPlottingRecipe)
    plots  = Vector{Plots.Plot}()
    nplots = length(r.recipes)
    for (recipe, title) in zip(r.recipes, r.titles)
        p = plot(; xaxis = true, legend = :outertopright, title)
        plot!(p, recipe)
        push!(plots, p)
    end
    plot(plots..., layout=(nplots,1), size=(1500, nplots*250), sharex=true, link=:x)
end

Plots.plot(data::National.Data, generated_posterior) =
    plot( OverviewPlottingRecipe(data, generated_posterior) )
# Plots.plot(data::National.Data, generated_posterior) = plot!(plot(), data, generated_posterior)















## ============================================================================
# out dated
function plot_results(data, expected_daily_cases, expected_daily_deaths, expected_daily_hospit, Rt, cumulative_cases, iar)
    l = @layout [a; b; c; d; e; f]


    xlims   = (data.dates[1], data.dates[end])
    # p1 = plot(; xaxis = false, xlims, legend = :outertopright)
    # plot_death_counts!(p1, data)
    p2 = plot(; xaxis = false, xlims, legend = :outertopright)
    plot_deaths_posterior!(p2, data, expected_daily_deaths)
    p3 = plot(; xaxis = false, xlims, legend = :outertopright)
    plot_cases_posterior!(p3, data, expected_daily_cases)
    p4 = plot(; xaxis = false, xlims, legend = :outertopright)
    plot_hospit_posterior!(p4, data, expected_daily_hospit)
    # plot_cumdeaths_posterior!(p5, data, edeaths)
    p5 = plot(; xaxis = true, xlims, legend = :outertopright)
    plot_cumcases_posterior!(p5, data, cumulative_cases)
    p6 = plot(; xaxis = false, xlims, legend = :outertopright)
    plot_Rt_posterior!(p6, data, Rt)
    p7 = plot(; xaxis = false, xlims, legend = :outertopright)
    plot_iar_posterior!(p7, data, iar)
    tickdates = data.dates[1]:Month(1):data.dates[end]
    p = plot(p3, p4, p2, p5, p6, p7, layout=l, size=(1500, 1000), sharex=true, link=:x, xticks = tickdates, legend=false)

    #dates = data.dates[1:20:end] .|> Symbol .|> String
    #Plots.xticks!(1:20:ntot, dates, xrotation=45)
    p
end


function plot_death_counts!(p, data)
    title!(p, "death counts")
    cases_start = data.num_observations - data.turing_data.cases_start
    tstart = data.dates[data.turing_data.epidemic_start]
    tNobs  = data.dates[data.num_observations]
    tcases = data.dates[cases_start]
    plot!(p, data.dates, data.deaths, lc=:match, c=:midnightblue, label=nothing, lw=4) #, label="Observed daily deaths"
    vline!(p, [tstart], label="epidemic start", linewidth=2, lc=:green, hover="$(tstart)")
    vline!(p, [tNobs], label="end of observations", linewidth=2, lc=:black, hover="$(tNobs)")
    vline!(p, [tcases], label="start with case observations", linewidth=2, lc=:black, ls=:dash, hover="$(tcases)")
    # xlims!(p, data.dates[1], data.dates[end])
end

function plot_deaths_posterior!(p, data, expected_daily_deaths)
    edeaths = hcat(expected_daily_deaths...)
    tNobs  = data.dates[data.num_observations]
    title!(p, "posterior death counts")
    plot!(p, data.dates, data.deaths, alpha=0.5, lc=:match, c=:midnightblue, label=nothing, lw=4) #, label="Recorded daily deaths (observed)"
    vline!(p, [tNobs], label="end of observations", linewidth=2, lc=:black, hover="$(tNobs)")
    plot_confidence_timeseries!(p, data.dates, edeaths; label = "Expected daily deaths") #Dict(hover=>strdates)
end

function plot_hospit_posterior!(p, data, expected_daily_hospit)
    ehospit  = hcat(expected_daily_hospit...)
    tNobs  = data.dates[data.num_observations]
    title!(p, "posterior hospitalization counts")
    plot!(p, data.dates, data.hospit, alpha=0.5, lc=:match, label=nothing, c=:midnightblue, lw=4) #, label="Recorded daily hospitalizations (observed)"
    vline!(p, [tNobs], label="end of observations", linewidth=2, lc=:black, hover="$(tNobs)")
    plot_confidence_timeseries!(p, data.dates, ehospit; label = "Expected daily hospitalizations") #Dict(hover=>strdates)
end

function plot_Rt_posterior!(p, data, Rt)
    eRt    = hcat(Rt...)
    tNobs  = data.dates[data.num_observations]
    title!(p, "R_t")
    plot_confidence_timeseries!(p, data.dates, eRt; no_label = true)
    qs = [quantile(v, [0.025, 0.975]) for v in eachrow(eRt)]
    lq, hq = (eachrow(hcat(qs...))..., )
    ylims!(p, 0, maximum(hq) + 0.1)
    vline!(p, [tNobs], label="end of observations", linewidth=2, lc=:black, hover="$(tNobs)")
end

function plot_iar_posterior!(p, data, iar)
    iar    = hcat(iar...)
    size(iar, 1) == 1 && (iar = repeat(iar, outer=( length(data.dates), 1)) )
    tNobs  = data.dates[data.num_observations]
    title!(p, "infection ascertainment rate")
    plot_confidence_timeseries!(p, data.dates, iar; no_label = true)
    qs = [quantile(v, [0.025, 0.975]) for v in eachrow(iar)]
    lq, hq = (eachrow(hcat(qs...))..., )
    ylims!(p, 0, maximum(hq) + 0.1)
    vline!(p, [tNobs], label="end of observations", linewidth=2, lc=:black, hover="$(tNobs)")
end

function plot_cases_posterior!(p, data, expected_daily_cases)
    ecases  = hcat(expected_daily_cases...)
    cases_start = data.num_observations - data.turing_data.num_case_obs
    tcases = data.dates[cases_start]
    tNobs  = data.dates[data.num_observations]
    title!(p, "posterior case counts")
    plot!(p, data.dates, data.cases, alpha=0.5, lc=:match, label=nothing, c=:midnightblue, lw=4)
    vline!(p, [tcases], label="start with case observations", linewidth=2, lc=:black, ls=:dash, hover="$(tcases)")
    vline!(p, [tNobs], label="end of observations", linewidth=2, lc=:black, hover="$(tNobs)")
    # plot!(p, [d1, d2, d2, d1], [low, low, up, up], fc=:orange, fα=0.4, leg=nothing, hover="$d1 - $d2", seriestype=:shape, lc=:orange)
    plot_confidence_timeseries!(p,data.dates, ecases; label = "observed daily cases")
    # plot_confidence_timeseries!(p,data.dates, ecases; label = "expected daily cases")
end

function plot_cumdeaths_posterior!(p, data, expected_daily_deaths)
    edeaths = hcat(expected_daily_deaths...)
    tNobs  = data.dates[data.num_observations]
    title!(p, "posterior cumulative death counts")
    vline!(p, [tNobs], label="end of observations", linewidth=2, lc=:black, hover="$(tNobs)")
    plot_confidence_timeseries!(p, data.dates, cumsum(edeaths; dims=1); label = "Expected cumulative deaths")
    Plots.plot!(p, data.dates, cumsum(data.deaths; dims=1), label="Recorded cumulative deaths", color=:red)
end

function plot_cumcases_posterior!(p, data, cumulative_cases)
    ecumul  = hcat(cumulative_cases...)
    tNobs  = data.dates[data.num_observations]
    title!(p, "posterior cumulative cases counts")
    vline!(p, [tNobs], label="end of observations", linewidth=2, lc=:black, hover="$(tNobs)")
    plot_confidence_timeseries!(p, data.dates, ecumul ./ data.pop .* 100; label = "Expected cumulative cases")
    # plot_confidence_timeseries!(p, data.dates, cumsum(ecases, dims=1) ./ data.pop .* 100; label = "test")
    Plots.plot!(p, data.dates, cumsum(data.cases; dims=1) ./ data.pop .* 100, label="Recorded cumulative cases", color=:red)
    plot_seroprev!(p, data)

    # plot_seroprev!(p, Date("2020-06-09"), Date("2020-06-10"), 1.2, .7, 1.7)
    # plot_seroprev!(p, Date("2020")+Week(34), Date("2020")+Week(36), 2.2, 1.8, 2.6)
    # plot_seroprev!(p, Date("2020")+Week(38), Date("2020")+Week(51), 3.9, 3.3, 4.6)
    #TODO include seroprevalence study: 34 (1.2%) positive with 95% CI: 0,7%-1,7% in week 20-21
    #TODO include seroprevalence study: 2.2% positive with 95% CI: 1,8% - 2,6% in week 34-36
    #TODO include seroprevalence study: 3.9% positive with 95% CI: 3,3% - 4,6% in week 38-51
end

function plot_seroprev!(p, data)
    seroprev = data.seroprev
    le = [s.country-s.CI[1] for s in eachrow(seroprev)] * 100
    ue = [s.CI[2]-s.country for s in eachrow(seroprev)] * 100
    scatter!(p, seroprev.date, seroprev.country * 100, yerror=(le,ue), hover=["$d" for d in seroprev.date])
end

function plot_seroprev!(p, d1, d2, med, up, low)
    # rec = [(d1, low), (d1, low), (d2, up), (d2, up)]
    plot!(p, [d1, d2, d2, d1], [low, low, up, up], fc=:orange, fα=0.4, leg=nothing, hover="$d1 - $d2", seriestype=:shape, lc=:orange)
    plot!(p, [d1,d2], [med, med], lc=:orange, leg=nothing, lw=4, hover="$med")
end
