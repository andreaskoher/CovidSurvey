using OrderedCollections

abstract type PlottingRecipe end

Plots.plot(pr::PlottingRecipe, args...; kwargs...) = plot!(plot(), pr, args...; kwargs...)

# =============================================================================
# for a single region

struct ObservationPlottingRecipe1{Tsd, Ted, To, Te, Tp, Tl} <: PlottingRecipe
    startdate ::Tsd
    enddate   ::Ted
    observed  ::To
    expected  ::Te
    predicted ::Tp
    label     ::Tl
end

struct ObservationWithPredictorsPlottingRecipe1{Tsd, Ted, To, Te, Tp, Tl, Psd} <: PlottingRecipe
    startdate ::Tsd
    enddate   ::Ted
    observed  ::To
    expected  ::Te
    predicted ::Tp
    label     ::Tl
    pstartdate::Psd
end

function expected(data, gp, region, label)
    @assert label in ["cases", "hospitalizations", "deaths"]
    i  = region isa Integer ? region : findfirst(==(region), data.regions)
    dates = data.dates[i]
    if label == "cases"
        values = gp.expected_daily_cases[i]
        # start  = data.turing_data.casemodel.starts[i]
        # stop   = data.turing_data.casemodel.stops[i]
        # dates  =
        return (; dates, values )
    elseif label == "hospitalizations"
        values = gp.expected_daily_hospits[i]
        return (; dates, values )
    else
        values = gp.expected_daily_deaths[i]
        return (; dates, values )
    end
end

function observed(data, region, label)
    @assert label in ["cases", "hospitalizations", "deaths"]
    i = region isa Integer ? region : findfirst(==(region), data.regions)
    r = region isa Integer ? Regional.regions[region] : region
    dates = data.hospit.date
    values = if label == "cases"
        data.cases[:,r]
    elseif label == "hospitalizations"
        data.hospit[:,r]
    else
        data.deaths[:,r]
    end
    return (; dates, values )
end

function predicted(data, gp, region, label)
    @assert label in ["cases", "hospitalizations", "deaths"]
    i  = region isa Integer ? region : findfirst(==(region), data.regions)
    dates = data.dates[i]
    if label == "cases"
        values = gp.predicted_daily_cases[i]
        # start  = data.turing_data.casemodel.starts[i]
        # stop   = data.turing_data.casemodel.stops[i]
        # dates  =
        return (; dates, values )
    elseif label == "hospitalizations"
        values = gp.predicted_daily_hospits[i]
        return (; dates, values )
    else
        values = gp.predicted_daily_deaths[i]
        return (; dates, values )
    end
end

function startdate(data::CovidSurvey.Data, region, label::String)
    @assert label in ["cases", "hospitalizations", "deaths"]
    i = region isa Integer ? region : findfirst(==(region), data.regions)
    s = if label == "cases"
        data.turing_data.casemodel.starts[i]
    elseif label == "hospitalizations"
        data.turing_data.hospitmodel.starts[i]
    else
        data.turing_data.deathmodel.starts[i]
    end
    return data.dates[i][s]
end

function enddate(data::CovidSurvey.Data, region, label::String)
    @assert label in ["cases", "hospitalizations", "deaths"]
    i = region isa Integer ? region : findfirst(==(region), data.regions)
    s = if label == "cases"
        data.turing_data.casemodel.stops[i]
    elseif label == "hospitalizations"
        data.turing_data.hospitmodel.stops[i]
    else
        data.turing_data.deathmodel.stops[i]
    end
    return data.dates[i][s]
end

function ObservationPlottingRecipe1(data::CovidSurvey.Data, gp, region, label)
    e = expected(data, gp, region, label)
    o = observed(data, region, label)
    p = predicted(data, gp, region, label)
    sd = startdate(data, region, label)
    ed = enddate(data, region, label)
    isnothing(data.predictors) && return ObservationPlottingRecipe1( sd, ed, o, e, p, label)
    pstartdate = Date("2020-11-10")
    return ObservationWithPredictorsPlottingRecipe1( sd, ed, o, e, p, label, pstartdate)
end

function Plots.plot!(p::Plots.Plot, r::ObservationPlottingRecipe1; plot_only_fit=false)
    o  = r.observed
    e  = r.expected
    pr = r.predicted
    ed = r.enddate
    sd = r.startdate

    if !plot_only_fit
        vline!(p, [ed], lab="end observations", lw=2, lc=:black, hover="$ed")
        vline!(p, [sd], lab="start observations", lw=2, lc=:black, hover="$sd", ls=:dash)
    end
    plot!(p, o.dates, o.values, α=0.5, lc=:match, lab="observed $(r.label)", c=:midnightblue, lw=4, ylab="cases")
    plot_confidence_timeseries!(p, pr.dates, pr.values; label = "predicted $(r.label)", c=:midnightblue) #Dict(hover=>strdates)
    plot_confidence_timeseries!(p, e.dates, e.values; label = "expected $(r.label)") #Dict(hover=>strdates)
    if plot_only_fit
        xlo = Dates.value.(sd)
        xup = Dates.value.(ed)
        yup = maximum( o.values[o.dates .< ed] ) * 1.1
        xlims!(p, xlo, xup)
        ylims!(p, 0, yup)
    end
    return p
end

function Plots.plot!(p::Plots.Plot, r::ObservationWithPredictorsPlottingRecipe1; plot_only_fit = false)
    o  = r.observed
    e  = r.expected
    pr = r.predicted
    ed = r.enddate
    sd = r.startdate
    psd= r.pstartdate

    if !plot_only_fit
        vline!(p, [ed], lab="end observations", lw=2, lc=:black, hover="$ed")
        vline!(p, [sd], lab="start observations", lw=2, lc=:black, hover="$sd", ls=:dash)
    end
    vspan!(p, [psd, ed], label="fit to survey data", α=0.2, fc=:midnightblue)
    bar!(p, o.dates, o.values, α=0.5, lc=:match, lab="observed $(r.label)", c=:midnightblue, lw=4, ylab="cases")
    plot_confidence_timeseries!(p, pr.dates, pr.values; label = "predicted $(r.label)", c=:royalblue ) #Dict(hover=>strdates)
    plot_confidence_timeseries!(p, e.dates, e.values; label = "expected $(r.label)", c=:peru) #Dict(hover=>strdates)
    if plot_only_fit
        xlo = Dates.value.(sd)
        xup = Dates.value.(ed)
        yup = maximum( o.values[o.dates .< ed] ) * 1.1
        xlims!(p, xlo, xup)
        ylims!(p, 0, yup)
    end
    return p
end

# ============================================================================
# seropos

struct SeroPlottingRecipe1{SD, ED, E, O, L} <: Regional.PlottingRecipe
    startdate  ::SD
    enddate    ::ED
    expecteds  ::E
    observeds  ::O
    label     ::L
end

function SeroPlottingRecipe1(data::CovidSurvey.Data, gp, region, label)
    @unpack dates, means, CIs, delay, populations = data.turing_data.seromodel
    i  = region isa Integer ? region : findfirst(==(region), data.regions)
    sd = first(data.dates[i])# startdate(data, region, "hospitalizations")
    ed = Date(data.observations_end)

    observed   = let
        mean = means[i] * 100
        error = [(m-l*100, u*100-m) for (m,(l,u)) in zip(mean,CIs[i])]
        (; dates, mean, error )
    end
    expected = let
        values = gp.expected_seropos[i]  ./ populations[i] .* 100
        dates  = data.dates[i] + Day(delay)
        (; dates, values)
    end
    SeroPlottingRecipe1( sd, ed, expected, observed, label)
end

function Plots.plot!(p::Plots.Plot, r::SeroPlottingRecipe1; plot_only_fit = false)
    sd= r.startdate
    ed= r.enddate
    e = r.expecteds
    o = r.observeds

    plot_confidence_timeseries!(p, e.dates, e.values; r.label)
    y  = o.mean
    ye = o.error
    scatter!(p, o.dates, y, yerror=ye, hover=["$d: $val" for (d, val) in zip(o.dates, y)], lab="serological survey")
    if plot_only_fit
        xlo = Dates.value.(sd)
        xup = Dates.value.(ed)
        yup = maximum([ maximum( v[e.dates .< ed] ) for v in e.values ]) * 1.1
        xlims!(p, xlo, xup)
        ylims!(p, 0, yup)
    end
    return p
end

# ============================================================================
# reproduction number for a single region

struct RtPlottingRecipe1{Tlo, Tst, Ted, Te, Tl} <: Regional.PlottingRecipe
    lockdown  ::Tlo
    startdate ::Tst
    enddate  ::Ted
    expected  ::Te
    label     ::Tl
end

struct RtWithPredictorsPlottingRecipe1{Tlo, Tst, Ted, Te, Tl, Psd} <: Regional.PlottingRecipe
    lockdown  ::Tlo
    startdate ::Tst
    enddate  ::Ted
    expected  ::Te
    label     ::Tl
    pstartdate::Psd
end

function RtPlottingRecipe1(data::CovidSurvey.Data, generated_posterior, region, label)
    i  = region isa Integer ? region : findfirst(==(region), data.regions)
    dates      = data.dates[i]
    values     = generated_posterior.Rts[i]
    expected   = (; dates, values )
    lo = Date(data.lockdown)
    sd = first(data.dates[i])
    ed = Date(data.observations_end)

    isnothing(data.predictors) && return RtPlottingRecipe1( lo, sd, ed, expected, label)
    psd = Date("2020-11-10")
    return RtWithPredictorsPlottingRecipe1( lo, sd, ed, expected, label, psd)
end

function Plots.plot!(p::Plots.Plot, r::RtPlottingRecipe1; plot_only_fit = false)
    e  = r.expected
    sd = r.startdate
    ed = r.enddate
    lo = r.lockdown

    if !plot_only_fit
        vline!(p, [ed], lab="end observations", lw=2, lc=:black, hover="$ed")
        vline!(p, [lo], lab="lockdown", lw=2, lc=:black, hover="$lo", ls=:dash)
    end
    plot_confidence_timeseries!(p, e.dates, e.values; label = r.label) #Dict(hover=>strdates)
    if plot_only_fit
        xlo = Dates.value.(sd)
        xup = Dates.value.(ed)
        # yup = maximum( o.values[o.dates .< ed] ) * 1.1
        xlims!(p, xlo, xup)
        # ylims!(p, 0, yup)
    end
    return p
end

function Plots.plot!(p::Plots.Plot, r::RtWithPredictorsPlottingRecipe1; plot_only_fit = false)
    e  = r.expected
    sd = r.startdate
    ed = r.enddate
    lo = r.lockdown
    psd= r.pstartdate

    if !plot_only_fit
        vline!(p, [ed], lab="end observations", lw=2, lc=:black, hover="$ed")
        vline!(p, [lo], lab="lockdown", lw=2, lc=:black, hover="$lo", ls=:dash)
    end
    vspan!(p, [psd, ed], label="fit to survey data", α=0.2, fc=:midnightblue)
    plot_confidence_timeseries!(p, e.dates, e.values; label = r.label) #Dict(hover=>strdates)
    if plot_only_fit
        xlo = Dates.value.(sd)
        xup = Dates.value.(ed)
        # yup = maximum( e.values[e.dates .< ed] ) * 1.1
        xlims!(p, xlo, xup)
        # ylims!(p, 0, yup)
    end
    return p
end
# ============================================================================
# region plot

struct RegionPlottingRecipe{Tr,Tt,T} <: Regional.PlottingRecipe
    recipes::Tr
    titles ::Tt
    region ::T
end

posterior2recipe = OrderedDict(
    :expected_daily_cases   => ObservationPlottingRecipe1,
    :expected_daily_hospits => ObservationPlottingRecipe1,
    :expected_daily_deaths  => ObservationPlottingRecipe1,
    :Rts                    => RtPlottingRecipe1,
    # :iar                    => IARPlottingRecipe,
    # :effective_Rt           => RtPlottingRecipe1,
    :expected_seropos       => SeroPlottingRecipe1,
)

posterior2label = OrderedDict(
    :expected_daily_cases   => "cases",
    :expected_daily_hospits => "hospitalizations",
    :expected_daily_deaths  => "deaths",
    :Rts                    => "reproduction number",
    # :iar                   => "infections ascertainment rate",
    # :effective_Rt          => "effective reproduction number",
    :expected_seropos      => "total infected"
)

function RegionPlottingRecipe(data::CovidSurvey.Data, generated_posterior, region)
    ks = keys(generated_posterior)
    recipes = Vector{PlottingRecipe}()
    titles  = Vector{String}()
    for (k,recipe) in posterior2recipe
        if k in ks
            label = posterior2label[k]
            r     = recipe(data, generated_posterior, region, label)
            title = label * " for the $region"
            push!(recipes, r)
            push!(titles, title)
        end
    end
    RegionPlottingRecipe(recipes, titles, region)
end

function Plots.plot(r::RegionPlottingRecipe; kwargs...)
    plots  = Vector{Plots.Plot}()
    nplots = length(r.recipes)
    for (recipe, title) in zip(r.recipes, r.titles)
        p = plot(; xaxis = true, legend = :outertopright, title)
        plot!(p, recipe; kwargs...)
        push!(plots, p)
    end
    plot(plots..., layout=(nplots,1), size=(1000, nplots*250), sharex=true, link=:x)
end

# Plots.plot(data::CovidSurvey.Data, generated_posterior) =
#     plot( RegionPlottingRecipe(data, generated_posterior) )
# Plots.plot(data::CovidSurvey.Data, generated_posterior) = plot!(plot(), data, generated_posterior)

# ============================================================================
# reproduction number for all regions

struct RtsPlottingRecipe{Tlo, Tsd, Ted, Tre, Te} <: Regional.PlottingRecipe
    lockdown  ::Tlo
    startdates::Tsd
    enddate   ::Ted
    regions   ::Tre
    expecteds ::Te
end

function RtsPlottingRecipe(data::CovidSurvey.Data, generated_posterior)
    dates      = data.dates
    samples    = generated_posterior.Rts
    expecteds  = [(dates = d, values = g) for (d,g) in zip(dates, samples)]
    lockdown   = Date(data.lockdown) # NOTE use data["lockdown"] instead
    startdates = first.(data.dates)
    enddate    = Date(data.observations_end) # NOTE use data["observations_end"] instead
    RtsPlottingRecipe( lockdown, startdates, enddate, data.regions, expecteds)
end

function Plots.plot(r::RtsPlottingRecipe; kwargs...)
    plots  = Vector{Plots.Plot}()
    nplots = length(r.regions)
    for (expected, region, startdate) in zip(r.expecteds, r.regions, r.startdates)
        recipe = RtPlottingRecipe1(r.lockdown, startdate, r.enddate, expected, "reproduction number")
        p = plot(; xaxis = true, legend = :outertopright, title="$region")
        plot!(p, recipe; kwargs...)
        push!(plots, p)
    end
    plot(plots..., layout=(nplots,1), size=(1000, nplots*200), sharex=true, link=:x)
end

# ============================================================================
# observations for all regions

struct RegionsOverviewPlottingRecipe{Tr,Tt,T} <: Regional.PlottingRecipe
    recipes::Tr
    titles ::Tt
    region ::T
end


function RegionsOverviewPlottingRecipe(data::CovidSurvey.Data, generated_posterior, posterior)
    # i  = region isa Integer ? region : findfirst(==(region), data.regions)
    recipes = Vector{PlottingRecipe}()
    titles  = Vector{String}()
    for (i,region) in enumerate(data.regions)
        label  = posterior2label[posterior]
        recipe = posterior2recipe[posterior]
        r      = recipe(data, generated_posterior, region, label)
        title  = i == 1 ? label*" for the $region" : "$region"
        push!(recipes, r)
        push!(titles, title)
    end
    return RegionsOverviewPlottingRecipe(recipes, titles, data.regions)
end

function Plots.plot(r::RegionsOverviewPlottingRecipe; kwargs...)
    # plots  = Vector{Plots.Plot}()
    nplots = length(r.recipes)

    title = r.titles[1]
    p = plot(; xaxis = true, legend = :outertopright, title)
    plot!(p, r.recipes[1]; kwargs...)
    plots = [p]

    for (recipe, title) in Iterators.drop(zip(r.recipes, r.titles),1)
        p = plot(; xaxis = true, legend = nothing, title)
        plot!(p, recipe; kwargs...)
        push!(plots, p)
    end
    plot(plots..., layout=(nplots,1), size=(1000, nplots*250), sharex=true, link=:x)
end






# # ============================================================================
# # reproduction number for a single region
#
# struct SingleRtPlottingRecipe1{Tlo, Ted, Te} <: Regional.PlottingRecipe
#     lockdown  ::Tlo
#     enddates  ::Ted
#     expected  ::Te
# end
#
# function SingleRtPlottingRecipe1(data::CovidSurvey.Data, generated_posterior, region)
#     i  = region isa Integer ? region : findfirst(==(region), data.regions)
#     dates      = data.dates[i]
#     values     = generated_posterior.Rts[i]
#     expected   = (; dates, values )
#     lockdown   = Date(data.lockdown)
#     enddates   = Date(data.observations_end)
#     SingleRtPlottingRecipe1( lockdown, enddates, expected)
# end
#
# function Plots.plot!(p::Plots.Plot, rt::SingleRtPlottingRecipe1)
#     e  = rt.expected
#     ed = rt.enddates
#     lo = rt.lockdown
#
#     vline!(p, [ed], lab="end observations", lw=2, lc=:black, hover="$ed")
#     vline!(p, [lo], lab="lockdown", lw=2, lc=:black, hover="$lo", ls=:dash)
#     plot_confidence_timeseries!(p, e.dates, e.values; label = "reproduction number") #Dict(hover=>strdates)
# end
#
# Plots.plot(pr::SingleRtPlottingRecipe1) = plot!(plot(), pr)

# # ============================================================================
# # region plot
#
# struct RegionPlottingRecipe{Tr,Tt,T} <: Regional.PlottingRecipe
#     recipes::Tr
#     titles ::Tt
#     region ::T
# end
#
# posterior2recipe = OrderedDict(
#     :expected_daily_hospits => Regional.SingleHospitPlottingRecipe,
#     :expected_daily_deaths  => Regional.SingleDeathsPlottingRecipe,
#     :Rts                    => Regional.SingleRtPlottingRecipe1
# )
#
# posterior2title = OrderedDict(
#     :expected_daily_hospits => "daily hospitalizations",
#     :expected_daily_deaths  => "daily deaths",
#     :Rts                    => "effective reproduction number"
# )
#
# function RegionPlottingRecipe(data::CovidSurvey.Data, generated_posterior, region)
#     ks = keys(generated_posterior)
#     recipes = Vector{Regional.PlottingRecipe}()
#     titles  = Vector{String}()
#     for (k,v) in posterior2recipe
#         if k in ks
#             recipe = v(data, generated_posterior, region)
#             title  = posterior2title[k] * " for the $region"
#             push!(recipes, recipe)
#             push!(titles, title)
#         end
#     end
#     RegionPlottingRecipe(recipes, titles, region)
# end
#
# function Plots.plot(r::RegionPlottingRecipe)
#     plots  = Vector{Plots.Plot}()
#     nplots = length(r.recipes)
#     for (recipe, title) in zip(r.recipes, r.titles)
#         p = plot(; xaxis = true, legend = :outertopright, title)
#         plot!(p, recipe)
#         push!(plots, p)
#     end
#     plot(plots..., layout=(nplots,1), size=(1000, nplots*250), sharex=true, link=:x)
# end
