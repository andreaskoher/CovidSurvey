using OrderedCollections

abstract type PlottingRecipe end

Plots.plot(pr::PlottingRecipe, args...) = plot!(plot(), pr, args...)

# =============================================================================
# for a single region

struct ObservationPlottingRecipe{Tsd, Ted, To, Te, Tl} <: PlottingRecipe
    startdate ::Tsd
    enddate   ::Ted
    observed  ::To
    expected  ::Te
    label     ::Tl
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

function startdate(data::Regional.Data, region, label::String)
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

function enddate(data::Regional.Data, region, label::String)
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

function ObservationPlottingRecipe(data::Regional.Data, gp, region, label)
    e = expected(data, gp, region, label)
    o = observed(data, region, label)
    sd = startdate(data, region, label)
    ed = enddate(data, region, label)
    return ObservationPlottingRecipe( sd, ed, o, e, label)
end

function Plots.plot!(p::Plots.Plot, r::ObservationPlottingRecipe)
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
# seropos

struct SeroPlottingRecipe{E, O, L} <: Regional.PlottingRecipe
    expecteds  ::E
    observeds  ::O
    label     ::L
end

function SeroPlottingRecipe(data::Regional.Data, gp, region, label)
    @unpack dates, means, CIs, delay, populations = data.turing_data.seromodel
    i  = region isa Integer ? region : findfirst(==(region), data.regions)
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
    SeroPlottingRecipe( expected, observed, label)
end

function Plots.plot!(p::Plots.Plot, r::SeroPlottingRecipe)
    e = r.expecteds
    o = r.observeds

    plot_confidence_timeseries!(p, e.dates, e.values; r.label)
    y  = o.mean
    ye = o.error
    scatter!(p, o.dates, y, yerror=ye, hover=["$d: $val" for (d, val) in zip(o.dates, y)])
end

# ============================================================================
# reproduction number for a single region

struct RtPlottingRecipe{Tlo, Ted, Te, Tl} <: Regional.PlottingRecipe
    lockdown  ::Tlo
    enddates  ::Ted
    expected  ::Te
    label     ::Tl
end

function RtPlottingRecipe(data::Regional.Data, generated_posterior, region, label)
    i  = region isa Integer ? region : findfirst(==(region), data.regions)
    dates      = data.dates[i]
    values     = generated_posterior.Rts[i]
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
    plot_confidence_timeseries!(p, e.dates, e.values; label = r.label) #Dict(hover=>strdates)
end

# ============================================================================
# region plot

struct RegionPlottingRecipe{Tr,Tt,T} <: Regional.PlottingRecipe
    recipes::Tr
    titles ::Tt
    region ::T
end

posterior2recipe = OrderedDict(
    :expected_daily_cases   => ObservationPlottingRecipe,
    :expected_daily_hospits => ObservationPlottingRecipe,
    :expected_daily_deaths  => ObservationPlottingRecipe,
    :Rts                    => RtPlottingRecipe,
    # :iar                    => IARPlottingRecipe,
    # :effective_Rt           => RtPlottingRecipe,
    :expected_seropos       => SeroPlottingRecipe,
)

posterior2label = OrderedDict(
    :expected_daily_cases   => "cases",
    :expected_daily_hospits => "hospitalizations",
    :expected_daily_deaths  => "deaths",
    :Rts                    => "reproduction number",
    # :iar                   => "infections ascertainment rate",
    # :effective_Rt          => "effective reproduction number",
    :expected_seropos      => "sero positive"
)

function RegionPlottingRecipe(data::Data, generated_posterior, region)
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

function Plots.plot(r::RegionPlottingRecipe)
    plots  = Vector{Plots.Plot}()
    nplots = length(r.recipes)
    for (recipe, title) in zip(r.recipes, r.titles)
        p = plot(; xaxis = true, legend = :outertopright, title)
        plot!(p, recipe)
        push!(plots, p)
    end
    plot(plots..., layout=(nplots,1), size=(1500, nplots*250), sharex=true, link=:x)
end

# Plots.plot(data::Data, generated_posterior) =
#     plot( RegionPlottingRecipe(data, generated_posterior) )
# Plots.plot(data::Data, generated_posterior) = plot!(plot(), data, generated_posterior)

# ============================================================================
# reproduction number for all regions

struct RtsPlottingRecipe{Tlo, Ted, Tre, Te} <: Regional.PlottingRecipe
    lockdown  ::Tlo
    enddate   ::Ted
    regions   ::Tre
    expecteds ::Te
end

function RtsPlottingRecipe(data::Regional.Data, generated_posterior)
    dates      = data.dates
    samples    = generated_posterior.Rts
    expecteds  = [(dates = d, values = g) for (d,g) in zip(dates, samples)]
    lockdown   = Date(data.lockdown) # NOTE use data["lockdown"] instead
    enddate    = Date(data.observations_end) # NOTE use data["observations_end"] instead
    RtsPlottingRecipe( lockdown, enddate, data.regions, expecteds)
end

function Plots.plot(r::RtsPlottingRecipe)
    plots  = Vector{Plots.Plot}()
    nplots = length(r.regions)
    for (expected, region) in zip(r.expecteds, r.regions)
        recipe = RtPlottingRecipe(r.lockdown, r.enddate, expected, "reproduction number")
        p = plot(; xaxis = true, legend = :outertopright, title="$region")
        plot!(p, recipe)
        push!(plots, p)
    end
    plot(plots..., layout=(nplots,1), size=(1000, nplots*200), sharex=true, link=:x)
end













# # ============================================================================
# # reproduction number for a single region
#
# struct SingleRtPlottingRecipe{Tlo, Ted, Te} <: Regional.PlottingRecipe
#     lockdown  ::Tlo
#     enddates  ::Ted
#     expected  ::Te
# end
#
# function SingleRtPlottingRecipe(data::Regional.Data, generated_posterior, region)
#     i  = region isa Integer ? region : findfirst(==(region), data.regions)
#     dates      = data.dates[i]
#     values     = generated_posterior.Rts[i]
#     expected   = (; dates, values )
#     lockdown   = Date(data.lockdown)
#     enddates   = Date(data.observations_end)
#     SingleRtPlottingRecipe( lockdown, enddates, expected)
# end
#
# function Plots.plot!(p::Plots.Plot, rt::SingleRtPlottingRecipe)
#     e  = rt.expected
#     ed = rt.enddates
#     lo = rt.lockdown
#
#     vline!(p, [ed], lab="end observations", lw=2, lc=:black, hover="$ed")
#     vline!(p, [lo], lab="lockdown", lw=2, lc=:black, hover="$lo", ls=:dash)
#     plot_confidence_timeseries!(p, e.dates, e.values; label = "reproduction number") #Dict(hover=>strdates)
# end
#
# Plots.plot(pr::SingleRtPlottingRecipe) = plot!(plot(), pr)

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
#     :Rts                    => Regional.SingleRtPlottingRecipe
# )
#
# posterior2title = OrderedDict(
#     :expected_daily_hospits => "daily hospitalizations",
#     :expected_daily_deaths  => "daily deaths",
#     :Rts                    => "effective reproduction number"
# )
#
# function RegionPlottingRecipe(data::Regional.Data, generated_posterior, region)
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
