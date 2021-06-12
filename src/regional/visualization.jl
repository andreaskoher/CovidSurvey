using OrderedCollections

abstract type PlottingRecipe end

Plots.plot(pr::PlottingRecipe, args...) = plot!(plot(), pr, args...)

# ============================================================================
# hospitalizations for all regions

struct HospitsPlottingRecipe{Tsd, Ted, Tr, To, Te} <: PlottingRecipe
    startdates::Tsd
    enddates  ::Ted
    regions   ::Tr
    observed  ::To
    expected  ::Te
end

function HospitsPlottingRecipe(data::Regional.Data, generated_posterior)
    dates      = data.dates
    samples    = generated_posterior.expected_daily_hospits
    expected   = [(dates = d, values = g) for (d,g) in zip(dates, samples)]
    observed   = [(dates = data.hospit.date, values = data.hospit[:,r]) for r in data.regions]
    startdates = Date.(data.startdates)
    enddates   = Date(data.observations_end)
    HospitsPlottingRecipe( startdates, enddates, data.regions, observed, expected)
end

function Plots.plot(hospit::HospitsPlottingRecipe)
    @unpack expected, observed, regions, startdates, enddates = hospit
    plots  = Vector{Plots.Plot}()
    nplots = length(regions)
    for (e, o, r, s) in zip(expected, observed, regions, startdates)
        recipe = SingleHospitPlottingRecipe(s, enddates, o, e)
        p = plot(; xaxis = true, legend = :outertopright, title="$r")
        plot!(p, recipe)
        push!(plots, p)
    end
    plot(plots..., layout=(nplots,1), size=(1000, nplots*200), sharex=true, link=:x)
end


# =============================================================================
# for a single region

struct SingleHospitPlottingRecipe{Tsd, Ted, To, Te} <: PlottingRecipe
    startdate ::Tsd
    enddate   ::Ted
    observed  ::To
    expected  ::Te
end

function SingleHospitPlottingRecipe(data::Regional.Data, generated_posterior, region)
    i  = region isa Integer ? region : findfirst(==(region), data.regions)
    expected = let
        dates  = data.dates[i]
        values = generated_posterior.expected_daily_hospits[i]
        (; dates, values )
    end
    observed = let
        dates  = data.hospit.date
        values = data.hospit[:,data.regions[i]]
        (; dates, values )
    end
    startdate = Date(data.startdates[i])
    enddate   = Date(data.observations_end)
    SingleHospitPlottingRecipe( startdate, enddate, observed, expected)
end

function Plots.plot!(p::Plots.Plot, hospit::SingleHospitPlottingRecipe)
    o  = hospit.observed
    e  = hospit.expected
    ed = hospit.enddate
    sd = hospit.startdate

    plot!(p, o.dates, o.values, α=0.5, lc=:match, lab="observed hospitalizations", c=:midnightblue, lw=4, ylab="cases")
    vline!(p, [ed], lab="end observations", lw=2, lc=:black, hover="$ed")
    vline!(p, [sd], lab="start observations", lw=2, lc=:black, hover="$sd", ls=:dash)
    plot_confidence_timeseries!(p, e.dates, e.values; label = "expected hospitalizations") #Dict(hover=>strdates)
end

# ============================================================================
# reproduction number for a single region

struct SingleRtPlottingRecipe{Tlo, Ted, Te} <: Regional.PlottingRecipe
    lockdown  ::Tlo
    enddates  ::Ted
    expected  ::Te
end

function SingleRtPlottingRecipe(data::Regional.Data, generated_posterior, region)
    i  = region isa Integer ? region : findfirst(==(region), data.regions)
    dates      = data.dates[i]
    values     = generated_posterior.Rts[i]
    expected   = (; dates, values )
    lockdown   = Date(data.lockdown)
    enddates   = Date(data.observations_end)
    SingleRtPlottingRecipe( lockdown, enddates, expected)
end

function Plots.plot!(p::Plots.Plot, rt::SingleRtPlottingRecipe)
    e  = rt.expected
    ed = rt.enddates
    lo = rt.lockdown

    vline!(p, [ed], lab="end observations", lw=2, lc=:black, hover="$ed")
    vline!(p, [lo], lab="lockdown", lw=2, lc=:black, hover="$lo", ls=:dash)
    plot_confidence_timeseries!(p, e.dates, e.values; label = "reproduction number") #Dict(hover=>strdates)
end

# ============================================================================
# reproduction number for all regions

struct RtsPlottingRecipe{Tlo, Ted, Tre, Te} <: Regional.PlottingRecipe
    lockdown  ::Tlo
    enddates  ::Ted
    regions   ::Tre
    expected  ::Te
end

function RtsPlottingRecipe(data::Regional.Data, generated_posterior)
    dates      = data.dates
    samples    = generated_posterior.Rts
    expected   = [(dates = d, values = g) for (d,g) in zip(dates, samples)]
    lockdown   = Date("2020-03-18") # NOTE use data["lockdown"] instead
    enddates   = Date(data.observations_end) # NOTE use data["observations_end"] instead
    RtsPlottingRecipe( lockdown, enddates, data.regions, expected)
end

function Plots.plot(rts::RtsPlottingRecipe)
    plots  = Vector{Plots.Plot}()
    nplots = length(rts.regions)
    for (expected, region) in zip(rts.expected, rts.regions)
        recipe = SingleRtPlottingRecipe(rts.lockdown, rts.enddates, expected)
        p = plot(; xaxis = true, legend = :outertopright, title="$region")
        plot!(p, recipe)
        push!(plots, p)
    end
    plot(plots..., layout=(nplots,1), size=(1000, nplots*200), sharex=true, link=:x)
end

# ============================================================================
# region plot

struct RegionPlottingRecipe{Tr,Tt,T} <: Regional.PlottingRecipe
    recipes::Tr
    titles ::Tt
    region ::T
end

posterior2recipe = OrderedDict(
    :expected_daily_hospits => Regional.SingleHospitPlottingRecipe,
    :Rts                    => Regional.SingleRtPlottingRecipe
)

posterior2title = OrderedDict(
    :expected_daily_hospits => "daily hospitalizations",
    :Rts                    => "effective reproduction number"
)

function RegionPlottingRecipe(data::Regional.Data, generated_posterior, region)
    ks = keys(generated_posterior)
    recipes = Vector{Regional.PlottingRecipe}()
    titles  = Vector{String}()
    for (k,v) in posterior2recipe
        if k in ks
            recipe = v(data, generated_posterior, region)
            title  = posterior2title[k] * " for the $region"
            push!(recipes, recipe)
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
    plot(plots..., layout=(nplots,1), size=(1000, nplots*250), sharex=true, link=:x)
end
