using DrWatson
quickactivate(@__DIR__)
@show projectdir()
##
using CovidSurvey
using CSV
using DataFrames
using Dates
# using ArviZ
using PrettyTables
import PyPlot as plt
# using RCall
using Turing
using StatsPlots
using Plots.PlotMeasures
using Random
using BSON
pgfplotsx()
## ============================================================================
# plotting utilities

function plot_effect_size(pp)
    c = group(pp.chain, "grouped_effect")
    effects = DataFrame(c)
    effects = select(effects, Not(["iteration","chain"]))
    rename!(effects, Dict(zip(Symbol.(names(effects)), Symbol.(pp.data.predictors)))...)

    p = plot(xlabel="effect size", legend=:outertopright)
    for (i,(v,n)) in enumerate(zip(eachcol(effects), names(effects)))
        label1 = i == 1 ? "95% HPDI" : nothing
        label2 = i == 1 ? "median" : nothing
        l,u = hpdi(v; alpha=0.05)
        m = median(v)
        plot!([l,u],[n,n], lw=2, c=:midnightblue, lab=label1)
        scatter!([m], [n], mc=:white, lab=label2, msw=2, lc=:black, ms=5)
    end
    return p
end

function last_latent_Rt(region, chain)
    i = Ns[region]
    c = group(chain, "latent_Rts")
    n = Symbol("latent_Rts[$region][$i]")
    j = findfirst(==(n), names(c))
    return Array(c[:,j,:]) |> vec
end

function sampled_effects(effect, region, chain)
    c = group(chain, "effects")
    n = Symbol("effects[$effect,$region]")
    j = findfirst(==(n), names(c))
    return Array(c[:,j,:]) |> vec
end

function sampled_effects(effect, chain)
    c = group(chain, "grouped_effect")
    n = Symbol("grouped_effect[$effect]")
    j = findfirst(==(n), names(c))
    return Array(c[:,j,:]) |> vec
end

# function sampled_effects(region, chain)
#     c = group(chain, "effects")
#     n = Symbol("effects[1,$region]")
#     j = findfirst(==(n), names(c))
#     return Array(c[:,j,:]) |> vec
# end

function relative_change_Rt(x, region, effect, pp)
    # r  = last_latent_Rt(region, pp.chain)
    # R0 = f.(r)
    α    = sampled_effects(effect, region, pp.chain)
    invf = pp.data.turing_data.invlink
    R0   = 1
    r    = invf(1.)
    f    = pp.data.turing_data.link
    RT   = f.(r .+ α * x)
    return @. ( RT / R0 - R0 ) * 100
end

function relative_change_Rt(x, effect, pp)
    # r  = last_latent_Rt(region, pp.chain)
    # R0 = f.(r)
    α    = sampled_effects(effect, pp.chain)
    invf = pp.data.turing_data.invlink
    f    = pp.data.turing_data.link
    R0   = 1
    r    = invf(1.)
    RT   = f.(r .+ α * x)
    return @. ( RT / R0 - R0 ) * 100
end

function plot_Rt_change(x, region, pp)
    predictors = pp.data.predictors
    chain = pp.chain
    title = Regional.regions[region]
    xlabel = region == 5 ? "reduction in Rt [\\%]" : ""
    legend = region == 1 ? :outertopright : nothing
    bottom_margin = region == 5 ? 0mm : 3mm
    top_margin = region == 1 ? 0mm : 3mm

    p = plot(; xlabel, legend, title, bottom_margin, top_margin)
    for (i,n) in enumerate(predictors)
        label1 = i == 1 ? "95% HPDI" : nothing
        label2 = i == 1 ? "median" : nothing
        v = relative_change_Rt(x, region, i, pp)
        l,u = hpdi(v; alpha=0.05)
        m = median(v)
        plot!([l,u],[n,n], lw=2, c=:midnightblue, lab=label1)
        scatter!([m], [n], mc=:white, lab=label2, msw=2, lc=:black, ms=5)
    end
    return p
end

function plot_Rt_change(x, pp)
    ps = Vector{Plots.Plot}()
    for region in 1:5
        p = plot_Rt_reduction(x, region, pp)
        push!(ps, p)
    end
    plot(ps..., layout=(5,1), size=(500, 1000), link=:x, sharex=true)
end


function plot_Rt_reduction_grouped_predictors(x, predictor, pp)
    predictors = pp.data.predictors
    num_predictors = length(predictors)
    title = predictors[predictor]
    chain = pp.chain
    xlabel = predictor == num_predictors ? "change in Rt [\\%]" : ""
    legend = predictor == 1 ? :outertopright : nothing
    bottom_margin = predictor == num_predictors ? 0mm : 3mm
    top_margin = predictor == 1 ? 0mm : 3mm

    p = plot(; xlabel, legend, title, bottom_margin, top_margin)
    for (i,n) in enumerate(Regional.regions)
        label1 = i == 1 ? "95% HPDI" : nothing
        label2 = i == 1 ? "median" : nothing
        v = relative_change_Rt(x, i, predictor, pp)
        l,u = hpdi(v; alpha=0.05)
        m = median(v)
        plot!([l,u],[n,n], lw=1, c=:midnightblue, lab=label1)
        scatter!([m], [n], mc=:white, lab=label2, msw=2, lc=:black, ms=5)
    end
    n = "grouped"
    v = relative_change_Rt(x, predictor, pp)
    l,u = hpdi(v; alpha=0.05)
    m = median(v)
    plot!([l,u],[n,n], lw=4, c=:midnightblue, lab=nothing)
    scatter!([m], [n], mc=:white, msw=3, lc=:black, ms=8, lab=nothing)
    return p
end

function plot_Rt_change_grouped_predictors(x, pp)
    ps = Vector{Plots.Plot}()
    predictors = pp.data.predictors
    N = length(predictors)
    for (i,n) in enumerate(predictors)
        p = plot_Rt_reduction_grouped_predictors(x, i, pp)
        push!(ps, p)
    end
    plot(ps..., layout=(N,1), size=(500, N*250), link=:x, sharex=true)
end

## =============================================================================
# load data
fname = projectdir("reports/regional_effect_estimation/specific_contacts/2021-08-11/CHAIN_chains=10_model=hospit_observ=2021-02-01_preds=CF,CC,CR,CS_regional=true_rwstep=7_std=false_steps=200_warmup=1500.jls")
# fname = projectdir("reports/regional_effect_estimation/specific_contacts/mixed_model/CHAIN_chains=10_model=hospit_observ=2021-02-01_preds=CF,CC,CR,CS_regional=true_rwstep=7_semipar=false_std=false_steps=200_warmup=1000.jls")
fdir, ps, ignores = Regional.parse_fname(fname)
pp = Regional.PostProcessing2(fdir, ps, ignores, [], fname)
pp = Regional.skip_warmup(pp)
# gp = Regional.generate_posterior(pp)
# fname = projectdir("reports/regional_effect_estimation/specific_contacts/mixed_model/GENERATED-QUANTITIES_chains=10_model=hospit_observ=2021-02-01_preds=CF,CC,CR,CS_regional=true_rwstep=7_semipar=false_std=false_steps=200_warmup=1000.bson")
fname = projectdir("reports/regional_effect_estimation/specific_contacts/2021-08-11/GENERATED-QUANTITIES_chains=10_model=hospit_observ=2021-02-01_preds=CF,CC,CR,CS_regional=true_rwstep=7_std=false_steps=200_warmup=1500.bson")
gp = BSON.load(fname) |> NamedTuple;

## =============================================================================
# plot fit result
default(titlefontsize = 20, legendfontsize = 18, labelfontsize = 18, guidefontsize = 18, tickfontsize = 12, framestyle = :zerolines, yminorgrid = true)
# dir = projectdir("figures/")

dir = projectdir("reports/regional_effect_estimation/specific_contacts/mixed_model/")
r = Regional.RegionsOverviewPlottingRecipe(
    pp.data,
    gp,
    :expected_daily_hospits
)
p = plot(r; plot_only_fit=true)
savefig(p, fdir*"overview_hospitalizations.png")
# savefig(p, fdir*"overview_hospitalizations.svg")

r = Regional.RegionsOverviewPlottingRecipe(
    pp.data,
    gp,
    :Rts
)
p = plot(r; plot_only_fit=true)
savefig(p, fdir*"overview_reproduction_number.png")
# savefig(p, fdir*"overview_reproduction_number.svg")

for region in Regional.regions
    r = Regional.RegionPlottingRecipe(pp.data, gp, region)
    p = plot(r; plot_only_fit=true)
    savefig(p, fdir*"overview_region=$(region).png")
    # savefig(p, fdir*"figures/overview_region=$(region).svg")
end

## ===========================================================================

p = plot_effect_size(pp)
savefig(p, fdir*"effect_sizes.png")
savefig(p, fdir*"effect_sizes.svg")

x = -0.5
# p = plot_Rt_change(x, pp)
p = plot_Rt_change_grouped_predictors(x, pp)
savefig(p, fdir*"reproduction_number_half_contacts.png")
savefig(p, fdir*"reproduction_number_half_contacts.svg")

x = 2.
# p = plot_Rt_change(x, pp)
p = plot_Rt_change_grouped_predictors(x, pp)
savefig(p, fdir*"reproduction_number_double_contacts.png")
savefig(p, fdir*"reproduction_number_double_contacts.svg")

x = -0.5
d = []
for (i,p) in enumerate(pp.data.predictors)
    for (j,r) in enumerate(Regional.regions)
        v = relative_change_Rt(x, j, i, pp)
        l,u = hpdi(v; alpha=0.05)
        m = median(v)
        push!(d, [m, l, u, p, r])
    end
    r = "grouped"
    v = relative_change_Rt(x, i, pp)
    l,u = hpdi(v; alpha=0.05)
    m = median(v)
    push!(d, [m, l, u, p, r])
end
results = DataFrame(
    (median=m, lower=l, upper=u, predictor=p, region=r)
    for (m, l, u, p, r) in d
)
CSV.write(projectdir("reports/HOPE/effect_sizes_half_contacts.csv"), results)
##
default()
p = let
    c = group(pp.chain, "grouped_effect")
    df = DataFrame(c)
    df = select(df, Not(["iteration","chain"]))
    rename!(df, Dict(zip(Symbol.(names(df)), Symbol.(pp.data.predictors)))...)

    xs = df.colleagues
    p1 = histogram(xs, normalize=true, title="contacts to colleagues", label=nothing, fc=:gray, lw=.1, xlabel="effect size", bottom_margin=3mm)
    plot!(Exponential(0.2), xlims=(0,1), lw=2, label="prior", legend=:topright);
    vline!([mean(xs)], lw=2, label="mean = $(round(mean(xs), sigdigits=2))", c=:black, ls=:dash)
    xs = df.strangers
    p2 = histogram(xs, normalize=true, title="contacts to strangers", label=nothing, fc=:gray, lw=.1, xlabel="effect size", bottom_margin=3mm);
    plot!(Exponential(0.2), xlims=(0,1), lw=2, label="prior", legend=:topright);
    vline!([mean(xs)], lw=2, label="mean = $(round(mean(xs), sigdigits=2))", c=:black, ls=:dash)
    xs = df.family
    p3 = histogram(xs, normalize=true, title="contacts to family", label=nothing, fc=:gray, lw=.1, xlabel="effect size", top_margin=3mm);
    plot!(Exponential(0.2), xlims=(0,1), lw=2, label="prior", legend=:topright);
    vline!([mean(xs)], lw=2, label="mean = $(round(mean(xs), sigdigits=2))", c=:black, ls=:dash)
    xs = df.friends
    p4 = histogram(xs, normalize=true, title="contacts to friends", label=nothing, fc=:gray, lw=.1, xlabel="effect size", top_margin=3mm);
    plot!(Exponential(0.2), xlims=(0,1), lw=2, label="prior", legend=:topright);
    vline!([mean(xs)], lw=2, label="mean = $(round(mean(xs), sigdigits=2))", c=:black, ls=:dash)
    plot(p1,p2,p3,p4, layout=(2,2))
end

savefig(p, projectdir("figures/prior_vs_posterior_effect.png"))
## ==========================================================================
#                         TOTAL CONTACTS
#  ==========================================================================
# load data
fname = projectdir("reports/regional_effect_estimation/total_contacts/2021-08-10/CHAIN_chains=10_model=hospit_observ=2021-02-01_preds=CT_regional=true_rwstep=7_std=false_steps=200_warmup=1000.jls")
fdir, ps, ignores = Regional.parse_fname(fname)
pp = Regional.PostProcessing2(fdir, ps, ignores, [], fname)
pp = Regional.skip_warmup(pp)

fname = projectdir("reports/regional_effect_estimation/total_contacts/2021-08-10/GENERATED-QUANTITIES_chains=10_model=hospit_observ=2021-02-01_preds=CT_regional=true_rwstep=7_std=false_steps=200_warmup=1000.bson")
gp = BSON.load(fname) |> NamedTuple;
## =============================================================================
# plot fit result
default(titlefontsize = 20, legendfontsize = 18, labelfontsize = 18, guidefontsize = 18, tickfontsize = 12, framestyle = :zerolines, yminorgrid = true)

r = Regional.RegionsOverviewPlottingRecipe(
    pp.data,
    gp,
    :expected_daily_hospits
)
p = plot(r; plot_only_fit=true)
savefig(p, projectdir("figures/overview_hospitalizations_TOTAL.png"))
savefig(p, projectdir("figures/overview_hospitalizations_TOTAL.svg"))

r = Regional.RegionsOverviewPlottingRecipe(
    pp.data,
    gp,
    :Rts
)
p = plot(r; plot_only_fit=true)
savefig(p, projectdir("figures/overview_reproduction_number_TOTAL.png"))
savefig(p, projectdir("figures/overview_reproduction_number_TOTAL.svg"))

for region in Regional.regions
    r = Regional.RegionPlottingRecipe(pp.data, gp, region)
    p = plot(r; plot_only_fit=true)
    savefig(p, projectdir("figures/overview_region=$(region)_TOTAL.png"))
end

## ============================================================================
# plot effects

x = -0.5
p = plot_Rt_change_grouped_predictors(x, pp)
savefig(p, projectdir("figures/reproduction_number_half_contacts_TOTAL.png"))
savefig(p, projectdir("figures/reproduction_number_half_contacts_TOTAL.svg"))

x = 2.
p = plot_Rt_change_grouped_predictors(x, pp)
savefig(p, projectdir("figures/reproduction_number_double_contacts_TOTAL.png"))
savefig(p, projectdir("figures/reproduction_number_double_contacts_TOTAL.svg"))

x = 2#-0.5
d = []
for (i,p) in enumerate(pp.data.predictors)
    for (j,r) in enumerate(Regional.regions)
        v = relative_change_Rt(x, j, i, pp)
        l,u = hpdi(v; alpha=0.05)
        m = median(v)
        push!(d, [m, l, u, p, r])
    end
    r = "grouped"
    v = relative_change_Rt(x, i, pp)
    l,u = hpdi(v; alpha=0.05)
    m = median(v)
    push!(d, [m, l, u, p, r])
end
results = DataFrame(
    (median=m, lower=l, upper=u, predictor=p, region=r)
    for (m, l, u, p, r) in d
)

# CSV.write(projectdir("reports/HOPE/effect_sizes_half_contacts_TOTAL.csv"), results)
##
default()
p = let
    xs = df.colleagues
    p1 = histogram(xs, normalize=true, title="contacts to colleagues", label=nothing, fc=:gray, lw=.1, xlabel="effect size", bottom_margin=3mm);
    plot!(Exponential(0.2), xlims=(0,1), lw=2, label="prior", legend=:topright);
    vline!([mean(xs)], lw=2, label="mean = $(round(mean(xs), sigdigits=2))", c=:black, ls=:dash)
    xs = df.strangers
    p2 = histogram(xs, normalize=true, title="contacts to strangers", label=nothing, fc=:gray, lw=.1, xlabel="effect size", bottom_margin=3mm);
    plot!(Exponential(0.2), xlims=(0,1), lw=2, label="prior", legend=:topright);
    vline!([mean(xs)], lw=2, label="mean = $(round(mean(xs), sigdigits=2))", c=:black, ls=:dash)
    xs = df.family
    p3 = histogram(xs, normalize=true, title="contacts to family", label=nothing, fc=:gray, lw=.1, xlabel="effect size", top_margin=3mm);
    plot!(Exponential(0.2), xlims=(0,1), lw=2, label="prior", legend=:topright);
    vline!([mean(xs)], lw=2, label="mean = $(round(mean(xs), sigdigits=2))", c=:black, ls=:dash)
    xs = df.friends
    p4 = histogram(xs, normalize=true, title="contacts to friends", label=nothing, fc=:gray, lw=.1, xlabel="effect size", top_margin=3mm);
    plot!(Exponential(0.2), xlims=(0,1), lw=2, label="prior", legend=:topright);
    vline!([mean(xs)], lw=2, label="mean = $(round(mean(xs), sigdigits=2))", c=:black, ls=:dash)
    plot(p1,p2,p3,p4, layout=(2,2))
end

savefig(p, projectdir("figures/prior_vs_posterior_effect.png"))
