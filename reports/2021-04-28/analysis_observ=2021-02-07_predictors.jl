using DrWatson
quickactivate(@__DIR__)
@show projectdir()
##
using CovidSurvey
using CSV
using DataFrames
using Dates
# using RCall
using Turing
using StatsPlots
using Plots.PlotMeasures
using Random
## ===========================================================================
# load data and model
# ============================================================================
fdir = projectdir("reports/2021-04-28/")
# fname = "chains_cases=2020-06-01_chains=10_hospit=true_model=contacts2_observ=2020-12-16_preds=CF,CC,CR,CS_seed=30000_sero=true_steps=1000_warmup=1000"
fname = "cases=2020-06-01_chains=10_hospit=true_model=contacts2_observ=2021-02-07_preds=CF,CC,CR,CS_seed=20000_sero=true_steps=2000_warmup=1000"
## read chain
chain = read(fdir*"chains_"*fname*".jls", Chains)
chain = chain[1:10:end,:,[1,2,3,4,6,7,8,9,10]]
diagnostics = gelmandiag(chain)
## read params
_, ps, _ = parse_savename(fname, parsetypes = (Int, Float64, Bool))
!("preds" in keys(ps)) && (ps["preds"] = nothing)
ps = NamedTuple{Tuple(Symbol.(keys(ps)))}(values(ps))
## read data
data = National.load_data(ps.observ, ps.preds |> CovidSurvey.parse_predictors, ps.cases, ps.hospit, ps.sero; update=false)
turing_data = data.turing_data;
## ===========================================================================
# plot results
# ============================================================================
# make predictions
m_pred = National.model_contacts_v2(turing_data..., true; link=KLogistic(3.), invlink=KLogit(3.) )
gq = Turing.generated_quantities(m_pred, chain)
generated_posterior = vectup2tupvec( reshape(gq, length(gq)) );
observed_daily_cases, expected_daily_deaths, expected_daily_hospit, Rt, expected_seropositive, iar = generated_posterior;
#---------------------------------------------------------------------------
# plot posterior predictions
plotlyjs()
p = National.plot_results(data, generated_posterior...)
savefig(p, "/home/and/tmp/figures/POSTERIOR_"*fname*".html")
# savefig(p, projectdir("figures/tmp", fname*".png") )
run(`firefox $("/home/and/tmp/figures/POSTERIOR_"*fname*".html")`, wait=false)
#---------------------------------------------------------------------------
# plot chain statistics
plotlyjs()
n = filter( x->!occursin(r"(?<!α)\[", x), String.(names(chain)))
p = plot(chain[n]);
savefig(p, "/home/and/tmp/figures/CHAINS_"*fname*".html" )
run(`firefox  $("/home/and/tmp/figures/CHAINS_"*fname*".html")`, wait=false)
#---------------------------------------------------------------------------
# plot more chain statistics (meanplot)
using MCMCChains
plotlyjs()
p = meanplot(chain[n]);
savefig(p, projectdir("/home/and/tmp/figures/MEANPLOT_", fname*".html") )
run(`firefox $(projectdir("/home/and/tmp/figures/MEANPLOT_", fname*".html"))`, wait=false)
## ===========================================================================
# evaluate effect size
# ============================================================================
pgfplotsx()
using LaTeXStrings
## posterior predictions
# scalefontsizes(2.)
p = let
    ocases  = hcat(observed_daily_cases...)#hcat([observed_daily_cases[t] for t = 1:data.num_total]...)
    ehospit  = hcat(expected_daily_hospit...)

    tickdates = data.dates[1]:Month(1):data.dates[end]
    p3 = plot(; legend = :outertopright, ylims=(0,7000), bottom_margin=3mm)
    National.plot_cases_posterior!(p3, data, ocases)
    p4 = plot(; legend = :outertopright, ylims=(0,500), top_margin=3mm)
    National.plot_hospit_posterior!(p4, data, ehospit)
    plot(p3, p4, layout=(2,1), sharex=true, link=:x, size=(1000, 700)) #, size=(1500, 1000)
end
# savefig(p, projectdir("figures/", "posterior_check.png") )
#---------------------------------------------------------------------------
# plot posterior effect size distribution vs. prior
pgfplotsx()
p = let
    xs = chain["α[1]"]|>vec
    p1 = histogram(xs, normalize=true, title="contacts to colleagues", label=nothing, fc=:gray, lw=.1, xlabel="effect size", bottom_margin=3mm);
    plot!(Beta(1,100), xlims=(0,0.03), lw=2, label="prior", legend=:topright);
    vline!([mean(xs)], lw=2, label="mean effect = $(round(mean(xs), sigdigits=2))", c=:black, ls=:dash)
    xs = chain["α[2]"]|>vec
    p2 = histogram(xs, normalize=true, title="contacts to strangers", label=nothing, fc=:gray, lw=.1, xlabel="effect size", bottom_margin=3mm);
    plot!(Beta(1,100), xlims=(0,0.03), lw=2, label="prior", legend=:topright);
    vline!([mean(xs)], lw=2, label="mean effect = $(round(mean(xs), sigdigits=2))", c=:black, ls=:dash)
    xs = chain["α[3]"]|>vec
    p3 = histogram(xs, normalize=true, title="contacts to family", label=nothing, fc=:gray, lw=.1, xlabel="effect size", top_margin=3mm);
    plot!(Beta(1,100), xlims=(0,0.03), lw=2, label="prior", legend=:topright);
    vline!([mean(xs)], lw=2, label="mean effect = $(round(mean(xs), sigdigits=2))", c=:black, ls=:dash)
    xs = chain["α[4]"]|>vec
    p4 = histogram(xs, normalize=true, title="contacts to friends", label=nothing, fc=:gray, lw=.1, xlabel="effect size", top_margin=3mm);
    plot!(Beta(1,100), xlims=(0,0.03), lw=2, label="prior", legend=:topright);
    vline!([mean(xs)], lw=2, label="mean effect = $(round(mean(xs), sigdigits=2))", c=:black, ls=:dash)
    plot(p1,p2,p3,p4, layout=(2,2))
end
# savefig(p, projectdir("figures/", "total_effect_analysis.png") )
#-----------------------------------------------------------------------------
# evaluate Rt
ib = findfirst(==(Date("2020-05-15")), data.dates)
ie = findfirst(==(Date(ps.observ)), data.dates)
Rt_array = hcat(Rt...)[ib:ie,:]

qs = [quantile(v, [0.025, 0.25, 0.5, 0.75, 0.975]) for v in eachrow(Rt_array)]
llq, lq, mq, uq, uuq = (eachrow(hcat(qs...))..., )

t = date = data.dates[ib:ie]
plot(t, mq, ribbon=(mq - llq, uuq - mq), linewidth=0, label="95% quantiles")
plot!(t, mq, ribbon=(mq - lq, uq - mq), linewidth=2, label="50% quantiles", hover=mq)
# save(projectdir(fdir, "Rt_"*fname*".csv"), DataFrame((;date, llq, lq, mq, uq, uuq)))
