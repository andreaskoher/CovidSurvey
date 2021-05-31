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
# load old model without predictors and use posterior samples as new priors
# ============================================================================
# fdir = projectdir("reports/2021-04-27/")
# fname = "cases=2020-06-01_chains=10_hospit=true_model=v2_observ=2020-12-16_seed=10000_sero=true_steps=2000_warmup=1000"
fdir = projectdir("reports/2021-04-28/")
fname = "cases=2020-06-01_chains=10_hospit=true_model=v2_observ=2021-02-07_seed=10000_sero=true_steps=2000_warmup=1000"
## read chain
chain = read(fdir*"chains_"*fname*".jls", Chains)
# chain = chain[1:10:end,:,:]
## read params
_, ps, _ = parse_savename(fname, parsetypes = (Int, Float64, Bool))
!("preds" in keys(ps)) && (ps["preds"] = nothing)
ps = NamedTuple{Tuple(Symbol.(keys(ps)))}(values(ps))
## read data
data = National.load_data(ps.observ, ps.preds |> CovidSurvey.parse_predictors, ps.cases, ps.hospit, ps.sero; update=false)
turing_data = data.turing_data;
#------------------------------------------------------------------------------
# use posterior samples as priors
elementary_priors = (
    :y => Gamma{Float64},
    :ϕ => Gamma{Float64},
    :ϕ2 => Normal{Float64},
    :ϕ3 => Normal{Float64},
    :R0 => Normal{Float64},
    :R1 => Normal{Float64},
    :ifr => Normal{Float64},
    :ihr => Normal{Float64},
    :iar0 => Gamma{Float64},
    :σ_rt => Normal{Float64},
    :σ_iar => Normal{Float64},
);
array_priors = (
    :latent_Rt => Normal{Float64},
    :seroprev_σ => Normal{Float64},
    :latent_iar => Normal{Float64},
);

function fitsamples(chain)
    dfits = Dict()
    for (n, d) in elementary_priors
        dfits[n] = fit(d, vec(chain[n]) )
    end
    for (n, d) in array_priors
        c = group(chain, n)
        ns = names(c)
        @assert issorted( [parse(Int, match(r"\d+" , s).match) for s in String.(ns)] )
        dfits[n] = [fit(d, c[n] |> vec) for n in ns]
    end
    # dfits[:logRt] = dfits[:logRt][1:272] #DEBUG
    # dfits[:logit_iar] = dfits[:logit_iar][1:29] #DEBUG
    dfits
end

priors = fitsamples(chain);
#------------------------------------------------------------------------------
# load prior for Rt
rt = load(projectdir("reports/test_contacts/RtPrior.csv")) |> DataFrame
rt_prior = [Normal(μ, σ) for (μ, σ) in zip(rt.y,rt.u)]
rt.t[end]
ps.observ
priors[:Rt] = rt_prior
## ===========================================================================
# setup new model
# ============================================================================
ps = (
    warmup = 2,
    steps  = 6,
    seed   = 40000,
    observ = "2021-02-07",#"2021-03-25"
    cases  = "2020-06-01",
    model  = "testcontacts",
    preds = "CC,CS,CF,CR",
    hospit = true,
    sero = true,
)

fname = savename("chains", ps, "")
Random.seed!(ps.seed);
data = National.load_data(ps.observ, ps.preds |> CovidSurvey.parse_predictors, ps.cases, ps.hospit, ps.sero; update=false)
turing_data = data.turing_data;
#----------------------------------------------------------------------------
# sample model
m = National.model_contacts_test(turing_data..., (; priors...), false; link=KLogistic(3.), invlink=KLogit(3.))
m()
@time chain = sample(m, NUTS(ps.warmup, 0.95), MCMCThreads(), ps.steps + ps.warmup, 4)
chain = chain[ps.warmup+1:end,:,:]
@info "Saving at: $(projectdir("out", fname))"
safesave(projectdir("out/tmp", fname*".jls"), chain)
#-----------------------------------------------------------------------------
# diagnostics
diagnostics = gelmandiag(chain)
fname_diagnostics = projectdir("out/tmp", fname*"_GELMANDIAG.csv")
safesave(fname_diagnostics, diagnostics)
run(`firefox $(fname_diagnostics)`, wait=false)
#-----------------------------------------------------------------------------
# make predictions
m_pred = model(turing_data..., priors, true)
gq = Turing.generated_quantities(m_pred, chain)
generated_posterior = vectup2tupvec( reshape(gq, length(gq)) );
observed_daily_cases, expected_daily_deaths, expected_daily_hospit, Rt, cumulative_cases, iar = generated_posterior;
#---------------------------------------------------------------------------
# plot posterior predictions
plotlyjs()
p = National.plot_results(data, generated_posterior...)
savefig(p, "/home/and/tmp/figures/"*fname*".html")
# savefig(p, projectdir("figures/tmp", fname*".png") )
run(`firefox $("/home/and/tmp/figures/"*fname*".html")`, wait=false)
#---------------------------------------------------------------------------
# plot chain statistics
plotlyjs()
n = filter( x->!occursin(r"\[", x), String.(names(chain)))
p = plot(chain[n]);
savefig(p, "/home/and/tmp/figures/"*fname*"_CHAINS.html" )
run(`firefox  $("/home/and/tmp/figures/"*fname*"_CHAINS.html")`, wait=false)
#---------------------------------------------------------------------------
# plot more chain statistics (meanplot)
using MCMCChains
plotlyjs()
p = meanplot(chain[n]);
savefig(p, projectdir("/home/and/tmp/figures/", fname*"_MEANPLOT.html") )
run(`firefox $(projectdir("/home/and/tmp/figures/", fname*"_MEANPLOT.html"))`, wait=false)
## ===========================================================================
# evaluation
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
    plot!(Beta(1,100), xlims=(0,0.015), lw=2, label="prior", legend=:topright);
    vline!([mean(xs)], lw=2, label="mean effect = $(round(mean(xs), sigdigits=2))", c=:black, ls=:dash)
    xs = chain["α[2]"]|>vec
    p2 = histogram(xs, normalize=true, title="contacts to strangers", label=nothing, fc=:gray, lw=.1, xlabel="effect size", bottom_margin=3mm);
    plot!(Beta(1,100), xlims=(0,0.015), lw=2, label="prior", legend=:topright);
    vline!([mean(xs)], lw=2, label="mean effect = $(round(mean(xs), sigdigits=2))", c=:black, ls=:dash)
    xs = chain["α[3]"]|>vec
    p3 = histogram(xs, normalize=true, title="contacts to family", label=nothing, fc=:gray, lw=.1, xlabel="effect size", top_margin=3mm);
    plot!(Beta(1,100), xlims=(0,0.015), lw=2, label="prior", legend=:topright);
    vline!([mean(xs)], lw=2, label="mean effect = $(round(mean(xs), sigdigits=2))", c=:black, ls=:dash)
    xs = chain["α[4]"]|>vec
    p4 = histogram(xs, normalize=true, title="contacts to friends", label=nothing, fc=:gray, lw=.1, xlabel="effect size", top_margin=3mm);
    plot!(Beta(1,100), xlims=(0,0.015), lw=2, label="prior", legend=:topright);
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

t = data.dates[ib:ie]
plot(t, mq, ribbon=(mq - llq, uuq - mq), linewidth=0, label="95% quantiles")
plot!(t, mq, ribbon=(mq - lq, uq - mq), linewidth=2, label="50% quantiles", hover=mq)
# save(projectdir("out", "Rt_corrected_with_contacts.csv"), DataFrame((;llq, lq, mq, uq, uuq)))
#------------------------------------------------------------------------------
# Rt_with_vs_without_predictors
pgfplotsx()
Rt_original = load(projectdir("out", "Rt.csv")) |> DataFrame

p = let
    p = plot(legend=:topright)
    plot!(t, mq, ribbon=(mq - llq, uuq - mq), linewidth=2, label="with predictors (contact data)")
    plot!(Rt_original.date, Rt_original.mq, ribbon=(Rt_original.mq - Rt_original.llq, Rt_original.uuq - Rt_original.mq), linewidth=2, label="without predictors")
    xlims!((t[1],t[end]))
end
#------------------------------------------------------------------------------
# Rt_vs_predictors
pgfplotsx()
cov = turing_data.covariates
p = let
    p = plot(legend=:topright)
    plot!(Rt_original.date, Rt_original.mq, ribbon=(Rt_original.mq - Rt_original.llq, Rt_original.uuq - Rt_original.mq), linewidth=2, label="Rt without predictors", c=:black)
    p2 = twinx()
    for (i, n) in enumerate(["colleagues","strangers","family","friends"])
        plot!(p2, t, cov[1:length(t),i], lw=2, label=n, grid=:off, xticks=:none)
    end
    xlims!((t[1],t[end]))
end
# savefig(p, projectdir("figures/", "Rt_vs_predictors.png") )
#------------------------------------------------------------------------------
# compare to SSI data to predictors
Rt_SSI = National.readsurvey("/home/and/data/covidsurvey/contacts.csv")
Rt_SSI = select( Rt_SSI, [:Rt,:uncertainty_lower,:uncertainty_upper, :date])
p = let
    p = plot(legend=:topright)
    plot!(Rt_SSI.date, Rt_SSI.Rt, ribbon=(Rt_SSI.Rt - Rt_SSI.uncertainty_lower, Rt_SSI.uncertainty_upper - Rt_SSI.Rt), linewidth=2, label="Rt (SSI) without predictors", c=:black)
    p2 = twinx()
    for (i, n) in enumerate(["colleagues","strangers","family","friends"])
        plot!(p2, t, cov[1:length(t),i], lw=2, label=n, grid=:off, xticks=:none)
    end
    xlims!((t[1],t[end]))
end
# savefig(p, projectdir("figures/", "Rt_from_SSI_vs_predictors.png") )
#------------------------------------------------------------------------------
# compare to SSI data to inferred Rt
p = let
    p = plot(legend=:topright)
    plot!(Rt_original.date, Rt_original.mq, ribbon=(Rt_original.mq - Rt_original.llq, Rt_original.uuq - Rt_original.mq), linewidth=2, label="Rt without predictors")
    plot!(Rt_SSI.date, Rt_SSI.Rt, ribbon=(Rt_SSI.Rt - Rt_SSI.uncertainty_lower, Rt_SSI.uncertainty_upper - Rt_SSI.Rt), linewidth=2, label="Rt (SSI)", c=:black)
    xlims!((t[1],t[end]))
end
# savefig(p, projectdir("figures/", "Rt_without_predictors_vs_Rt_from_SSI.png") )
#------------------------------------------------------------------------------
# compare to Rt from SSI or model to predictors
survey = National.readsurveys_raw()
p = let
    ps = []
    ts = survey.date
    for (i, n) in [(3,:family), (1,:colleagues), (4,:friends), (2,:strangers)] #"CC,CS,CF,CR"
        p = plot()
        plot!(
         #Rt_SSI.date, Rt_SSI.Rt, ribbon=(Rt_SSI.Rt - Rt_SSI.uncertainty_lower, Rt_SSI.uncertainty_upper - Rt_SSI.Rt),
         Rt_original.date, Rt_original.mq, ribbon=(Rt_original.mq - Rt_original.llq, Rt_original.uuq - Rt_original.mq),
         linewidth=2, label=nothing, c=:black, α=.5, ylabel="\$R_t\$",
         right_margin=20mm, bottom_margin=10mm)
        p2 = twinx()
        xs = survey[:, n]
        x = cov[1:length(t),i]
        National.surveynorm!(xs)
        plot!(p2, ts, xs / std(xs) * std(x), lw=2,
            label="$n (raw)", grid=:off, xticks=:none, c=:darkred,
            ylabel="contacts (normalized)",
            right_margin=20mm, bottom_margin=10mm,
            extra_kwargs=:subplot, legend=:outertop, legend_columns=-1)
        plot!(p2, t, x, lw=4, label="$n (smoothed)",
            grid=:off, xticks=:none, c=:darkorange,
            right_margin=20mm, bottom_margin=10mm,
            extra_kwargs=:subplot, legend=:outertop, legend_columns=-1)
        xlims!((t[1],t[end]))
        push!(ps, p)
    end
    #l = @layout [a{0.2w} b{0.2w}; c d]
    plot(ps..., layout=4, size=(1300, 1000))
end
# savefig(p, projectdir("figures/", "Rt_vs_predictors_smoothed_and_raw.png") )
