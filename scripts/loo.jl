using Turing
using ArviZ
import PyPlot as plt
using PrettyTables
using LinearAlgebra
## ========================================================================
# data
J = 8
y = [28.0, 8.0, -3.0, 7.0, -1.0, 1.0, 18.0, 12.0]
σ = [15.0, 10.0, 16.0, 11.0, 9.0, 11.0, 10.0, 18.0]
schools = [
    "Choate",
    "Deerfield",
    "Phillips Andover",
    "Phillips Exeter",
    "Hotchkiss",
    "Lawrenceville",
    "St. Paul's",
    "Mt. Hermon",
];

nwarmup, nsamples, nchains = 10_000, 1000, 4;

## ============================================================================
# model

Turing.@model function turing_model(y, σ, J=length(y))
    μ ~ Normal(10, 15)
    τ ~ InverseGamma2(5, 1)
    θ ~ filldist(Normal(μ, τ), J)
    for i in 1:J
        y[i] ~ Normal(θ[i], σ[i])
    end
end

param_mod = turing_model(y, σ)
sampler = NUTS(nwarmup, 0.8)

rng = Random.MersenneTwister()#16653
turing_chns = sample(
    rng, param_mod, sampler, MCMCThreads(), nsamples, nchains; progress=false
);

plot(turing_chns)
meanplot(turing_chns)

## ============================================================================
# autocorr
plot_autocorr(turing_chns; var_names=["μ", "τ"]);
plt.gcf()

# remove autocorr
turing_chns = turing_chns[1:20:end]
plot_autocorr(turing_chns; var_names=["μ", "τ"]);
plt.gcf()

##
idata = from_mcmcchains(
    turing_chns;
    #coords=Dict("school" => schools),
    #dims=Dict("y" => ["school"], "σ" => ["school"], "θ" => ["school"]),
    library="Turing",
)
idata.posterior

plot_trace(idata);
plt.gcf()

summarystats(idata) |> pretty_table
## ============================================================================
# analyse divergences
plot_pair(
    idata;
    #coords=Dict("school" => ["Choate", "Deerfield", "Phillips Andover"]),
    divergences=true,
);
plt.gcf()
## ============================================================================
# extract additional info
prior = sample(param_mod, Prior(), nsamples; progress=false)

# Instantiate the predictive model
param_mod_predict = turing_model(similar(y, Missing), σ)
# and then sample!
prior_predictive = predict(param_mod_predict, prior)
posterior_predictive = predict(param_mod_predict, turing_chns)

loglikelihoods = Turing.pointwise_loglikelihoods(
    param_mod, MCMCChains.get_sections(turing_chns, :parameters)
)

## ============================================================================
# add to ArviZ
# Ensure the ordering of the loglikelihoods matches the ordering of `posterior_predictive`
ynames = string.(keys(posterior_predictive))
loglikelihoods_vals = getindex.(Ref(loglikelihoods), ynames)
# Reshape into `(nchains, nsamples, size(y)...)`
loglikelihoods_arr = permutedims(cat(loglikelihoods_vals...; dims=3), (2, 1, 3))

idata = from_mcmcchains(
    turing_chns;
    posterior_predictive=posterior_predictive,
    log_likelihood=Dict("y" => loglikelihoods_arr),
    prior=prior,
    prior_predictive=prior_predictive,
    observed_data=Dict("y" => y),
    coords=Dict("school" => schools),
    #dims=Dict("y" => ["school"], "σ" => ["school"], "θ" => ["school"]),
    library="Turing",
)

loo(idata)
