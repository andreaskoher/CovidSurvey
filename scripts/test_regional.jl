using DrWatson
quickactivate(@__DIR__)
@show projectdir()
## ============================================================================
@info "load packages"
using CovidSurvey
using Random, Dates, Turing
using Base.Threads
using StatsPlots
using PrettyTables
using DataFrames
using BSON

using Memoization
using ReverseDiff
Turing.setrdcache(true)
setadbackend(:reversediff)

plotlyjs()
@info "number of threads available: $(nthreads())"
## ============================================================================

ps = (
    warmup = 100,
    model = "hospit",
    steps  = 100,
    seed   = nothing,
    observ = "2021-02-01",#"2021-02-06",#"2021-03-25"
    chains = 1,
    preds = nothing,#"CT,MNRES",#"CF,CC,CR,CS"
    semipara=false,
    rwstep = 7,
    include= false,
    std = true,
)
## ============================================================================
# load data
@info "load data"

data_params = (
      observationsend  = ps.observ
    , predictors        = ps.preds|> CovidSurvey.parse_predictors
    # , cases_start       = ps.cases
    # , hospitmodel       = Regional.HospitInit1(obs_stop="2020-07-01")
    , casemodel         = Regional.CaseInit1(obs_start="2020-06-01")
    , seromodel         = Regional.SeroInit2(delay=0, std=1.)
    , rwstep            = ps.rwstep
    , epidemicstart    = 20
    , numimpute        = 6
    , include_early_dynamic = ps.include
    , link              = KLogistic(5.)
    , invlink           = KLogit(5.)
    , predictive        = false
    , covariates_kwargs = Dict(
      :semiparametric => false,
      :fname          => [projectdir("data", "mean_contact_rates_region=$(r).csv" ) for r in 1:Regional.nregions],#[projectdir("data", "mean_contact_rates_region=$r.csv" ) for r in 1:Regional.nregions],
      :shift          => -1,
      :startdate      => "2020-11-10", # >= 1000 cases / day
      :datecol        => "dates",
      :aggregation    => ps.rwstep,
      :standartize    => ps.std,
      :mobility       => [projectdir("data/google/mobility_region=$(Regional.regions[i]).csv") for i in 1:Regional.nregions]
      )
    )

data = Regional.load_data(; data_params... )
d = turing_data = data.turing_data;
## ============================================================================
@info "prepare model"

model = Regional.name2model[ps.model]
# model = Regional.model_hospit2
#!isnothing(ps.seed) && Random.seed!(ps.seed);
@info ps
m = model(turing_data, vectorize=true)
m();
## ============================================================================
@info "start sampling"

@time chain = let #460s; 366s
    thinning = 1
    if ps.chains > 1
        sample(m, NUTS(ps.warmup, 0.95; max_depth=5), MCMCThreads(), ps.steps + ps.warmup, ps.chains; progress=true, thinning)
    else
        sample(m, NUTS(ps.warmup, 0.95; max_depth=5), ps.steps + ps.warmup; progress=true, thinning) #; max_depth=15
    end
end #112.942085
## ============================================================================
# improve initial parameters
function initparams(param_mod)
    varinfo = DynamicPPL.VarInfo(param_mod)
    # varinfo.metadata.R0s.ranges
    f = turing_data.invlink
    param_mod(varinfo, Turing.SampleFromPrior(),
        Turing.PriorContext((
            # R0s = ones(turing_data.num_regions),
            latent_Rts_z = [f.(ones(turing_data.num_rt_steps[m])) for m in 1:turing_data.num_regions],
        ))
    )
    varinfo[Turing.SampleFromPrior()]
end
init_theta = initparams(m)
chain = sample(m, Turing.NUTS(1000, 0.8), 1000; init_params = init_theta, discard_adapt = false)
## ==========================================================================
@info "save"
fdir = projectdir("reports/tmp", "$(today())")
fname = Regional.save_results(fdir, ps, data, chain)

## ==========================================================================
@info "post processing"
Regional.postprocessing(fname; plot_results=false)

##
fname = "/home/and/dev/CovidSurvey/reports/tmp/2021-08-30/CHAIN_chains=1_include=true_model=hospit_observ=2021-02-01_preds=CT,MNRES_rwstep=7_semipara=false_std=true_steps=10_warmup=10.jls"
fdir, ps, ignores = Regional.parse_fname(fname)
pp = Regional.PostProcessing2(fdir, ps, ignores, [], fname)
fname = projectdir("/home/and/dev/CovidSurvey/reports/tmp/2021-08-30/GENERATED-QUANTITIES_chains=1_include=true_model=hospit_observ=2021-02-01_preds=CT,MNRES_rwstep=7_semipara=false_std=true_steps=10_warmup=10.bson")
gp = BSON.load(fname) |> NamedTuple;
# gp = Regional.generate_posterior(pp)









mean(Exponential(100.))





# chain = deepcopy(chain)
turing_data_predictive = Regional.predictive(d)

keys(turing_data_predictive)
turing_data_predictive.hospits

predictive_model = model(turing_data_predictive, false, false)
posterior_predictive = predict(predictive_model, chain)

prior = sample(predictive_model, Prior(), 1000; progress=false)
prior_predictive = predict(predictive_model, prior)

loglikelihoods = Turing.pointwise_loglikelihoods(
    param_mod, MCMCChains.get_sections(chain, :parameters)
)


data_params_dct = convert(Dict, data_params)
data_params_dct[:predictive] = true
data_params = namedtuple(data_params_dct)
data = Regional.load_data(; data_params... )
turing_data = data.turing_data;

predictive_model = model1(turing_data, false)
posterior_predictive = predict(predictive_model, chain);

prior = sample(m, Prior(), 1000; progress=false)
prior_predictive = predict(param_mod_predict, prior):

loglikelihoods = Turing.pointwise_loglikelihoods(
    m, MCMCChains.get_sections(chain, :parameters)
)

using LinearAlgebra
# Ensure the ordering of the loglikelihoods matches the ordering of `posterior_predictive`
ynames = string.(keys(posterior_predictive))
loglikelihoods_vals = getindex.(Ref(loglikelihoods), ynames)
# Reshape into `(nchains, nsamples, size(y)...)`
loglikelihoods_arr = permutedims(cat(loglikelihoods_vals...; dims=3), (2, 1, 3))

using LinearAlgebra
# Ensure the ordering of the loglikelihoods matches the ordering of `posterior_predictive`
ynames = string.(keys(posterior_predictive))
loglikelihoods_vals = getindex.(Ref(loglikelihoods), ynames)
# Reshape into `(nchains, nsamples, size(y)...)`
loglikelihoods_arr = permutedims(cat(loglikelihoods_vals...; dims=3), (2, 1, 3))

## ==========================================================================
@info "post processing"
fdir, ps, ignores = Regional.parse_fname(fname)
p = Regional.PostProcessing2(fdir, ps, ignores, [], fname)
Regional.savechain(p)
## ==========================================================================
@info "plot chain"
plot_results = true
Regional.plot_chains(p; plot_results)

## ==========================================================================
@info "meanplot"
Regional.plot_means(p; plot_results)
## ==========================================================================
# perform diagnostics
p = Regional.skip_warmup(p)
Regional.diagnostics(p)
## ==========================================================================
@info "make predictions"
gp = Regional.generate_posterior(p);
## ==========================================================================
@info "plot regions"
Regional.plot_regions(p, gp; plot_results=true)

recipe = Regional.RegionPlottingRecipe(p.data, gp, "capital")
plot(recipe)
## -----------------------------------------------------------------------------
@info "store reproduction number"
Regional.save_rt(p, gp)
## ===========================================================================
# ArviZ

using ArviZ
import PyPlot as plt

c = MCMCChains.get_sections(chain, :parameters)
n = filter(
    x-> !occursin("latent_Rts", x) &&
        !occursin("effect", x)     &&
        !occursin("ys", x)     &&
        !occursin("R0", x), String.(names(c)))
plot_autocorr(c; var_names=n);
plt.gcf()
##
idata = from_mcmcchains(
    p.chain;
    #coords=Dict("school" => schools),
    #dims=Dict("y" => ["school"], "σ" => ["school"], "θ" => ["school"]),
    library="Turing",
)
plot_pair(
    idata;
    #coords=Dict("school" => ["Choate", "Deerfield", "Phillips Andover"]),
    divergences=true,
);

fn = fdir*"tmp.png"
plt.gcf().savefig(fn)
run(`firefox $(fn)`, wait=false)
##


# function posterior(model, turing_data, chain)
m_pred = model(turing_data, predict = true, context=:prediction)
gq = Turing.generated_quantities(m_pred, chain)
cat(gq...; dims=3)

gp = vectup2tupvec(reshape(gq, length(gq)))
groupregions(gp)
# end

turing_data_predictive = Regional.predictive(d)
predictive_model = model(turing_data_predictive,
    turing_data_predictive.hospits,
    vectorize = true,
    predict = false)
posterior_predictive = Turing.predict(predictive_model, chain)

predictive_model()

m = 1
s = d.hospitmodel.starts[m]
e = d.hospitmodel.stops[m]
ts= s:e
length(ts)

prior = sample(predictive_model, Prior(), 100; progress=false)
prior_predictive = predict(predictive_model, prior)

param_mod = model(turing_data, false, false)
loglikelihoods = Turing.pointwise_loglikelihoods(
    param_mod, MCMCChains.get_sections(chain, :parameters)
)



using LinearAlgebra
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
    dims=Dict("y" => ["school"], "σ" => ["school"], "θ" => ["school"]),
    library="Turing",
)
ArviZ.loo(idata)

using MCMCChains
MCMCChains.header(chain, section = :parameters)
MCMCChains.sections(chain)

idata = from_mcmcchains(
    turing_chns,
    coords = Dict("school" => schools),
    dims = Dict(
        "y" => ["school"],
        "σ" => ["school"],
        "θ" => ["school"],
    ),
    library = "Turing",
)

plot_pair(
    idata;
    coords=Dict("school" => ["Choate", "Deerfield", "Phillips Andover"]),
    divergences=true,
);
plt.gcf()


plot_rank(post; var_names=["μ", "τ"]);
gcf()

plot_density(
    [idata.posterior_predictive, idata.prior_predictive];
    data_labels=["Post-pred", "Prior-pred"],
    var_names=["y"],
)
gcf()















##
TV = Vector{Float64}
vectorize = true
θ = turing_data
@unpack num_observations, num_total_days, num_regions, num_rt_steps, invlink, hospitmodel, seromodel, num_covariates, include_early_dynamic = θ
# If we don't want to predict the future, we only need to compute up-to time-step `num_obs_countries[m]`
num_time_steps = num_observations

############# 2.) time varying reproduction number

# κ          ~ truncated(Normal(0, 1.), 0, Inf)
# R0s        ~ filldist(truncated(Normal(2., κ), 1., 5.), num_regions)
# σ_rt       ~ truncated(Normal(0.15, .05), 0, .25)
# R0         = truncated(Normal(3.0, 0.5), 1., 4.5) |> rand
# R0s        = filldist(truncated(Normal(R0, 0.2), 1., 4.5), num_regions) |> rand
R0s        = filldist(truncated(Normal(3., 1.), 1., 5.), num_regions) |> rand
σ_rt       = truncated(Normal(0.15*2, .05*2), 0, .25*3) |> rand

init_rw = if include_early_dynamic
        R0s
    else
        R1s = filldist(truncated(Normal(.8, .1), .5, 1.1), num_regions) |> rand
end

latent_Rts_z = [TV(undef, num_rt_steps[m]) for m in 1:num_regions]
latent_Rts = [TV(undef, num_rt_steps[m]) for m in 1:num_regions]
for m in 1:num_regions
    latent_Rts_z[m] = CovidSurvey.RandomWalk(num_rt_steps[m]) |> rand
    latent_Rts[m] = latent_Rts_z[m] * σ_rt .+ invlink(init_rw[m])
end

Rts = TV[TV(undef, num_time_steps[m]) for m in 1:num_regions]
if num_covariates > 0
    grouped_effect = filldist( Laplace(0,0.2), num_covariates) |> rand
    effect_std = filldist( GammaMeanCv(0.1, 0.5), num_covariates) |> rand
    effects_z = filldist( MvNormal( num_covariates, 1.), num_regions) |> rand
    effects = [ effects_z[:,m] .* effect_std .+ grouped_effect for m in 1:num_regions]

    # grouped_effect = filldist( Exponential(0.2), num_covariates)
    # effect_std ~ filldist( GammaMeanCv(0.1, 0.5), num_covariates)
    # pooled_effects_distr_raw = Normal.( grouped_effect, effect_std )
    # pooled_effects_distr = arraydist( truncated.( pooled_effects_distr_raw, zero(V), V(Inf) ) )
    # effects ~ filldist( pooled_effects_distr, num_regions )

    if θ.semiparametric
        Regional.semiparametric!(Rts, θ, false, latent_Rts, R0s, σ_rt, effects)
    else
        Regional.mixed!(Rts, θ, false, latent_Rts, R0s, σ_rt, effects)
    end
else
    Regional.random_walks!(Rts, θ, false, latent_Rts, R0s, σ_rt)
end

############ 3.) infection dynamics
τ  = Exponential(100 / 5)  |> rand# `Exponential` has inverse parameterization of the one in Stan
T  = typeof(τ)
ys = filldist(truncated(Normal(τ, 30),T(0),T(500)), num_regions) |> rand

newly_infecteds      = TV[TV(undef, num_time_steps[m]) for m in 1:num_regions]
cumulative_infecteds = TV[TV(undef, num_time_steps[m]) for m in 1:num_regions]
# effective_Rts        = TV[TV(undef, num_time_steps[m]) for m in 1:num_regions]

Regional.infections!(newly_infecteds, cumulative_infecteds, θ, ys, Rts)
# infections!(newly_infecteds, cumulative_infecteds, effective_Rts, θ, τ, ys, Rts)
########### 4.) derive observables
μ_i2h = truncated(Normal(9.5, .5), 7, 15) |> rand
σ_i2h = truncated(Normal(5.41,1), 1, 14) |> rand
ihr   = truncated(Normal(1.8/100,0.5/100), 1/100, 5/100) |> rand
ϕ_h   = truncated(Normal(50, 10), 20, Inf) |> rand

expected_daily_hospits   = TV[TV(undef, num_time_steps[m]) for m in 1:num_regions]
hospit_observation_model = Regional.SimpleObsModel2(hospitmodel, μ_i2h, σ_i2h, ihr, ϕ_h, expected_daily_hospits, vectorize)
sero_observation_model   = Regional.SimpleSeroObsModel(seromodel, cumulative_infecteds)

Regional.expected!(hospit_observation_model, newly_infecteds)
plot(expected_daily_hospits[1], yticks=:native)

########### 4.) compare model to observations
## 4.1) observe hospitalizations

V = Float64
ℓ  = zero(V)
ℓ += logpdf(hospit_observation_model, θ.hospits)
ℓ += logpdf(sero_observation_model, θ.sero)
Turing.@addlogprob! ℓ

@code_warntype logpdf(hospit_observation_model, θ.hospits)
@code_warntype logpdf(sero_observation_model, θ.sero)


m = 1
start = hospitmodel.starts[m]
stop  = hospitmodel.stops[m]
population = hospitmodel.populations[m]

ts    = start:stop
μs = expected_daily_hospits[m][ts]
ys = θ.hospits[m][ts]

Regional._out_of_bounds(μs, population) && (return V(Inf))
# if vectorize
dist = arraydist(NegativeBinomial2.(μs, Ref(ϕ_h)))
findall(μ-> !(0 .< (1 ./ (1 .+ μ ./ ϕ_h)) .< 1), μs)
1 ./ (1 .+ μs[190]./ ϕ_h)
rand(dist)
logpdf(dist, ys)
# else
    # ys .= NegativeBinomial2.(μs, Ref(ϕ_h))
let
    ℓ = 0.
    for i in eachindex(ys)# zip(μs,ys)
        @show i
        dist = NegativeBinomial2(μs[i], ϕ_h)
        rand(dist)
        ℓ += logpdf(dist, ys[i])
    end
    ℓ
end

any( 1 .<= 1 ./ (1 .+ μs / ϕ_h) .<= 0)
any(μs .<= 0)

dist = NegativeBinomial2.(μs, Ref(ϕ_h))
rand.(dist)
