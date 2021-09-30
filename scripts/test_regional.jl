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
setadbackend(:reversediff)
Turing.setrdcache(true)

plotlyjs()
@info "number of threads available: $(nthreads())"
## ============================================================================

ps = (
    warmup = 100,
    model = "hospit",
    steps  = 100,
    seed   = nothing,
    observ = "2021-03-01",#"2021-02-06",#"2021-03-25"
    chains = 1,
    preds = nothing,#"CT,MG",#"CF,CC,CR,CS"
    semipara=true,
    rwstep = 1,
    include= false,
    std = false,
)
## ============================================================================
# load data
@info "load data"

data_params = (
      observationsend  = ps.observ
    , predictors        = ps.preds|> CovidSurvey.parse_predictors
    # , cases_start       = ps.cases
    # , hospitmodel       = Regional.HospitInit1(obs_stop="2020-07-01")
    , casemodel         = Regional.CaseInit(obs_start="2020-06-01")
    , seromodel         = Regional.SeroInit(delay=0, std=1.)
    , rwstep            = ps.rwstep
    , epidemicstart    = 20
    , numimpute        = 6
    , include_early_dynamic = ps.include
    , link              = KLogistic(4.5)
    , invlink           = KLogit(4.5)
    , predictive        = false
    , covariates_kwargs = Dict(
      :semiparametric => ps.semipara,
      :fname          => [projectdir("data", "mean_contact_rates_region=$(r).csv" ) for r in 1:Regional.nregions],#[projectdir("data", "mean_contact_rates_region=$r.csv" ) for r in 1:Regional.nregions],
      :shift          => -1,
      :startdate      => "2020-11-10", # >= 1000 cases / day
      :datecol        => "dates",
      :aggregation    => ps.rwstep,
      :standartize    => ps.std,
      :mobility       => [projectdir("data/mobility/mobility_region=$(Regional.regions[i]).csv") for i in 1:Regional.nregions]
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
m = model(turing_data; vectorize=true)
m();
## ============================================================================
@info "start sampling"

@time chain = let #460s; 366s
    thinning = 1
    if ps.chains > 1
        sample(m, NUTS(ps.warmup, 0.9; max_depth=5), MCMCThreads(), ps.steps + ps.warmup, ps.chains; progress=true, thinning)
    else
        sample(m, NUTS(ps.warmup, 0.9; max_depth=5), ps.steps + ps.warmup; progress=true, thinning) #; max_depth=15
    end
end #112.942085

divervences = sum( chain[ps.warmup+1:end, :numerical_error, :] )
@show divervences

## =========================================================================
# MODE ESTIMATION WITH OPTI
using Optim
using Underscores
using Accessors

function initvals(m, chain)
    varinfo = Turing.VarInfo(m)

    param = Dict()
    for key in keys(varinfo.metadata)
        val = @_ group(chain, key) |>
            DataFrame |>
            DataFrames.select(__, Not([:iteration, :chain])) |>
            Array |>
            mean(__, dims=1) |>
            vec
        if key == :latent_Rts_z
            val = reshape(val, (length(val)÷5, 5))
        end

        param[key] = length(val) == 1 ? first(val) : val
    end
    return NamedTuple(param)
end

init_tuple = initvals(
    pp.model(pp.data.turing_data),
    Turing.MCMCChains.get_sections(pp.chain, :parameters)
)

@set θ0.latent_Rts_z = similar(θ0.latent_Rts_z)

function optim_args(model, init_tuple)
    varinfo = Turing.VarInfo(model)
    model(varinfo, Turing.SampleFromPrior(), Turing.PriorContext(θ0))
    init_vec = varinfo[Turing.SampleFromPrior()]

    ctx = Turing.OptimizationContext(DynamicPPL.DefaultContext())
    f = Turing.OptimLogDensity(model, ctx)
    return f, init_vec
end

m = pp.model(pp.data.turing_data)
f, init_vec = optim_args(
      m
    , init_tuple
    )
f(init_vec)
optimize(m, MAP(), init_vec, ConjugateGradient(), Optim.Options(iterations=1000, allow_f_increases=true, show_trace=true))
##
using Accessors, NamedTupleTools
delete(pp.data, :num_total)
d.num_total_days

m = pp.model(pp.data.turing_data)
varinfo = Turing.VarInfo(m)
m(varinfo, Turing.SampleFromPrior(), Turing.PriorContext())
initθ = varinfo[Turing.SampleFromPrior()]

chain = sample(model, NUTS(), 100, init_theta=initθ)

##

x = NamedTuple(name => mode.values.array[i] for (name, i) in pairs(mode.values.dicts[1]))
DynamicPPL.setval_and_resample!(varinfo, x)
model(varinfo)
## ============================================================================
# improve initial parameterschain
function initparams(m)
    param_mod = deepcopy(m)
    varinfo = DynamicPPL.VarInfo(param_mod)
    # varinfo.metadata.R0s.ranges
    f = turing_data.invlink
    param_mod(varinfo, Turing.SampleFromPrior(),
        Turing.PriorContext((
            R0 = 1.,
            #latent_Rts_z = [MvNormal(turing_data.num_rt_steps[m], 0.01) .+ f(1) for m in 1:turing_data.num_regions],
        ))
    )
    varinfo[Turing.SampleFromPrior()]
end

chain = sample(m, NUTS(ps.warmup, 0.95; max_depth=5), ps.steps + ps.warmup; progress=true, discard_adapt = false)
initp = initparams(m)
chain = sample(m, NUTS(ps.warmup, 0.95; max_depth=5), ps.steps + ps.warmup; progress=true, init_params = initp)
plot(chain)
chain = sample(m, Turing.NUTS(1000, 0.8), 1000; init_params = init_theta, discard_adapt = false)

chains_params = Turing.MCMCChains.get_sections(chain, :parameters)
m_pred = model(turing_data, predict = true, context=:prediction)
gq = Turing.generated_quantities(m_pred, chains_params)
generated_posterior = Regional.posterior(model, turing_data, chains_params)

## ==========================================================================
@info "save"
fdir = projectdir("reports/tmp", "$(today())")
fname = Regional.save_results(fdir, ps, data, chain)

## ==========================================================================
@info "post processing"
Regional.postprocessing(fname; plot_results=false)
##
# fname = "/home/and/dev/CovidSurvey/reports/regional_effect_estimation/mobility/apple_totalcontacts/2021-09-05/CHAIN_chains=10_model=hospit_observ=2021-02-01_preds=CT,MA_regional=true_rwstep=7_semipar=true_std=false_steps=500_warmup=2000.jls"
fname = "/home/and/dev/CovidSurvey/reports/regional_prediction_end_of_peak/2021-09-07/CHAIN_chains=10_model=hospit_observ=2021-03-01_regional=true_rwstep=1_semipar=true_std=false_steps=500_warmup=2000.jls"
# fname = "/home/and/dev/CovidSurvey/reports/tmp/2021-08-31/CHAIN_cases=2020-06-01_chains=1_include=true_model=hospit_observ=2021-02-01_preds=CT,MNRES_rwstep=7_semipara=false_std=true_steps=10_warmup=10.jls"
fdir, ps, ignores = Regional.parse_fname(fname)
pp = Regional.PostProcessing2(fdir, ps, ignores, [], fname)
gp = Regional.generate_posterior(pp)
# gp = BSON.load(fname) |> NamedTuple;




## ===========================================================================
#  ARVIZ
# chain = deepcopy(chain)
turing_data_predictive = Regional.predictive(d)

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
## ==========================================================================
@info "plot predictors"
# pgfplotsx()
# default(titlefontsize = 20, legendfontsize = 18, labelfontsize = 18, guidefontsize = 18, tickfontsize = 12, framestyle = :zerolines, yminorgrid = true)
Regional.plot_effects(pp, gp; plot_results , grouped = false, effect_on_Rt = 0.)
Regional.plot_effects(p, gp; plot_results, grouped = false, effect_on_Rt = 2.)
Regional.plot_effects(p, gp; plot_results, grouped = false, effect_on_Rt = -0.5)

Regional.plot_effects(p, gp; plot_results, grouped = true, effect_on_Rt = 0.)
Regional.plot_effects(p, gp; plot_results, grouped = true, effect_on_Rt = 2.)
Regional.plot_effects(p, gp; plot_results, grouped = true, effect_on_Rt = -0.5)

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
@unpack num_observations, num_total_days, num_regions, num_rt_steps, hospitmodel, invlink, seromodel, num_covariates, rwscale, include_early_dynamic = θ
# If we don't want to predict the future, we only need to compute up-to time-step `num_obs_countries[m]`
num_time_steps = num_observations

############# 2.) time varying reproduction number

# κ          ~ truncated(Normal(0, 1.), 0, Inf)
# R0s        ~ filldist(truncated(Normal(2., κ), 1., 5.), num_regions)
# σ_rt       ~ truncated(Normal(0.15, .05), 0, .25)
R0         = truncated(Normal(3.0, 0.5), 1., 4.5) |> rand
R0s        = filldist(truncated(Normal(R0, 0.2), 1., 4.5), num_regions)  |> rand#CovidSurvey.CustomMvNormal1(R0, 0.2, 1., 4.5, num_regions) #filldist(truncated(Normal(R0, 0.2), 1., 4.5), num_regions)
σ_rt       = truncated(Normal(0.15*rwscale, .05*rwscale), 0, .3*rwscale) |> rand

init_rw = if include_early_dynamic
        R0s
    else
        # R1s ~ filldist(truncated(Normal(.8, .1), .5, 1.1), num_regions)
        R1  = truncated(Normal(0.8, 0.1), 0., 1.2) |> rand
        R1s = filldist(GammaMeanCv(0.8, 0.05), num_regions) |> rand
end

latent_Rts = if include_early_dynamic
    latent_Rts_z = [TV(undef, num_rt_steps[m]) for m in 1:num_regions]
    latent_Rts = [TV(undef, num_rt_steps[m]) for m in 1:num_regions]
    for m in 1:num_regions
        latent_Rts_z[m] = CovidSurvey.RandomWalk(num_rt_steps[m])
        @. latent_Rts[m] = latent_Rts_z[m] * σ_rt + invlink(init_rw[m])
    end
    latent_Rts
else
    n = first( num_rt_steps )
    latent_Rts_z = filldist( CovidSurvey.RandomWalk( n ), num_regions) |> rand
    Regional.rescale( latent_Rts_z, σ_rt, invlink.(init_rw) )
end
findall(isnan, latent_Rts_z)

Rts = TV[TV(undef, num_time_steps[m]) for m in 1:num_regions]
if num_covariates > 0
    grouped_effects = filldist( Laplace(0,0.2), num_covariates) |> rand
    effect_std = filldist( GammaMeanCv(0.1, 0.5), num_covariates) |> rand
    effects_z = filldist( MvNormal( num_covariates, 1.), num_regions) |> rand
    effects = [ effects_z[:,m] .* effect_std .+ grouped_effects for m in 1:num_regions]

    # grouped_effect ~ filldist( Exponential(0.2), num_covariates)
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
findall(isnan, Rts[5])

############ 3.) infection dynamics
τ  = Exponential(20) |> rand# `Exponential` has inverse parameterization of the one in Stan
T  = typeof(τ)
ys = filldist(truncated(Normal(τ, 30),T(0),T(500)), num_regions) |> rand

newly_infecteds      = TV[TV(undef, num_time_steps[m]) for m in 1:num_regions]
cumulative_infecteds = TV[TV(undef, num_time_steps[m]) for m in 1:num_regions]
# effective_Rts        = TV[TV(undef, num_time_steps[m]) for m in 1:num_regions]

Regional.infections!(newly_infecteds, cumulative_infecteds, θ, ys, Rts)

let
    i = 1
    plot(cumulative_infecteds[i], yticks=:native)
    hline!([hospitmodel.populations[i]])
end
plot(newly_infecteds[5], yticks=:native)
# findall(isnan, newly_infecteds[5])

# infections!(newly_infecteds, cumulative_infecteds, effective_Rts, θ, τ, ys, Rts)
########### 4.) derive observables
μ_i2h = truncated(Normal(11.7, .5), 8, 14) |> rand
σ_i2h = truncated(Normal(6.5,0.5), 1, 14) |> rand
ihr   = GammaMeanCv(0.025, 0.1) |> rand#truncated(Normal(1.8/100,0.5/100), 1/100, 5/100)
ϕ_h   = truncated(Normal(50, 10), 20, Inf) |> rand

expected_daily_hospit    = TV[TV(undef, num_time_steps[m]) for m in 1:num_regions]
hospit_observation_model = Regional.SimpleObsModel2(hospitmodel, μ_i2h, σ_i2h, ihr, ϕ_h, expected_daily_hospit, vectorize)
sero_observation_model   = Regional.SimpleSeroObsModel(seromodel, cumulative_infecteds)
Regional.expected!(hospit_observation_model, newly_infecteds)
plot(expected_daily_hospit[5], yticks=:native)
all(isfinite.(expected_daily_hospit[5]))
########### 4.) compare model to observations
## 4.1) observe hospitalizations

ℓ  = 0.
ℓ += logpdf(hospit_observation_model, θ.hospits)
ℓ += logpdf(sero_observation_model, θ.sero)
