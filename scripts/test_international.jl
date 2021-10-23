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

ps = ( warmup = 100,
    model  = "intdeaths",
    steps  = 100,
    seed   = nothing,
    observ = nothing,
    chains = 1,
    preds = nothing,#"CF,CC,CR,CS",#"CS,MG,MA"
    semipara=true,
    rwstep = 7,
    include= true,
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
    , rwstep            = ps.rwstep
    , epidemicstart     = Date("2020-08-01")
    , numimpute         = 20
    , include_early_dynamic = ps.include
    , link              = KLogistic(4.5)
    , invlink           = KLogit(4.5)
    , predictive        = false
    , covariates_kwargs = Dict(
      :semiparametric => ps.semipara,
      :fname          => [projectdir("data/contacts/int/smoothed_contact_rates_region=$(r).csv" ) for r in International.regions],#[projectdir("data", "mean_contact_rates_region=$r.csv" ) for r in 1:Regional.nregions],
      :shift          => -1,
      :conditions     => :date => x->x>Date("2020-11-10"), # >= 1000 cases / day
      # :datecol        => "date",
      :aggregation    => ps.rwstep,
      :standardize    => ps.std,
      :normalize      => true,
      :mobility       => [projectdir("data/mobility/mobility_country=$(r).csv") for r in International.regions]
      )
    )

data = International.load_data(; data_params... )
d = turing_data = data.turing_data;
## ============================================================================
@info "prepare model"

model = Regional.model_deaths
model = Regional.model_international_deaths
#!isnothing(ps.seed) && Random.seed!(ps.seed);
@info ps
m = model(turing_data; vectorize=true)
m();
## ============================================================================
@info "start sampling"

@time chain = let #460s; 366s
    thinning = 1
    if ps.chains > 1
        sample(m, NUTS(ps.warmup, 0.9; max_depth=6), MCMCThreads(), ps.steps + ps.warmup, ps.chains; progress=true, thinning)
    else
        sample(m, NUTS(ps.warmup, 0.9; max_depth=6), ps.steps + ps.warmup; progress=true, thinning) #; max_depth=15
    end
end

divervences = sum( chain[ps.warmup+1:end, :numerical_error, :] )
@show divervences

## ==========================================================================
@info "save"
fdir = projectdir("reports/tmp", "$(today())")
fname = Regional.save_results(fdir, ps, data, chain)
## ==========================================================================
@info "post processing"
Regional.postprocessing(fname; plot_results=false)
## ==========================================================================
## ==========================================================================
@info "post processing"
fdir, ps, ignores = Regional.parse_fname(fname)
p = Regional.PostProcessing2(fdir, ps, ignores, [], fname)
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
data.cases
## ==========================================================================
@info "make predictions"
gp = Regional.generate_posterior(p);
## ==========================================================================
@info "plot regions"
Regional.plot_regions(p, gp; plot_results=true)
## ==========================================================================
@info "plot predictors"
# pgfplotsx()
# default(titlefontsize = 20, legendfontsize = 18, labelfontsize = 18, guidefontsize = 18, tickfontsize = 12, framestyle = :zerolines, yminorgrid = true)
Regional.plot_effects(p, gp; plot_results , grouped = false, effect_on_Rt = 0.)
Regional.plot_effects(p, gp; plot_results, grouped = false, effect_on_Rt = 2.)
Regional.plot_effects(p, gp; plot_results, grouped = false, effect_on_Rt = -0.5)

Regional.plot_effects(p, gp; plot_results, grouped = true, effect_on_Rt = 0.)
Regional.plot_effects(p, gp; plot_results, grouped = true, effect_on_Rt = 2.)
Regional.plot_effects(p, gp; plot_results, grouped = true, effect_on_Rt = -0.5)



























## ===================================================================

TV = Vector{Float64}
V = Float64
vectorize = true
θ = turing_data
@unpack num_observations, num_total_days, num_regions, num_rt_steps, deathmodel, num_covariates, rwscale, include_early_dynamic,init_infected = θ
  # If we don't want to predict the future, we only need to compute up-to time-step `num_obs_countries[m]`
num_time_steps = num_observations

############# 2.) time varying reproduction number

R0_std     = truncated(Normal(.5, .2), 0.01, Inf) |> rand
R0s        = filldist(truncated(Normal(0.8, R0_std), 0., 4.5), num_regions) |> rand
σ_rt       = truncated(Normal(0.3*rwscale, .02*rwscale), 0, .5*rwscale) |> rand

n = first( num_rt_steps )
latent_Rts_z = filldist( CovidSurvey.RandomWalk( n ), num_regions) |> rand
latent_Rts = CovidSurvey.rescale( latent_Rts_z, σ_rt, d.invlink.(R0s) )



Rts = TV[TV(undef, num_time_steps[m]) for m in 1:num_regions]
if num_covariates > 0
    grouped_effect = filldist( Laplace(0,0.2), num_covariates) |> rand
    effect_std = filldist( GammaMeanCv(0.1, 0.5), num_covariates) |> rand
    effects_z = filldist( MvNormal( num_covariates, 1.), num_regions) |> rand
    effects = [ effects_z[:,m] .* effect_std .+ grouped_effect for m in 1:num_regions]

    # grouped_effect = filldist( Exponential(0.2), num_covariates)
    # effect_std = filldist( GammaMeanCv(0.1, 0.5), num_covariates)
    # pooled_effects_distr_raw = Normal.( grouped_effect, effect_std )
    # pooled_effects_distr = arraydist( truncated.( pooled_effects_distr_raw, zero(V), V(Inf) ) )
    # effects = filldist( pooled_effects_distr, num_regions )

    if θ.semiparametric
        CovidSurvey.semiparametric!(Rts, θ, false, latent_Rts, R0s, σ_rt, effects)
    else
        CovidSurvey.mixed!(Rts, θ, false, latent_Rts, R0s, σ_rt, effects)
    end
else
    CovidSurvey.random_walks!(Rts, θ, false, latent_Rts, R0s, σ_rt)
end
plot(latent_Rts_z[:,3], ticks=:native)
plot(θ.link.(latent_Rts[:,1]), ticks=:native)
plot(d.dates[1], Rts[1], ticks=:native)
############ 3.) infection dynamics
ys = arraydist(Exponential.(init_infected)) |> rand


newly_infecteds      = TV[TV(undef, num_time_steps[m]) for m in 1:num_regions]
CovidSurvey.infections!(newly_infecteds, θ, ys, Rts)
plot(d.dates[1], newly_infecteds[1], ticks=:native)

# infections!(newly_infecteds, cumulative_infecteds, effective_Rts, θ, τ, ys, Rts)
########### 4.) derive observables
μ_i2d = truncated(Normal(21.82, 1.01), 18, 24) |> rand
σ_i2d = truncated(Normal(4.26, 5.18), 1, 14) |> rand
idr   = truncated(Normal(8/1000, 1/1000), 1/1000, 15/1000) |> rand
ϕ_d   = truncated(Normal(50, 10), 20, Inf) |> rand


weekdayeffect_simplex = filldist(Dirichlet([7,7,7,7,7,3,1]), num_regions) |> rand
weekdayeffect = TV[TV(undef, 7) for m in 1:num_regions]
CovidSurvey.weekdayeffect!(weekdayeffect, weekdayeffect_simplex)

### holiday effect
holidayeffect = filldist(Beta(1,1), num_regions) |> rand

expected_daily_deaths    = TV[TV(undef, num_time_steps[m]) for m in 1:num_regions]
deaths_observation_model = CovidSurvey.WeekdayHolidayObsModel2(
    deathmodel, μ_i2d, σ_i2d, idr, ϕ_d, expected_daily_deaths, weekdayeffect, holidayeffect, vectorize
)

# deaths_observation_model = CovidSurvey.SimpleObsModel2(deathmodel, μ_i2d, σ_i2d, idr, ϕ_d, expected_daily_deaths, vectorize)

CovidSurvey.expected!(deaths_observation_model, newly_infecteds)
plot(d.dates[1], expected_daily_deaths[1], ticks=:native)


########### 4.) compare model to observations
## 4.1) observe hospitalizations

ℓ  = zero(V)
ℓ += CovidSurvey.logpdf(deaths_observation_model, θ.deaths)
