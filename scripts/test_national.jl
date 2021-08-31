using DrWatson
quickactivate(@__DIR__)
@show projectdir()
#-----------------------------------------------------------------------------
# load packages
@info "load packages"
using ArgParse
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
#-----------------------------------------------------------------------------
ps = (
    warmup = 10,
    model = "cases",
    steps  = 10,
    seed   = nothing,
    observ = "2021-01-13",#"2021-02-06",#"2021-03-25"
    chains = 1,
    preds = "CT",#"CF,CC,CR,CS",
    semipara=false,
)
#----------------------------------------------------------------------------
# load data
@info "load data"
data = National.load_data(
      observationsend  = ps.observ
    , predictors        = ps.preds|> CovidSurvey.parse_predictors
    , rwstep           = 1
    , epidemicstart    = 30
    , numimpute        = 6
    , deathmodel        = ps.model == "cases" ? National.DeathInit(obs_stop="2020-07-01") : National.DeathInit()
    , casemodel         = National.CaseInit(obs_start="2020-06-01")
    # , hospitmodel       = National.HospitInit()
    , seromodel         = National.SeroInit1(delay=0, std=.5)
    , link              = KLogistic(3.)
    , invlink           = KLogit(3.)
    , lockdown          = "2020-03-18"
    , covariates_kwargs = Dict(
        :semiparametric => ps.semipara,
        :fname => projectdir("data","raw_contact_rates.csv"),
        :shift => -1,
        :startdate => "2020-10-01",
        :enddate => nothing))

turing_data = data.turing_data;
## ============================================================================


@info "prepare model"

model1 = National.name2model[ps.model]
!isnothing(ps.seed) && Random.seed!(ps.seed);
m = model1(turing_data, false)
p = m();
## ============================================================================
@info "start sampling"

@time chain = let
    thinning = 10
    if ps.chains > 1
        sample(m, NUTS(ps.warmup, 0.95; max_depth=5), MCMCThreads(), ps.steps + ps.warmup, ps.chains; progress=true, thinning)
    else
        sample(m, NUTS(ps.warmup, 0.95; max_depth=5), ps.steps + ps.warmup; progress=true, thinning) #; max_depth=15
    end
end
## ==========================================================================
@info "save"

fdir = projectdir("reports/tmp", "$(today())")
fname = National.save_results(fdir, ps, data, chain)

## ==========================================================================
@info "post processing"
National.postprocessing(fname; plot_results=true)

## ==========================================================================
@info "post processing"
fdir, ps, ignores = National.parse_fname(fname)
p = National.PostProcessing3(fdir, ps, ignores, [], fname)
National.savechain(p)
## ==========================================================================
@info "plot chain"
plot_results = true
National.plot_chains(p; plot_results)

## ==========================================================================
@info "meanplot"
National.plot_means(p; plot_results)
## ==========================================================================
# perform diagnostics
p = National.skip_warmup(p)
National.diagnostics(p)
## ==========================================================================
@info "make predictions"
gp = National.generate_posterior(p);
## ==========================================================================
@info "plot regions"
National.plot_prediction(p, gp; plot_results=true)
## -----------------------------------------------------------------------------
@info "store reproduction number"
National.save_rt(p, gp)
