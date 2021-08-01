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
    warmup = 10,
    model = "hospit",
    steps  = 100,
    seed   = nothing,
    observ = "2021-01-13",#"2021-02-06",#"2021-03-25"
    chains = 1,
    preds = "CF,CC,CR,CS",
    semipara=true,
)

## ============================================================================
# load data
@info "load data"

data_params = (
      observations_end  = ps.observ
    , predictors        = nothing#ps.preds|> CovidSurvey.parse_predictors
    # , cases_start       = ps.cases
    # , hospitmodel       = Regional.HospitInit1(obs_stop="2020-07-01")
    , casemodel         = Regional.CaseInit1(obs_start="2020-06-01")
    , seromodel         = Regional.SeroInit2(delay=0, std=1.)
    , rw_step           = 1
    , epidemic_start    = 10
    , num_impute        = 6
    , link              = KLogistic(3.)
    , invlink           = KLogit(3.)
    , covariates_kwargs = Dict(
      :semiparametric => true,
      :fname => normpath( homedir(), "data/covidsurvey/smoothed_contacts.csv" ),
      :shift => -1,
      :startdate => "2020-11-10", # >= 1000 cases / day
    )
)

data = Regional.load_data(; data_params... )
turing_data = data.turing_data;
fdir = "/home/and/tmp/figures/"

## ============================================================================
@info "prepare model"

model = Regional.name2model[ps.model]
!isnothing(ps.seed) && Random.seed!(ps.seed);
@info ps
m = model(turing_data, false)
m();
## ============================================================================
@info "start sampling"

@time chain = if ps.chains > 1
    sample(m, NUTS(ps.warmup, 0.95; max_depth=5), MCMCThreads(), ps.steps + ps.warmup, ps.chains; progress=true, thinning=10)
else
    sample(m, NUTS(ps.warmup, 0.95; max_depth=5), ps.steps + ps.warmup; progress=true, thinning=10) #; max_depth=15
end

chain = chain[ps.warmup+1:end,:,:]

## ============================================================================
@info "make predictions"
generated_posterior = Regional.posterior(model, turing_data, chain);
## ============================================================================
@info "plot regions"

for r in Regional.regions
    recipe = Regional.RegionPlottingRecipe(data, generated_posterior, r)
    p = plot(recipe)
    fname = normpath( fdir, savename("FIG-$(uppercase(r))", ps, "html") )
    savefig( p, fname )
    run(`firefox $fname`, wait=false)
end
## ============================================================================
@info "plot rt"

let
    recipe = Regional.RtsPlottingRecipe(data, generated_posterior)
    p = plot(recipe)
    fname = normpath( fdir, savename("FIG-RT", ps, "html") )
    savefig( p, fname )
    run(`firefox $fname`, wait=false)
end

## ============================================================================
@info "store reproduction number"

for r in Regional.regions
    rt = rtstats(
            generated_posterior, r;
            startdate = Date("2020-05-15"),
            stopdate  = ps.observ
    )
    fname = normpath( fdir, savename("Rt-$(uppercase(r))", ps, "csv") )
    save(fname, rt)
end

## ============================================================================
@info "perform diagnostics"

if ps.chains > 1
    @info "gelman diagnostics"
    diagnostics = gelmandiag(chain)
    fname = normpath( fdir, savename(ps.prefix*"GELMANDIAG", ps, "csv") )
    safesave( fname, diagnostics)
    pretty_table(diagnostics; crop=:none)
end

## ============================================================================
@info "plot chain"

n = filter( x->!occursin(r"latent_Rts", x), String.(names(chain)))
let
    p = plot(chain[n])
    fname = normpath( fdir, savename("CHAINSPLOT", ps, "html") )
    savefig(p, fname )
    run(`firefox $(fname)`, wait=false)
end

## ============================================================================
@info "meanplot"

let
    p = meanplot(chain[n]);
    fname = normpath( fdir, savename("MEANPLOT", ps, "html") )
    savefig(p, fname )
    run(`firefox $(fname)`, wait=false)
end

#-----------------------------------------------------------------------------
# @info "plot prior vs. posterior"
# pchain = sample(m, Prior(), ps.steps + ps.warmup; progress=true)[ps.warmup+1:end,:,:]
# n = filter( x->!occursin(r"\[", x), String.(names(pchain)))
# p = density(chainscat(pchain[n], chain[n]))
# savefig(p, projectdir("figures/tmp", fname*"_PRIORvsPOSTERIOR.html") )
# run(`firefox $(projectdir("figures/tmp", fname*"_PRIORvsPOSTERIOR.html"))`, wait=false)
