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
using BSON
using PrettyTables
## ===========================================================================
# load data and model
# ============================================================================
fdir = projectdir("reports/regional/")
fname = "cases_start=2020-06-01_chains=10_epidemic_start=30_folder=regional_model=hospit_num_impute=6_observations_end=2021-01-13_prefix=_rw_step=1_steps=4000_warmup=1000"
PREFIX = ""
#-----------------------------------------------------------------------------
# read params
_, parsed_args, _ = parse_savename(fname, parsetypes = (Int, Float64, Bool))
!("preds" in keys(parsed_args)) && (parsed_args["preds"] = nothing)
# ps = NamedTuple{Tuple(Symbol.(keys(ps)))}(values(ps))

#------------------------------------------------------------------------------
# read data
data_params = (
      observations_end  = parsed_args["observations_end"]
    , predictors        = parsed_args["preds"]|> CovidSurvey.parse_predictors
    , cases_start       = parsed_args["start"]
    , rw_step           = parsed_args["prefix=_rw_step"]
    , epidemic_start    = parsed_args["epidemic_start"]
    , num_impute        = parsed_args["num_impute"]
    , link              = KLogistic(3.)
    , invlink           = KLogit(3.)
    , lockdown          = "2020-03-18"
    , covariates_kwargs = Dict(
      :fname => normpath( homedir(), "data/covidsurvey/smoothed_contacts.csv" ),
      :shift => -1,
      :startdate => "2020-11-10", # >= 1000 cases / day
    )
)

data        = Regional.load_data(; data_params... )
turing_data = data.turing_data;
model       = Regional.name2model[parsed_args["model"]]
#-----------------------------------------------------------------------------
# read chain
chain = read(fdir*PREFIX*"CHAIN_"*fname*".jls", Chains)
chain = chain[1:10:end,:,[1,2,3,5,6,7,9,10]]
# fname = normpath( fdir, savename(PREFIX*"CHAIN", ps, "jls") )
# safesave( fname, chain)

#-----------------------------------------------------------------------------
# plot chain
n = filter( x->!occursin(r"latent_Rts", x), String.(names(chain)))
let
    p = plot(chain[n])
    fname = normpath( fdir, savename(PREFIX*"CHAINSPLOT", data_params, "html") )
    savefig(p, fname )
    run(`firefox $(fname)`, wait=false)
end

## ===========================================================================
# plot results
# ============================================================================
@info "make predictions"
generated_posterior = Regional.posterior(model, turing_data, chain)
let
    dic = Dict( zip( keys(generated_posterior), values(generated_posterior) ) )
    bson( normpath( fdir, "GENERATED-QUANTITIES_"*fname*".bson" ) ,  dic )
end
## ==========================================================================
for r in Regional.regions
    recipe = Regional.RegionPlottingRecipe(data, generated_posterior, r)
    p = plot(recipe)
    savefig( p, normpath( fdir, "FIG-$(uppercase(r))_"*fname*".html" ))
end
##
let
    r = Regional.RtsPlottingRecipe(data, generated_posterior)
    p = plot(r)
    savefig( p, normpath( fdir, "FIG-RT_"*fname*".html" ) )
end
##
let
    r = Regional.HospitsPlottingRecipe(data, generated_posterior)
    p = plot(r)
    savefig( p, normpath( fdir, "FIG-HOSPIT_"*fname*".html" ) )
end

#-----------------------------------------------------------------------------
@info "store reproduction number"
rt = let
    rt = let
        ib = findfirst(==(Date("2020-05-15")), data.dates)
        ie = findfirst(==(Date(ps["observ"])), data.dates)
        Rt_array = hcat(posterior.Rt...)[ib:ie,:]

        qs = [quantile(v, [0.025, 0.25, 0.5, 0.75, 0.975]) for v in eachrow(Rt_array)]
        llq, lq, mq, uq, uuq = (eachrow(hcat(qs...))..., )

        date = data.dates[ib:ie]
        DataFrame((;date, llq, lq, mq, uq, uuq))
    end
    fname = normpath( fdir, savename(PREFIX*"Rt", ps, "csv") )
    save(fname, rt)
    rt
end
#-----------------------------------------------------------------------------
@info "plot results"
let
    p = National.plot_results(data, posterior...);
    fname = normpath( fdir, savename(PREFIX*"PREDICTION", ps, "html") )
    savefig(p, fname )
    # savefig(p, projectdir("figures", fname*".png") )
    run(`firefox $(fname)`, wait=false)
end
#-----------------------------------------------------------------------------
# perform diagnostics
if ps["chains"] > 1
    @info "gelman diagnostics"
    diagnostics = gelmandiag(chain)
    fname = normpath( fdir, savename(PREFIX*"GELMANDIAG", ps, "csv") )
    safesave( fname, diagnostics)
    pretty_table(diagnostics; crop=:none)
end
#-----------------------------------------------------------------------------
@info "meanplot"
let
    p = meanplot(chain[n]);
    fname = normpath( fdir, savename(PREFIX*"MEANPLOT", ps, "html") )
    savefig(p, fname )
    run(`firefox $(fname)`, wait=false)
end
