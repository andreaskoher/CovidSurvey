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
# load packages
argtable = ArgParseSettings(
    description="This script samples from National.model using NUTS."
)
@add_arg_table! argtable begin
    "--chunksize"
        help = "chunksize to be used by ForwardDiff.jl"
        arg_type = Int
        default = 40
    "--num-samples", "-n"
        help = "number of samples"
        arg_type = Int
        default = 1000
    "--num-chains", "-t"
        help = "number of chains sampled with multi-threading"
        arg_type = Int
        default = 1
    "--num-warmup", "-w"
        help = "number of samples to use for warmup/adaptation"
        arg_type = Int
        default = 1000
    "--observations-end", "-o"
        help = "end of observation date as string yyyy-mm-dd. Use full data set by defaults"
        arg_type = String
        default = nothing
    "--cases-start", "-s"
        help = "start date for case observations as string yyyy-mm-dd. No case observations by default"
        arg_type = String
        default = nothing
    "--add-covariates", "-c"
        help = "list of covariate names (default empty string = no covariates). Choose: C: contacts, R:"
        arg_type = String
        default = nothing
    "--model", "-m"
        help = "choose from: 'v1', 'gp'. Defaul: 'v1'"
        arg_type = String
        default = "v1"
    "--plot-results", "-p"
        arg_type = Bool
        default = false
    "--foldername", "-f"
        arg_type = String
        default = "$(today())"
    "--name-prefix", "-u"
        arg_type = String
        default = ""
    "--seed"
        help = "random seed to use"
        arg_type = Int
        default = nothing
end
parsed_args = parse_args(ARGS, argtable)

#----------------------------------------------------------------------------
# load data
@info "load data"

data_params = (
      observations_end  = parsed_args["observations-end"]
    , predictors        = parsed_args["add-covariates"]|> CovidSurvey.parse_predictors
    , cases_start       = parsed_args["cases-start"]
    , rw_step           = 1
    , epidemic_start    = 30
    , num_impute        = 6
    , link              = KLogistic(3.)
    , invlink           = KLogit(3.)
    , covariates_kwargs = Dict(
      :fname => normpath( homedir(), "data/covidsurvey/smoothed_contacts.csv" ),
      :shift => -1,
      :startdate => "2020-11-10", # >= 1000 cases / day
    )
)

data = Regional.load_data(; data_params... )

turing_data = data.turing_data;
#----------------------------------------------------------------------------
# prepare model
ps = Dict( zip( keys(data_params), values(data_params) ) )
ps[:model]  = parsed_args["model"]
ps[:steps]  = parsed_args["num-samples"]
ps[:warmup] = parsed_args["num-warmup"]
ps[:chains] = parsed_args["num-chains"]
ps[:seed]   = parsed_args["seed"]
ps[:folder] = parsed_args["foldername"]
ps[:prefix] = isempty(parsed_args["name-prefix"]) ? "" : parsed_args["name-prefix"]*"-"
ps = NamedTuple(ps)

const model = Regional.name2model[ps.model]
!isnothing(ps.seed) && Random.seed!(ps.seed);
@info ps
m = model(turing_data, false)
#----------------------------------------------------------------------------
# sampling

@time chain = if ps.chains > 1
    sample(m, NUTS(ps.warmup, 0.95; max_depth=5), MCMCThreads(), ps.steps + ps.warmup, ps.chains; progress=true)
else
    sample(m, NUTS(ps.warmup, 0.95; max_depth=5), ps.steps + ps.warmup; progress=true) #; max_depth=15
end
#-----------------------------------------------------------------------------
# save params
fdir = projectdir("reports/", ps.folder)
mkpath(fdir)
@info "Saving at: $fdir"
let
    dic = Dict( zip( keys(ps), values(ps) ) )
    safesave( normpath(fdir, ps.prefix*"params.csv"), DataFrame( dic ) )
    bson( normpath(fdir, ps.prefix*"params.bson") ,  dic )
end
#-----------------------------------------------------------------------------
# save chain
chain = let
    chain[ps.warmup+1:end,:,:]
    fname = normpath( fdir, savename(ps.prefix*"CHAIN", ps, "jls") )
    safesave( fname, chain )
    chain[1:10:end,:,:]
end
#-----------------------------------------------------------------------------
@info "make predictions"
generated_posterior = Regional.posterior(model, turing_data, chain)
let
    fname = normpath( fdir, savename(ps.prefix*"GENERATED-QUANTITIES", ps, "bson") )
    dic = Dict( zip( keys(generated_posterior), values(generated_posterior) ) )
    bson( fname ,  dic )
end
## ==========================================================================
@info "plot regions"
for r in Regional.regions
    recipe = Regional.RegionPlottingRecipe(data, generated_posterior, r)
    p = plot(recipe)
    savefig( p, savename(ps.prefix*"FIG-$(uppercase(r))", ps, "html") )
end
##
@info "plot rt"
let
    recipe = Regional.RtsPlottingRecipe(data, generated_posterior)
    p = plot(recipe)
    savefig( p, savename(ps.prefix*"FIG-RT", ps, "html") )
end
##
@info "plot hospitalizations"
let
    recipe = Regional.HospitsPlottingRecipe(data, generated_posterior)
    p = plot(recipe)
    savefig( p, savename(ps.prefix*"FIG-HOSPIT", ps, "html") )
end

#-----------------------------------------------------------------------------
# @info "store reproduction number"
# rt = let
#     ib = findfirst(==(Date("2020-05-15")), data.dates)
#     ie = findfirst(==(Date(ps.observ)), data.dates)
#     Rt_array = hcat(Rt...)[ib:ie,:]
#
#     qs = [quantile(v, [0.025, 0.25, 0.5, 0.75, 0.975]) for v in eachrow(Rt_array)]
#     llq, lq, mq, uq, uuq = (eachrow(hcat(qs...))..., )
#
#     date = data.dates[ib:ie]
#     DataFrame((;date, llq, lq, mq, uq, uuq))
# end
# fname = normpath( fdir, savename(ps.prefix*"Rt", ps, "csv") )
# save(fname, rt)

#-----------------------------------------------------------------------------
# perform diagnostics
if ps.chains > 1
    @info "gelman diagnostics"
    diagnostics = gelmandiag(chain)
    fname = normpath( fdir, savename(ps.prefix*"GELMANDIAG", ps, "csv") )
    safesave( fname, diagnostics)
    pretty_table(diagnostics; crop=:none)
end
#-----------------------------------------------------------------------------
@info "plot chain"
n = filter( x->!occursin(r"latent_Rts", x), String.(names(chain)))
let
    p = plot(chain[n])
    fname = normpath( fdir, savename(ps.prefix*"CHAINSPLOT", ps, "html") )
    savefig(p, fname )
    parsed_args["plot-results"] && run(`firefox $(fname)`, wait=false)
end
#-----------------------------------------------------------------------------
@info "meanplot"
let
    p = meanplot(chain[n]);
    fname = normpath( fdir, savename(ps.prefix*"MEANPLOT", ps, "html") )
    savefig(p, fname )
    parsed_args["plot-results"] && run(`firefox $(fname)`, wait=false)
end
#-----------------------------------------------------------------------------
# @info "plot prior vs. posterior"
# pchain = sample(m, Prior(), ps.steps + ps.warmup; progress=true)[ps.warmup+1:end,:,:]
# n = filter( x->!occursin(r"\[", x), String.(names(pchain)))
# p = density(chainscat(pchain[n], chain[n]))
# savefig(p, projectdir("figures/tmp", fname*"_PRIORvsPOSTERIOR.html") )
# run(`firefox $(projectdir("figures/tmp", fname*"_PRIORvsPOSTERIOR.html"))`, wait=false)
