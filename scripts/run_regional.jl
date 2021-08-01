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
# read input
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
        default = 100
    "--num-chains", "-t"
        help = "number of chains sampled with multi-threading"
        arg_type = Int
        default = 1
    "--num-warmup", "-w"
        help = "number of samples to use for warmup/adaptation"
        arg_type = Int
        default = 100
    "--observations-end", "-o"
        help = "end of observation date as string yyyy-mm-dd. Use full data set by defaults"
        arg_type = String
        default = nothing
    "--cases-start", "-s"
        help = "start date for case observations as string yyyy-mm-dd. No case observations by default"
        arg_type = String
        default = ""
    "--add-covariates", "-c"
        help = "list of covariate names (default empty string = no covariates). Choose: C: contacts, R:"
        arg_type = String
        default = nothing
    "--model", "-m"
        help = "choose from: 'hospit', 'cases', 'deaths'. Defaul: 'hospit'"
        arg_type = String
        default = "hospit"
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
      observationsend   = parsed_args["observations-end"]
    , predictors        = parsed_args["add-covariates"]|> CovidSurvey.parse_predictors
    # , hospitmodel       = Regional.HospitInit1(obs_stop="2020-07-01")
    , casemodel         = Regional.CaseInit1(obs_start=parsed_args["cases-start"])
    , seromodel         = Regional.SeroInit2(delay=0, std=1.)
    , rwstep           = 1
    , epidemicstart    = 10
    , numimpute        = 6
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
m();
## ==========================================================================
# sampling
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
fdir = projectdir("reports/", ps.folder)
ignores = [k for (k,v) in pairs(ps) if ((v isa String) && isempty(v))]

mkpath(fdir)
@info "Saving at: $fdir"
fname = let
    dic = Dict( zip( keys(ps), values(ps) ) )
    fname = normpath(fdir, savename(ps.prefix*"PARAMS", ps, "csv"; ignores) )
    safesave( fname, DataFrame( dic ) )
    bson( normpath(fdir, ps.prefix*"params.bson") ,  dic )

    fname = normpath( fdir, savename(ps.prefix*"DATA", ps, "bson"; ignores) )
    bson( fname, Dict("data"=>data) )

    fname = normpath( fdir, savename(ps.prefix*"CHAIN", ps, "jls"; ignores) )
    safesave( fname, chain )
    fname
end

## ==========================================================================
@info "post-processing"
Regional.postprocessing(fname)
