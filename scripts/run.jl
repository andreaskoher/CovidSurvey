using DrWatson
quickactivate(@__DIR__)
@show projectdir()
#-----------------------------------------------------------------------------
# load parameters
using ArgParse
using Dates
argtable3 = ArgParseSettings(
    description="This script samples from National.model using NUTS."
)
@add_arg_table! argtable3 begin
    "--chunksize"
        help = "chunksize to be used by ForwardDiff.jl"
        arg_type = Int
        default = 40
    "--num-samples", "-n"
        help = "number of samples"
        arg_type = Int
        default = 10
    "--num-chains", "-t"
        help = "number of chains sampled with multi-threading"
        arg_type = Int
        default = 1
    "--num-warmup", "-w"
        help = "number of samples to use for warmup/adaptation"
        arg_type = Int
        default = 10
    "--observations-end", "-o"
        help = "end of observation date as string yyyy-mm-dd. Use full data set by defaults"
        arg_type = String
        default = nothing
    "--cases-start", "-s"
        help = "start date for case observations as string yyyy-mm-dd. No case observations by default"
        arg_type = String
        default = "2020-06-01"
    "--covariates-start", "-p"
        help = "start date for covariates as string yyyy-mm-dd. starts on 2020-11-10 by default"
        arg_type = String
        default = "2020-11-10"
    "--add-covariates", "-c"
        help = "list of covariate names (default empty string = no covariates). Choose: C: contacts, R:"
        arg_type = String
        default = nothing
    "--model", "-m"
        help = "choose from: 'v1', 'gp'. Defaul: 'v1'"
        arg_type = String
        default = "hospit"
    "--semiparametric", "-S"
        arg_type = Bool
        default = true
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
parsed_args = parse_args(ARGS, argtable3)
#-----------------------------------------------------------------------------
# load packages
@info "load packages"
using CovidSurvey
using Random, Turing
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
#----------------------------------------------------------------------------
# load data
@info "load data"
data = National.load_data(
      observationsend  = parsed_args["observations-end"]
    , predictors        = parsed_args["add-covariates"]|> CovidSurvey.parse_predictors
    , rwstep           = 1
    , epidemicstart    = 30
    , numimpute        = 6
    , deathmodel        = parsed_args["model"] == "cases" ? National.DeathInit(obs_stop="$( Date(parsed_args["cases-start"]) + Day(30))") : National.DeathInit()
    , casemodel         = National.CaseInit(obs_start=parsed_args["cases-start"])
    # , hospitmodel       = National.HospitInit()
    , seromodel         = National.SeroInit1(delay=0, std=.5)
    , link              = KLogistic(3.)
    , invlink           = KLogit(3.)
    , lockdown          = "2020-03-18"
    , covariates_kwargs = Dict(
        :semiparametric => parsed_args["semiparametric"],
        :fname => projectdir("data","raw_contact_rates.csv"),
        :shift => -1,
        :startdate => parsed_args["covariates-start"],
        :enddate => nothing))

turing_data = data.turing_data;
#----------------------------------------------------------------------------
# prepare model
ps = (
    regional = false,
    model    = parsed_args["model"],
    preds    = parsed_args["add-covariates"],
    steps    = parsed_args["num-samples"],
    cases    = parsed_args["cases-start"],
    warmup   = parsed_args["num-warmup"],
    observ   = parsed_args["observations-end"],
    chains   = parsed_args["num-chains"],
    seed     = parsed_args["seed"],
    semipar  = parsed_args["semiparametric"],
    predstart= parsed_args["covariates-start"]
)
PREFIX = isempty(parsed_args["name-prefix"]) ? "" : parsed_args["name-prefix"]*"-"

model = National.name2model[ps.model]
# model = name2model["hospit"]
!isnothing(ps.seed) && Random.seed!(ps.seed);
@info ps
m = model(turing_data, false)
#----------------------------------------------------------------------------
@time chain = let
    thinning = 10
    if ps.chains > 1
        sample(m, NUTS(ps.warmup, 0.95; max_depth=5), MCMCThreads(), ps.steps + ps.warmup, ps.chains; progress=true, thinning)
    else
        sample(m, NUTS(ps.warmup, 0.95; max_depth=5), ps.steps + ps.warmup; progress=true, thinning) #; max_depth=15
    end
end
#-----------------------------------------------------------------------------
@info "save result"
fdir = projectdir("reports/", parsed_args["foldername"])
fname = National.save_results(fdir, ps, data, chain)

#-----------------------------------------------------------------------------
@info "post processing"
National.postprocessing(fname; plot_results=false)
