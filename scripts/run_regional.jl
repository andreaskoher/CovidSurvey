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
using OrderedCollections

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
        default = 10
    "--rwstep", "-r"
        help = "random walk step size. Defaults to 7"
        arg_type = Int
        default = 7
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
        default = "2021-02-01"
    "--cases-start", "-s"
        help = "start date for case observations as string yyyy-mm-dd. case observations start on 2020-06-01 by default"
        arg_type = String
        default = "2020-06-01"
    "--add-covariates", "-c"
        help = "list of covariate names (default empty string = no covariates). Choose: C: contacts, R:"
        arg_type = String
        default = nothing
    "--model", "-m"
        help = "choose from: 'hospit', 'cases', 'deaths'. Defaul: 'hospit'"
        arg_type = String
        default = "hospit"
    "--standardize", "-z"
        help = "normalize predictors to unit standard deviation"
        arg_type = Bool
        default = false
    "--plot-results", "-p"
        arg_type = Bool
        default = false
    "--foldername", "-f"
        arg_type = String
        default = "$(today())"
    "--name-prefix", "-u"
        arg_type = String
        default = ""
    "--include-early-dynamics", "-i"
        arg_type = Bool
        default = false
    "--semiparametric", "-S"
        arg_type = Bool
        default = true
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
    , predictors        = parsed_args["add-covariates"] |> CovidSurvey.parse_predictors
    # , hospitmodel       = Regional.HospitInit1(obs_stop="2020-07-01")
    , casemodel         = Regional.CaseInit1(obs_start=parsed_args["cases-start"])
    , seromodel         = Regional.SeroInit2(delay=0, std=.5)
    , rwstep           = parsed_args["rwstep"]
    , epidemicstart    = 20
    , numimpute        = 6
    , include_early_dynamic = parsed_args["include-early-dynamics"]
    , link              = KLogistic(4.5)
    , invlink           = KLogit(4.5)
    , covariates_kwargs = Dict(
      :fname => [normpath( projectdir("data"), "mean_contact_rates_region=$r.csv" ) for r in 1:Regional.nregions],
      :mobility => [projectdir("data/mobility/mobility_region=$(Regional.regions[i]).csv") for i in 1:Regional.nregions],
      :semiparametric => parsed_args["semiparametric"],
      :shift          => -1,
      :startdate      => "2020-11-10", # >= 1000 cases / day
      :datecol        => "dates",
      :aggregation    => 1,
      :standartize    => parsed_args["standardize"],
    )
    )

data = Regional.load_data(; data_params... )
turing_data = data.turing_data;
#----------------------------------------------------------------------------
# prepare model
# ps = OrderedDict( zip( keys(data_params), values(data_params) ) )
# ps[:regional] = true
# ps[:model]  = parsed_args["model"]
# ps[:steps]  = parsed_args["num-samples"]
# ps[:warmup] = parsed_args["num-warmup"]
# ps[:chains] = parsed_args["num-chains"]
# ps[:rwstep] = parsed_args["rwstep"]
# ps[:seed]   = parsed_args["seed"]
# ps[:prefix] = isempty(parsed_args["name-prefix"]) ? "" : parsed_args["name-prefix"]*"-"
# ps = NamedTuple(ps)

ps = (
        regional = true,
        model    = parsed_args["model"],
        preds    = parsed_args["add-covariates"],
        steps    = parsed_args["num-samples"],
        warmup   = parsed_args["num-warmup"],
        observ   = parsed_args["observations-end"],
        chains   = parsed_args["num-chains"],
        rwstep   = parsed_args["rwstep"],
        seed     = parsed_args["seed"],
        std      = parsed_args["standardize"],
        semipar  = parsed_args["semiparametric"],
        prefix   = isempty(parsed_args["name-prefix"]) ? "" : parsed_args["name-prefix"]*"-"
)
# cases    = parsed_args["cases-start"],
# predstart= parsed_args["covariates-start"]


const model = Regional.name2model[ps.model]
!isnothing(ps.seed) && Random.seed!(ps.seed);
@info ps
m = model(turing_data, vectorize=true)
m();
## ==========================================================================
# sampling
@time chain = let
    thinning = 10
    if ps.chains > 1
        sample(m, NUTS(ps.warmup, 0.95; max_depth=6), MCMCThreads(), ps.steps + ps.warmup, ps.chains; progress=true, thinning)
    else
        sample(m, NUTS(ps.warmup, 0.95; max_depth=6), ps.steps + ps.warmup; progress=true, thinning) #; max_depth=15
    end
end
## ==========================================================================
@info "save"
fdir = projectdir("reports", parsed_args["foldername"])
fname = Regional.save_results(fdir, ps, data, chain)

## ==========================================================================
@info "post-processing"
Regional.postprocessing(fname)
