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
        default = ""
    "--add-covariates", "-c"
        help = "list of covariate names (default empty string = no covariates). Choose: C: contacts, R:"
        arg_type = String
        default = nothing
    "--model", "-m"
        help = "choose from: 'v1', 'gp'. Defaul: 'v1'"
        arg_type = String
        default = "hospit"
    "--plot-results", "-p"
        arg_type = Bool
        default = false
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
# helper functions for parsing

name2model = Dict(
    "v3" => National.model_v3,
    "v2" => National.model_v2,
    "contacts2" => National.model_contacts_v2,
    # "contacts" => National.model_contacts,
    "contacts3" => National.model_contacts_v3,
    "contacts4" => National.model_contacts_v4,
    "const-iar" => National.model_constant_iar,
    "const-iar-c" => National.model_constant_iar_contacts,
    "contacts-shifted" => National.model_contacts_shifted,
    "deaths" => National.model_deaths,
    "cases" => National.model_cases,
    "hospit" => National.model_hospit,
    "contacts3" => National.model_contacts_v3,
    # "parametric-cases" => National.model_parametric_cases,
    # "parametric-hospit" => National.model_parametric_hospit,
    # "parametric-deaths" => National.model_parametric_deaths
)
#----------------------------------------------------------------------------
# load data
@info "load data"
data = National.load_data(
      observations_end  = parsed_args["observations-end"]
    , predictors        = parsed_args["add-covariates"]|> CovidSurvey.parse_predictors
    , rw_step           = 1
    , epidemic_start    = 30
    , num_impute        = 6
    # , deathmodel        = National.DeathInit(obs_stop="$( Date(parsed_args["cases-start"]) + Day(1))")
    # , casemodel         = National.CaseInit(obs_start=parsed_args["cases-start"])
    # , hospitmodel       = National.HospitInit()
    , link              = KLogistic(3.)
    , invlink           = KLogit(3.)
    , lockdown          = "2020-03-18"
    , covariates_kwargs = Dict(
        :semiparametric => parsed_args["semiparametric"],
        :fname => projectdir("data","smoothed_contact_rates.csv"),
        :shift => -1,
        :startdate => "2020-11-10",
        :enddate => nothing))

turing_data = data.turing_data;
#----------------------------------------------------------------------------
# prepare model
ps = (
    model    = parsed_args["model"],
    preds    = parsed_args["add-covariates"],
    steps    = parsed_args["num-samples"],
    cases    = parsed_args["cases-start"],
    warmup   = parsed_args["num-warmup"],
    observ   = parsed_args["observations-end"],
    chains   = parsed_args["num-chains"],
    seed     = parsed_args["seed"],
    semipar  = parsed_args["semiparametric"],
)
PREFIX = isempty(parsed_args["name-prefix"]) ? "" : parsed_args["name-prefix"]*"-"

const model = name2model[ps.model]
# model = name2model["hospit"]
!isnothing(ps.seed) && Random.seed!(ps.seed);
@info ps
m = model(turing_data, false)
#----------------------------------------------------------------------------
# start sampling
# function trysample(m, ps)
#     try
#         return sample(m, NUTS(ps.warmup, 0.95), ps.steps + ps.warmup; progress=true)
#     catch e
#         @error e
#         return nothing
#     end
# end
#
# function runsim(m,ps)
#     for i in 1:100
#         @info "trial number $i"
#         chain = trysample(m,ps)
#         !isnothing(chain) && return chain
#     end
#     @error "exceeded number of failed trials"
# end
#
# @time chain = runsim(m,ps)
@time chain = if ps.chains > 1
    sample(m, NUTS(ps.warmup, 0.95; max_depth=5), MCMCThreads(), ps.steps + ps.warmup, ps.chains; progress=true)
else
    sample(m, NUTS(ps.warmup, 0.95; max_depth=5), ps.steps + ps.warmup; progress=true) #; max_depth=15
end
#-----------------------------------------------------------------------------
# save params
fdir = projectdir("reports/", parsed_args["foldername"])
mkpath(fdir)
@info "Saving at: $fdir"
safesave( normpath(fdir, PREFIX*"params.csv"), DataFrame( parameter=collect(keys(ps)), value=collect(values(ps)) ) )
#-----------------------------------------------------------------------------
@info "plot chain"
# n = filter( x->!occursin(r"(?<!α)\[", x), String.(names(chain)))
let
    n = filter( x->!occursin(r"latent_Rt", x), String.(names(chain)))
    p = plot(chain[n]);
    fname = normpath( fdir, savename(PREFIX*"FIG-CHAINSPLOT", ps, "html") )
    savefig(p, fname )
    parsed_args["plot-results"] && run(`firefox $(fname)`, wait=false)
end
#-----------------------------------------------------------------------------
@info "meanplot"
let
    n = filter( x->!occursin(r"latent_Rt", x), String.(names(chain)))
    p = meanplot(chain[n]);
    fname = normpath( fdir, savename(PREFIX*"FIG-MEANPLOT", ps, "html") )
    savefig(p, fname )
    parsed_args["plot-results"] && run(`firefox $(fname)`, wait=false)
end
#-----------------------------------------------------------------------------
# save chain
chain = let chain = chain[ps.warmup+1:10:end,:,:]
    fname = normpath( fdir, savename(PREFIX*"CHAIN", ps, "jls") )
    safesave( fname, chain)
    chain
end
#-----------------------------------------------------------------------------
@info "make predictions"
generated_posterior = let
    m_pred = model(turing_data, true)
    gq = Turing.generated_quantities(m_pred, chain)
    vectup2tupvec(reshape(gq, length(gq)))
end
#-----------------------------------------------------------------------------
@info "store generated quantities"
let
    fname = normpath( fdir, savename(PREFIX*"GENERATED-QUANTITIES", ps, "bson") )
    dic = Dict( zip( keys(generated_posterior), values(generated_posterior) ) )
    bson( fname ,  dic )
end
#-----------------------------------------------------------------------------
@info "store reproduction number"
let
    rt = let
        ib = findfirst(==(Date("2020-05-15")), data.dates)
        de = isnothing(ps.observ) ? last(data.cases.date) : Date(ps.observ)
        ie = findfirst(==(de), data.dates)
        Rt_array = hcat(generated_posterior.Rt...)[ib:ie,:]

        qs = [quantile(v, [0.025, 0.25, 0.5, 0.75, 0.975]) for v in eachrow(Rt_array)]
        llq, lq, mq, uq, uuq = (eachrow(hcat(qs...))..., )

        date = data.dates[ib:ie]
        DataFrame((;date, llq, lq, mq, uq, uuq))
    end
    fname = normpath( fdir, savename(PREFIX*"Rt", ps, "csv") )
    save(fname, rt)
end
#-----------------------------------------------------------------------------
@info "plot results"
let
    p = National.plot(data, generated_posterior);
    fname = normpath( fdir, savename(PREFIX*"FIG-PREDICTION", ps, "html") )
    savefig(p, fname )
    # savefig(p, projectdir("figures", fname*".png") )
    parsed_args["plot-results"] && run(`firefox $(fname)`, wait=false)
end
#-----------------------------------------------------------------------------
# perform diagnostics
if ps.chains > 1
    @info "gelman diagnostics"
    diagnostics = gelmandiag(chain)
    fname = normpath( fdir, savename(PREFIX*"GELMANDIAG", ps, "csv") )
    safesave( fname, diagnostics)
    pretty_table(diagnostics; crop=:none)
end
#-----------------------------------------------------------------------------
# @info "plot prior vs. posterior"
# pchain = sample(m, Prior(), ps.steps + ps.warmup; progress=true)[ps.warmup+1:end,:,:]
# n = filter( x->!occursin(r"\[", x), String.(names(pchain)))
# p = density(chainscat(pchain[n], chain[n]))
# savefig(p, projectdir("figures/tmp", fname*"_PRIORvsPOSTERIOR.html") )
# run(`firefox $(projectdir("figures/tmp", fname*"_PRIORvsPOSTERIOR.html"))`, wait=false)
