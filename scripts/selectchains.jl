using DrWatson
quickactivate(@__DIR__)
@show projectdir()
##
using CovidSurvey
using CSV
using DataFrames
using Dates
# using RCall
# using Turing
using StatsPlots
using Plots.PlotMeasures
using Random
using BSON
using PrettyTables
## ===========================================================================
# load data and model
# ============================================================================
fdir = projectdir("reports/parametric-model-on-2nd-wave_#2/")
fname = "cases=2020-06-01_chains=10_hospit=true_model=parametric-cases_observ=2021-01-13_preds=CF,CC,CR,CS_sero=true_steps=4000_warmup=1000"
PREFIX = ""
epimodel = CovidSurvey.National.model_parametric_cases
#-----------------------------------------------------------------------------
# read params
_, ps, _ = parse_savename(fname, parsetypes = (Int, Float64, Bool))
!("preds" in keys(ps)) && (ps["preds"] = nothing)
# ps = NamedTuple{Tuple(Symbol.(keys(ps)))}(values(ps))

#------------------------------------------------------------------------------
# read data
data = National.load_data(
      ps["observ"]
    , ps["preds"]|> CovidSurvey.parse_predictors
    , ps["cases"]
    , ps["hospit"]
    , ps["sero"]
    , iar_step = 1
    , covariates_kwargs = Dict(
        :fname => normpath( homedir(), "data/covidsurvey/smoothed_contacts.csv" ),
        :shift => -1,
        :startdate => "2020-11-10", # >= 1000 cases / day
        # :enddate => "2021-01-13" # <= 1000 cases / day
    )
)
turing_data = data.turing_data;

#-----------------------------------------------------------------------------
# read chain
chain = read(fdir*PREFIX*"CHAIN_"*fname*".jls", Chains)
chain = chain[1:10:end,:,[2,4,5,6,7,8,9]]
# fname = normpath( fdir, savename(PREFIX*"CHAIN", ps, "jls") )
# safesave( fname, chain)

#-----------------------------------------------------------------------------
# plot chain
n = filter( x->!occursin(r"(?<!α)\[", x), String.(names(chain)))
let
    p = plot(chain[n])
    fname = normpath( fdir, savename(PREFIX*"CHAINSPLOT", ps, "html") )
    savefig(p, fname )
    run(`firefox $(fname)`, wait=false)
end

## ===========================================================================
# plot results
# ============================================================================
# make predictions
posterior = let
    m_pred = epimodel(turing_data..., true; link=KLogistic(3.), invlink=KLogit(3.) )
    gq = Turing.generated_quantities(m_pred, chain)
    vectup2tupvec( reshape(gq, length(gq)) )
end;
# observed_daily_cases, expected_daily_deaths, expected_daily_hospit, Rt, expected_seropositive, iar = posterior;

#-----------------------------------------------------------------------------
@info "store generated quantities"
let
    dic = Dict( zip( keys(posterior), values(posterior) ) )
    fname = normpath( fdir, savename(PREFIX*"GENERATED-QUANTITIES", ps, "bson") )
    bson( fname ,  dic )
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
