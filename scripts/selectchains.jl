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
using Turing
plotlyjs()
## ===========================================================================
# load data and model
# ============================================================================
fdir = projectdir("reports/parametric-model-on-2nd-wave-newdata/")
# fname = "cases=2020-06-01_chains=10_hospit=true_model=parametric-cases_observ=2021-01-13_preds=CF,CC,CR,CS_sero=true_steps=2000_warmup=1000"
fname = "cases=2020-06-01_chains=10_hospit=true_model=parametric-cases_observ=2021-01-13_preds=CF,CC,CR,CS_sero=true_steps=2000_warmup=1000"
PREFIX = "RAW-"
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
        :fname => projectdir("data", "inferred_contact_rates.csv"),
        #:fname => projectdir("data", "smoothed_contact_rates.csv"), #normpath( homedir(), "data/covidsurvey/smoothed_contacts.csv" ),
        # :fname => normpath( homedir(), "data/covidsurvey/smoothed_contacts.csv" ),
        :shift => -1,
        :startdate => "2020-11-10",
        # :enddate => "2021-01-13"
    )
)
turing_data = data.turing_data;

#-----------------------------------------------------------------------------
# read chain
function not(c::Chains, i)
    n = size(c, 3)
    s = filter(x->x∉i, 1:n)
    return c[:,:,s]
end
chain = read(fdir*PREFIX*"CHAIN_"*fname*".jls", Chains)
chain = chain[1:10:end,:,:]
chain = not(chain, [2,4])
# fname = normpath( fdir, savename(PREFIX*"CHAIN", ps, "jls") )
# safesave( fname, chain)

#-----------------------------------------------------------------------------
# plot chain
n = filter( x->!occursin(r"(?<!α)\[", x), String.(names(chain)))
let
    p = plot(chain[n])
    saveto = normpath( fdir, PREFIX*"CHAINSPLOT_"*fname*".html")
    savefig(p, saveto )
    run(`firefox $(saveto)`, wait=false)
end

## ===========================================================================
# plot results
# ============================================================================
# make predictions
m_pred = epimodel(turing_data..., true; link=KLogistic(3.), invlink=KLogit(3.))
gq = Turing.generated_quantities(m_pred, chain)
generated_posterior = vectup2tupvec(reshape(gq, length(gq)))

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
    saveto = normpath( fdir, PREFIX*"GENERATED-QUANTITIES_"*fname*".bson")
    bson( saveto ,  dic )
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
    saveto = normpath( fdir, PREFIX*"Rt_"*fname*".csv")
    save(saveto, rt)
    rt
end
#-----------------------------------------------------------------------------
@info "plot results"
let
    p = National.plot_results(data, posterior...);
    saveto = normpath( fdir, PREFIX*"PREDICTION_"*fname*".html")
    savefig(p, saveto )
    # savefig(p, projectdir("figures", fname*".png") )
    run(`firefox $(saveto)`, wait=false)
end
#-----------------------------------------------------------------------------
# perform diagnostics
if ps["chains"] > 1
    @info "gelman diagnostics"
    diagnostics = gelmandiag(chain)
    saveto = normpath( fdir, PREFIX*"GELMANDIAG_"*fname*".csv")
    safesave( saveto, diagnostics)
    pretty_table(diagnostics; crop=:none)
end
#-----------------------------------------------------------------------------
@info "meanplot"
let
    p = meanplot(chain[n]);
    saveto = normpath( fdir, PREFIX*"MEANPLOT_"*fname*".html")
    savefig(p, saveto )
    run(`firefox $(saveto)`, wait=false)
end
