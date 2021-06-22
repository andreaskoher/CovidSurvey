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
## ============================================================================

## ===========================================================================
# load data and model
# ============================================================================
fdir_p = projectdir("reports/2021-06-07/")
fdir_np = projectdir("reports/variable-delays/")
# fname = "cases=2020-06-01_chains=10_hospit=true_model=parametric-cases_observ=2021-01-01_preds=CF,CC,CR,CS_sero=true_steps=4000_warmup=1000"
# fname = "cases=2020-06-01_chains=10_hospit=true_model=parametric-hospit_observ=2021-01-01_preds=CF,CC,CR,CS_sero=true_steps=4000_warmup=1000"
fname_p = "cases=2020-06-01_chains=10_hospit=true_model=parametric-deaths_observ=2021-01-01_preds=CF,CC,CR,CS_sero=true_steps=4000_warmup=1000"
fname_np = "cases=2020-06-01_chains=10_hospit=true_model=deaths_observ=2021-02-06_sero=true_steps=4000_warmup=1000"

## read chain
chain = read(fdir*"CHAIN_"*fname*".jls", Chains)
chain = chain[1:10:end,:,:]
# chain = chain[1:10:end,:,[1,2,3,4,6,7,8,9,10]]
diagnostics = gelmandiag(chain)
"/home/and/dev/CovidSurvey/reports/2021-06-07/Rt_"
Plots.scalefontsizes(1.3)
pgfplotsx()
p = let
    rtp = load(fdir*"Rt_"*fname*".csv") |> DataFrame
    p = plot(legend=:bottomright, ylabel="Rt")
    plot!(rtp.date, rtp.mq, ribbon=(rtp.mq - rtp.llq, rtp.uuq - rtp.mq), lw=2, label="with predictors (contact data)")
    plot!(rt.date, rt.mq, ribbon=(rt.mq - rt.llq, rt.uuq - rt.mq), lw=2, label="without predictors")
    # xlims!((t[1],t[end]))
end
savefig(projectdir("figures/compare_Rt.png"))
