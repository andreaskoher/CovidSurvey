using DrWatson
quickactivate(@__DIR__)
@show projectdir()
##
using CSV
using DataFrames
using Turing
using Dates
using StatsBase
using Distributions
using CovidSurvey
# using Gadfly
using StatsPlots
using LaTeXStrings
using NamedTupleTools
using StatsFuns
using Random
using TransformVariables, LogDensityProblems
using Optim
using ReverseDiff
using BSON
using TimeSeriesDecompositions
plotlyjs()

# using DataFramesMeta
# plotlyjs()
const dateform = "yyyymmdd"
include( projectdir("scripts/survey/rawsurvey_utils.jl") )

## ============================================================================
region = 4
rawcontacts = readcontacts( "/home/and/data/covidsurvey/rawcontacts.csv", select_region=region )
let
    gp = groupby(rawcontacts[:,[:date]], :date, sort=true)
    cb = combine(gp, nrow)
    @show minimum(cb.nrow)
    histogram(cb.nrow, bins=20)
end
## =============================================================================
# remove outliers
# pgfplotsx()
# scalefontsizes(1/1.5)
threshold = (family=50, colleagues=100, friends=100, strangers=1000)
contacts = filteroutliers(rawcontacts, threshold) #NOTE National: p = 99.9
p = contact_histograms(contacts; bins=0:40, fc=:gray)
# saveto = projectdir("figures","survey_histogram.png")
# savefig(p, saveto)
## =============================================================================
# plot daily aggregated time series
# plotlyjs()
p = plot_confidence_timeseries(contacts);

## =============================================================================
# zero contacts
zerocontacts = countzeros(contacts)
plot_timeseries(zerocontacts)

## =============================================================================
# inference with zero inflated model

surveyproblem = SurveyInferenceProblem(
    ZINegBinomialProblem,
    ctype,
    contacts
)

results = optimize(surveyproblem, LBFGS(),
    Optim.Options(show_trace=true, iterations=2000);
)
contactrates = DataFrame(results)

let
    fname = projectdir("scripts/survey/","survey_inference_results_method=$(results.problemtype)_region=$region.bson")
    save(fname, θ = results)
end
plot_confidence_timeseries(results)
## save and plots results
using FillArrays
BSON.@load projectdir("scripts/survey/survey_inference_results_method=ZINegBinomialProblem_region=$region.bson") results
# save(projectdir("data","inferred_contact_rates.csv"), contactsrates)
# plot_generated_quantities(results; compare_old=true)
pgfplotsx()
p = plot_survey_histograms(results, day=nothing)
savefig(p, projectdir("figures/histogram_survey_data_vs_fake_data.png"))
## load results
contactrates = load(projectdir("data","inferred_contact_rates.csv")) |> DataFrame
## =============================================================================
# decomposition
import TimeSeriesDecompositions as TSD

slowtrends = TSD.decompose(contactrates; fdir=projectdir("figures/"))
save(projectdir("data","smoothed_contact_rates_region=$region.csv"), slowtrends)
# bson("contacts_decomposition=$(results.problemtype)_region=$region", Dict(:slowtrends => slowtrends))
## load results
slowtrends = load(projectdir("data","smoothed_contact_rates_region=$region.csv"))
# BSON.load("survey_inference_results_method=ZINegBinomialProblem.bson")
##
p = plot_comparison_with_SSI(contactrates);
firefox(p; fname="compare_SSI_with_contactrates.html");

## =============================================================================

olddata = CSV.File("/home/and/data/covidsurvey/smoothed_contacts_0425_2m.csv") |> DataFrame
rename!(olddata, :date=>:dates)
p = plot_comparison(olddata, slowtrends);
firefox(p; fname="compare_contactrates_with_olddata.html");

olddata = CSV.File("/home/and/data/covidsurvey/contacts_0425_2m.csv") |> DataFrame
rename!(olddata, :date=>:dates)
p = plot_comparison(olddata, contactrates);
firefox(p; fname="compare_contactrates_with_olddata.html");


##
