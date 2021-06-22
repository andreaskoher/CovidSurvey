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
plotlyjs()

# using DataFramesMeta
# plotlyjs()
const dateform = "yyyymmdd"
include( projectdir("scripts/survey/rawsurvey_utils.jl") )

## ============================================================================
contacts = readcontacts( "/home/and/data/covidsurvey/rawcontacts.csv" )
## =============================================================================
# remove outliers
contacts = filteroutliers(contacts)
contact_histograms(contacts; day=nothing, bins=0:10)
## =============================================================================
# plot daily aggregated time series
plot_confidence_timeseries(contacts)
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
## save and plots results
# bson("survey_inference_results_method=$(results.problemtype)", Dict(:results => results))
# BSON.load("survey_inference_results_method=ZINegBinomialProblem.bson")
# save(projectdir("data","inferred_contact_rates.csv"), contactsrates)
plot_confidence_timeseries(results)
plot_generated_quantities(results, compare_old=true)
plot_survey_histograms(results, day=nothing)
## =============================================================================
using TimeSeriesDecompositions

contactrates = load(projectdir("data","inferred_contact_rates.csv")) |> DataFrame

slowtrends = decompose(contactrates; fdir=projectdir("figures/"))
save(projectdir("data","smoothed_contact_rates.csv"), slowtrends)
bson("contacts_decomposition=$(results.problemtype)", Dict(:slowtrends => slowtrends))
BSON.load("survey_inference_results_method=ZINegBinomialProblem.bson")
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
