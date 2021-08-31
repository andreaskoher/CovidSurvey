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
import TimeSeriesDecompositions as TSD
plotlyjs()

# using DataFramesMeta
# plotlyjs()
const dateform = "yyyymmdd"
include( projectdir("scripts/survey/rawsurvey_utils.jl") )

## ============================================================================
function plot_contacts_per_day(df; region=0)
    gp = groupby(df[:,[:date]], :date, sort=true)
    cb = combine(gp, nrow)
    p = plot(cb.date, cb.nrow, ylab="# contacts", xticks=:native)
    savefig(p, projectdir("scripts/survey/","contacts_per_day_region=$region.html"))
    return p
end

function main()
    plotlyjs()
    for region in 1:5
        @info "load data region $region"
        rawcontacts = readcontacts( "/home/and/data/covidsurvey/rawcontacts.csv", select_region=region )
        plot_contacts_per_day(rawcontacts; region)
        threshold = (family=50, colleagues=100, friends=100, strangers=1000)
        contacts = filteroutliers(rawcontacts, threshold) #NOTE National: p = 99.9

        @info "zero inflated regression region $region"
        surveyproblem = SurveyInferenceProblem(
            ZINegBinomialProblem,
            ctype,
            contacts
        )

        results = optimize(surveyproblem, LBFGS(),
            Optim.Options(show_trace=true, iterations=2000);
        )

        contactrates = DataFrame(results)
        save(projectdir("data","raw_contact_rates_region=$region.csv"), contactrates)

        let
            fname = projectdir("scripts/survey/","survey_inference_results_method=$(results.problemtype)_region=$region.bson")
            save(fname, θ = results)
        end

        plot_confidence_timeseries(results)

        # pyplot()
        # p = plot_survey_histograms(results, day=nothing)
        # savefig(p, projectdir("figures/histogram_survey_data_vs_fake_data_region=$region.png"))
        # plotlyjs()

        @info "decomposition region $region"
        slowtrends = TSD.decompose(contactrates, Optim.Options(show_trace=true, iterations=2000); fdir=projectdir("figures/"))
        save(projectdir("data","smoothed_contact_rates_region=$region.csv"), slowtrends)
    end
end

# main()

## =========================================================================================================
@info "load data"
rawcontacts = readcontacts( normpath( homedir(), "data/covidsurvey/rawcontacts.csv"), select_region=0 )
p = plot_contacts_per_day(rawcontacts)
threshold = 99.9 #NOTE National
contacts = filteroutliers(rawcontacts, threshold)
contacts[!,:total] .= 0
for c in ctype
    contacts[!,:total] += contacts[:,c]
end
contacts = contacts[:, [:total, :strangers, :date]]
## =========================================================================================================
@info "zero inflated regression"
surveyproblem = SurveyInferenceProblem(
    ZINegBinomialProblem,
    datacols(contacts),
    contacts
)

results = optimize(surveyproblem, LBFGS(),
    Optim.Options(show_trace=true, iterations=2000);
)

contactrates = DataFrame(results)
save(projectdir("data","raw_total_contact_rates.csv"), contactrates)

let
    fname = projectdir("scripts/survey/","survey_inference_results_method=$(results.problemtype)_TOTAL.bson")
    save(fname, θ = results)
end

plot_confidence_timeseries(results)

pyplot()
p = plot_survey_histograms(results, day=nothing)
# savefig(p, projectdir("figures/histogram_survey_data_vs_fake_data_region=$region.png"))
plotlyjs()

@info "decomposition region $region"
slowtrends = TSD.decompose(contactrates, Optim.Options(show_trace=true, iterations=2000); fdir=projectdir("figures/"))
save(projectdir("data","smoothed_total_contact_rates.csv"), slowtrends)
