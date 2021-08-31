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
# util functions

function plot_contacts_per_day(df)
    gp = groupby(df[:,[:date]], :date, sort=true)
    cb = combine(gp, nrow)
    return plot(cb.date, cb.nrow, ylab="# contacts", xticks=:native)
end

function support_index(stepsize, support_position)
    support_position == :start && return 1
    support_position == :end && return stepsize
    iseven(stepsize) && return stepsize ÷ 2
    return stepsize ÷ 2 + 1
end

function sampledown(dates, stepsize=1, support_position=:middle)
    @assert issorted(dates)
    @assert all(unique(dates) .== first(dates):Day(1):last(dates))

    s = support_index(stepsize, support_position)
    n = length(dates)

    new_dates = Vector{Date}(undef, n)
    date = first(dates) + Day(s - 1)
    i = 1
    d_old = first(dates)
    for (j,d) in enumerate(dates)
        new_dates[j] = date
        if d == (d_old + Day(1))
            i += 1
            i % stepsize == 0 && ( date += Day(stepsize))
            d_old = date
        end
    end
    new_dates
end

## =============================================================================
# zero inflated analysis

function main(aggregate=1)
    plotlyjs()
    for region in 1:5
        @info "load data region $region"
        rawcontacts = readcontacts( normpath( homedir(), "data/covidsurvey/rawcontacts.csv"), select_region=region )
        plot_contacts_per_day(rawcontacts) |> display
        threshold = (family=50, colleagues=100, friends=100, strangers=1000)
        contacts = filteroutliers(rawcontacts, threshold) #NOTE National: p = 99.9
        contacts[!,:date] = sampledown(contacts.date, aggregate)
        contacts[!,:total] .= 0
        for c in ctype
            contacts[!,:total] += contacts[:,c]
        end
        # contacts = contacts[:, [:total, :strangers, :date]]

        @info "zero inflated regression region $region"
        surveyproblem = SurveyInferenceProblem(
            ZINegBinomialProblem,
            datacols(contacts),
            contacts
        )

        results = optimize(surveyproblem, LBFGS(),
            Optim.Options(show_trace=true, iterations=2000);
        )

        contactrates = DataFrame(results)
        save(projectdir("data","raw_contact_rates_region=$(region)_aggregation=$(aggregate).csv"), contactrates)

        let
            fname = projectdir("scripts/survey/","survey_inference_results_method=$(results.problemtype)_region=$(region)_aggregation=$(aggregate).bson")
            save(fname, θ = results)
        end

        plot_confidence_timeseries(results) |> display
        # pyplot()
        # p = plot_survey_histograms(results, day=nothing)
        # savefig(p, projectdir("figures/histogram_survey_data_vs_fake_data_region=$region.png"))
        # plotlyjs()

        # @info "decomposition region $region"
        # slowtrends = TSD.decompose(contactrates, Optim.Options(show_trace=true, iterations=2000); fdir=projectdir("figures/"))
        # save(projectdir("data","smoothed_contact_rates_region=$(region)_aggregation=$(aggregate).csv"), slowtrends)
    end
end

main(7)


## =========================================================================================================
#  Simple data cleaning with mean instead of zero inflated Negative Binomial
function main(aggregate=1)
    plotlyjs()
    @progress for region in 1:5
        @info "load data region $region"
        rawcontacts = readcontacts( normpath( homedir(), "data/covidsurvey/rawcontacts.csv"), select_region=region )
        threshold = (family=50, colleagues=100, friends=100, strangers=1000)
        contacts = filteroutliers(rawcontacts, threshold) #NOTE National: p = 99.9
        contacts[!,:date] = sampledown(contacts.date, aggregate)
        contacts[!,:total] .= 0
        for c in ctype
            contacts[!,:total] += contacts[:,c]
        end
        # contacts = contacts[:, [:total, :strangers, :date]]


        gp = groupby(contacts, :date; sort=true)
        contactrates = combine(gp, names(gp, Not("date")) .=> mean; renamecols=false)
        rename!(contactrates, "date"=>"dates")
        save(projectdir("data","mean_contact_rates_region=$(region)_aggregation=$(aggregate).csv"), contactrates)


        # @info "decomposition region $region"
        # slowtrends = TSD.decompose(contactrates, Optim.Options(show_trace=true, iterations=2000); fdir=projectdir("figures/"))
        # save(projectdir("data","smoothed_mean_contact_rates_region=$region.csv"), slowtrends)
    end
end

main(7)
