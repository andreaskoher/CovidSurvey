using DrWatson
quickactivate(@__DIR__)
@show projectdir()
##
using CSV#
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
using Underscores
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
function rollingmean(xs::AbstractVector, window)
    n  = length(xs)-window+1
    ys = Vector{Float64}(undef, n)
    for i in 1:n
        x = xs[i:i+window-1]
        ys[i] = mean(skipmissing(x))
        @assert isfinite(ys[i]) "infinite at i=$i"
    end
    return ys
end

function rollingmean(df, window)
    @assert isodd(window)
    Δ = window ÷ 2
    smoothed = DataFrame( :date => df.date[1+Δ:end-Δ] )
    for col in names(df, Not(:date))
        @info col
        smoothed[!,col] = rollingmean(df[:,col], window)
        @assert all( isfinite.(smoothed[!,col]) ) "$(findall(x-> !isfinite(x), smoothed[!,col]))"
    end
    smoothed
end

function main(aggregate=1)
    plotlyjs()
    threshold = (family=50, colleagues=100, friends=100, strangers=1000)
    @progress for region in 1:5
        @info "load data region $region"
        contacts = @_ normpath( homedir(), "data/covidsurvey/rawcontacts.csv") |>
            readcontacts(__ , select_region=region ) |>
            filteroutliers(__, threshold) #NOTE National: p = 99.9

        contacts[!,:date] = sampledown(contacts.date, aggregate)
        contacts[!,:total] .= 0
        for c in ctype
            contacts[!,:total] += contacts[:,c]
        end
        # contacts = contacts[:, [:total, :strangers, :date]]


        contactrates = @_ contacts |>
             groupby(__, :date; sort=true) |>
             combine(__, names(__, Not("date")) .=> mean; renamecols=false) |>
             rollingmean(__, 7) |>
             rename(__, "date"=>"dates")

        save(projectdir("data/contacts/dk","averaged_contact_rates_region=$(region).csv"), contactrates)


        # @info "decomposition region $region"
        # slowtrends = TSD.decompose(contactrates, Optim.Options(show_trace=true, iterations=2000); fdir=projectdir("figures/"))
        # save(projectdir("data","smoothed_mean_contact_rates_region=$region.csv"), slowtrends)
    end
end

main(1)
