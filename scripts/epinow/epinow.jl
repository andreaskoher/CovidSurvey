using DrWatson
quickactivate(@__DIR__)
@show projectdir()
##
using RCall
using CSV
using DataFrames
using Dates
using Plots
using Underscores
using StatsBase
using ColorSchemes
plotlyjs()
## ========================================================================

@rlibrary var"data.table"
@rlibrary EpiNow2
@rlibrary covidregionaldata

import NetworkOptions
ENV["JULIA_SSL_CA_ROOTS_PATH"] = NetworkOptions.bundled_ca_roots()

begin
    R"""
    library(data.table)
    library(EpiNow2)
    library(covidregionaldata)
    country <- "france"
    options(mc.cores = 4)
    generation_time <- get_generation_time(disease = "SARS-CoV-2", source = "ganyani")
    incubation_period <- get_incubation_period(disease = "SARS-CoV-2", source = "lauer")
    reporting_delay <- list(mean = convert_to_logmean(4, 1),
                            mean_sd = 0.1,
                            sd = convert_to_logsd(4, 1),
                            sd_sd = 0.1,
                            max = 15)
    # reported_cases <- read.csv(file = 'WHO-COVID-19-global-data.csv') #covidregionaldata::get_national_data(country)
    # reported_cases <- data.table::setDT(reported_cases)
    # reported_cases <- reported_cases[, .(date, confirm = cases_new)]
    """
end

regions = ["Hungary", "The United Kingdom", "France", "Sweden", "Denmark", "Germany", "Italy", "United States of America"]
old2new = Dict(
    "Hungary" => "Hungary",
    "The United Kingdom" => "UK",
    "France" => "France",
    "Sweden" => "Sweden",
    "Denmark" => "Denmark",
    "Germany" => "Germany",
    "Italy" => "Italy",
    "United States of America" => "USA"
)

function load_data(regions)
    fname = projectdir("data/epinow/WHO-COVID-19-global-data.csv")
    who = CSV.read(fname, DataFrame)
    observations = Dict{String, DataFrame}()
    for region in regions
        regional = @_ who |>
                filter(:Country => ==(region), __) |>
                select(__, :Date_reported => :date, :New_cases => :cases, :New_deaths => :deaths) #|>
        #__[Date("2020-08-01") <. __.date <. Date("2021-02-01") ]
        observations[old2new[region]] = regional
    end
    observations
end

obs = load_data(regions)


fname = projectdir("data/epinow/WHO-COVID-19-global-data.csv")
who = CSV.read(fname, DataFrame)
## ========================================================================
# for region in regions
function firefox(p; fname="tmp.html", fdir=normpath(homedir(), ".tmp"))
    # mkpath(dir)
    filepath = normpath(fdir, fname)
    savefig(p, filepath)
    run(`firefox $(filepath)`, wait=false)
    return p
end

function plot_regions(contacts, observations, mobility)
    regions = keys(contacts)
    region = first(regions)
    clr = palette(:tab10)
    for region in keys(contacts)
        o = observations[region]
        c = contacts[region]
        m = mobility[region]
        p1 = plot(o.date, o.cases, lab="cases", lw=3, xticks=:native, yticks=:native, c=clr[1])
        p2 = plot(o.date, o.deaths, lab="deaths", lw=3, xticks=:native, yticks=:native, c=clr[2])
        p3 = plot(c.date, c.total |> zscore, lab="total contacts", lw=3, xticks=:native, yticks=:native, c=clr[3])
        plot!(p3, m.date, m.apple |> zscore, lab="apple", lw=3, xticks=:native, yticks=:native, c=clr[4])
        plot!(p3, m.date, m.google |> zscore, lab="google", lw=3, xticks=:native, yticks=:native, c=clr[5])
        p = plot(p1, p2, p3, layout=(3,1), size=(1000, 3*250), sharex=true, link=:x)
        firefox(p, fname="$region.html")
    end
end


plot_regions(
    BSON.load(projectdir("data/contacts/int/mean_contact_rates_aggregation=14.bson")),
    obs,
    BSON.load(projectdir("data/mobility/", "mobility_international.bson"))
    )
## ========================================================================
fname = projectdir("data/epinow/rt.csv")
regions = ["Hungary", "United Kingdom", "France", "Sweden", "Denmark", "Germany", "Italy", "United States"]
old2new = Dict(
    "Hungary" => "Hungary",
    "United Kingdom" => "UK",
    "France" => "France",
    "Sweden" => "Sweden",
    "Denmark" => "Denmark",
    "Germany" => "Germany",
    "Italy" => "Italy",
    "United States" => "USA"
)

raw_rt = CSV.read(fname, DataFrame)
region = "Hungary"
rt = @_ raw_rt |>
        filter(:country => ==(region), __) |>
        select(__, :date, :mean)


all_regions = unique(raw_rt.country)

let
    c = contacts[:Hungary]
    plot(c.date, c.total ./ c.nrow |> zscore)
    plot!(rt.date, rt.mean)
end
