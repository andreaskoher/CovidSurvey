using DrWatson
quickactivate(@__DIR__)
@show projectdir()
##
using CovidSurvey
using StatsBase
using StatsPlots
using DataFrames
using Underscores
using HTTP
using Dates
using RData
using CSV
## ===========================================================================
# load data sets

fname = "/home/and/data/google-mobility/Global_Mobility_Report.csv"
mobil = CSV.File(fname) |> DataFrame

dateformat = "dduuuyyyy"
url = "https://raw.githubusercontent.com/OxCGRT/covid-policy-tracker/master/data/timeseries/stringency_index.csv"
stringency = @_ CSV.File(HTTP.get(url).body) |>
            DataFrame |>
            DataFrames.stack(__, Not([:Column1, :country_code, :country_name]), value_name=:stringency, variable_name=:date)
stringency[!,:date] = Date.(stringency.date, dateformat)

##
const mobility_names = [
    "retail_and_recreation_percent_change_from_baseline",
    "grocery_and_pharmacy_percent_change_from_baseline",
    "workplaces_percent_change_from_baseline",
    "residential_percent_change_from_baseline"
]

const new_mobility_names = [
    "retail",
    "grocery",
    "workplaces",
    "residential"
]
## ============================================================================
# dataset Denmark
mobil_dk = @_ mobil |>
                filter(:country_region => ==("Denmark"), __) |>
                filter(:sub_region_1 => ismissing, __) |>
                select(__, :date, mobility_names .=> new_mobility_names)
mobil_dk[!,:nonresidential] = [ mean(r) for r in eachrow( select(mobil_dk, Not([:date,:residential])) ) ]
@assert issorted(mobil_dk.date)

string_dk = @_ stringency |>
                filter(:country_name => ==("Denmark"), __) |>
                select(__, [:date, :stringency])

fname_epi = Dict(
    :cases => normpath( homedir(), "data/Statens-Serum-Institut/dashboard/Regionalt_DB/08_bekraeftede_tilfaelde_pr_dag_pr_regions.csv" ),
    :hospit => normpath( homedir(), "data/Statens-Serum-Institut/dashboard/Regionalt_DB/06_nye_indlaeggelser_pr_region_pr_dag.csv" ),
    :deaths => normpath( homedir(), "data/Statens-Serum-Institut/dashboard/Regionalt_DB/07_antal_doede_pr_dag_pr_region.csv" )
)

epi = Dict(
    n => @_ CSV.File( fn ) |>
            DataFrame |>
            National.process_ssi |>
            DataFrames.select(__, :date, :country => n) |>
            sort(__, :date)
    for (n, fn) in pairs(fname_epi)
)

data = let
    data = @_ RData.load("/home/and/code/epidemia/data/EuropeCovid2.RData")["EuropeCovid2"]["data"] |>
                filter(:country => ==("Denmark"), __) |>
                select(__, Not([:cases, :deaths, :id])) |>
                leftjoin(__, mobil_dk, on=:date) |>
                leftjoin(__, string_dk, on=:date) |>
                leftjoin(__, epi[:cases], on=:date) |>
                leftjoin(__, epi[:hospit], on=:date) |>
                leftjoin(__, epi[:deaths], on=:date) |>
                sort(__, :date)

    for n in [keys(epi)..., new_mobility_names..., "nonresidential"]
        data[!,n] = coalesce.(data[:,n], 0)
    end
    data[!,:pop] .= 5_837_213
    disallowmissing!(data)
    data
end

plot(data.residential)
plot!(data.nonresidential)

CSV.write("/home/and/Datathon/data/denmark.csv", data)

# f = KLogistic(.05)
# histogram( f.(rand(Normal(0,0.2), 10_000)) ,xticks=:native)
# histogram(rand(Normal(0.02,0.05), 10_000))
#
# let
#     plot(data.date, data.schools_universities, lab="schools")
#     plot!(data.date, data.public_events, lab="events")
#     plot!(data.date, data.self_isolating_if_ill, lab="isolating")
#     plot!(data.date, data.social_distancing_encouraged, lab="distancing")
#     plot!(data.date, data.lockdown, lab="lockdown")
# end
#
# plot(data.date, data.stringency, lab="stringency")
#
# let
#     plot(data.date, data.residential, lab="residential")
#     plot!(data.date, data.nonresidential, lab="nonresidential")
# end

## ============================================================================
# dataset Europe

data = RData.load("/home/and/code/epidemia/data/EuropeCovid2.RData")["EuropeCovid2"]["data"]
countries = unique(data.country)


function country_data(data, mobil, stringency, country)
    data_c  = filter(:country => ==(country), data)
    mobil_c = @_ mobil |>
                filter(:country_region => ==(country), __) |>
                filter(:sub_region_1 => ismissing, __) |>
                select(__, :date, mobility_names .=> new_mobility_names)
    mobil_c[!,:nonresidential] = [ mean(r) for r in eachrow( select(mobil_c, Not([:date,:residential])) ) ]
    @assert issorted(mobil_c.date)

    string_c = @_ stringency |>
                filter(:country_name => ==(country), __) |>
                select(__, [:date, :stringency])
    data_c = @_ data_c |>
                leftjoin(__, mobil_c, on=:date) |>
                leftjoin(__, string_c, on=:date)

    for n in [new_mobility_names..., "nonresidential"]
        data_c[!,n] = coalesce.(data_c[:,n], 0)
    end
    #disallowmissing!(data_c, :stringency)
    return data_c
end

newdata = let
    newdata = []
    for c in countries
        push!(newdata, country_data(data, mobil, stringency, c) )
    end
    vcat(newdata...)
end

CSV.write("/home/and/Datathon/data/EuropeCovid2.csv", newdata)

## ============================================================================
# regional hospitalizations

ps = (
    model = "hospit",
    observ = "2021-02-01",#"2021-02-06",#"2021-03-25"
    preds = nothing,#"CT,MRES,MNRES",#"CF,CC,CR,CS"
    semipara=false,
    rwstep = 7,
    include= true,
    std = true,
)

data_params = (
      observationsend  = ps.observ
    , predictors        = ps.preds|> CovidSurvey.parse_predictors
    # , cases_start       = ps.cases
    # , hospitmodel       = Regional.HospitInit1(obs_stop="2020-07-01")
    , casemodel         = Regional.CaseInit1(obs_start="2020-06-01")
    , seromodel         = Regional.SeroInit2(delay=0, std=1.)
    , rwstep            = ps.rwstep
    , epidemicstart    = 20
    , numimpute        = 6
    , include_early_dynamic = ps.include
    , link              = KLogistic(5.)
    , invlink           = KLogit(5.)
    , predictive        = false
    , covariates_kwargs = Dict(
      :semiparametric => false,
      :fname          => [projectdir("data", "mean_contact_rates_region=$(r).csv" ) for r in 1:Regional.nregions],#[projectdir("data", "mean_contact_rates_region=$r.csv" ) for r in 1:Regional.nregions],
      :shift          => -1,
      :startdate      => "2020-11-10", # >= 1000 cases / day
      :datecol        => "dates",
      :aggregation    => ps.rwstep,
      :standartize    => ps.std,
      :mobility       => [projectdir("data/google/mobility_region=$(Regional.regions[i]).csv") for i in 1:Regional.nregions]
      )
    )
latent_Rts_z[m] ~ RandomWalk(num_rt_steps[m])

data = Regional.load_data(; data_params... )
turing_data = data.turing_data;

using DataFrames
hospit = data.hospit
dates  = hospit.date

df = DataFrame(
      date=Vector{Date}()
    , week=Vector{Int64}()
    , hosp=Vector{Union{Int64, Missing}}()
    , region=Vector{String}()
)

for (i,(s,r)) in enumerate(zip(data.startdates, Regional.regions))
    mask = dates .>= s
    # date = (s-Day(30-1)):Day(1):last(dates) |> collect
    date = s:Day(1):last(dates) |> collect
    # hosp = vcat(fill(Missing, 30), hospit[mask, r])
    hosp = hospit[mask, r]
    region = fill(r, length(date))
    week = Regional.stepindex(length(date), 7)
    d    = DataFrame((; date, week, hosp, region))
    df = vcat(df, d)
end

hospits = copy(data.hospit)
hospits[!,:week] = Regional.stepindex(size(hospits,1), 7)
df = DataFrames.stack(hospits, Not([:week, :date]); variable_name=:region, value_name=:hospitalizations)
CSV.write("/home/and/Datathon/data/hopitalizations_regional.csv", df)

fdir = "/home/and/tmp/figures/"

using RData
fname = projectdir("data/COVID19/DK.rds")
c19 = RData.load(fname)

fname = projectdir("data/COVID19/DK.csv")
c19 = CSV.read(fname, DataFrame)
plot(c19.date, c19.confirmed)

df = copy(data.hospit)
df[!,:denmark] = sum(Array(df[:,Regional.regions]), dims=2 ) |> vec
plot!(data.deaths.date, df.denmark)


##                  OLD
# ## ============================================================================
# #                    data set for Aarhus Datathon
# #  ============================================================================
# using HTTP
# url = "https://raw.githubusercontent.com/OxCGRT/covid-policy-tracker/master/data/timeseries/stringency_index.csv"
# string = CSV.File(HTTP.get(url).body) |> DataFrame
#
# string = DataFrames.stack(string, Not([:Column1, :country_code, :country_name]), value_name=:stringency, variable_name=:date)
# string = filter(:country_name => ==("Denmark"), string)
# dateformat = "dduuuyyyy"
# string[!,:date] = Date.(string.date, dateformat)
# string = select(string, [:date, :stringency])
#
# fname_cases  = normpath( homedir(), "data/Statens-Serum-Institut/dashboard/Regionalt_DB/08_bekraeftede_tilfaelde_pr_dag_pr_regions.csv" )
# fname_hospit = normpath( homedir(), "data/Statens-Serum-Institut/dashboard/Regionalt_DB/06_nye_indlaeggelser_pr_region_pr_dag.csv" )
# fname_deaths = normpath( homedir(), "data/Statens-Serum-Institut/dashboard/Regionalt_DB/07_antal_doede_pr_dag_pr_region.csv" )
#
# @_ data |> DataFrames.select(__, :date)
#
# using Underscores
#
# cases = @_ CSV.File( fname_cases ) |>
#     DataFrame |>
#     National.process_ssi |>
#     DataFrames.select(__, :date, :country => :cases)
#
# hospit = @_ CSV.File( fname_hospit ) |>
#     DataFrame |>
#     National.process_ssi |>
#     DataFrames.select(__, :date, :country => :hospit)
#
# deaths = @_ CSV.File( fname_deaths ) |>
#     DataFrame |>
#     National.process_ssi |>
#     DataFrames.select(__, :date, :country => :deaths)
#
# using RData
# data = RData.load("/home/and/code/epidemia/data/EuropeCovid2.RData")["EuropeCovid2"]["data"]
# data = filter(:country => ==("Denmark"), data)
# data = select(data, Not([:cases, :deaths, :id, :country]))
# data = leftjoin(data, mobil_dk, on=:date)
# data = leftjoin(data, string, on=:date)
# data = leftjoin(data, cases, on=:date)
# data = leftjoin(data, hospit, on=:date)
# data = leftjoin(data, deaths, on=:date)
#
# CSV.write("/home/and/Datathon/data/denmark.csv", data)
