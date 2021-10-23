using DrWatson
quickactivate(@__DIR__)
@show projectdir()

## ============================================================================
@info "load packages"
using CovidSurvey
using Random, Dates, Turing
using Base.Threads
using StatsPlots
using PrettyTables
using DataFrames
using BSON
using ColorSchemes
colors = ColorSchemes.tableau_10.colors

using Memoization
using ReverseDiff
setadbackend(:reversediff)
Turing.setrdcache(true)

plotlyjs()
@info "number of threads available: $(nthreads())"
## ============================================================================

ps = (
    warmup = 100,
    model = "deaths",
    steps  = 100,
    seed   = nothing,
    observ = "2021-03-01",#"2021-02-06",#"2021-03-25"
    chains = 1,
    preds = "CS,MG,MA",#"CF,CC,CR,CS"
    semipara=true,
    rwstep = 7,
    include= false,
    std = false,
)
## ============================================================================
# load data
@info "load data"

data_params = (
      observationsend  = ps.observ
    , predictors        = ps.preds|> CovidSurvey.parse_predictors
    # , cases_start       = ps.cases
    # , hospitmodel       = Regional.HospitInit1(obs_stop="2020-07-01")
    , casemodel         = Regional.CaseInit(obs_start="2020-06-01")
    , seromodel         = Regional.SeroInit(delay=0, std=1.)
    , rwstep            = ps.rwstep
    , epidemicstart    = 20
    , numimpute        = 6
    , include_early_dynamic = ps.include
    , link              = KLogistic(4.5)
    , invlink           = KLogit(4.5)
    , predictive        = false
    , covariates_kwargs = Dict(
      :semiparametric => ps.semipara,
      :fname          => [projectdir("data/contacts/dk/", "averaged_contact_rates_region=$(r).csv" ) for r in 1:Regional.nregions],#[projectdir("data", "mean_contact_rates_region=$r.csv" ) for r in 1:Regional.nregions],
      :shift          => -1,
      :startdate      => "2020-11-10", # >= 1000 cases / day
      :datecol        => "dates",
      :aggregation    => ps.rwstep,
      :standartize    => ps.std,
      :mobility       => [projectdir("data/mobility/mobility_region=$(Regional.regions[i]).csv") for i in 1:Regional.nregions]
      )
    )
data = Regional.load_data(; data_params... )
d = turing_data = data.turing_data;

gethospits(d,i) = DataFrame( :hospits => d.hospits[i] )
getdeaths(d,i) = DataFrame( :deaths => d.deaths[i] )
getcases(d,i) = DataFrame( :cases => d.cases[i] )
getdates(d,i) = DataFrame( :date => d.dates[i] )

function getrw(d, i)
    no = d.num_observations[i]
    cs = d.covariates_starts[i]
    ri = d.rt_step_indices[i]
    li = d.lockdown_indices[i]

    rw = ones(Int64, no)
    rw[li+1:cs-1] = ri .+ 1
    rw[cs:no] .=  rw[cs-1]
    return DataFrame(:rw => rw)
end

function getcovariates(d, i)
    no = d.num_observations[i]
    cs = d.covariates_starts[i]
    names = data.predictors
    co = d.covariates[i]
    np = length(names)

    dat = zeros(no, np)
    dat[cs:no, :] = co
    return DataFrame(dat, names)
end

function epidata(d, i, r)
    df = hcat(
          getdates(d,i)
        , getcases(d,i)
        , gethospits(d,i)
        , getdeaths(d,i)
        , getcovariates(d, i)
        , getrw(d, i)
    )
    df[!,:region] .= r
    return df
end

function epidata(data)
    return vcat( [epidata(data.turing_data, i, r) for (i, r) in enumerate(data.regions)]... )
end

epid = epidata(data)
CSV.write(projectdir("data/epidemia", "denmark_predictors=CS-MG-MA.csv"), epid)


plotlyjs()
let
    p = plot()
    for (i, r) in enumerate(data.regions)
        df = epidata(d, i, Regional.regions[i])
        plot!(df.date, cumsum(df.hospit), hover=["$i, cases: $n" for (i,n) in enumerate(cumsum(df.hospits))], c=colors[i])
        vline!([df.date[20]], c=colors[i])
    end
    hline!([30], c=colors[end])
    p
end
