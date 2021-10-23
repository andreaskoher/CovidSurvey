# using ..CovidSurvey
#include("../data.jl")

const countries = ["Sweden","France","Hungary","UK","USA","Italy","Germany","Denmark"]
const ncountries = length(countries)

function regions!(data)
    data["regions"] = countries
    data["num_regions"] = ncountries
    return nothing
end

function observables!(data)
    epilong = data["data"] = @_ projectdir("data/epidata/", "epidata_longformat.csv") |>
        CSV.read( __, DataFrame ) |>
        preprocess


    data["cases"] = @_ DataFrames.select(epilong, :date, :cases, :region) |>
        unstack(__, :region, :cases) |>
        disallowmissing(__)

    data["hospit"] = nothing

    data["deaths"] = @_ DataFrames.select(epilong, :date, :deaths, :region) |>
        unstack(__, :region, :deaths) |>
        disallowmissing(__)

    return nothing
end

function preprocess(df::DataFrame)
    startdate = Date("2020-01-05") #Date("2020-01-05") #avoid missing value at very early date
    enddate = Date("2021-04-01")
    @_ df |>
        filter(:date => >=(startdate), __) |>
        filter(:date => <=(enddate), __) |>
        disallowmissing
end

# max_vacc_rate = 0.05
# min_case_rate = 20
# epidemic = CSV.read(projectdir("data/epidata/epidata_region=$(region).csv"), DataFrame)
# is = findfirst(>(22), epidemic.cases_per_100000)
# ds = epidemic.date[is]
# de1 = epidemic.date[is:end][findfirst(<(10), epidemic.cases_per_100000[is:end])]
# de2 = epidemic.date[findfirst(>(0.05), epidemic.fraction_vaccinated)]
# de  = min(de1, de2)

# ============================================================================
# observation model defaults
# ============================================================================
@kwdef struct CaseInit <: CovidSurvey.ObservationInit
    obs_start        ::String  = ""
    obs_stop         ::String  = ""
    delay_dist                 = NegativeBinomial2
    delay_length     ::Int64   = 40
    delay_dispersion ::Float64 = 5.41
    populations      ::Vector{Int64} = Regional.populations
end

@kwdef struct HospitInit <: CovidSurvey.ObservationInit
    obs_start        ::String  = ""
    obs_stop         ::String  = ""
    delay_dist                 = NegativeBinomial2
    delay_length     ::Int64   = 40
    delay_dispersion ::Float64 = 5.41
    populations      ::Vector{Int64} = Regional.populations
end

@kwdef struct DeathInit <: CovidSurvey.ObservationInit
    obs_start        ::String  = ""
    obs_stop         ::String  = ""
    delay_dist                 = NegativeBinomial2
    delay_length     ::Int64   = 40
    delay_dispersion ::Float64 = 14.26
    populations      ::Vector{Int64} = Regional.populations
end

@kwdef struct SeroInit# <: ObservationInit
    obs_start        ::String  = ""
    obs_stop         ::String  = "2020-07-01"
    delay            ::Int64   = 0
    std              ::Float64 = 1.5
    # population       ::Int64   = National.populations
    # dist                       = InverseGamma2
    # cv               ::Float64 = 0.1
end

ObservParams!(data, o::CaseInit)   = data["casemodel"]   = CovidSurvey.ObservParams(data, o)
ObservParams!(data, o::HospitInit) = data["hospitmodel"] = CovidSurvey.ObservParams(data, o)
ObservParams!(data, o::DeathInit)  = data["deathmodel"]  = CovidSurvey.ObservParams(data, o)


# ============================================================================

function lockdown!(data, lockdowns, include_early_dynamic)
    @unpack dates_turing, num_regions = data
    include_early_dynamic && (data["lockdown_indices"] = fill(0, num_regions); return)
    li = Vector{Int64}()
    for (d, lockdown) in zip(dates_turing, lockdowns)
        i = findfirst(==(Date(lockdown)), d)
        push!(li, i)
    end
    data["lockdown_indices"] = li
    return nothing
end

# ===============================================================================
#                           main function
# ===============================================================================
function load_international_data(;
    observationsend  = nothing,
    predictors       = nothing,
    rwstep           = 7,
    epidemicstart    = Date("2020-06-01"),
    epidemic_threshold = 30,
    numimpute        = 6,
    casemodel         = CaseInit(),
    deathmodel        = DeathInit(),
    link              = KLogistic(3.),
    invlink           = KLogit(3.),
    lockdowns         = nothing,
    predictive        = false,
    include_early_dynamic = true,
    covariates_kwargs = Dict(
        :semiparametric => false,
        :fname => normpath( homedir(), "data/covidsurvey/smoothed_contacts.csv" ),
        :shift => 0,
        :startdate => nothing,
        :enddate => nothing,
        :aggregation => 1,
        :mobility => nothing,
        :standardize => false
    )
)

    data = Dict{String, Any}()
    regions!(data)
    observables!(data)
    CovidSurvey.consistent!(data; startdates = epidemicstart)
    CovidSurvey.turingformat!(data, observationsend, predictive)
    lockdown!(data, lockdowns, include_early_dynamic)
    # cases_start_date!(data, cases_start)
    CovidSurvey.covariates!(data, predictors; covariates_kwargs...)
    CovidSurvey.randomwalk!(data, rwstep, covariates_kwargs)
    ObservParams!(data, casemodel)
    ObservParams!(data, deathmodel)

    # iar_idx = get_iar_idx(num_tot, num_observations, cases_start_idx, iar_step)
    # num_iar_steps = length(unique(iar_idx))


    turing_data = (
        num_impute                  = numimpute,
        num_total_days              = data["num_tot"],
        dates                       = data["dates_turing"],
        cases                       = data["cases_turing"],
        casemodel                   = data["casemodel"],
        deaths                      = data["deaths_turing"],
        deathmodel                  = data["deathmodel"],
        epidemic_start              = epidemicstart,
        serial_interval             = CovidSurvey.serialinterval(15),#padzeros(serialinterval(30), 0, num_tot), #data_usa.turing_data.serial_intervals[1:50],
        num_si                      = 15,
        lockdown_indices            = data["lockdown_indices"],
        # cases_start_indices         = data["cases_start_idx"],
        # num_case_observations       = data["num_case_obs"],
        num_rt_steps                = data["num_rt_steps"],
        rt_step_indices             = data["rt_step_index"],
        num_regions                 = length(regions),
        num_observations            = data["num_observations"],
        link                        = link,
        invlink                     = invlink,
        num_covariates              = data["num_covariates"],
        covariates                  = data["covariates"],
        covariates_starts           = data["covariates_starts"],
        semiparametric              = covariates_kwargs[:semiparametric],
        include_early_dynamic       = include_early_dynamic,
        rwscale                     = sqrt(rwstep),
        # num_i2h                     = 40,
        # ϕ_i2h                       = 5.41,
        # num_i2d                     = 60,
        # ϕ_i2d                       = 14.26,
        # num_i2c                     = 40,
        # ϕ_i2c                       = 5.41,
    )

    dates = [ collect(d:Day(1):d+Day(n-1)) for (d, n) in zip(data["startdates"], data["num_tot"])]
    startdates = [d[data["epidemicstart"]] for d in data["dates_turing"]]

    CovidSurvey.Data(
        turing_data,
        data["cases"],
        data["deaths"],
        data["hospit"],
        data["num_observations"], #num_observations
        data["num_tot"],
        dates,
        nothing,
        predictors,
        nothing,
        countries,
        nothing,
        data["observationsend"],
        nothing,
        startdates
    )
end
