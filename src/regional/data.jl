# using ..CovidSurvey
#include("../data.jl")

const regions = ["capital", "zealand", "south", "central", "north"]
const nregions = length(regions)
const populations = [1_855_084, 838_840, 1_223_634, 1_332_048, 590_439] # from statistics denmark

dan2eng = Dict(
    "Dato"          => "date",
    "Total"         => "total",
    "Hovedstaden"   => "capital",
    "Sj\xe6lland"   => "zealand",
    "Sjælland"      => "zealand",
    "Syddanmark"    => "south",
    "Midtjylland"   => "central",
    "Nordjylland"   => "north",
    "Ukendt Region" => "unknown"
)

function regions!(data)
    data["regions"] = regions
    data["num_regions"] = nregions
    data["populations"] = populations
    return nothing
end

# ===========================================================================
#                     SSI DATA
# ===========================================================================

function process_ssi(df)
    df = @_ DataFrames.select(df, 1=>:region, 2=>:date, 3=>:counts) |>
        filter(:region => x->!ismissing(x), __)
    df.region = replace(df.region, dan2eng...)
    df = filter(:region => x -> x ∈ regions, df)
    df = unstack(df, :date, :region, :counts)
    CovidSurvey.insert_missing_dates!(df)
    CovidSurvey.replace_missing!(df)
    df[:, ["date", regions...]]
    disallowmissing!(df)
    return df
end

function observables!(data,
    fname_cases  = normpath( homedir(), "data/Statens-Serum-Institut/dashboard/Regionalt_DB/08_bekraeftede_tilfaelde_pr_dag_pr_regions.csv" ),
    fname_hospit = normpath( homedir(), "data/Statens-Serum-Institut/dashboard/Regionalt_DB/06_nye_indlaeggelser_pr_region_pr_dag.csv" ),
    fname_deaths = normpath( homedir(), "data/Statens-Serum-Institut/dashboard/Regionalt_DB/07_antal_doede_pr_dag_pr_region.csv" )
)

    data["cases"] = CSV.File( fname_cases ) |>
        DataFrame |> process_ssi


    data["hospit"] = CSV.File( fname_hospit ) |>
        DataFrame |> process_ssi

    data["deaths"] = CSV.File( fname_deaths ) |>
        DataFrame |> process_ssi

    return nothing
end

# ============================================================================
#                        SERO-PREVALENCE
# ============================================================================

struct SeroParams{M,S,C,I,A,Y,P,N}
    means::M
    std::S
    CIs::C
    # dstd::D
    indexs::I
    dates::A
    delay::Y
    populations::P
    num_regions::N
end

function SeroParams!(data, sero=SeroInit(); later_dataset = false)
    @unpack dates_turing, dates, regions, populations = data

    df = readseroprev(;later_dataset)
    s  = isempty(sero.obs_start) ? dates[1] : Date(sero.obs_start)
    e  = isempty(sero.obs_stop) ? dates[end] : Date(sero.obs_stop)
    df = df[ s .<= df.date .<= e, :]
    data["seroprev"] = df

    means  = Vector{Vector{Float64}}()
    CIs    = Vector{Vector{Tuple{Float64,Float64}}}()
    indexs = Vector{Vector{Int64}}()
    for (i,r) in enumerate(regions)
        index = [findfirst(==(d), dates_turing[i]) for d in df.date]
        push!( indexs, index )
        push!( means, df[:, "$(r)_mean"] )
        push!( CIs, df[:, "$(r)_CI"] )
    end

    params = (; means,
        std  = sero.std, CIs, indexs,
        dates = df.date,
        delay = sero.delay,
        populations = populations,
        num_regions = length(regions)
    )
    data["sero"] = means
    data["seromodel"] = SeroParams(params...)

    return nothing
end


function parsepercent(x::AbstractString)
    x = strip(x, ['%',')','('])
    x = replace(x, "," => ".")
    parse(Float64, x) / 100
end

function parseweek(w)
    y = w > 40 ? Date("2020") : Date("2021")
    y + Week(w) - Day(3) #center at day 4
end

parse_mean(s) = parsepercent(first(split(s)))

function parse_CI(s)
    s_m, s_cil, _, s_cir = split(s)
    μ = parsepercent(s_m)
    cil = parsepercent(s_cil)
    cir = parsepercent(s_cir)
    # σ = max( abs(μ - cil), abs(μ - cir)) / 1.96
    return (cil, cir) #σ
end

function readseroprev(; later_dataset = false)
    df = CSV.File(projectdir("data/seroprevalence/", "seroprev_early.csv"), delim="\t") |> DataFrame
    for r in regions
        col = df[:,r]
        df[!,"$(r)_CI"] = parse_CI.(col)
        df[!,"$(r)_mean"] = parse_mean.(col)
    end
    return df
end

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
    delay_length     ::Int64   = 60
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

function lockdown!(data, lockdown, include_early_dynamic)
    @unpack dates_turing, num_regions = data
    include_early_dynamic && (data["lockdown_indices"] = fill(0, num_regions); return)
    li = Vector{Int64}()
    for d in dates_turing
        i = findfirst(==(Date(lockdown)), d)
        push!(li, i)
    end
    data["lockdown_indices"] = li
    return nothing
end

# ===============================================================================
#                           main function
# ===============================================================================

function load_data(;
    observationsend  = nothing,
    predictors       = nothing,
    rwstep           = 1,
    epidemicstart    = 30,
    epidemic_threshold = 30,
    numimpute        = 6,
    casemodel         = CaseInit(),
    hospitmodel       = HospitInit(),
    deathmodel        = DeathInit(),
    seromodel         = SeroInit(),
    link              = KLogistic(3.),
    invlink           = KLogit(3.),
    lockdown          = "2020-03-18",
    predictive        = false,
    include_early_dynamic = false,
    covariates_kwargs = Dict(
        :semiparametric => false,
        :fname => normpath( homedir(), "data/covidsurvey/smoothed_contacts.csv" ),
        :shift => 0,
        :startdate => nothing,
        :enddate => nothing,
        :aggregation => 1,
        :mobility => nothing,
        :standardize => false,
        :normalize => true,
    )
)

    data = Dict{String, Any}()
    regions!(data)
    observables!(data)
    CovidSurvey.consistent!(data; epidemicstart, epidemic_threshold)
    CovidSurvey.turingformat!(data, observationsend, predictive)
    # cases_start_date!(data, cases_start)
    CovidSurvey.covariates!(data, predictors; covariates_kwargs...)
    lockdown!(data, lockdown, include_early_dynamic)
    CovidSurvey.randomwalk!(data, rwstep, covariates_kwargs)
    ObservParams!(data, casemodel)
    ObservParams!(data, hospitmodel)
    ObservParams!(data, deathmodel)
    SeroParams!(data, seromodel)

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
        populations                 = populations,
        serial_interval             = CovidSurvey.serialinterval(15),#padzeros(serialinterval(30), 0, num_tot), #data_usa.turing_data.serial_intervals[1:50],
        num_si                      = 15,
        lockdown_indices            = data["lockdown_indices"],
        # cases_start_indices         = data["cases_start_idx"],
        # num_case_observations       = data["num_case_obs"],
        hospits                     = data["hospit_turing"],
        hospitmodel                 = data["hospitmodel"],
        num_rt_steps                = data["num_rt_steps"],
        rt_step_indices             = data["rt_step_index"],
        num_regions                 = length(regions),
        num_observations            = data["num_observations"],
        link                        = link,
        invlink                     = invlink,
        seromodel                   = data["seromodel"],
        sero                        = data["sero"],
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
    startdates = [d[epidemicstart] for d in data["dates_turing"]]

    CovidSurvey.Data(
        turing_data,
        data["cases"],
        data["deaths"],
        data["hospit"],
        data["num_observations"], #num_observations
        data["num_tot"],
        dates,
        populations,
        predictors,
        data["seroprev"],
        regions,
        lockdown,
        data["observationsend"],
        nothing,
        startdates
    )
end
