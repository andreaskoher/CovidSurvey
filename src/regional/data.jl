const regions = ["capital", "zealand", "south", "central", "north"]
const nregions = length(regions)
const populations = [1_855_084, 838_840,	1_223_634, 1_332_048, 590_439] # from statistics denmark

struct Data
    turing_data
    cases
    deaths
    hospit
    num_observations
    num_total
    dates
    pop
    predictors
    seroprev
    regions
    lockdown
    observations_end
    cases_start
    startdates
end

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

function replace_missing!(df, v=0)
    @assert !any( ismissing.( df.date ) )
    for c in eachcol(df)
        replace!(c, missing=>v)
    end
    disallowmissing!(df)
end

function insert_missing_dates!(df)
    ts = df.date[1] : Day(1) : df.date[end]
    missing_date = .! in.(ts, Ref(df.date))
    if any(missing_date)
        for t in ts[missing_date]
            insertrow!(df, t)
        end
    end
end

function insertrow!(df, d::Date, v=0)
    i = findfirst(==(d-Day(1)), df.date)
    v = [eltype(c) == Date ? d : v for c in eachcol(df)]
    insert!.(eachcol(df), i+1, v)
    nothing
end

function process_ssi(df)
    rename!(df, 1=>:region, 2=>:date, 3=>:counts)
    df.region = replace(df.region, dan2eng...)
    dropmissing!(df, :region)
    df = unstack(df, :date, :region, :counts)
    insert_missing_dates!(df)
    replace_missing!(df)
    df[:, ["date", regions...]]
end

function startdate(df; epidemicstart = 30)
    start_idx = zeros(Int64, nregions)
    for (i, r) in enumerate(regions)
        cumulative = cumsum(df[:,r])
        start_idx[i] = findfirst(>=(20), cumulative)
    end
    df.date[start_idx] .- Day(epidemicstart)
end

# ============================================================================

enddate(df1, df2) = min(df1.date[end], df2.date[end])

enddate(df, args...) = min(df.date[end], enddate(args...))

# ============================================================================
function limit(df::DataFrame, s::Date, e::Date, pad=true)
    pad && df.date[1] > s && ( df = padzeros(df, s) )
    is = findfirst(==(s), df.date)
    ie = findlast(==(e), df.date)
    df[is:ie, :]
end

function padzeros(df::DataFrame, s::Date)
    npad = (df.date[1] - s).value
    dfpad = DataFrame( :date => s:Day(1):df.date[end] )
    for n in names(df)
        n == "date" && continue
        dfpad[!,n] = padzeros(df[!,n], npad)
    end
    dfpad
end

function padzeros(xs::AbstractVector, padfirst=0, total=length(xs)+padfirst)
    n  = length(xs)
    ys = zeros(eltype(xs), total)
    ys[padfirst+1:n+padfirst] = xs
    ys
end

# ============================================================================

function readcovariates(; fname=nothing, shift=0, startdate=nothing, enddate=nothing)
    isnothing(fname) && (@error "please give a file name to covariates_kwargs")
    df = CSV.File(fname)|>DataFrame
    df.date += Day(shift)
    !isnothing(startdate) && filter!(row -> row.date >= Date(startdate), df)
    !isnothing(enddate) && filter!(row -> row.date <= Date(enddate), df)
    ts = df.date[1] : Day(1) : df.date[end]
    @assert all(df.date .== ts )
    @assert !any( ismissing.(Array(df)) )
    @assert all( isfinite.(Array(df)) )
    return df
end


# ============================================================================

function holiday(dates)
    specialdays = [Date("2020-12-24"), Date("2020-12-25"), Date("2020-12-31"), Date("2021-01-01")]
    holidays = dates .∈ Ref(specialdays)
    convert(Vector{Int64}, holidays)
end

# ============================================================================

function nconvolve(f1, f2, ts)
    nbins = length(ts)
    f(τ, t) = f1(τ) * f2(t - τ)
    f3 = zeros(nbins)
    for (i, t) in enumerate(ts)
        integral, err = quadgk(x->f(x, t), 0, t, rtol=1e-8)
        f3[i] = integral
    end
    return f3
end

function inf2hosp(nbins=50, ihr = 1)
    # SSI: ihr ∈ 0.05%-0.5%
    # derived from Verity et al. See parameters.jl for details
    # incubatine time ~ gamma(5.1, .86) see Flaxman et al
    # onset to hospitalization see Zhang et al. Lancet (2020):
    # "Evolving epidemiology and transmission dynamics of coronavirus disease 2019 outside Hubei province, China: a descriptive and modelling study"
    ts = 1.5:1:(nbins + .5)
    i2c(x) = pdf(GammaMeanCv(5.1, .86), x)
    c2h(x) = pdf(GammaMeanCv(2.6, .92), x) * ihr
    p =  nconvolve(i2c, c2h, ts)
    return p / sum(p)
end

function inf2death(nbins=60, ihr = 1)
    # ifr = 89 / 100_000 for age < 70 see Erikstrup et al. in Clin. Inf. Dis.

    ## report34: onset to death = 19.8, Cv = .85
    ts = 1.5:1:(nbins + .5)
    i2c(x) = pdf(GammaMeanCv(5.1, .86), x)
    c2d(x) = pdf(GammaMeanCv(19.8, .85), x) * ihr
    return nconvolve(i2c, c2d, ts)

    # Brauer et al.
    # r = vcat([0], 1.5:1:(nbins + .5))
    # p = diff(cdf.(NegativeBinomial2(21.82, 14.26), r))
    # return p / sum(p)

    ## S. Abbott
    # i2o(x) = pdf(LogNormal(1.621, 0.418),x)
    # o2d(x) = pdf(LogNormalMeanStd(13.1, 11.7),x)
    # return nconvolve(i2o, o2d, ts)

    ## Nouvelette
    # GammaMeanStd(18.8, 8.48)
end

function inf2case(nbins = 30)
    # Brauner et al.
    r = vcat([0], 1.5:1:(nbins + .5))
    p = diff(cdf.(NegativeBinomial2(10.92, 5.41), r))
    return p / sum(p)
    ## Imperial Report
    # r = vcat([0], 1.5:1:(nbins + .5))
    # return diff(cdf.(gamma(5.1, .86), r))
    ## Brauner et al.
    # return diff(cdf.(NegativeBinomial2(10.92, 5.41), r))
    ## S. Abbott
    # i2o(x) = pdf(LogNormal(1.621, 0.418),x)
    # o2c(x) = pdf(LogNormalMeanStd(6.5, 17),x)
    # return nconvolve(i2o, o2c, ts)
end

"""
seropos()[t] is the probablity of being seropositive t days after infection.
For details see imperial report 34
"""
function seropos(nbins=200)
    # λ = 13.3
    # μ = 300 #143.7 - λ
    # κ = 3.67
    # f(x, t, κ, μ, λ) = 1/λ*exp(-x/λ-((t-x)/μ)^κ)
    # c2p(t) = quadgk(x->f(x, t, κ, μ, λ), 0, t, rtol=1e-8)[1]
    # i2c(t) = pdf(GammaMeanCv(5.1, .86), t)
    # ts = 1.5:1:(nbins + .5)
    # p = nconvolve(i2c, c2p, ts)
    # return p / sum(p)

    i2s = zeros(nbins)
    delay = 14
    i2s[delay:end] .= 1.
    return i2s
end


function serialinterval(nbins = 30)
    # Imperial Report
    r = vcat([0], 1.5:1:(nbins + .5))
    # diff(cdf.(GammaMeanStd(6.5, .62), r))
    # return p / sum(p)
    ## Brauer et al. Science (2020)
    p = diff(cdf.(GammaMeanStd(5.06, 2.11), r))
    return p# / sum(p)
end

# ============================================================================

function surveynorm!(xs)
    i = findfirst(x->!ismissing(x), xs)
    xs[i] == 0. && return nothing
    xs ./= abs(xs[i])
    xs .-= xs[i]
    return nothing
end

function padlast(xs, l)
    n  = length(xs)
    i  = findlast(x -> typeof(x) <: Real, xs) #avoid catching a missing value
    ys = Vector{Union{Float64, Missing}}(undef, l)
    ys[1:n] = xs
    ys[n+1:end] .= xs[i]
    ys
end

function preproc(xs::AbstractVector, num_observations, num_tot; normalize=nothing)
    mask = 1:num_observations
    xs = padlast(xs[mask], num_tot)
    !isnothing(normalize) && normalize(xs)
    xs
end

function select_predictors(covariates, predictors, num_observations = nothing, num_tot = nothing)
    predictors = String.(predictors)

    cov_names = filter(x->x!=="date", names(covariates))
    for k in cov_names
        if isnothing(num_observations) || isnothing(num_tot)
            surveynorm!(covariates[!,k])
        else
            covariates[!,k] = preproc(covariates[:,k], num_observations, num_tot; normalize=surveynorm!)
        end
    end

    miss = .!( in.(predictors, Ref(cov_names) ) )
    any(miss) && (@error "predictor $(predictors[miss]) not found. Select either nonresidential_mobility, residential_mobility or from survey: $(surveyvars)")
    Array( covariates[!,predictors] )
end
# ============================================================================

@kwdef struct SeroInit2# <: ObservationInit
    obs_start        ::String  = ""
    obs_stop         ::String  = "2020-07-01"
    delay            ::Int64   = 0
    std              ::Float64 = 1.5
    # population       ::Int64   = National.populations
    # dist                       = InverseGamma2
    # cv               ::Float64 = 0.1
end

struct SeroParams3{M,S,C,I,A,Y,P,N}
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

function SeroParams3!(data, sero=SeroInit2(); later_dataset = false)
    @unpack dates_turing, dates = data

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
        num_regions = length(populations)
    )
    data["sero"] = means
    data["seromodel"] = SeroParams3(params...)

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

function df2vec(df, startdates, observationsend)
    vs = Vector{Vector{Int64}}()
    for (s,r) in zip(startdates, regions)
        is = findfirst(==(s), df.date)
        ie = findfirst(==(observationsend), df.date)
        v = vec(df[is:ie, r])
        push!(vs, v)
    end
    vs
end

function regional_timeseries(dates, startdates, observationsend)
    ts = Vector{Vector{Dates.Date}}()
    for s in startdates
        is = findfirst(==(s), dates)
        ie = findfirst(==(observationsend), dates)
        t = dates[is:ie]
        push!(ts, t)
    end
    ts
end

function totaldays(dates, startdates)
    observationsend = last(dates)
    regional_dates = regional_timeseries(dates, startdates, observationsend)
    return length.(regional_dates)
end

function lockdown!(data, lockdown)
    @unpack dates_turing = data
    li = Vector{Int64}()
    for d in dates_turing
        i = findfirst(==(Date(lockdown)), d)
        push!(li, i)
    end
    data["lockdown_indices"] = li
    return nothing
end

function turingformat!(data, observationsend = nothing)
    @unpack dates, cases, hospit, deaths = data
    e = data["observationsend"] = isnothing(observationsend) ? last(dates) : Date(observationsend)
    s = data["startdates"]
    data["num_tot"]       = totaldays(dates, s)
    ts = data["dates_turing"]  = regional_timeseries(dates, s, e)
    cs = data["cases_turing"]  = df2vec(cases, s, e)
    hs = data["hospit_turing"] = df2vec(hospit, s, e)
    ds = data["deaths_turing"] = df2vec(deaths, s, e)
    for (t,c,h,d) in zip(ts,cs,hs,ds)
        @assert length(t) == length(c) == length(h) == length(d)
    end
    data["num_observations"]  = length.(data["dates_turing"])
    return nothing
end

function observables!(turing_data,
    fname_cases  = normpath( homedir(), "data/Statens-Serum-Institut/dashboard/Regionalt_DB/08_bekraeftede_tilfaelde_pr_dag_pr_regions.csv" ),
    fname_hospit = normpath( homedir(), "data/Statens-Serum-Institut/dashboard/Regionalt_DB/06_nye_indlaeggelser_pr_region_pr_dag.csv" ),
    fname_deaths = normpath( homedir(), "data/Statens-Serum-Institut/dashboard/Regionalt_DB/07_antal_doede_pr_dag_pr_region.csv" )
)

    turing_data["cases"] = CSV.File( fname_cases ) |>
        DataFrame |> process_ssi


    turing_data["hospit"] = CSV.File( fname_hospit ) |>
        DataFrame |> process_ssi

    turing_data["deaths"] = CSV.File( fname_deaths ) |>
        DataFrame |> process_ssi

    return nothing
end

function consistent!(data; epidemicstart = 30)
    @unpack cases, hospit, deaths = data
    data["epidemicstart"] = epidemicstart
    startdates = data["startdates"] = startdate(hospit; epidemicstart)
    s      = minimum(startdates)
    e      = enddate(deaths, cases, hospit)
    data["deaths"] = limit(deaths, s, e)
    data["hospit"] = limit(hospit, s, e)
    data["cases"]  = limit(cases, s, e)
    data["dates"]  = s:Day(1):e
    @assert length(data["dates"]) == size(data["cases"],1) == size(data["hospit"],1) == size(data["deaths"],1)
    return nothing
end

# function cases_start_date!(data, cases_start)
#     @unpack dates_turing, num_observations = data
#     if isnothing(cases_start)
#         data["num_case_obs"] = zeros(Int64, length(regions))
#         data["cases_start_idx"] = nothing
#     else
#         is = Vector{Int64}()
#         ns = Vector{Int64}()
#         for (d, no) in zip(dates_turing, num_observations)
#             i  = findfirst(==(Date.(cases_start)), d)
#             n = no - i + 1
#             push!(is, i)
#             push!(ns, n)
#         end
#         data["cases_start_idx"] = is
#         data["num_case_obs"] = ns
#     end
#     return nothing
# end

function covariates!(data, predictors; kwargs...)
    data["semiparametric"] = kwargs[:semiparametric]
    isnothing(predictors) && (data["num_covariates"] = 0; return)
    # @unpack dates_turing, num_observations = data
    # covariates = readcovariates(; covariates_kwargs... )
    # covariates = leftjoin(DataFrame(:date=>dates_turing), covariates, on=:date)
    # sort!(covariates, :date)
    # covariates_start = findfirst(x->!ismissing(x), covariates[:,Not(:date)][:,1])
    # covariates = covariates[covariates_start:num_observations,:]
    # covariates = select_predictors(covariates, predictors) #num_observations+shift_covariates, num_tot
    # covariates = convert(Array{Float64,2}, covariates)
    # data["covariates"] = covariates
    # data["covariates_start"] = covariates_start
    return nothing
end

# ============================================================================

function rw_step_idx(num_observations, start_idx, iar_step)
    # iar_idx = zeros(Int64, num_tot)
    # iar_idx[1:start_idx-1] .= 1
    idx = 2
    for (j,i) in enumerate(start_idx:num_observations)
         iar_idx[i] = idx
         j % iar_step == 0 && (idx += 1)
    end
    if num_observations < num_tot
        iar_idx[num_observations+1:num_tot] .= iar_idx[num_observations]
    end
    iar_idx
end

function stepindex(n, stepsize)
    index = Vector{Int64}(undef, n)
    step = 1
    for i in 1:n
        index[i] = step
        i % stepsize == 0 && ( step += 1 )
    end
    index
end

function randomwalk!(data, stepsize=1)
    @unpack num_observations, lockdown_indices, num_covariates = data
    ns = Vector{Int64}()
    is = Vector{Vector{Int64}}()
    for (i, (no, l)) in enumerate(zip(num_observations, lockdown_indices))
        n = if data["semiparametric"] && data["num_covariates"] > 0
            data["covariates_start"][i] - l - 1
        else
            no - l
        end
        i = stepindex(n, stepsize)
        push!(ns, n)
        push!(is, i)
    end
    data["num_rt_steps"] = ns
    data["rt_step_index"] = is
end

# ============================================================================
# observation model defaults

@kwdef struct CaseInit1 <: ObservationInit
    obs_start        ::String  = ""
    obs_stop         ::String  = ""
    delay_dist                 = NegativeBinomial2
    delay_length     ::Int64   = 40
    delay_dispersion ::Float64 = 5.41
    populations      ::Vector{Int64} = Regional.populations
end

@kwdef struct HospitInit1 <: ObservationInit
    obs_start        ::String  = ""
    obs_stop         ::String  = ""
    delay_dist                 = NegativeBinomial2
    delay_length     ::Int64   = 40
    delay_dispersion ::Float64 = 5.41
    populations      ::Vector{Int64} = Regional.populations
end

@kwdef struct DeathInit1 <: ObservationInit
    obs_start        ::String  = ""
    obs_stop         ::String  = ""
    delay_dist                 = NegativeBinomial2
    delay_length     ::Int64   = 60
    delay_dispersion ::Float64 = 14.26
    populations      ::Vector{Int64} = Regional.populations
end
# ==============================================================================
# observation model

struct ObservParams3{SA,SO,N,DD,DL,DP,P,H,W}
    starts::SA
    stops::SO
    num_regions::N
    delay_dist::DD
    delay_length::DL
    delay_dispersion::DP
    populations::P
    holidays::H
    weekdays::W
end

function ObservParams3(data, o::ObservationInit)
    @unpack dates_turing, dates = data
    ds  = [collect(first(dt):Day(1):last(dates)) for dt in dates_turing]

    params = (
        starts           = startindices(data, o),
        stops            = stopindices(data, o),
        num_regions      = length(regions),
        delay_dist       = o.delay_dist,
        delay_length     = o.delay_length,
        delay_dispersion = o.delay_dispersion,
        # observations     = observations(data, o),
        populations      = o.populations,
        holidays         = holiday.(ds),
        weekdays         = [dayofweek.(d) for d in ds]
    )
    return ObservParams3( params... )
end

function startindices(data, o)
    @unpack dates_turing = data
    observation_startdates = if isempty(o.obs_start)
        data["startdates"] .+ Day(data["epidemicstart"])
    else
        fill(Date(o.obs_start), length(regions))
    end

    is = Vector{Int64}()
    for m in 1:length(regions)
        d = observation_startdates[m]
        i = findfirst(==(d), dates_turing[m])
        push!(is,i)
    end
    return is
end

function stopindices(data, o)
    @unpack dates_turing = data
    if isempty(o.obs_stop)
        return data["num_observations"]
    else
        is = Vector{Int64}()
        d = Date(o.obs_stop)
        for m in 1:length(regions)
            i = findfirst(==(d), dates_turing[m])
            push!(is,i)
        end
        return is
    end
end

function holiday(dates)
    specialdays = [Date("2020-12-24"), Date("2020-12-25"), Date("2020-12-31"), Date("2021-01-01")]
    holidays = dates .∈ Ref(specialdays)
    return convert(Vector{Int64}, holidays)
end

ObservParams3!(data, o::CaseInit1)   = data["casemodel"]   = ObservParams3(data, o)
ObservParams3!(data, o::HospitInit1) = data["hospitmodel"] = ObservParams3(data, o)
ObservParams3!(data, o::DeathInit1)  = data["deathmodel"]  = ObservParams3(data, o)

# ===============================================================================
# main function

function load_data(;
    observationsend  = nothing,
    predictors        = nothing,
    rwstep           = 1,
    epidemicstart    = 30,
    numimpute        = 6,
    casemodel         = CaseInit1(),
    hospitmodel       = HospitInit1(),
    deathmodel        = DeathInit1(),
    seromodel         = SeroInit2(),
    link              = KLogistic(3.),
    invlink           = KLogit(3.),
    lockdown          = "2020-03-18",
    covariates_kwargs = Dict(
        :semiparametric => false,
        :fname => normpath( homedir(), "data/covidsurvey/smoothed_contacts.csv" ),
        :shift => 0,
        :startdate => nothing,
        :enddate => nothing
    )
)

    data = Dict{String, Any}()
    observables!(data)
    consistent!(data; epidemicstart)
    turingformat!(data, observationsend)
    # cases_start_date!(data, cases_start)
    covariates!(data, predictors; covariates_kwargs...)
    lockdown!(data, lockdown)
    randomwalk!(data, rwstep)
    ObservParams3!(data, casemodel)
    ObservParams3!(data, hospitmodel)
    ObservParams3!(data, deathmodel)
    SeroParams3!(data, seromodel)

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
        serial_interval             = serialinterval(15),#padzeros(serialinterval(30), 0, num_tot), #data_usa.turing_data.serial_intervals[1:50],
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
        # num_i2h                     = 40,
        # ϕ_i2h                       = 5.41,
        # num_i2d                     = 60,
        # ϕ_i2d                       = 14.26,
        # num_i2c                     = 40,
        # ϕ_i2c                       = 5.41,
    )

    dates = [ collect(d:Day(1):d+Day(n-1)) for (d, n) in zip(data["startdates"], data["num_tot"])]
    startdates = [d[epidemicstart] for d in data["dates_turing"]]

    Data(
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
