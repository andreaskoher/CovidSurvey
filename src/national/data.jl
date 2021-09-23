const population = 5_837_213

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

function missing2zero(xs)
    i = findfirst(!ismissing, xs)
    xs[1:i] .= 0
    convert(Vector{Int64}, xs)
end
# weekly_average(xs) = [mean(xs[i:(i+6)]) for i in 1:div(length(xs),7)]
# weekly_average(deaths)

function padlast(xs, l)
    n  = length(xs)
    i  = findlast(x -> typeof(x) <: Real, xs) #avoid catching a missing value
    ys = Vector{Union{Float64, Missing}}(undef, l)
    ys[1:n] = xs
    ys[n+1:end] .= xs[i]
    ys
end

function padmissing(xs, l)
    n  = length(xs)
    ys = Vector{Union{eltype(xs), Missing}}(missing, l)
    ys[1:n] = xs
    ys
end

function cum2counts(x)
    x = x |> missing2zero |> diff
    pushfirst!(x, 0)
    x[x.<0] .= 0 # ignore data inconsistencies
    x
end

const dformat1 = DateFormat("d u Y")
const dformat2 = DateFormat("d-u-Y")
function parsetime(ds)
    ds = replace(ds, "Mrz"=>"Mar")
    ds = replace(ds, "Dez"=>"Dec")
    ds = replace(ds, "."=>"")
    d = try
        Date(ds, dformat1)
    catch
        try
            Date(ds, dformat2)
        catch
            @error "Not a valid time format: $ds"
        end
    end
    d + Year(2000)
end

function znorm!(xs)
    xs .-= mean(skipmissing(xs))
    xs ./= std(skipmissing(xs))
    nothing
end

function minmaxnorm!(xs)
    xs .-= minimum(skipmissing(xs))
    xs ./= maximum(skipmissing(xs))
    nothing
end

function surveynorm!(xs)
    i = findfirst(x->!ismissing(x), xs)
    xs[i] == 0. && return nothing
    xs ./= abs(xs[i])
    xs .-= xs[i]
    return nothing
end

# findall(ismissing, xs[mask])
function preproc(xs::AbstractVector, num_obs, Ntot; normalize=nothing)
    mask = 1:num_obs
    xs = padlast(xs[mask], Ntot)
    !isnothing(normalize) && normalize(xs)
    xs
end

# preproc(dk.Rt, num_obs, Ntot; normalize=surveynorm!)
# function select_predictors(dk, predictors, num_obs, Ntot, surveyvars)
#     predictors = String.(predictors)
#
#     #preprocess mobility data
#     dk[!,:nonresidential_mobility] = preproc(
#         (
#             dk.retail_and_recreation_percent_change_from_baseline +
#             dk.grocery_and_pharmacy_percent_change_from_baseline +
#             dk.workplaces_percent_change_from_baseline
#         ) / 3 / 100 , num_obs, Ntot; normalize=nothing)
#     # begin
#     #     Mavg = (
#     #     preproc(dk.retail_and_recreation_percent_change_from_baseline, num_obs, Ntot) +
#     #     preproc(dk.grocery_and_pharmacy_percent_change_from_baseline, num_obs, Ntot) +
#     #     preproc(dk.workplaces_percent_change_from_baseline, num_obs, Ntot)
#     #     ) ./ -3
#     # end
#     dk[!,:residential_mobility] = preproc(dk.residential_percent_change_from_baseline, num_obs, Ntot; normalize=nothing) ./ 100
#
#     #preprocess survey data
#     for k in surveyvars
#         dk[!,k] = preproc(dk[:,k], num_obs, Ntot; normalize=surveynorm!)
#     end
#
#     dk[!, :lockdown] .= 0.
#     t = findfirst(==(Date("2020-03-14")), dk.date) # lockdown date in DK: 11. (partial), 17. (full)
#     dk[t:end, :lockdown] .+= 1
#     # select predictors
#     miss = .!( in.(predictors, Ref(names(dk)) ) )
#     any(miss) && (@error "predictor $(predictors[miss]) not found. Select either nonresidential_mobility, residential_mobility or from survey: $(surveyvars)")
#     Array( dk[!,predictors] )
# end

function select_predictors(covariates, predictors, num_obs = nothing, num_tot = nothing)
    predictors = String.(predictors)

    cov_names = filter(x->x!=="date", names(covariates))
    for k in cov_names
        if isnothing(num_obs) || isnothing(num_tot)
            surveynorm!(covariates[!,k])
        else
            covariates[!,k] = preproc(covariates[:,k], num_obs, num_tot; normalize=surveynorm!)
        end
    end

    miss = .!( in.(predictors, Ref(cov_names) ) )
    any(miss) && (@error "predictor $(predictors[miss]) not found. Select either nonresidential_mobility, residential_mobility or from survey: $(surveyvars)")
    Array( covariates[!,predictors] )
end

rolling(xs) = begin
    n  = length(xs)
    ys = Vector{Union{Missing, Float64}}(missing, n)
    ys[1:n-6] = [maximum(xs[i:i+6]) for i in 1:n-6]
    ys
end

function gamma(μ, cv)
    α = 1/cv^2
    θ = μ/α
    Gamma(α, θ)
end

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
    i2c(x) = pdf(gamma(5.1, .86), x)
    # c2h(x) = pdf(gamma(2.6, .92), x) * ihr
    c2h(x) = pdf(gamma(6.49, .92), x) * ihr
    return nconvolve(i2c, c2h, ts)
end

function inf2death(nbins=75, ihr = 1)
    # ifr = 89 / 100_000 for age < 70 see Erikstrup et al. in Clin. Inf. Dis.

    ## report34: onset to death = 19.8, Cv = .85
    ts = 1.5:1:(nbins + .5)
    i2c(x) = pdf(gamma(5.1, .86), x)
    c2d(x) = pdf(gamma(19.8, .85), x) * ihr
    return nconvolve(i2c, c2d, ts)

    # Brauer et al.
    # r = vcat([0], 1.5:1:(nbins + .5))
    # diff(cdf.(NegativeBinomial2(21.82, 14.26), r))

    ## S. Abbott
    # i2o(x) = pdf(LogNormal(1.621, 0.418),x)
    # o2d(x) = pdf(LogNormalMeanStd(13.1, 11.7),x)
    # return nconvolve(i2o, o2d, ts)

    ## Nouvelette
    # GammaMeanStd(18.8, 8.48)
end

function inf2case(nbins = 30)
    ## Imperial Report
    r = vcat([0], 1.5:1:(nbins + .5))
    # return diff(cdf.(gamma(5.1, .86), r))
    ## Brauner et al.
    return diff(cdf.(NegativeBinomial2(10.92, 5.41), r))
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
    # i2c(t) = pdf(gamma(5.1, .86), t)
    # ts = 1.5:1:(nbins + .5)
    # return nconvolve(i2c, c2p, ts)

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

# function inf2serocon(nbins)
#     N = 10_000_000
#     # onset to seroconversion time from Imperial Report 34
#     x = rand(gamma(5.1, .86), N) .+ rand(Exponential(18.3), N)
#     h = fit(Histogram, x, vcat([0], 1.5:1:150.5))
#     h.weights / N * ihr
# end

function aggregateregions!(df, mun2reg)
  regions = values(mun2reg) |> unique
  for r in regions
    df[!, r] .= 0
  end
  for m in keys(mun2reg)
    df[!, mun2reg[m]] += df[!,m]
  end
end

# function getcases()
#     municipalities = CSV.File(normpath( homedir(), "data/Statens-Serum-Institut/municipalities.csv"); delim="\t", )|>DataFrame
#     mun2reg = Dict(municipalities.municipality[i] => municipalities.region[i]
#                     for i in 1:size(municipalities,1))
#     cases    = CSV.File(normpath( homedir(), "data/Statens-Serum-Institut/Municipality_cases_time_series.csv") )|>DataFrame
#     cases = select(cases, municipalities.municipality, :date_sample => :date)
#     cases[!, :total] = sum(select(cases, municipalities.municipality)|>eachcol)
#     aggregateregions!(cases, mun2reg)
#     cases
# end

function insert_missing_dates!(df)
    ts = df.date[1] : Day(1) : df.date[end]
    missing_date = .! in.(ts, Ref(df.date))
    if any(missing_date)
        for t in ts[missing_date]
            insertrow!(df, t)
        end
    end
end

function replace_missing!(df, v=0)
    @assert !any( ismissing.( df.date ) )
    for c in eachcol(df)
        replace!(c, missing=>v)
    end
    disallowmissing!(df)
end

function process_ssi(df)
    rename!(df, 1=>:region, 2=>:date, 3=>:counts)
    df.region = replace(df.region, dan2eng...)
    dropmissing!(df, :region)
    df = unstack(df, :date, :region, :counts)
    insert_missing_dates!(df)
    replace_missing!(df)
    DataFrames.transform(df, AsTable(Not(:date)) => sum => :country)
end

# function getdeaths()
#     deaths   = CSV.File(normpath( homedir(), "data/Statens-Serum-Institut/Deaths_over_time.csv"))|>DataFrame
#     deaths = deaths[1:end-1,:]
#     rename!(deaths, "Dato"=>"date", "Antal_døde"=>"total")
#     deaths.date = Date.(deaths.date, Ref(dateformat"y-m-d"))
#     deaths
# end
#
# function gethospit()
#     hosp     = CSV.File(normpath( homedir(), "data/Statens-Serum-Institut/Newly_admitted_over_time.csv"))|>DataFrame
#     dan2eng = Dict(
#         "Dato"          => "date",
#         "Total"         => "total",
#         "Hovedstaden"   => "capital",
#         "Sjælland"      => "zealand",
#         "Syddanmark"    => "south",
#         "Midtjylland"   => "central",
#         "Nordjylland"   => "north",
#         "Ukendt Region" => "unknown"
#     )
#     rename!(hosp, dan2eng)
#     hosp
# end

enddate(df1, df2) = min(df1.date[end], df2.date[end])
enddate(df, args...) = min(df.date[end], enddate(args...))

function padzeros(xs::AbstractVector, padfirst=0, total=length(xs)+padfirst)
    n  = length(xs)
    ys = zeros(eltype(xs), total)
    ys[padfirst+1:n+padfirst] = xs
    ys
end

function insertrow!(df, d::Date, v=0)
    i = findfirst(==(d-Day(1)), df.date)
    v = [eltype(c) == Date ? d : v for c in eachcol(df)]
    insert!.(eachcol(df), i+1, v)
    nothing
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

function limit(df::DataFrame, s::Date, e::Date, pad=true)
    pad && df.date[1] > s && ( df = padzeros(df, s) )
    is = findfirst(==(s), df.date)
    ie = findlast(==(e), df.date)
    df[is:ie, :]
end

function parseweek(s)
    _,w = split(s)
    w = parse(Int, w)
    y = w > 40 ? Date("2020") : Date("2021")
    y + Week(w) - Day(3) #center at day 4
end

function parsepercent(x::String)
    x = strip(x, '%')
    x = strip(x)
    x = replace(x, "," => ".")
    parse(Float64, x) / 100
end

function parseCI(x)
    x = strip(x, '%')
    x = replace(x, "," => ".")
    lq, uq = split(x, '-')
    parse(Float64, lq) / 100, parse(Float64, uq) / 100
end

function parsepercent!(df)
    regions = ["capital", "zealand", "south", "central", "north"]
    df.country = parsepercent.(df.country)
    # for r in regions
    #     df[!,r] = parsepercent.(df[:,r])
    # end
end

function findindices(dates, ds)
    is = Int[]
    for d in ds
        i = findfirst(==(d), dates)
        push!(is, i)
    end
    is
end

function readsurvey(fname)
    s = CSV.File(fname)|>DataFrame
    s.date = s.date .|> parsetime
    s.date -= Day(1) # survey setup: contacts in past 24h
    dropmissing!(s)
    for n in names(s)
        c = s[:,n]
        if eltype(c) == String
            c = replace.(c, ","=>".")
            s[!,n] = parse.(Ref(Float64), c)
        end
    end
    s
end
#
# function readsurveys_raw()
#     s1 = normpath( homedir(), "data/covidsurvey/contacts.csv" )  |> readsurvey #projectdir("data", "survey", "timeseries_colleagues_strangers.csv") |> readsurvey
#     s1 = select(s1, Not([:Rt, :uncertainty_lower, :uncertainty_upper]))
#     s2 = normpath( homedir(), "data/covidsurvey/predictors.csv" )  |> readsurvey #projectdir("data", "survey", "predictors.csv") |> readsurvey
#     s = innerjoin(s1, s2, on=:date)
#     s[3:end,:] # skip first for data quality
# end
function readcovariates(; fname=nothing, shift=0, startdate=nothing, enddate=nothing, datecol=:date, kwargs...)
    isnothing(fname) && (@error "please give a file name to covariates_kwargs")
    df = CSV.File(fname)|>DataFrame
    df[!,datecol] += Day(shift)
    !isnothing(startdate) && filter!(row -> row[datecol] >= Date(startdate), df)
    !isnothing(enddate) && filter!(row -> row[datecol] <= Date(enddate), df)
    ts = df[1,datecol] : Day(1) : df[end,datecol]
    @assert all(df[!,datecol] .== ts )
    @assert !any( ismissing.(Array(df)) )
    @assert all( isfinite.(Array(df)) )
    rename!(df, datecol=>:date)
    return df
end

country(df) = DataFrame(:date=>df.date, :country=>df.country)

function holiday(dates)
    specialdays = [Date("2020-12-24"), Date("2020-12-25"), Date("2020-12-31"), Date("2021-01-01")]
    holidays = dates .∈ Ref(specialdays)
    convert(Vector{Int64}, holidays)
end

function observables!(turing_data,
    fname_cases  = normpath( homedir(), "data/Statens-Serum-Institut/dashboard/Regionalt_DB/08_bekraeftede_tilfaelde_pr_dag_pr_regions.csv" ),
    fname_hospit = normpath( homedir(), "data/Statens-Serum-Institut/dashboard/Regionalt_DB/06_nye_indlaeggelser_pr_region_pr_dag.csv" ),
    fname_deaths = normpath( homedir(), "data/Statens-Serum-Institut/dashboard/Regionalt_DB/07_antal_doede_pr_dag_pr_region.csv" )
)

    turing_data["cases"] = CSV.File( fname_cases ) |>
        DataFrame |> process_ssi |> country

    turing_data["hospit"] = CSV.File( fname_hospit ) |>
        DataFrame |> process_ssi |> country

    turing_data["deaths"] = CSV.File( fname_deaths ) |>
        DataFrame |> process_ssi |> country

    return nothing
end

function consistent!(data; epidemicstart = 30)
    @unpack cases, hospit, deaths = data
    data["epidemicstart"] = epidemicstart
    s = data["startdate"]  = startdate(deaths; epidemicstart)
    e = enddate(deaths, cases, hospit)
    data["deaths"] = limit(deaths, s, e)
    data["hospit"] = limit(hospit, s, e)
    data["cases"]  = limit(cases, s, e)
    data["dates"]  = s:Day(1):e
    @assert length(data["dates"]) == size(data["cases"],1) == size(data["hospit"],1) == size(data["deaths"],1)
    return nothing
end

function startdate(df; epidemicstart = 30)
    cumulative = cumsum(df[:,:country])
    start_idx = findfirst(>=(10), cumulative)
    return df.date[start_idx] - Day(epidemicstart)
end

function national_timeseries(dates, startdate, observationsend)
    is = findfirst(==(startdate), dates)
    ie = findfirst(==(observationsend), dates)
    return dates[is:ie]
end

function totaldays(dates, startdates)
    observationsend = last(dates)
    national_dates = national_timeseries(dates, startdates, observationsend)
    return length(national_dates)
end

function df2vec(df, startdate, observationsend)
    vs = Vector{Vector{Int64}}()
    is = findfirst(==(startdate), df.date)
    ie = findfirst(==(observationsend), df.date)
    v = vec(df[is:ie, :country])
    return v
end

function turingformat!(data, observationsend = nothing)
    @unpack dates, cases, hospit, deaths = data
    e = data["observationsend"] = isnothing(observationsend) ? last(dates) : Date(observationsend)
    s = data["startdate"]
    data["num_tot"]       = totaldays(dates, s)
    data["dates_turing"]  = national_timeseries(dates, s, e)
    data["cases_turing"]  = df2vec(cases, s, e)
    data["hospit_turing"] = df2vec(hospit, s, e)
    data["deaths_turing"] = df2vec(deaths, s, e)
    data["num_obs"]       = length(data["cases_turing"])
    return nothing
end

# function cases_start_date!(data, cases_start)
#     @unpack dates_turing, num_obs = data
#     if isnothing(cases_start)
#         data["num_case_obs"] = 0
#         data["cases_start_idx"] = nothing
#     else
#         data["cases_start_idx"] = i = findfirst(==(Date.(cases_start)), dates_turing)
#         data["num_case_obs"] = num_obs - i + 1
#     end
#     return nothing
# end

function covariates!(data, predictors; covariates_kwargs...)
    @unpack dates_turing, num_obs = data
    if isnothing(predictors)
        data["covariates"]       = Array{Float64,2}(undef, 1,1)
        data["num_covariates"]   = 0
        data["covariates_start"] = 0
    else
        covariates       = readcovariates(; covariates_kwargs... )
        covariates       = leftjoin(DataFrame(:date=>dates_turing), covariates, on=:date)
        sort!(covariates, :date)
        covariates_start = findfirst(x->!ismissing(x), covariates[:,Not(:date)][:,1])
        covariates       = covariates[covariates_start:num_obs,:]
        covariates       = select_predictors(covariates, predictors) #num_obs+shift_covariates, num_tot
        covariates       = convert(Array{Float64,2}, covariates)
        data["covariates"]       = covariates
        data["covariates_start"] = covariates_start
        data["num_covariates"]   = size(covariates,2)
    end
end

function lockdown!(data, lockdown)
    @unpack dates_turing = data
    i = findfirst(==(Date(lockdown)), dates_turing)
    data["lockdown_index"] = i
    return nothing
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

function randomwalk!(data, stepsize, covariates_kwargs)
    @unpack num_obs, lockdown_index, covariates_start, num_covariates = data
    n = if covariates_kwargs[:semiparametric] && data["num_covariates"] > 0
        covariates_start-lockdown_index-1
    else
        num_obs - lockdown_index
    end
    i = stepindex(n, stepsize)
    data["num_rt_steps"] = n
    data["rt_step_index"] = i
    return nothing
end

function time_varying_iar!(data, casemodel, iarstep=1)
    @unpack num_obs, num_tot = data
    data["iar_start_idx"] = startindex(data, casemodel)
    data["num_iarsteps"] = num_obs - data["iar_start_idx"] + 1
    # data["iar_idx"] = idx = stepindex(n, stepsize)
    # data["num_iarsteps"] = length(unique(idx[1:num_obs]))
    # iar_idx = get_iar_idx(num_tot, num_obs, cases_start_idx, iarstep)
    # num_iarsteps = length(unique(iar_idx))
    return nothing
end
# ============================================================================
# observation model defaults
@kwdef struct CaseInit <: ObservationInit
    obs_start        ::String  = ""
    obs_stop         ::String  = ""
    delay_dist                 = NegativeBinomial2
    delay_length     ::Int64   = 40
    delay_dispersion ::Float64 = 5.41
    population       ::Int64   = National.population
end

@kwdef struct HospitInit <: ObservationInit
    obs_start        ::String  = ""
    obs_stop         ::String  = ""
    delay_dist                 = NegativeBinomial2
    delay_length     ::Int64   = 40
    delay_dispersion ::Float64 = 5.41
    population       ::Int64   = National.population
end

@kwdef struct DeathInit <: ObservationInit
    obs_start        ::String  = ""
    obs_stop         ::String  = ""
    delay_dist                 = NegativeBinomial2
    delay_length     ::Int64   = 60
    delay_dispersion ::Float64 = 14.26
    population       ::Int64   = National.population
end

# ==============================================================================
# observation model

struct ObservParams{SA,SO,DD,DL,DP,P,H,W}
    start::SA
    stop::SO
    delay_dist::DD
    delay_length::DL
    delay_dispersion::DP
    population::P
    holiday::H
    weekday::W
end

function startindex(data, o)
    dates   = data["dates_turing"]
    if isempty(o.obs_start)
        date = Date(data["startdate"]) + Day(data["epidemicstart"])
        i = findfirst(==(date), dates)
        return i
    else
        date = Date(o.obs_start)
        i = findfirst(==(date), dates)
        return i
    end
end

function stopindex(data, o)
    dates   = data["dates_turing"]
    if isempty(o.obs_stop)
        return data["num_obs"]
    else
        date = Date(o.obs_stop)
        i = findfirst(==(date), dates)
        return i
    end
end

function ObservParams(data, o::ObservationInit)
    params = (
        start            = startindex(data, o),
        stop             = stopindex(data, o),
        delay_dist       = o.delay_dist,
        delay_length     = o.delay_length,
        delay_dispersion = o.delay_dispersion,
        # observations     = observations(data, o),
        population       = o.population,
        holiday          = holiday(data["dates"]),
        weekday          = dayofweek.(data["dates"])
    )
    return ObservParams( params... )
end

ObservParams!(data, o::CaseInit)   = data["casemodel"]   = ObservParams(data, o)
ObservParams!(data, o::HospitInit) = data["hospitmodel"] = ObservParams(data, o)
ObservParams!(data, o::DeathInit)  = data["deathmodel"]  = ObservParams(data, o)
# ObservParams!(data, o::Sero)    = data["seromodel"]   = ObservParams(data, o)

# observations(data, ::CaseInit)   = data["cases_turing"]
# observations(data, ::HospitInit) = data["hospit_turing"]
# observations(data, ::DeathInit)  = data["deaths_turing"]

# ===========================================================================
# sero

# ≈ 15days delay from infection to antibody production
# see Ferguson Report and Zhao 2020: "Antibody responses to SARS-CoV-2 in patients of novel coronavirus disease 2019"
function readsero(;later_dataset = true)
    df0 = CSV.File(projectdir("data/seroprevalence/", "seroprev_early.csv"), delim="\t")|>DataFrame
    df1 = CSV.File(projectdir("data/seroprevalence/", "seroprev.csv"), delim="\t")|>DataFrame
    df1[!,:date] = df1.week .|> parseweek
    select!(df1, Not(:week))
    df = later_dataset ? vcat(df0, df1) : df0
    parsepercent!(df)
    df.CI = df.CI .|> parseCI
    df[!,:std] = [max( abs.(df.country[i] .- df.CI[i])... ) / 1.96 for i in 1:length(df.CI)]
    return df
end

function SeroParams!(data, sero=SeroInit1(); later_dataset = true)
    @unpack dates_turing = data

    df = readsero(;later_dataset)
    s = isempty(sero.obs_start) ? dates_turing[1] : Date(sero.obs_start)
    e = isempty(sero.obs_stop) ? dates_turing[end] : Date(sero.obs_stop)
    df = df[ s .<= df.date .<= e, :]
    data["seroprev"] = df

    # σ = df.std
    # uncertainty_dist = sero.dist.(σ, Ref(sero.cv))
    params = (
        mean  = df.country,
        std   = sero.std,
        CI    = df.CI,
        # dstd  = arraydist( uncertainty_dist),
        index = [findfirst(==(d), dates_turing) for d in df.date],
        dates = df.date,
        delay = sero.delay,
        population = population
    )
    data["sero"] = df.country
    data["seromodel"] = SeroParams(params...)

    return nothing
end

@kwdef struct SeroInit1# <: ObservationInit
    obs_start        ::String  = ""
    obs_stop         ::String  = "2020-07-01"
    delay            ::Int64   = 0
    std              ::Float64 = 0.5
    # population       ::Int64   = National.population
    # dist                       = InverseGamma2
    # cv               ::Float64 = 0.1
end

struct SeroParams{M,S,C,I,A,Y,P}
    mean::M
    std::S
    CIs::C
    # dstd::D
    index::I
    dates::A
    delay::Y
    population::P
end

# function hospitalizations!(data)
#     params = (
#         start            = data["cases_start_idx"],
#         stop             = data["num_observations"]
#         delay_dist       = NegativeBinomial2,
#         delay_num        = 40,
#         delay_dispersion = 5.41,
#         observations     = data["cases_turing"],
#         population       = population,
#     )
#     data["hospit_model"] = Observations( params... )
#     return nothing
# end
#

function load_data(;
    observationsend  = nothing,
    predictors        = nothing,
    rwstep           = 1,
    iarstep          = 1,
    epidemicstart    = 30,
    numimpute        = 6,
    casemodel         = CaseInit(),
    hospitmodel       = HospitInit(),
    deathmodel        = DeathInit(),
    seromodel         = SeroInit1(),
    link              = KLogistic(3.),
    invlink           = KLogit(3.),
    lockdown          = "2020-03-18",
    covariates_kwargs = Dict(
        :semiparametric => false,
        :fname => projectdir("data","smoothed_contact_rates.csv"),
        :shift => -1,
        :startdate => "2020-11-10",
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
    randomwalk!(data, rwstep, covariates_kwargs)
    time_varying_iar!(data, casemodel, iarstep)
    ObservParams!(data, casemodel)
    ObservParams!(data, hospitmodel)
    ObservParams!(data, deathmodel)
    SeroParams!(data, seromodel)


    # dk = begin
    #     update && run(`Rscript $(datadir())/COVID19.R`)
    #     dk = load(datadir("COVID19", "DK.rds"))
    #     limit(dk, s, e, false)
    # end

    turing_data = (
        num_impute                  = numimpute,
        num_total_days              = data["num_tot"],
        casemodel                   = data["casemodel"],
        cases                       = data["cases_turing"],
        deathmodel                  = data["deathmodel"],
        deaths                      = data["deaths_turing"],
        epidemic_start              = epidemicstart,
        population                  = population, #denmark statistics 2020 Q4
        serial_interval             = serialinterval(15),#padzeros(serialinterval(30), 0, num_tot), #data_usa.turing_data.serial_intervals[1:50],
        num_si                      = 15,
        lockdown_index              = data["lockdown_index"],
        hospitmodel                 = data["hospitmodel"],
        hospit                      = data["hospit_turing"],
        num_rt_steps                = data["num_rt_steps"],
        rt_step_indices             = data["rt_step_index"],
        iar_start_idx               = data["iar_start_idx"],
        num_iarsteps               = data["num_iarsteps"],
        num_observations            = data["num_obs"],
        link                        = link,
        invlink                     = invlink,
        seromodel                   = data["seromodel"],
        sero                        = data["sero"],
        num_covariates              = data["num_covariates"],
        covariates                  = data["covariates"],
        covariates_start            = data["covariates_start"],
        semiparametric              = covariates_kwargs[:semiparametric],
        rwscale                     = sqrt(rwstep),
        # seroprev_mean               = data["seroprev_mean"],
        # seroprev_std                = data["seroprev_std"],
        # seroprev_idx                = data["seroprev_idx"],

    )

    Data(
        turing_data,
        data["cases"],
        data["deaths"],
        data["hospit"],
        data["num_obs"], #num_observations
        data["num_tot"],
        data["dates"],
        population,
        predictors,
        data["seroprev"],
        lockdown,
        data["observationsend"],
        -1,
        data["startdate"]
    )
end
