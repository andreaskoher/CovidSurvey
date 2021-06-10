const regions = ["capital", "zealand", "south", "central", "north"]
const nregions = length(regions)
const population = [1_855_084, 838_840,	1_223_634, 1_332_048, 590_439] # from statistics denmark

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

function startdate(df; epidemic_start = 30)
    start_idx = zeros(Int64, nregions)
    for (i, r) in enumerate(regions)
        cumulative = cumsum(df[:,r])
        start_idx[i] = findfirst(>=(10), cumulative)
    end
    df.date[start_idx] .- Day(epidemic_start)
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
    return p / sum(p)
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

function preproc(xs::AbstractVector, num_obs, num_tot; normalize=nothing)
    mask = 1:num_obs
    xs = padlast(xs[mask], num_tot)
    !isnothing(normalize) && normalize(xs)
    xs
end

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
# ============================================================================

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

function parse_std(s)
    s_m, s_cil, _, s_cir = split(s)
    μ = parsepercent(s_m)
    cil = parsepercent(s_cil)
    cir = parsepercent(s_cir)
    σ = max( abs(μ - cil), abs(μ - cir)) / 1.96
    return σ
end

function readseroprev(dates)
    df0 = CSV.File(projectdir("data/seroprevalence/", "seroprev_regional.csv"), delim="\t") |> DataFrame
    #df1 = CSV.File(projectdir("data/seroprevalence/", "seroprev.csv"), delim="\t")|>DataFrame
    #df1[!,:date] = df1.week .|> parseweek
    #select!(df1, Not(:week))
    #df = vcat(df0, df1)

    df = df0
    df = df[ dates[1] .<= df.date .<= dates[end], :]
    means = Vector{Vector{Float64}}()
    stds  = Vector{Vector{Float64}}()
    idxs  = Vector{Vector{Int64}}()
    idx   = [findfirst(==(d), dates) for d in df.date]
    for r in regions
        col = df[:,r]
        push!( means, parse_mean.(col) )
        push!( stds, parse_std.(col) )
        push!( idxs, idx )
    end

    turing_data = (
        std  = stds,
        mean = means,
        idx  = idxs
    )
    return turing_data
end

# ============================================================================

function df2vec(df, startdates, observations_end)
    vs = Vector{Vector{Int64}}()
    for (s,r) in zip(startdates, regions)
        is = findfirst(==(s), df.date)
        ie = findfirst(==(observations_end), df.date)
        v = vec(df[is:ie, r])
        push!(vs, v)
    end
    vs
end

function regional_timeseries(dates, startdates, observations_end)
    ts = Vector{Vector{Dates.Date}}()
    for s in startdates
        is = findfirst(==(s), dates)
        ie = findfirst(==(observations_end), dates)
        t = dates[is:ie]
        push!(ts, t)
    end
    ts
end

function totaldays(dates, startdates)
    observations_end = last(dates)
    regional_dates = regional_timeseries(dates, startdates, observations_end)
    return length.(regional_dates)
end

function lockdown_index(dates)
    li = Vector{Int64}()
    for d in dates
        i = findfirst(==(Date("2020-03-18")), d)
        push!(li, i)
    end
    return li
end

function turingformat!(data, observations_end = nothing)
    @unpack dates, cases, hospit, deaths = data
    e = data["observations_end"] = isnothing(observations_end) ? last(dates) : Date(observations_end)
    s = data["startdates"]
    data["num_tot"]       = totaldays(dates, s)
    data["dates_turing"]  = regional_timeseries(dates, s, e)
    data["cases_turing"]  = df2vec(cases, s, e)
    data["hospit_turing"] = df2vec(hospit, s, e)
    data["deaths_turing"] = df2vec(deaths, s, e)
    data["num_obs"]  = length.(data["cases_turing"])
    data["lockdown"] = lockdown_index(data["dates_turing"])
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

function consistent!(data; epidemic_start = 30)
    @unpack cases, hospit, deaths = data
    startdates = data["startdates"] = startdate(deaths; epidemic_start)
    s      = minimum(startdates)
    e      = enddate(deaths, cases, hospit)
    data["deaths"] = limit(deaths, s, e)
    data["hospit"] = limit(hospit, s, e)
    data["cases"]  = limit(cases, s, e)
    data["dates"]  = s:Day(1):e
    @assert length(data["dates"]) == size(data["cases"],1) == size(data["hospit"],1) == size(data["deaths"],1)
    return nothing
end

function cases_start_date!(data, cases_start)
    @unpack dates_turing, num_obs = data
    if isnothing(cases_start)
        data["num_case_obs"] = zeros(Int64, length(regions))
        data["cases_start_idx"] = nothing
    else
        is = Vector{Int64}()
        ns = Vector{Int64}()
        for (d, no) in zip(dates_turing, num_obs)
            i  = findfirst(==(Date.(cases_start)), d)
            n = no - i + 1
            push!(is, i)
            push!(ns, n)
        end
        data["cases_start_idx"] = is
        data["num_case_obs"] = ns
    end
    return nothing
end

function covariates!(data, predictors; kwargs...)
    isnothing(predictors) && return
    # covariates = readcovariates(; kwargs... )
    # covariates = leftjoin(DataFrame(:date=>dates), covariates, on=:date)
    # sort!(covariates, :date)
    # covariates_start = findfirst(x->!ismissing(x), covariates[:,end]) # NOTE: assiming all covariates start at the same date
    # covariates = covariates[covariates_start:num_obs,:]
    #
    # covariates = select_predictors(covariates, predictors) #num_obs+shift_covariates, num_tot
    # covariates = convert(Array{Float64,2}, covariates)
    # # preds = select_predictors(dk, predictors, num_obs+shift_covariates, num_tot, surveyvars)
    # # preds = convert(Array{Float64,2}, preds[covariates_start:end,:] )
    # turing_data[:covariates] = covariates
    # turing_data[:covariates_start] = covariates_start
    return nothing
end

# ============================================================================

function rw_step_idx(num_obs, start_idx, iar_step)
    # iar_idx = zeros(Int64, num_tot)
    # iar_idx[1:start_idx-1] .= 1
    idx = 2
    for (j,i) in enumerate(start_idx:num_obs)
         iar_idx[i] = idx
         j % iar_step == 0 && (idx += 1)
    end
    if num_obs < num_tot
        iar_idx[num_obs+1:num_tot] .= iar_idx[num_obs]
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
    @unpack num_obs, lockdown = data
    ns = Vector{Int64}()
    is = Vector{Vector{Int64}}()
    for (no, l) in zip(num_obs, lockdown)
        n = no - l
        i = stepindex(n, stepsize)
        push!(ns, n)
        push!(is, i)
    end
    data["num_rt_steps"] = ns
    data["rt_step_index"] = is
end

function load_data(;
    observations_end=nothing,
    predictors=nothing,
    cases_start=nothing,
    addhospital=true,
    addseroprev=false,
    addtests=false,
    update=false,
    iar_step = 1,
    rw_step  = 1,
    epidemic_start = 30,
    num_impute = 6,
    link = exp,
    invlink = log,
    covariates_kwargs = Dict(
        :fname => normpath( homedir(), "data/covidsurvey/smoothed_contacts.csv" ),
        :shift => 0,
        :startdate => nothing,
        :enddate => nothing
    )
)

    data = Dict{String, Any}()
    observables!(data)
    consistent!(data; epidemic_start)
    turingformat!(data, observations_end)
    cases_start_date!(data, cases_start)
    covariates!(data, predictors; covariates_kwargs...)
    randomwalk!(data, rw_step)


    # iar_idx = get_iar_idx(num_tot, num_obs, cases_start_idx, iar_step)
    # num_iar_steps = length(unique(iar_idx))

    seroprev = nothing
    # seroprev, turing_seroprev = readseroprev(dates[1:num_obs])
    # if addseroprev
    #     #turing_data[:πs] = padzeros(inf2serocon(50, 1), 0,  num_tot)
    #     turing_data[:seroprev_mean] = turing_seroprev.mean .* population[:,:]'
    #     turing_data[:seroprev_std]  = turing_seroprev.std .* population[:,:]'
    #     turing_data[:seroprev_idx]  = turing_seroprev.idx
    #     turing_data[:i2s]            = seropos(num_tot)
    # end

    turing_data = (
        num_impute                  = num_impute,
        num_total_days              = data["num_tot"],
        cases                       = data["cases_turing"],
        deaths                      = data["deaths_turing"],
        epidemic_start              = epidemic_start,
        populations                 = population,
        serial_interval             = serialinterval(15),#padzeros(serialinterval(30), 0, num_tot), #data_usa.turing_data.serial_intervals[1:50],
        num_si                      = 15,
        lockdown_indices            = data["lockdown"],
        cases_start_indices         = data["cases_start_idx"],
        hospits                     = data["hospit_turing"],
        num_rt_steps                = data["num_rt_steps"],
        rt_step_indices             = data["rt_step_index"],
        num_regions                 = length(regions),
        num_observations            = length.(data["cases_turing"]),
        link                        = link,
        invlink                     = invlink,
        num_i2h                     = 40,
        ϕ_i2h                       = 5.41,
    )

    Data(
        turing_data,
        data["cases"],
        data["deaths"],
        data["hospit"],
        data["num_obs"], #num_observations
        data["num_tot"],
        data["startdates"],
        population,
        predictors,
        seroprev,
        regions
    )
end
