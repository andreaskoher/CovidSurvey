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

function startdate(df)
    start_idx = zeros(Int64, nregions)
    for (i, r) in enumerate(regions)
        cumulative = cumsum(df[:,r])
        start_idx[i] = findfirst(>=(10), cumulative)
    end
    df.date[start_idx] .- Day(30)
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

function readcovariates(fname, shift_covariates=0)
    # fname = normpath( homedir(), "data/covidsurvey/smoothed_contacts_0425_2m.csv" )
    # fname = normpath( homedir(), "data/covidsurvey/contacts_0425_2m.csv" )
    df = CSV.File(fname)|>DataFrame
    df.date += Day(shift_covariates)
    ts = df.date[1] : Day(1) : df.date[end]
    @assert all(df.date .== ts )
    @assert !any( ismissing.(Array(df)) )
    @assert all( isfinite.(Array(df)) )
    return df
end

# ============================================================================

function get_iar_idx(num_tot, num_obs, cases_start_idx, iar_step)
    iar_idx = zeros(Int64, num_tot)
    iar_idx[1:cases_start_idx-1] .= 1
    idx = 2
    for (j,i) in enumerate(cases_start_idx:num_obs)
         iar_idx[i] = idx
         j % iar_step == 0 && (idx += 1)
    end
    if num_obs < num_tot
        iar_idx[num_obs+1:num_tot] .= iar_idx[num_obs]
    end
    iar_idx
end

# ============================================================================

function nconvolve(f1, f2, ts)
    nbins = length(ts)
    f(τ, t) = f1(τ) * f2(t - τ)
    f3 = zeros(nbins)
    ts = 1.:1:nbins
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
    # ts = 1.5:1:(nbins + .5)
    # i2c(x) = pdf(gamma(5.1, .86), x)
    # c2d(x) = pdf(gamma(19.8, .85), x) * ihr
    # return nconvolve(i2c, c2d, ts)

    # Brauer et al.
    r = vcat([0], 1.5:1:(nbins + .5))
    p = diff(cdf.(NegativeBinomial2(21.82, 14.26), r))
    return p / sum(p)
end

function inf2case(nbins = 30)
    # Brauner et al.
    r = vcat([0], 1.5:1:(nbins + .5))
    p = diff(cdf.(NegativeBinomial2(10.92, 5.41), r))
    return p / sum(p)
end

"""
seropos()[t] is the probablity of being seropositive t days after infection.
For details see imperial report 34
"""
function seropos(nbins=200)
    λ = 13.3
    μ = 300 #143.7 - λ
    κ = 3.67
    f(x, t, κ, μ, λ) = 1/λ*exp(-x/λ-((t-x)/μ)^κ)
    c2p(t) = quadgk(x->f(x, t, κ, μ, λ), 0, t, rtol=1e-8)[1]
    i2c(t) = pdf(GammaMeanCv(5.1, .86), t)
    ts = 1.5:1:(nbins + .5)
    p = nconvolve(i2c, c2p, ts)
    return p / sum(p)
end


function serialinterval(nbins = 30)
    # Imperial Reports
    r = vcat([0], 1.5:1:(nbins + .5))
    #diff(cdf.(gamma(6.5, .62), r))
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
    nothing
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

function select_predictors(covariates, predictors, num_obs, num_tot)
    predictors = String.(predictors)

    cov_names = filter(x->x!=="date", names(covariates))
    for k in cov_names
        covariates[!,k] = preproc(covariates[:,k], num_obs, num_tot; normalize=surveynorm!)
    end

    miss = .!( in.(predictors, Ref(cov_names) ) )
    any(miss) && (@error "predictor $(predictors[miss]) not found. Select either nonresidential_mobility, residential_mobility or from survey: $(surveyvars)")
    Array( covariates[!,predictors] )
end

# ============================================================================

function parsepercent(x::String)
    x = strip(x, '%')
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
    df.country = parsepercent.(df.country)
    for r in regions
        df[!,r] = parsepercent.(df[:,r])
    end
end

function readseroprev(dates)
    df0 = CSV.File(projectdir("data/seroprevalence/", "seroprev_early.csv"), delim="\t")|>DataFrame
    #df1 = CSV.File(projectdir("data/seroprevalence/", "seroprev.csv"), delim="\t")|>DataFrame
    #df1[!,:date] = df1.week .|> parseweek
    #select!(df1, Not(:week))
    #df = vcat(df0, df1)
    df = df0
    df = df[ dates[1] .<= df.date .<= dates[end], :]
    # df.date .-= Day(delay)
    parsepercent!(df)
    df.CI = df.CI .|> parseCI
    # df[!,:std_u] = [(CIs[i][2] - df.country[i]) / 1.96 for i in 1:length(CIs)]
    # estimates of std are not symmetric with respect to lower / upper CI bound
    # we choose the larger estimate, i.e. the lower bound
    df[!,:std] = [max( abs.(df.country[i] .- df.CI[i])... ) / 1.96 for i in 1:length(df.CI)]

    turing_data = (
        std  = repeat(df.std, outer=(1,length(regions))), #DEBUG ask for better estimates of uncertainty
        mean = Array{Float64}(df[:, regions]),
        idx  = [findfirst(==(d), dates) for d in df.date]
    )
    df, turing_data
end

# ============================================================================

function load_data(
    observations_end=nothing,
    predictors=nothing,
    cases_start=nothing,
    addhospital=false,
    addseroprev=false,
    addtests=false;
    update=false,
    fname_covariates = normpath( homedir(), "data/covidsurvey/smoothed_contacts.csv" ),
    iar_step = 7,
    shift_covariates = 1
)

    cases = CSV.File(normpath( homedir(), "data/Statens-Serum-Institut/dashboard/Regionalt_DB/08_bekraeftede_tilfaelde_pr_dag_pr_regions.csv" ) ) |>
        DataFrame |> process_ssi

    deaths = CSV.File(normpath( homedir(), "data/Statens-Serum-Institut/dashboard/Regionalt_DB/07_antal_doede_pr_dag_pr_region.csv" ) ) |>
        DataFrame |> process_ssi

    hospit = CSV.File(normpath( homedir(), "data/Statens-Serum-Institut/dashboard/Regionalt_DB/06_nye_indlaeggelser_pr_region_pr_dag.csv" ) ) |>
        DataFrame |> process_ssi

    seed_dates = startdate(deaths)
    s      = minimum(seed_dates)
    e      = enddate(deaths, hospit, cases)
    dates  = collect( s : Day(1) : e )
    deaths = limit(deaths, s, e)
    hospit = limit(hospit, s, e)
    cases  = limit(cases, s, e)
    num_tot= length(dates)

    covariates = readcovariates(fname_covariates, shift_covariates)
    covariates = leftjoin(DataFrame(:date=>dates), covariates, on=:date)
    sort!(covariates, :date)

    num_obs = if isnothing(observations_end)
        num_tot
    else
        findfirst(==(Date.(observations_end)), dates)
    end

    lockdown = findfirst(==(Date("2020-03-18")), dates)

    num_case_obs, cases_start_idx = if isnothing(cases_start)
        0, nothing
    else
        cases_start_idx = findfirst(==(Date.(cases_start)), dates)
        num_obs - cases_start_idx + 1, cases_start_idx
    end

    iar_idx = get_iar_idx(num_tot, num_obs, cases_start_idx, iar_step)
    num_iar_steps = length(unique(iar_idx))


    turing_data = OrderedDict(
        :num_impute                  => 6,
        :num_total_days              => num_tot,
        :cases                       => Array{Int64}(cases[1:num_obs, regions]), #padmissing(cases[mask], num_tot),
        :deaths                      => Array{Int64}(deaths[1:num_obs, regions]), #padmissing(deaths[mask], num_tot),
        :i2d                         => inf2death(60),
        :i2c                         => inf2case(40),
        :epidemic_start              => 31,
        :population                  => population,
        :serial_intervals            => serialinterval(15), #data_usa.turing_data.serial_intervals[1:50],
        :num_iar_steps               => num_iar_steps,
        :iar_idx                     => iar_idx,
        :lockdown                    => lockdown,
        :num_case_obs                => num_case_obs,
        )

    if !isnothing(predictors)
        preds = select_predictors(covariates, predictors, num_obs+shift_covariates, num_tot)
        covariates_start = findfirst(x->!ismissing(x), preds[:,end])
        preds = convert(Array{Float64,2}, preds[covariates_start:end,:] )
        turing_data[:covariates] = preds
        turing_data[:covariates_start] = covariates_start
    end

    if addhospital
        turing_data[:πh] = inf2hosp(50, 1)
        turing_data[:hospit] = Array{Int64}(hospit[1:num_obs, regions])
    end

    seroprev, turing_seroprev = readseroprev(dates[1:num_obs])
    if addseroprev
        #turing_data[:πs] = padzeros(inf2serocon(50, 1), 0,  num_tot)
        turing_data[:seroprev_mean] = turing_seroprev.mean .* population[:,:]'
        turing_data[:seroprev_std]  = turing_seroprev.std .* population[:,:]'
        turing_data[:seroprev_idx]  = turing_seroprev.idx
        turing_data[:i2s]            = seropos(num_tot)
    end

    turing_data = (;turing_data...)
    Data(
        turing_data,
        cases,
        deaths,
        hospit,
        num_obs, #num_observations
        num_tot,
        dates,
        population,
        predictors,
        seroprev,
        regions
    )
end
