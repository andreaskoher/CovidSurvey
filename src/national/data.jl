#----------------------------------------------------------------------------
# load data from Imperial College
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
    # plot(LogNormalMeanStd(5.2, 1.52), label="Abbott")
    # plot!(GammaMeanStd(5.06, 2.11), label="Brauer")
    # plot!(gamma(6.5, .62), label="Flaxman")
    #
    # Imperial Reports
    r = vcat([0], 1.5:1:(nbins + .5))
    return diff(cdf.(gamma(6.5, .62), r))
    ## Brauer et al. Science (2020)
    #return diff(cdf.(gamma(5.06, 2.11/5.06), r))
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

function startdate(deaths)
    cumulative = cumsum(deaths.country)
    i = findfirst(>=(10), cumulative)
    deaths.date[i] - Day(30)
end

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

# ≈ 15days delay from infection to antibody production
# see Ferguson Report and Zhao 2020: "Antibody responses to SARS-CoV-2 in patients of novel coronavirus disease 2019"
function readseroprev(dates)
    df0 = CSV.File(projectdir("data/seroprevalence/", "seroprev_early.csv"), delim="\t")|>DataFrame
    df1 = CSV.File(projectdir("data/seroprevalence/", "seroprev.csv"), delim="\t")|>DataFrame
    df1[!,:date] = df1.week .|> parseweek
    select!(df1, Not(:week))
    df = vcat(df0, df1)
    # df = df0
    df = df[ dates[1] .<= df.date .<= dates[end], :]
    # df.date .-= Day(delay)
    parsepercent!(df)
    df.CI = df.CI .|> parseCI
    # df[!,:std_u] = [(CIs[i][2] - df.country[i]) / 1.96 for i in 1:length(CIs)]
    # estimates of std are not symmetric with respect to lower / upper CI bound
    # we choose the larger estimate, i.e. the lower bound
    df[!,:std] = [max( abs.(df.country[i] .- df.CI[i])... ) / 1.96 for i in 1:length(df.CI)]
    turing_data = (
        std  = df.std,
        mean = df.country,
        idx  = [findfirst(==(d), dates) for d in df.date]
    )
    df, turing_data
end

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
    regions = ["capital", "zealand", "south", "central", "north"]
    df.country = parsepercent.(df.country)
    for r in regions
        df[!,r] = parsepercent.(df[:,r])
    end
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

country(df) = df.country

function get_iar_idx(num_tot, num_obs, start_idx, iar_step)
    iar_idx = zeros(Int64, num_tot)
    iar_idx[1:start_idx-1] .= 1
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

function load_data(
    observations_end=nothing,
    predictors=nothing,
    cases_start=nothing,
    addhospital=false,
    addseroprev=false,
    addtests=false;
    update=false,
    iar_step = 7,
    covariates_kwargs = Dict(
        :fname => normpath( homedir(), "data/covidsurvey/smoothed_contacts.csv" ),
        :shift => 0,
        :startdate => nothing,
        :enddate => nothing
    )
)

    cases = CSV.File(normpath( homedir(), "data/Statens-Serum-Institut/dashboard/Regionalt_DB/08_bekraeftede_tilfaelde_pr_dag_pr_regions.csv" ) ) |>
        DataFrame |> process_ssi

    deaths = CSV.File(normpath( homedir(), "data/Statens-Serum-Institut/dashboard/Regionalt_DB/07_antal_doede_pr_dag_pr_region.csv" ) ) |>
        DataFrame |> process_ssi

    hospit = CSV.File(normpath( homedir(), "data/Statens-Serum-Institut/dashboard/Regionalt_DB/06_nye_indlaeggelser_pr_region_pr_dag.csv" ) ) |>
        DataFrame |> process_ssi

    # tests = CSV.File(normpath( homedir(), "data/Statens-Serum-Institut/dashboard/Regionalt_DB/06_nye_indlaeggelser_pr_region_pr_dag.csv" ) ) |>
    #     DataFrame |> process_ssi

    s      = startdate(deaths)
    e      = enddate(deaths, hospit, cases)
    dates  = collect( s : Day(1) : e )
    deaths = limit(deaths, s, e) |> country
    hospit = limit(hospit, s, e) |> country
    cases  = limit(cases, s, e)  |> country
    num_tot= length(dates)

    dk = begin
        update && run(`Rscript $(datadir())/COVID19.R`)
        dk = load(datadir("COVID19", "DK.rds"))
        limit(dk, s, e, false)
    end

    # survey = readsurveys(shift_covariates)
    # surveyvars = filter(x->x!=="date", names(survey))
    # dk = leftjoin(dk, survey, on = :date; validate=(true,true))
    # Rt = CSV.File(
    #     projectdir(
    #         "data/Rt.csv")
    # )  |> DataFrame
    # push!(surveyvars, filter(x->x!=="date", names(Rt))...)
    # dk = leftjoin(dk, Rt, on = :date; validate=(true,true))
    # sort!(dk, :date)

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
        :cases                       => cases[1:num_obs], #padmissing(cases[mask], num_tot),
        :deaths                      => deaths[1:num_obs], #padmissing(deaths[mask], num_tot),
        :π                           => inf2death(90),#padzeros(inf2death(90), 0, num_tot),#[1:100], #length = 100
        :π2                          => inf2case(40),#padzeros(inf2case(30), 0, num_tot),#[1:30], #WARNING US-NY DATA
        :epidemic_start              => 31,
        :population                  => dk.population[1],
        :serial_intervals            => serialinterval(15),#padzeros(serialinterval(30), 0, num_tot), #data_usa.turing_data.serial_intervals[1:50],
        :num_iar_steps               => num_iar_steps,
        :iar_idx                     => iar_idx,
        :lockdown                    => lockdown,
        :num_case_obs                => num_case_obs,
        :weekday                     => dayofweek.(dates)
        )

    if !isnothing(predictors)
        covariates = National.readcovariates(; covariates_kwargs... )
        covariates = leftjoin(DataFrame(:date=>dates), covariates, on=:date)
        sort!(covariates, :date)
        covariates_start = findfirst(x->!ismissing(x), covariates[:,end]) # NOTE: assiming all covariates start at the same date
        covariates = covariates[covariates_start:num_obs,:]

        covariates = select_predictors(covariates, predictors) #num_obs+shift_covariates, num_tot
        covariates = convert(Array{Float64,2}, covariates)
        # preds = select_predictors(dk, predictors, num_obs+shift_covariates, num_tot, surveyvars)
        # preds = convert(Array{Float64,2}, preds[covariates_start:end,:] )
        turing_data[:covariates] = covariates
        turing_data[:covariates_start] = covariates_start
    end

    if addhospital
        turing_data[:πh] = inf2hosp(50, 1)#padzeros(inf2hosp(50, 1), 0,  num_tot) #length 50
        turing_data[:hospit] = hospit[1:num_obs]
    end

    seroprev, turing_seroprev = readseroprev(dates[1:num_obs])
    if addseroprev
        #turing_data[:πs] = padzeros(inf2serocon(50, 1), 0,  num_tot)
        turing_data[:seroprev_mean] = turing_seroprev.mean * dk.population[1]
        turing_data[:seroprev_std]  = turing_seroprev.std * dk.population[1]
        turing_data[:seroprev_idx]  = turing_seroprev.idx
        turing_data[:πs]            = seropos(num_tot)
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
        dk.population,
        predictors,
        seroprev,
    )
end
