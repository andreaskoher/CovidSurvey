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

# ============================================================================
#                      OBSERVABLES
# ============================================================================
# data cleaning

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

# ============================================================================
# determine start and end date

function startdate(df; epidemicstart = 30, epidemic_threshold = 30, regions = names(df, Not(:date)))
    start_idx = zeros(Int64, length(regions))
    for (i, r) in enumerate(regions)
        cumulative = cumsum(df[:,r])
        start_idx[i] = findfirst(>=(epidemic_threshold), cumulative) #
    end
    df.date[start_idx] .- Day(epidemicstart)
end

function enddate(df, args...)
    isnothing(df) && enddate(args...)
    min(df.date[end], enddate(args...))
end

function enddate(df1, df2)
    t1 = isnothing(df1) ? Date(10^5) : df1.date[end]
    t2 = isnothing(df2) ? Date(10^5) : df2.date[end]
    min(t1, t2)
end
# ============================================================================
# make observables consistent
# limit observations and add zero padding if necessary

function consistent!(data; epidemicstart = 30, epidemic_threshold = 30, startdates = nothing)
    @unpack cases, hospit, deaths, regions = data
    if isnothing(startdates)
        startdates = startdate(hospit; epidemicstart, epidemic_threshold, regions)
    elseif startdates isa Date
        startdates = fill(startdates, length(regions))
        epidemicstart = 1 # start epidemic with given startdate
    else
        @assert startdates isa Vector{Date}
        epidemicstart = 1 # start epidemic with given startdates
    end
    data["epidemicstart"] = epidemicstart
    data["startdates"] = startdates
    s      = minimum(startdates)
    e      = enddate(deaths, cases, hospit)
    data["deaths"] = limit(deaths, s, e)
    data["hospit"] = limit(hospit, s, e)
    data["cases"]  = limit(cases, s, e)
    data["dates"]  = s:Day(1):e
    # @assert length(data["dates"]) == size(data["cases"],1) == size(data["hospit"],1) == size(data["deaths"],1)
    return nothing
end

# function consistent!(data; epidemicstart = Date("2020-06-01"))
#     @unpack cases, hospit, deaths, regions = data
#     data["epidemicstart"] = epidemicstart
#     s      = epidemicstart
#     e      = enddate(deaths, hospit, cases)
#     data["deaths"] = limit(deaths, s, e)
#     data["hospit"] = limit(hospit, s, e)
#     data["cases"]  = limit(cases, s, e)
#     data["dates"]  = s:Day(1):e
#     # @assert length(data["dates"]) == size(data["cases"],1) == size(data["hospit"],1) == size(data["deaths"],1)
#     return nothing
# end

function limit(df, s::Date, e::Date, pad=true)
    isnothing(df) && return
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
# transform observables to Turing format

function turingformat!(data, observationsend = nothing, predictive=false)
    @unpack dates, cases, hospit, deaths = data
    e = data["observationsend"] = isnothing(observationsend) ? last(dates) : Date(observationsend)
    s = data["startdates"]
    data["num_tot"]            = totaldays(dates, s)
    ts = data["dates_turing"]  = regional_timeseries(dates, s, e)
    cs = df2vec(cases, s, e)
    hs = df2vec(hospit, s, e)
    ds = df2vec(deaths, s, e)
    for (t,c,h,d) in zip(ts,cs,hs,ds)
        @assert length(t) == length(c) == length(h) == length(d)
    end

    data["cases_turing"]  = !predictive ? cs : [similar(c, Missing) for c in cs]
    data["hospit_turing"] = !predictive ? hs : [similar(h, Missing) for h in hs]
    data["deaths_turing"] = !predictive ? ds : [similar(d, Missing) for d in ds]
    data["num_observations"]  = length.(data["dates_turing"])
    return nothing
end

function df2vec(df, startdates, observationsend)
    vs = Vector{Vector{Int64}}()
    isnothing(df) && return vs
    regions = names(df, Not(:date))
    for (s,r) in zip(startdates, regions)
        is = findfirst(==(s), df.date)
        ie = findfirst(==(observationsend), df.date)
        v = vec(df[is:ie, r])
        push!(vs, v)
    end
    vs
end

function totaldays(dates, startdates)
    observationsend = last(dates)
    regional_dates = regional_timeseries(dates, startdates, observationsend)
    return length.(regional_dates)
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

# ============================================================================
#                        COVARIATES
# ============================================================================
# main function
function covariates!(data, predictors; covariates_kwargs...)
    @unpack dates_turing, num_regions = data
    if isnothing(predictors)
        data["covariates"]       = [Array{Float64,2}(undef, 1,1) for i in 1:num_regions]
        data["num_covariates"]   = 0
        data["covariates_starts"] = fill(0, num_regions)
    else
        regional_covariates = Vector{Array{Float64,2}}()
        covariates_start_indices = Vector{Int64}()
        covariates_end_indices = Vector{Int64}()

        for i in 1:num_regions
            covariates = readcovariates(
                ; fname_survey = covariates_kwargs[:fname][i]
                , fname_mobility = covariates_kwargs[:mobility][i]
                , shift = covariates_kwargs[:shift]
                , datecol = covariates_kwargs[:datecol]
                #, aggregation = covariates_kwargs[:aggregation]
            )

            limit!(covariates, data, covariates_kwargs[:conditions])

            covariates_array = select_predictors(
                              covariates, predictors
                            ; standardize = covariates_kwargs[:standardize]
                            , normalize = covariates_kwargs[:normalize]
            ) #num_obs+shift_covariates, num_tot

            covariates_start_index = findfirst(==(covariates.date[1]), dates_turing[i])
            push!(regional_covariates, covariates_array)
            push!(covariates_start_indices, covariates_start_index)
        end
        data["covariates"] = regional_covariates
        @assert length(unique([size(regional_covariates[i],2) for i in num_regions])) == 1
        data["num_covariates"]   = size(regional_covariates[1],2)
        data["covariates_starts"] = covariates_start_indices
    end
    return nothing
end

# ============================================================================
# read covariates

function readcovariates(
        ; fname_survey=nothing
        , fname_mobility=nothing
        , shift=0
        , datecol=:date
        # , aggregation=1
        , kwargs...
)

    @_ fname_survey |>
        CSV.read(__, DataFrame) |>
        DataFrames.rename!(__, datecol=>"date") |>
        transform(__, :date =>( x -> x + Day(shift) )=> :date) |>
        leftjoin(__,
            CSV.read(fname_mobility, DataFrame); on=:date) |>
        sort(__, :date)
end

function limit!(covariates, data, conditions)
    @unpack observationsend = data
    filter!(:date => x->x<=Date(observationsend), covariates)
    if conditions isa Pair
        filter!(conditions, covariates)
    elseif conditions isa Vector && !isempty(conditions)
        filter!.(conditions, Ref(covariates))
    end
    return nothing
end

function consistencycheck!(covariates, data)
    @assert issorted(df.date)
    @assert !any( ismissing.(Array(df)) )
    disallowmissing!(df)
end

# ============================================================================
# preprocess covariates

function surveynorm!(xs; standardize=false)
    i = findfirst(x->!ismissing(x), xs)
    xs[i] == 0. && return nothing
    xs ./= abs(xs[i])
    xs .-= xs[i]
    standardize && (xs ./= std(xs))
    return nothing
end

function select_predictors(covariates, predictors, num_observations = nothing, num_tot = nothing; standardize=false, normalize = true)
    predictors = String.(predictors)

    cov_names = filter(x->x!=="date", names(covariates))
    for k in cov_names
        covariates[!,k] = convert.(Float64, covariates[!,k])
        normalize && surveynorm!(covariates[!,k]; standardize)
        # if isnothing(num_observations) || isnothing(num_tot)
        #     surveynorm!(covariates[!,k]; standartize)
        # else
        #     covariates[!,k] = preproc(covariates[:,k], num_observations, num_tot; normalize=surveynorm!)
        # end
    end

    miss = .!( in.(predictors, Ref(cov_names) ) )
    any(miss) && (@error "predictor $(predictors[miss]) not found. Select either nonresidential_mobility, residential_mobility or from survey: $(surveyvars)")
    covdata = Array( covariates[!,predictors] )
    return convert(Array{Float64,2}, covdata)
end

# ============================================================================

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
#               RANDOM-WALK
# ============================================================================

function randomwalk!(data, stepsize, covariates_kwargs)
    @unpack num_observations, lockdown_indices, num_covariates = data
    ns = Vector{Int64}()
    is = Vector{Vector{Int64}}()
    for (i, (no, l)) in enumerate(zip(num_observations, lockdown_indices))
        n = if covariates_kwargs[:semiparametric] && data["num_covariates"] > 0
            data["covariates_starts"][i] - l - 1
        else
            no - l
        end
        i = stepindex(n, stepsize)
        push!(ns, length(unique(i)))
        push!(is, i)
    end
    data["num_rt_steps"] = ns
    data["rt_step_index"] = is
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

# ==============================================================================
# observation model

struct ObservParams{SA,SO,N,DD,DL,DP,P,H,W}
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

function ObservParams(data, o::ObservationInit)
    @unpack dates_turing, dates, num_regions = data
    ds  = [collect(first(dt):Day(1):last(dates)) for dt in dates_turing]

    params = (
        starts           = startindices(data, o),
        stops            = stopindices(data, o),
        num_regions      = num_regions,
        delay_dist       = o.delay_dist,
        delay_length     = o.delay_length,
        delay_dispersion = o.delay_dispersion,
        # observations     = observations(data, o),
        populations      = o.populations,
        holidays         = holiday.(ds),
        weekdays         = [dayofweek.(d) for d in ds]
    )
    return ObservParams( params... )
end

function startindices(data, o)
    @unpack dates_turing, num_regions = data
    observation_startdates = if isempty(o.obs_start)
        data["startdates"] .+ Day(data["epidemicstart"])
    else
        fill(Date(o.obs_start), num_regions)
    end

    is = Vector{Int64}()
    for m in 1:num_regions
        d = observation_startdates[m]
        i = findfirst(==(d), dates_turing[m])
        push!(is,i)
    end
    return is
end

function stopindices(data, o)
    @unpack dates_turing, num_regions = data
    if isempty(o.obs_stop)
        return data["num_observations"]
    else
        is = Vector{Int64}()
        d = Date(o.obs_stop)
        for m in 1:num_regions
            i = findfirst(==(d), dates_turing[m])
            push!(is,i)
        end
        return is
    end
end


# ============================================================================

function holiday(dates)
    specialdays = [Date("2020-12-24"), Date("2020-12-25"), Date("2020-12-31"), Date("2021-01-01")]
    holidays = dates .∈ Ref(specialdays)
    convert(Vector{Int64}, holidays)
end









## ============================================================================
#               depricated
# =============================================================================

# function nconvolve(f1, f2, ts)
#     nbins = length(ts)
#     f(τ, t) = f1(τ) * f2(t - τ)
#     f3 = zeros(nbins)
#     for (i, t) in enumerate(ts)
#         integral, err = quadgk(x->f(x, t), 0, t, rtol=1e-8)
#         f3[i] = integral
#     end
#     return f3
# end
#
# function inf2hosp(nbins=50, ihr = 1)
#     # SSI: ihr ∈ 0.05%-0.5%
#     # derived from Verity et al. See parameters.jl for details
#     # incubatine time ~ gamma(5.1, .86) see Flaxman et al
#     # onset to hospitalization see Zhang et al. Lancet (2020):
#     # "Evolving epidemiology and transmission dynamics of coronavirus disease 2019 outside Hubei province, China: a descriptive and modelling study"
#     ts = 1.5:1:(nbins + .5)
#     i2c(x) = pdf(GammaMeanCv(5.1, .86), x)
#     c2h(x) = pdf(GammaMeanCv(2.6, .92), x) * ihr
#     p =  nconvolve(i2c, c2h, ts)
#     return p / sum(p)
# end
#
# function inf2death(nbins=60, ihr = 1)
#     # ifr = 89 / 100_000 for age < 70 see Erikstrup et al. in Clin. Inf. Dis.
#
#     ## report34: onset to death = 19.8, Cv = .85
#     ts = 1.5:1:(nbins + .5)
#     i2c(x) = pdf(GammaMeanCv(5.1, .86), x)
#     c2d(x) = pdf(GammaMeanCv(19.8, .85), x) * ihr
#     return nconvolve(i2c, c2d, ts)
#
#     # Brauer et al.
#     # r = vcat([0], 1.5:1:(nbins + .5))
#     # p = diff(cdf.(NegativeBinomial2(21.82, 14.26), r))
#     # return p / sum(p)
#
#     ## S. Abbott
#     # i2o(x) = pdf(LogNormal(1.621, 0.418),x)
#     # o2d(x) = pdf(LogNormalMeanStd(13.1, 11.7),x)
#     # return nconvolve(i2o, o2d, ts)
#
#     ## Nouvelette
#     # GammaMeanStd(18.8, 8.48)
# end
#
# function inf2case(nbins = 30)
#     # Brauner et al.
#     r = vcat([0], 1.5:1:(nbins + .5))
#     p = diff(cdf.(NegativeBinomial2(10.92, 5.41), r))
#     return p / sum(p)
#     ## Imperial Report
#     # r = vcat([0], 1.5:1:(nbins + .5))
#     # return diff(cdf.(gamma(5.1, .86), r))
#     ## Brauner et al.
#     # return diff(cdf.(NegativeBinomial2(10.92, 5.41), r))
#     ## S. Abbott
#     # i2o(x) = pdf(LogNormal(1.621, 0.418),x)
#     # o2c(x) = pdf(LogNormalMeanStd(6.5, 17),x)
#     # return nconvolve(i2o, o2c, ts)
# end
#
# """
# seropos()[t] is the probablity of being seropositive t days after infection.
# For details see imperial report 34
# """
# function seropos(nbins=200)
#     # λ = 13.3
#     # μ = 300 #143.7 - λ
#     # κ = 3.67
#     # f(x, t, κ, μ, λ) = 1/λ*exp(-x/λ-((t-x)/μ)^κ)
#     # c2p(t) = quadgk(x->f(x, t, κ, μ, λ), 0, t, rtol=1e-8)[1]
#     # i2c(t) = pdf(GammaMeanCv(5.1, .86), t)
#     # ts = 1.5:1:(nbins + .5)
#     # p = nconvolve(i2c, c2p, ts)
#     # return p / sum(p)
#
#     i2s = zeros(nbins)
#     delay = 14
#     i2s[delay:end] .= 1.
#     return i2s
# end

# function padlast(xs, l)
#     n  = length(xs)
#     i  = findlast(x -> typeof(x) <: Real, xs) #avoid catching a missing value
#     ys = Vector{Union{Float64, Missing}}(undef, l)
#     ys[1:n] = xs
#     ys[n+1:end] .= xs[i]
#     ys
# end
#
# function preproc(xs::AbstractVector, num_observations, num_tot; normalize=nothing)
#     mask = 1:num_observations
#     xs = padlast(xs[mask], num_tot)
#     !isnothing(normalize) && normalize(xs)
#     xs
# end



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
#
#
# function rw_step_idx(num_observations, start_idx, iar_step)
#     # iar_idx = zeros(Int64, num_tot)
#     # iar_idx[1:start_idx-1] .= 1
#     idx = 2
#     for (j,i) in enumerate(start_idx:num_observations)
#          iar_idx[i] = idx
#          j % iar_step == 0 && (idx += 1)
#     end
#     if num_observations < num_tot
#         iar_idx[num_observations+1:num_tot] .= iar_idx[num_observations]
#     end
#     iar_idx
# end
