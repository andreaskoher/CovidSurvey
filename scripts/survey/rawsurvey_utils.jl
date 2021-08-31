Plots.histogram(d::Distributions.Distribution, args...; kwargs...) =
    histogram(rand(d, 1000), args...; label=nothing, kwargs...)

datacols(df) = Symbol.(names(df, Not(:date)))

# ===========================================================================
# read data
const ctype = [:family, :colleagues, :friends, :strangers]
parsedate(x) = Date("$x", dateform)
function readcontacts(fname; select_region=0)
    raw = CSV.File(fname)|>DataFrame
    raw = select_region > 0 ? raw[raw.region .== select_region, :] : raw
    data = DataFrames.select(raw, :Q4a_1 => :family, :Q4a_2 => :colleagues, :Q4a_3 => :friends, :Q4a_4 => :strangers )
    data[!, :date] = parsedate.(raw.Timings_yyyymmdd)
    sort!(data, :date)

    d = data.date
    @assert length(unique(d)) == length(first(d):Day(1):last(d))
    return data
end
# =============================================================================
# remove outliers
function maskoutliers(data, p=99.9)
    outliers = zeros(size(data,1))
    for c in datacols(data)
        threshold = p isa Number ? percentile(data[:,c], p) : p[c]
        @show "threshold for $c: $threshold"
        outliers .+= (threshold .< data[!,c])
        outliers .+= (data[!,c] .< 0)
    end
    return outliers .== 0
end

function rmfirstday(data)
    dates = data.date
    @assert issorted(dates)
    firstday = first(dates)
    i = findfirst(>(firstday), dates)
    data[i:end, :]
end

function filteroutliers(data, p=99.9)
    mask = maskoutliers(data, p)
    @show "remove $(sum(mask .== 0)/size(data,1)*100) percent"
    data = data[mask,:]
    data = rmfirstday(data)#NOTE first day seems unusual
    return data
end

function contact_histograms(data; day=nothing, kwargs...)

    df = if isnothing(day)
        data
    else
        byday = groupby(data, :date)
        byday[day]
    end

    ps = Vector{Plots.Plot}()
    for c in datacols(data)
        p = histogram( df[!, c]; label=nothing, xticks=:native, yticks=:native, title="$c", kwargs...)
        push!(ps, p)
    end
    plot(ps..., layout=(2,2), size=(800, 500))
end

# =============================================================================
# plot timeseries stats

lowerquant(x) = percentile(x, 25)
upperquant(x) = percentile(x, 75)
lowestquant(x) = percentile(x, 2.5)
upperstquant(x) = percentile(x, 97.5)

function stats(df::DataFrame, c; datecol=:date)
    combine(
        groupby(df, datecol),
        c => mean => :m,
        c => median => :mq,
        c => lowestquant => :llq,
        c => upperstquant => :uuq,
        c => lowerquant => :lq,
        c => upperquant => :uq,
        datecol => :date,
        nrow;
    )
end
function CovidSurvey.plot_confidence_timeseries(df::DataFrame, col; datecol=:date, kwargs...)
    s = stats(df, col)
    @unpack llq, lq, mq, uq, uuq, m, date = s
    plot(title="$col", ylim=(0,15), legend=:outertopright)
    plot!(date, mq, ribbon=(mq - llq, uuq - mq), c=:peachpuff, lw=2, lab="95% CI", kwargs...)
    plot!(date, mq, ribbon=(mq - lq, uq - mq), c=:peru, lw=2, lab="medean & 50% CI", kwargs...)
    plot!(date, m, lw=1, c=:black, ls=:solid ,lab="mean", kwargs...)
end

function firefox(p; fname="tmp.html", fdir=normpath(homedir(), ".tmp"))
    # mkpath(dir)
    filepath = normpath(fdir, fname)
    savefig(p, filepath)
    run(`firefox $(filepath)`, wait=false)
    return p
end

function CovidSurvey.plot_confidence_timeseries(df::DataFrame; datecol=:date, cols=datacols(df), kwargs...)
    ps = Vector{Plots.Plot}()
    n  = length(cols)
    for c in cols
        p = CovidSurvey.plot_confidence_timeseries(df, c; datecol, kwargs...)
        push!(ps, p)
    end
    p = plot(ps..., layout=(n,1), size=(1000, n*200))
    firefox(p)
end

# =============================================================================
# zero data
function countzeros(data)
    nzeros(x) = count( ==(0), x )
    grouped = groupby(data, :date; sort=true)
    combine(
        grouped,
        Pair.(datacols(data), Ref(nzeros))...,
        renamecols = false
    )
end

function plot_timeseries(df)
    t = df.date
    ps = Vector{Plots.Plot}()
    colnames = DataFrames.names(df, Not(:date))
    n = length(colnames)
    for colname in colnames
        y = df[!,colname]
        p = plot(t, y, label=nothing, title="$colname")
        push!(ps,p)
    end
    p = plot(ps..., layout=(n,1), size=(1000, n*200))
    firefox(p)
end

# =============================================================================

function Turing.Chains(q::MultivariateTransformed, model, samples=1000; ignorekeys=[])
    _, sym2range = bijector(model, Val(true))
    z = rand(q, samples)
    !isempty(ignorekeys) && (sym2range = delete(sym2range, ignorekeys...))
    d = Dict{Symbol, AbstractVecOrMat}()
    for (sym,rnge) in convert(Vector{Pair}, sym2range)
        i = union(rnge...)
        if length(i) == 1
            i = only(i)
            d[sym] = z[i,:]
        else
            d[sym] = transpose(z[i,:])
        end
    end
    Chains(d)
end
# ===========================================================================
# zero inflated distributions

struct ZeroInflatedDistribution{E,P} <: Distribution{Univariate, Discrete}
    encdist::E
    posdist::P
end

Base.rand(
    rng::AbstractRNG,
    zil::ZeroInflatedDistribution
) = rand(rng, zil.encdist) * rand(rng, zil.posdist)

Base.rand(zil::ZeroInflatedDistribution) = rand(Random.GLOBAL_RNG, zil)

function Distributions.logpdf(
    zil::ZeroInflatedDistribution{E,P},
    x::Integer
) where {E,P}
    @unpack encdist, posdist = zil
    p = succprob(encdist)
    logprob = log(p) + logpdf(posdist, x)

    if x > 0
        return logprob
    elseif x == 0
        return logaddexp( log1p(-p), logprob )
    else
        return oftype(logprob, -Inf)
    end
end

# test inference with zero inflated distributions
# μ = 2.
# ϕ = 10.
# p = 0.2
# posdist = NegativeBinomial2(μ, ϕ)
# encdist = Bernoulli(p)
# d = ZeroInflatedDistribution(encdist, posdist)
# y = rand(d,1000)
# histogram(rand(d,1000), label=nothing)
#
# @code_warntype logpdf(d, 1)
#
# @model function test(y, n)
#     p  ~ Beta()
#     ϕ  ~ Exponential(10.)
#     μ  ~ Exponential(3.)
#     pd = NegativeBinomial2(μ, ϕ)
#     ed = Bernoulli(p)
#     d  = ZeroInflatedDistribution(ed, pd)
#     y  ~ filldist(d, n)
# end
#
# c = sample(test(y, length(y)), NUTS(1000, 0.8), 2000; progress=true)
# plot(c[1001:end,:,:])

# =============================================================================
struct FixedRandomWalk{N, S} <: ContinuousMultivariateDistribution
    n::N
    s::S
end

Distributions.rand(rng::AbstractRNG, d::FixedRandomWalk{N,S}) where {N,S} = begin
    x = Vector{S}(undef, d.n)
    x[1] = rand(Normal(0, d.s))
    for i in 2:d.n
        x[i] = x[i-1] + rand(Normal(0, d.s))
    end
    return x
end

function Distributions.logpdf(
    d::FixedRandomWalk{N,S}, x::AbstractVector{X}
) where {N,S,X}
    ℓ  = logpdf( Normal(0, d.s) )
    ℓ += logpdf( MvNormal( d.n-1, d.s ), diff(x) )
    return ℓ
end
Bijectors.bijector(d::FixedRandomWalk) = Bijectors.Identity{1}()

Base.length(d::FixedRandomWalk) = d.n

# =============================================================================
function date2index(ds)
    Δds = unique(diff(ds))
    @assert length(Δds) == 2 #0 and time step
    Δd = last(Δds)
    i  = 1
    is = [i]
    d  = first(ds)
    for j in 2:length(ds)
        d_new = ds[j]
        if d_new == d
            push!(is, i)
        else d_new == d + Δd
            i += 1
            d = d_new
            push!(is, i)
        end
    end
    return is
end

# =============================================================================
# Hidden markov model with zero inflated likelihood
abstract type SurveyProblem end
include(projectdir("scripts","survey","problems","zipoisson.jl"))
include(projectdir("scripts","survey","problems","zinegbinomial.jl"))
# include(projectdir("scripts","survey","problems","randomwalk.jl"))
# include(projectdir("scripts","survey","problems","weekdayeffect.jl"))


# =============================================================================
function Optim.optimize(problem::SurveyProblem, args...; kwargs...)

    t  = problem_transformation(problem)
    P  = TransformedLogDensity(t, problem)
    ∇P = ADgradient(Val(:ReverseDiff), P)
    getlossf(P) = f(x) = -LogDensityProblems.logdensity(P, x)
    getgradf(∇P) = g(x) = -LogDensityProblems.logdensity_and_gradient(∇P, x)[2]
    f = getlossf(P)
    g = getgradf(∇P)

    θ  = rand(problem.prior)
    x  = TransformVariables.inverse(t, θ)

    optim = optimize(f, g, x, args...;
        inplace = false, kwargs...
    )

    return optim
end

struct SurveyInferenceProblem{P,N,D}
    problemtype::P
    names::N
    data::D
end

struct SurveyInferenceResults{P,N,D,T,R,M,O,A}
    problemtype::P
    names::N
    data::D
    θ::T
    problem::R
    μ::M
    optim::O
    dates::A
end

function DrWatson.save(file, results::SurveyInferenceResults)
    @unpack θ, μ, data, names, dates, problemtype, problem = results
    r = SurveyInferenceResults(problemtype, names, data, θ, problem, μ, nothing, dates)
    bson( file, Dict(:results => r) )
end

function Optim.optimize(survey::SurveyInferenceProblem, args...; kwargs...)
    @unpack date = survey.data
    dates = sort(unique(date))#first(date) : Day(1) : last(date)

    θs       = []
    problems = []
    μs       = []
    optims   = []
    dates    = dates

    for n in survey.names
        # runoptim!(results, contacts, c)
        @info "===== $n ====="
        problem     = survey.problemtype(survey.data, n)
        t           = problem_transformation(problem)
        @time optim = optimize(problem, args...; kwargs...)

        θ = TransformVariables.transform(t, optim.minimizer)
        μ = generate_quantities(problem, θ)

        push!(θs, θ)
        push!(problems, problem)
        push!(μs, μ)
        push!(optims, optim)
    end
    SurveyInferenceResults(
        survey.problemtype, survey.names, survey.data,
        vcat(θs...), vcat(problems...), hcat(μs...), vcat(optims...), dates
    )
end

function CovidSurvey.plot_confidence_timeseries(results::SurveyInferenceResults)
    @unpack data, names, problem, θ, μ = results
    for (i, n) in enumerate(names)
        y = predict(problem[i], θ[i])
        label_data = "data ($n)"
        label_fake = "fake data from posterior ($n)"

        df = DataFrame(
            label_fake=>y,
            label_data=>problem[i].y,
            "date"=>data.date
        )

        p = CovidSurvey.plot_confidence_timeseries(df; cols=[label_fake, label_data])

        t = sort(unique(df.date))
        plot!(t, μ[:,i], lw=2, c=:black, label="effective contacts")
        p = firefox(p; fname="inference_result_$n.html")
        display(p)
    end
end

function plot_generated_quantities(
    results::SurveyInferenceResults;
    fdir = normpath(homedir(), ".tmp"),
    fname="generated_quantities.html",
    compare_old=false,
    kwargs...
)
    @unpack names, θ, dates = results
    ps = Vector{Plots.Plot}()
    data_old = National.readsurvey("/home/and/data/covidsurvey/contacts_0425_2m.csv"; kwargs...)
    for (i, n) in enumerate(names)
        μ = θ[i].μ
        p = plot(dates, μ, xticks=:native, title="$n", lab="inferred contact rate")
        if compare_old
            plot!(p, data_old.date, data_old[!,n], xticks=:native, lab="old data set")
        end
        push!(ps,p)
    end
    n = length(names)
    p = plot(ps..., layout=(n,1), size=(1000, n*200))
    firefox(p; fname, fdir)
end

function plot_survey_histograms(
    results::SurveyInferenceResults;
    fdir = normpath(homedir(), ".tmp"),
    fname="generated_quantities.html",
    day = nothing
)
    @unpack names, θ, problem, data, dates = results
    ps = Vector{Plots.Plot}()
    for (i, n) in enumerate(names)
        ŷ = predict(problem[i], θ[i])
        y = problem[i].y

        if !isnothing(day)
            t = data.date
            mask = t .== dates[day]
            ŷ = ŷ[mask]
            y = y[mask]
        end

        p = histogram( ŷ, bins=0:20, label="posterior", fc=:gray, lw=0, yticks=:native, leg=:topright, normalize=true)
        histogram!(y, bins=0:20, label="data", fα=0., lc=:blue, lw=2, yticks=:native, normalize=true)
        push!(ps, p)
    end
    p = plot(ps..., layout=(length(names),1), size=(800, 500))
    # firefox(p; fname, fdir)
    p
end

function DataFrame(results::SurveyInferenceResults)
    @unpack names, θ, dates = results
    μs = NamedTupleTools.select.(θ, :μ)
    cols = vcat(:dates, names)
    dats = vcat([dates], μs)
    DataFrame(dats, cols)
end
