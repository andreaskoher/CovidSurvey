struct WeekDayEffectProblem{Y,N,I,W,L,IL,P} <: SurveyProblem
    y::Y
    n::N
    indices::I
    weekday::W
    lnk::L
    invlnk::IL
    prior::P
end

function WeekDayEffectProblem(y,n,indices, weekday, lnk, invlnk)
    prior = PriorWeekDayEffectProblem(n, invlnk)
    WeekDayEffectProblem(y,n,indices, weekday, lnk, invlnk, prior)
end

struct PriorWeekDayEffectProblem{P,L,F,S,W,N,I}
    prior_p::P
    prior_μ0::L
    prior_ϕ::F
    prior_σ::S
    prior_weekday_effect::W
    n::N
    invlnk::I
end

function PriorWeekDayEffectProblem(n, invlnk)
    prior_p  = Beta()
    prior_μ0 = truncated(Normal(5,10),0,Inf)
    prior_ϕ  = GammaMeanCv(20,.5)
    prior_σ  = truncated(Normal(0.1, .05), 0., Inf)
    prior_weekday_effect = Dirichlet(7, 1)
    PriorWeekDayEffectProblem( prior_p, prior_μ0, prior_ϕ, prior_σ, prior_weekday_effect, n, invlnk)
end

function Base.rand(priors::PriorWeekDayEffectProblem)
    @unpack prior_p, prior_μ0, prior_ϕ, prior_σ, n, invlnk, prior_weekday_effect = priors
    p        = rand(prior_p)
    μ0       = rand(prior_μ0)
    ϕ        = rand(prior_ϕ)
    σ        = rand(prior_σ)
    latent_μ = rand( CovidSurvey.RandomWalk(n-1, σ, invlnk(μ0)) )
    weekday_effect = rand(prior_weekday_effect)
    return (; p, μ0, ϕ, σ, latent_μ, weekday_effect)
end

function Distributions.logpdf(prior::PriorWeekDayEffectProblem, θ)
    @unpack p, μ0, ϕ, σ, latent_μ, weekday_effect = θ
    @unpack prior_p, prior_μ0, prior_ϕ, prior_σ, n, invlnk, prior_weekday_effect = prior
    ℓ  = logpdf(prior_p, p)
    ℓ += logpdf(prior_μ0, μ0)
    ℓ += logpdf(prior_ϕ, ϕ)
    ℓ += logpdf(prior_σ, σ)
    ℓ += logpdf(CovidSurvey.RandomWalk(n-1, σ, invlnk(μ0)), latent_μ)
    ℓ += logpdf(prior_weekday_effect, weekday_effect)
    return ℓ
end

function lnk!(μs, problem::WeekDayEffectProblem, θ)
    @unpack lnk, weekday = problem
    @unpack μ0, latent_μ, weekday_effect = θ

    μs[1] = μ0 * weekday_effect[first(weekday)]
    for (i, x) in enumerate(latent_μ)
        μs[i+1] = lnk(x) * weekday_effect[weekday[i+1]] * 7
    end
    return nothing
end

function (problem::WeekDayEffectProblem)(θ)
    @unpack p, ϕ = θ               # extract the parameters
    @unpack y, n, indices, prior = problem
    !(ϕ > 0) && (@error "not ϕ>0"; return -Inf)
    !(0 <= p <= 1) && (@error "not 0 <= p <= 1"; return -Inf)

    ℓ  = logpdf(prior, θ)

    μ = Vector{typeof(ϕ)}(undef, n)
    lnk!(μ, problem, θ)
    !all(0 .<= μ ) && (@error "not all(0 .<= μ )"; return -Inf)

    d = likelihood_dist(p, μ, ϕ)
    ℓ += logpdf(arraydist(d[indices]),y)
    return ℓ
end

function StatsBase.predict(problem::WeekDayEffectProblem, θ)
    @unpack p, ϕ = θ               # extract the parameters
    @unpack y, n, indices = problem

    μ = Vector{typeof(ϕ)}(undef, n)
    lnk!(μ, problem, θ)

    d = likelihood_dist(p, μ, ϕ)
    y = rand(arraydist(d[indices]))
    return y
end

function problem_transformation(p::WeekDayEffectProblem)
    as((
        p = as𝕀,
        μ0 = as(Real, 0, 20),
        ϕ = asℝ₊,
        σ = asℝ₊,
        latent_μ = as(Array, p.n-1),
        weekday_effect = TransformVariables.UnitSimplex(7) #QUESTION
    ))
end

function generate_quantities(problem::WeekDayEffectProblem, θ::NamedTuple)
    @unpack n = problem

    μ = zeros(n)
    lnk!(μ, problem, θ)
    return μ
end

function WeekDayEffectProblem(data, col)
    y = data[:,col]
    indices = date2index( data.date )
    weekday = dayofweek.(data.date)
    n = last(indices)
    lnk = KLogistic(50.)
    invlnk = KLogit(50.)
    return WeekDayEffectProblem(y, n, indices, weekday, lnk, invlnk)
end
