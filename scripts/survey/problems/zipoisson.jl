## =============================================================================
# Hidden markov model with zero inflated likelihood

struct ZIPoissonProblem{Y,N,I,P} <: SurveyProblem
    y::Y
    n::N
    indices::I
    prior::P
end

function ZIPoissonProblem(y,n,indices; c=nothing)
    prior = ZIPoissonPrior(n; c)
    ZIPoissonProblem(y,n,indices, prior)
end

struct ZIPoissonPrior{P,L,N}
    prior_p::P
    prior_μ::L
    n::N
end

function ZIPoissonPrior(n; c=nothing)
    if isnothing(c)
        prior_p  = filldist(Beta(), n)
        prior_μ = filldist(GammaMeanCv(15,.8), n)
        return ZIPoissonPrior( prior_p, prior_μ, n)
    elseif c == :family
        prior_p  = filldist(Beta(), n)
        # prior_p  = truncated(Normal(0.51,0.05), 0, 1)
        prior_μ = filldist(truncated(Normal(3.5,1), 0, Inf), n)
        return ZIPoissonPrior( prior_p, prior_μ, n)
    elseif c == :colleagues
        prior_p  = filldist(Beta(), n)
        # prior_p  = truncated(Normal(0.33,0.05), 0, 1)
        prior_μ = filldist(truncated(Normal(5,1.5), 0, Inf), n)
        return ZIPoissonPrior( prior_p, prior_μ, n)
    elseif c == :friends
        prior_p  = filldist(Beta(), n)
        # prior_p  = truncated(Normal(0.46,0.05), 0, 1)
        prior_μ = filldist(truncated(Normal(4.5,2), 0, Inf), n)
        return ZIPoissonPrior( prior_p, prior_μ, n)
    elseif c == :strangers
        prior_p  = filldist(Beta(), n)
        # prior_p  = truncated(Normal(0.32,0.05), 0, 1)
        prior_μ = filldist(truncated(Normal(20,8), 0, Inf), n)
        return ZIPoissonPrior( prior_p, prior_μ, n)
    else
        error( "wrong choice" )
    end

end

function Base.rand(priors::ZIPoissonPrior)
    @unpack prior_p, prior_μ, n = priors
    p  = rand(prior_p)
    μ  = rand(prior_μ)
    return (; p, μ)
end

function Distributions.logpdf(prior::ZIPoissonPrior, θ)
    @unpack p, μ = θ
    @unpack prior_p, prior_μ = prior
    ℓ  = logpdf(prior_p, p)
    ℓ += logpdf(prior_μ, μ)
    return ℓ
end

# likelihood_dist(p, μ, ϕ) =
#     ZeroInflatedDistribution.(
#         Ref(Bernoulli(p)),
#         NegativeBinomial2.(μ, Ref(ϕ)
#     )
# )
likelihood_dist(p, μ) =
    ZeroInflatedDistribution.(
        Bernoulli.(p),
        Poisson.(μ)
)

function (problem::ZIPoissonProblem)(θ)
    @unpack p, μ  = θ               # extract the parameters
    @unpack y, indices, prior = problem
    # !(ϕ > 0) && (@error "not ϕ>0"; return -Inf)
    # !(0 <= p <= 1) && (@error "not 0 <= p <= 1"; return -Inf)

    ℓ  = logpdf(prior, θ)
    # !all(0 .<= μ ) && (@error "not all(0 .<= μ )"; return -Inf)

    d = likelihood_dist(p, μ)
    ℓ += logpdf(arraydist(d[indices]),y)
    return ℓ
end

function problem_transformation(problem::ZIPoissonProblem)
    as((
        p = as(Vector, as𝕀, problem.n),
        μ = as(Vector, asℝ₊, problem.n)
    ))
end

function ZIPoissonProblem(data, col)
    y = data[:,col]
    indices = date2index( data.date )
    n = last(indices)
    return ZIPoissonProblem(y, n, indices; c=col)
end

function StatsBase.predict(problem::ZIPoissonProblem, θ)
    @unpack p, μ = θ               # extract the parameters
    @unpack y, indices = problem

    d = likelihood_dist(p, μ)
    y = rand(arraydist(d[indices]))
    return y
end

generate_quantities(problem::ZIPoissonProblem, θ) = θ.μ
