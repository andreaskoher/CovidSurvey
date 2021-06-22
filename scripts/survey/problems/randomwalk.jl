## =============================================================================
# Hidden markov model with zero inflated likelihood

struct RandomWalkProblem{Y,N,I,L,IL,P} <: SurveyProblem
    y::Y
    n::N
    indices::I
    lnk::L
    invlnk::IL
    prior::P
end

function RandomWalkProblem(y,n,indices, lnk, invlnk)
    prior = PriorRandomWalkProblem(n, invlnk)
    RandomWalkProblem(y,n,indices, lnk, invlnk, prior)
end

struct PriorRandomWalkProblem{P,L,F,S,N,I}
    prior_p::P
    prior_μ0::L
    prior_ϕ::F
    prior_σ::S
    n::N
    invlnk::I
end

function PriorRandomWalkProblem(n, invlnk)
    prior_p  = Beta()
    prior_μ0 = truncated(Normal(10,10),0,100)
    prior_ϕ  = truncated(GammaMeanCv(50,.5),1.,Inf)
    prior_σ  = truncated(Normal(0.2, .1), 0., Inf)
    PriorRandomWalkProblem( prior_p, prior_μ0, prior_ϕ, prior_σ, n, invlnk)
end

function Base.rand(priors::PriorRandomWalkProblem)
    @unpack prior_p, prior_μ0, prior_ϕ, prior_σ, n, invlnk = priors
    p        = rand(prior_p)
    μ0       = rand(prior_μ0)
    ϕ        = rand(prior_ϕ)
    σ        = rand(prior_σ)
    latent_μ = rand( CovidSurvey.RandomWalk(n-1, σ, invlnk(μ0)) )
    return (; p, μ0, ϕ, σ, latent_μ)
end

function Distributions.logpdf(prior::PriorRandomWalkProblem, θ)
    @unpack p, μ0, ϕ, σ, latent_μ = θ
    @unpack prior_p, prior_μ0, prior_ϕ, prior_σ, n, invlnk = prior
    ℓ  = logpdf(prior_p, p)
    ℓ += logpdf(prior_μ0, μ0)
    ℓ += logpdf(prior_ϕ, ϕ)
    ℓ += logpdf(prior_σ, σ)
    ℓ += logpdf(CovidSurvey.RandomWalk(n-1, σ, invlnk(μ0)), latent_μ)
    return ℓ
end

likelihood_dist(p, μ, ϕ) =
    ZeroInflatedDistribution.(
        Ref(Bernoulli(p)),
        NegativeBinomial2.(μ, Ref(ϕ)
    )
)

function lnk!(μ, problem::RandomWalkProblem, θ)
    @unpack μ0, latent_μ = θ
    @unpack lnk = problem

    μ[1] = μ0
    for (i, x) in enumerate(latent_μ)
        μ[i+1] = lnk(x)
    end
    return nothing
end

function (problem::RandomWalkProblem)(θ)
    @unpack p, μ0, ϕ, σ, latent_μ  = θ               # extract the parameters
    @unpack y, n, indices, lnk, invlnk, prior = problem
    !(ϕ > 0) && (@error "not ϕ>0"; return -Inf)
    !(0 <= p <= 1) && (@error "not 0 <= p <= 1"; return -Inf)

    ℓ  = logpdf(prior, θ)

    μ = Vector{eltype(latent_μ)}(undef, n)
    lnk!(μ, problem, θ)
    !all(0 .<= μ ) && (@error "not all(0 .<= μ )"; return -Inf)

    d = likelihood_dist(p, μ, ϕ)
    ℓ += logpdf(arraydist(d[indices]),y)
    return ℓ
end

function problem_transformation(p::RandomWalkProblem)
    as((p = as𝕀, μ0 = as(Real, 0., 100.), ϕ = asℝ₊, σ = asℝ₊, latent_μ = as(Array, p.n-1)))
end

function RandomWalkProblem(data, col)
    y = data[:,col]
    indices = date2index( data.date )
    n = last(indices)
    lnk = KLogistic(100.)
    invlnk = KLogit(100.)
    return RandomWalkProblem(y, n, indices, lnk, invlnk)
end

function generate_quantities(problem::RandomWalkProblem, θ::NamedTuple)
    @unpack n = problem

    μ = zeros(n)
    lnk!(μ, problem, θ)
    return μ
end

function StatsBase.predict(problem::RandomWalkProblem, θ)
    @unpack p, ϕ = θ               # extract the parameters
    @unpack y, n, indices = problem

    μ = Vector{typeof(ϕ)}(undef, n)
    lnk!(μ, problem, θ)

    d = likelihood_dist(p, μ, ϕ)
    y = rand(arraydist(d[indices]))
    return y
end
