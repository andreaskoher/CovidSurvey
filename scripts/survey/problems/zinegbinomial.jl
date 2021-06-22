## =============================================================================
# Hidden markov model with zero inflated likelihood

struct ZINegBinomialProblem{Y,N,I,P} <: SurveyProblem
    y::Y
    n::N
    indices::I
    prior::P
end

function ZINegBinomialProblem(y,n,indices; c=nothing)
    prior = ZINegBinomialPrior1(n; c)
    ZINegBinomialProblem(y,n,indices, prior)
end

struct ZINegBinomialPrior1{P,L,F,N}
    prior_p::P
    prior_μ::L
    prior_ϕ::F
    n::N
end

function ZINegBinomialPrior1(n; c=nothing)
    # if isnothing(c)
    prior_p = Beta()
    prior_μ = filldist(truncated(GammaMeanCv(15,.8), 0, 1000), n)
    prior_ϕ = filldist(truncated(Normal(0, 20), 0, Inf), n)
    return ZINegBinomialPrior1( prior_p, prior_μ, prior_ϕ, n)
    # elseif c == :family
    #     prior_p  = truncated(Normal(0.51,0.05), 0, 1)
    #     prior_μ = truncated(Normal(3.5,1), 0, Inf)
    #     prior_ϕ = filldist(truncated(Normal(0, 20), 0, Inf), n)
    #     return ZINegBinomialPrior1( prior_p, prior_μ, prior_ϕ, n)
    # elseif c == :colleagues
    #     prior_p  = truncated(Normal(0.33,0.05), 0, 1)
    #     prior_μ = truncated(Normal(5,1.5), 0, Inf)
    #     prior_ϕ = filldist(truncated(Normal(0, 20), 0, Inf), n)
    #     return ZINegBinomialPrior1( prior_p, prior_μ, prior_ϕ, n)
    # elseif c == :friends
    #     prior_p  = truncated(Normal(0.46,0.05), 0, 1)
    #     prior_μ = truncated(Normal(4.5,2), 0, Inf)
    #     prior_ϕ = filldist(truncated(Normal(0, 20), 0, Inf), n)
    #     return ZINegBinomialPrior1( prior_p, prior_μ, prior_ϕ, n)
    # elseif c == :strangers
    #     prior_p  = truncated(Normal(0.32,0.05), 0, 1)
    #     prior_μ = truncated(Normal(20,8), 0, Inf)
    #     prior_ϕ = filldist(truncated(Normal(0, 20), 0, Inf), n)
    #     return ZINegBinomialPrior1( prior_p, prior_μ, prior_ϕ, n)
    # else
    #     error( "wrong choice" )
    # end
end

function Base.rand(priors::ZINegBinomialPrior1)
    @unpack prior_p, prior_μ, prior_ϕ, n = priors
    p  = rand(prior_p)
    μ  = rand(prior_μ)
    ϕ  = rand(prior_ϕ)
    return (; p, μ, ϕ)
end

function Distributions.logpdf(prior::ZINegBinomialPrior1, θ)
    @unpack p, μ, ϕ = θ
    @unpack prior_p, prior_μ, prior_ϕ = prior
    ℓ  = logpdf(prior_p, p)
    ℓ += logpdf(prior_μ, μ)
    ℓ += logpdf(prior_ϕ, ϕ)
    return ℓ
end

# likelihood_dist(p, μ, ϕ) =
#     ZeroInflatedDistribution.(
#         Ref(Bernoulli(p)),
#         NegativeBinomial2.(μ, Ref(ϕ)
#     )
# )
likelihood_dist(p, μ, ϕ) =
    ZeroInflatedDistribution.(
        Ref( Bernoulli(p) ),
        NegativeBinomial2.(μ, ϕ .+ 0.1)
)

function (problem::ZINegBinomialProblem)(θ)
    @unpack p, μ, ϕ  = θ               # extract the parameters
    @unpack y, indices, prior = problem
    # !(ϕ > 0) && (@error "not ϕ>0"; return -Inf)
    # !(0 <= p <= 1) && (@error "not 0 <= p <= 1"; return -Inf)

    ℓ  = logpdf(prior, θ)
    # !all(0 .<= μ ) && (@error "not all(0 .<= μ )"; return -Inf)

    d = likelihood_dist(p, μ, ϕ)
    ℓ += logpdf(arraydist(d[indices]),y)
    return ℓ
end

function problem_transformation(p::ZINegBinomialProblem)
    as((p = as𝕀, μ = as(Vector, as(Real, 0, 1000), p.n), ϕ = as(Vector, asℝ₊, p.n)))
end

function ZINegBinomialProblem(data, col)
    y = data[:,col]
    indices = date2index( data.date )
    n = last(indices)
    return ZINegBinomialProblem(y, n, indices; c=col)
end

function StatsBase.predict(problem::ZINegBinomialProblem, θ)
    @unpack p, μ, ϕ = θ               # extract the parameters
    @unpack y, indices = problem

    d = likelihood_dist(p, μ, ϕ)
    y = rand(arraydist(d[indices]))
    return y
end

generate_quantities(problem::ZINegBinomialProblem, θ) = θ.μ
