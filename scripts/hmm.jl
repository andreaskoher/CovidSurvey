using Turing
using Memoization
using ReverseDiff
Turing.setrdcache(true)
setadbackend(:reversediff)


n  = 100
x  = 1:n
λs = sin.(2π/100 * x)*10 .+ 10
p  = .8
m  = 100
is = repeat(x, inner=m)
dist(x...) = ZeroInflatedDistribution(
    Bernoulli( first(x) ),
    Poisson( last(x) )
)
ys = [rand(dist(p, λs[i])) for i in is]

##
function lnk!(ys, y0, xs, lnk)
    ys[1] = y0
    for (i, x) in enumerate(xs)
        ys[i+1] = lnk(x)
    end
    return nothing
end


##
@model function test(y, n, i, lnk, invlnk, ::Type{V}  = Float64) where V
    p  ~ Beta()
    λ0 ~ truncated(Normal(10,1.), 0, 20.)
    σ  ~ truncated(Normal(0.1, .05), 0, Inf)
    latent_λ  ~ CovidSurvey.RandomWalk(n-1, σ, invlnk(λ0))

    λ = Vector{V}(undef, n)
    lnk!(λ, λ0, latent_λ, lnk)

    d = ZeroInflatedDistribution.(
        Ref(Bernoulli(p)),
        Poisson.(λ)
    )

    y ~ arraydist(d[i])
    return λ
end
lnk = KLogistic(20.)
invlnk = KLogit(20.)
m = test(ys, n, is, lnk, invlnk)
c = sample(m, NUTS(10, 0.8), 20; progress=true)
λs = generated_quantities(m, c)
##
plot(c[[:p, :λ0, :σ]])
CovidSurvey.plot_confidence_timeseries(λs[:,1])
## ADVI
using Turing: Variational
include(projectdir("scripts/utils.jl"))

advi = ADVI(10, 1000)
q0 = Variational.meanfield(m)
q = vi(m, advi, q0)
c = Chains(q, m; ignorekeys=[])
mixeddensity(c)
λs = generated_quantities(m, c)
CovidSurvey.plot_confidence_timeseries(λs[:,1])
## Optim
# using Optim
# mle = optimize(m, MLE(), LBFGS(), Optim.Options(iterations = 1000, allow_f_increases = true))
# opt = optimize(m, MAP(), Optim.Options(show_trace=true))
##
using TransformVariables, LogDensityProblems
include(projectdir("scripts/rawsurvey_utils.jl"))

df = let
    n  = 100
    x  = 1:n
    μ = sin.(2π/100 * x)*10 .+ 20
    p  = .8
    m  = 100
    ϕ  = 10.
    is = repeat(x, inner=m)
    d  = Date("2020-12-31") .+ Day.(is)
    ys = [rand(likelihood_dist(p, μ[i], ϕ)) for i in is]
    DataFrame(:date=>d, :y=>ys)
end

problem = ZeroInflatedProblem(df, :y)
t  = problem_transformation(problem)
P  = TransformedLogDensity(t, problem)
∇P = ADgradient(Val(:ReverseDiff), P)
getlossf(P) = f(x) = -LogDensityProblems.logdensity(P, x)
getgradf(∇P) = g(x) = -LogDensityProblems.logdensity_and_gradient(∇P, x)[2]
f = getlossf(P)
g = getgradf(∇P)

θ  = rand(problem.prior)
x  = TransformVariables.inverse(t, θ)
##
res = optimize(f, g, x,
    LBFGS(),
    Optim.Options(show_trace=true, iterations=4000);
    inplace = false
)
##
θ = TransformVariables.transform(t, res.minimizer)
μ = generate_quantities(problem, θ)
let
    dates = first(df.date) : Day(1) : last(df.date)
    p = plot(dates, μ, xticks=:native)
end
##
let
    y = predict(problem, θ)
    df = DataFrame(:posterior=>y, :data=>problem.y, :date=>df.date)
    p = CovidSurvey.plot_confidence_timeseries(df, :data; cols=[:posterior, :data])
    firefox(p)
end

## ============================================================================
# DynamicHMC
using DynamicHMC

results = mcmc_with_warmup(Random.GLOBAL_RNG, ∇P, 1000;
    initialization = (
        q = x,
        κ = GaussianKineticEnergy(length(x), 0.1)
    ),
)

plot( mean( getindex.(θs, :rw) ) )

posterior = TransformVariables.transform.(t, results.chain)

using NamedTupleTools
df = delete.(posterior, Ref(:latent_λ)) |> DataFrame
c = Chains(Array(df), names(df))
plot(c)

generate_quantities(problem, θs::Vector{<:NamedTuple}) =
    generate_quantities.(Ref(problem), θs)
λs = generate_quantities(problem, posterior)
plot_confidence_timeseries(λs)
## ==========================================================================
