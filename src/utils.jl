# ==============================================================================
using DistributionsAD
using Random: AbstractRNG
import Bijectors

struct RandomWalk{Tn, Ts, Tx} <: ContinuousMultivariateDistribution
    n::Tn
    s::Ts
    x0::Tx
end

Distributions.rand(rng::AbstractRNG, d::RandomWalk{Tn, Ts, Tx}) where {Tn, Ts, Tx} = begin
    x = Vector{Tx}(undef, d.n)
    Distributions._rand!(rng, x, d)
    return x
end

Distributions._rand!(rng::AbstractRNG, x::AbstractVector, d::RandomWalk) = begin
	x[1] = rand(Normal(d.x0, d.s))
	for i in 2:d.n
		x[i] = rand(Normal(x[i-1], d.s))
	end
	return nothing
end

Distributions.logpdf(
    d::RandomWalk{Tn, Ts, Tx}, x::AbstractVector{T}
) where {Tn, Ts, Tx, T} = begin
	ℓ  = logpdf( Normal(d.x0, d.s), first(x) )
	ℓ += logpdf( MvNormal( d.n-1, d.s ), diff(x) )
	return ℓ
end

Bijectors.bijector(d::RandomWalk) = Bijectors.Identity{1}()

Bijectors.bijector(
    ::DistributionsAD.VectorOfMultivariate{Continuous, RandomWalk{Tn, Ts, Tx}, Vector{RandomWalk{Tn, Ts, Tx}}}
) where {Tn, Ts, Tx} = Bijectors.Identity{2}()


Base.length(d::RandomWalk) = d.n

# ==============================================================================
"""
Mean standard deviation parametrization:

mean: m = exp(μ+σ^2/2)
std:  s^2 = [exp(σ^2)-1]exp(2μ+σ^2)
"""
function LogNormalMeanStd(m, s)
    σ = sqrt( log( s^2 / m^2 + 1 ) )
    μ = log(m) - σ^2/2
    LogNormal(μ, σ)
end

"""
    NegativeBinomial2(μ, ϕ)

Mean-variance parameterization of `NegativeBinomial`.

## Derivation
`NegativeBinomial` from `Distributions.jl` is parameterized following [1]. With the parameterization in [2], we can solve
for `r` (`n` in [1]) and `p` by matching the mean and the variance given in `μ` and `ϕ`.

We have the following two equations

(1) μ = r (1 - p) / p
(2) μ + μ^2 / ϕ = r (1 - p) / p^2

Substituting (1) into the RHS of (2):
  μ + (μ^2 / ϕ) = μ / p
⟹ 1 + (μ / ϕ) = 1 / p
⟹ p = 1 / (1 + μ / ϕ)
⟹ p = (1 / (1 + μ / ϕ)

Then in (1) we have
  μ = r (1 - (1 / 1 + μ / ϕ)) * (1 + μ / ϕ)
⟹ μ = r ((1 + μ / ϕ) - 1)
⟹ r = ϕ

Hence, the resulting map is `(μ, ϕ) ↦ NegativeBinomial(ϕ, 1 / (1 + μ / ϕ))`.

## References
[1] https://reference.wolfram.com/language/ref/NegativeBinomialDistribution.html
[2] https://mc-stan.org/docs/2_20/functions-reference/nbalt.html
"""
function NegativeBinomial2(μ, ϕ)
    p = 1 / (1 + μ / ϕ)
    r = ϕ
	# !(0 < p <= 1) && (@show "$μ $ϕ")
    return NegativeBinomial(r, p)
end

"""
Negative Binomial parametrized the mean μ and variance σ2.

In terms of Turings parametrisation we have the following relationship:
μ = r (1 - p) / p
σ2 =  = r (1 - p) / p²
"""
function NegativeBinomial3(μ, σ2)
    p = μ/σ2
    r = μ^2 / (σ2 - μ)
    NegativeBinomial(r, p)
end

"""
InverseGamma parametrized by the mean μ and coefficient of variation cv = σ/μ.

In terms of Turings parametrisation we have the following relationship:
μ = θ / (α - 1)
cv = (α - 2)^(-1/2)
"""
function InverseGamma2(μ, cv)
    α = cv^(-2)+2
    β = μ*(α-1)
    InverseGamma(α, β)
end

"""
    GammaMeanCv(mean, cv)

Gamma parametrized by the mean μ and coefficient of variation cv = σ/μ.

In terms of Turings parametrisation we have the following relationship:
μ = αθ
cv = 1 / √α

## References
- https://www.rdocumentation.org/packages/EnvStats/versions/2.3.1/topics/GammaAlt
"""
function GammaMeanCv(mean, cv)
    α = cv^(-2)
    θ = mean / α
    return Gamma(α, θ)
end

"""
    GammaMeanStd(mean, std)

Gamma parametrized by the mean μ and standard deviation σ.

In terms of Turings parametrisation we have the following relationship:
μ = αθ
σ² = αθ²
"""
function GammaMeanStd(μ, σ)
    α = (μ/σ)^2
    θ = σ^2/μ
    return Gamma(α, θ)
end

"""
This is a generalization of the logit-link, which is referred to as the scaled-logit.
That is to say we expect the quantity of interest x (the reproduction number R,
for instance) to have some carry capacity.

```math
\\sigma_K = KLogistic(K)
```

```math
\\sigma_K(x) = \\frac{K}{e^{-x} + 1}
```
"""
struct KLogistic{T}
	k::T
end
(f::KLogistic)(x::T)  where {T} = inv(exp(-x) + one(x)) * T(f.k)

struct KLogit{T}
	k::T
end
(f::KLogit)(x::T)  where {T} = log(x / (T(f.k) - x))

"Converts a vector of tuples to a tuple of vectors."
function vectup2tupvec(ts::AbstractVector{<:Tuple})
    k = length(first(ts))

    return tuple([[t[i] for t in ts] for i = 1:k]...)
end

"Converts a vector of named tuples to a tuple of vectors."
function vectup2tupvec(ts::AbstractVector{<:NamedTuple})
    ks = keys(first(ts))

    return (; (k => [t[k] for t in ts] for k ∈ ks)...)
end


"""
    rename!(d::Dict, names::Pair...)

Renames the keys given by `names` of `d`.
"""
function rename!(d::Dict, names::Pair...)
    # check that keys are not yet present before updating `d`
    for k_new in values.(names)
        @assert k_new ∉ keys(d) "$(k_new) already in dictionary"
    end

    for (k_old, k_new) in names
        d[k_new] = pop!(d, k_old)
    end
    return d
end

function arrarrarr2arr(a::AbstractVector{<:AbstractVector{<:AbstractVector{T}}}) where {T<:Real}
    n1, n2, n3 = length(a), length(first(a)), length(first(first(a)))

    A = zeros(T, (n1, n2, n3))
    for i = 1:n1
        for j = 1:n2
            for k = 1:n3
                A[i, j, k] = a[i][j][k]
            end
        end
    end

    return A
end

"""
    Chains(d::Dict)

Converts a `Dict` into a `Chains`, assuming the values of `d` is either
- `AbstractVector`: samples for a `Real` variable
- `AbstractMatrix`: samples for a `Vector` variable
- `AbstractArray{<:Any, 3}`: samples for a `Matrix` variable
"""
function MCMCChains.Chains(d::Dict)
    vals = []
    names = []

    for (k, v) in pairs(d)
        if v isa AbstractVector
            push!(vals, v)
            push!(names, k)
        elseif v isa AbstractMatrix
            push!(vals, v)
            append!(names, ["$(k)[$(i)]" for i = 1:size(v, 2)]) # assuming second dimension is dimensionality
        elseif v isa AbstractArray{<:Any, 3}
            indices = CartesianIndices(v[1, :, :])

            # The ordering is such that when you call `reshape` on the vector, the symbols will correspond with
            # the actual indices in the matrix, e.g. `X[i, j]` will be the same as `reshape(X, size)[i, j]`.
            for i = 1:size(indices, 2)
                for j = 1:size(indices, 1)
                    push!(vals, v[:, j, i])
                    push!(names, "$(k)[$j, $i]")
                end
            end
        else
            throw(ArgumentError("I'm so, so sorry but I can't handle $(typeof(v)) :("))
        end
    end

    return Chains(reduce(hcat, vals[2:end]; init = vals[1]), reduce(vcat, names[2:end]; init = names[1]))
end

"""
parse string and return a list of predictors.
Example:

```julia
parse_predictors( "C,T,H" )
```

`>> ["contact", "threat", "hygiene"]`
"""
function parse_predictors(str = nothing)
    isnothing(str) && return
    ps = String[]
    for sraw in split(str, ",")
        s = sraw |> strip |> Symbol
        if s in keys(short2long)
			push!(ps, short2long[s])
		else
			push!(ps, String(s))
		end
    end
    ps
end

short2long = Dict(
    :C => "contact",
    :T => "threat",
    :H => "hygiene",
    :D => "distance",
    :M => "motivation",
    :SE => "self_efficay",
    :RC => "response_cost",
    :RE => "response_efficacy",
    :RM => "residential_mobility",
    :NM => "nonresidential_mobility",
    :L  => "lockdown",
    :CR => "friends",
    :CC => "colleagues",
    :CF => "family",
    :CS => "strangers",
)

function parseval(sval)
    isempty(sval) && return nothing
    val = tryparse(Int, sval)
    val = isnothing(val) ? tryparse(Bool, sval) : val
    val = isnothing(val) ? sval : val
    return val
end

"""
parse file name with parameters assuming the format
*<parameter>=<value>*, where *<value>* can be either an Int, Bool, or String.
Example:

 ```julia
ps  = (
    a=nothing,
    b="B",
    c=true,
    d=1
)

fname = DrWatson.savename(ps)

ps = parseparams(fname)
 ```
"""
function parseparams(s)
    sp = split(s, "_")
    popfirst!(sp)
    sp[end] = split(sp[end], ".")[1]
    ps = Dict()
    for keyval in sp
        key, sval = split(keyval, "=")
        key = Symbol(key)
        ps[key] = parseval(sval)
    end
    (; ps...)
end

function firefox(p)
    dir = normpath(homedir(), ".tmp")
    mkpath(dir)
    fname = normpath(dir, "tmp.html")
    savefig(p, fname)
    run(`firefox $(fname)`, wait=false)
    return nothing
end
