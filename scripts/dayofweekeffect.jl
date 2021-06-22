struct WeekDayEffect1{Tα} <: ContinuousMultivariateDistribution
	α::Tα
end
WeekDayEffect1() = WeekDayEffect1(ones(Int64, 7))

Distributions.rand(rng::AbstractRNG, d::WeekDayEffect1{Tα}) where {Tα} = begin
	weekdayeffect_simplex = rand( Dirichlet(d.α) )
	weekdayeffect_simplex .* length(d.α)
end

Distributions.logpdf(d::WeekDayEffect1{Tα}, x::AbstractVector{T}) where {Tα,T} =
    logpdf( Dirichlet( d.α ), x ./ length(d.α) )


Bijectors.bijector(d::WeekDayEffect1) =
	Bijectors.bijector(Dirichlet(d.α)) ∘ Bijectors.Scale(1/7, dim=Val(1))


#Bijectors.bijector(Dirichlet( d.α ))
@model function test(y, d, ::Type{TV} = Vector{Float64}) where TV
	dwe ~ WeekDayEffect1()
	e ~ Normal(4, 0.1)
	ŷ = TV(undef, length(y))
	for i in eachindex(ŷ)
		ŷ = e + dwe[d[i]]
	end
	ŷ ~ MvNormal(y, 0.1)
	return ŷ
end

m = 10
d = repeat(1:7, outer=m)
n = length(d)
y = repeat(1:7, outer=m) + randn(n) * 0.1

m = test(y, d)
plot!(m())
c = sample(m, NUTS(1000, 0.8), 2000; progress=true)

gq = generated_quantities(m, c[1001:end])|>vec
ŷ = mean(gq)
plot(y)
plot!(ŷ)

plot(group(c, "dwe"))

logpdf(Dirichlet(ones(7)), ones(7) / 7.)
logpdf(WeekDayEffect1(ones(7)), ones(7))



bs = Bijectors.Scale(1/7, dim=Val(1))
bs isa Bijectors.Bijector
x = 1.
y = bs(x)
inv(bs)(y) ≈ x

bd = bijector(Dirichlet(ones(7)))
bd isa Bijectors.Bijector
x = ones(7)/7
y = bd(x)
inv(bd)(y) ≈ x

b = bd ∘ bs
x = ones(7)
y = b(x)
inv(b)(y) ≈ x
