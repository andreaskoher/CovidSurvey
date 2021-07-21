# using DrWatson
# quickactivate(@__DIR__)
# @show projectdir()
# using Plots
##
using CovidSurvey
using UnPack
using Test
using Turing
using Optim

θ = (
	num_rt_steps      = 100,
	lockdown_index    = 30,
	num_time_steps    = 130,
	num_observations  = 130,
	link              = KLogistic(3.),
    invlink           = KLogit(3.),
	population        = 5_837_213,
	num_impute        = 6,
	serial_interval   = National.serialinterval(15),
	num_si            = 15,
)

rt  = let
	n = θ.num_rt_steps
	N = θ.lockdown_index+n
	t = 1:N

	frt(t) = t <= θ.lockdown_index ? 2. : frt1(t)
	frt1(t) = 0.5 + (t-θ.lockdown_index-1)*1/100
	rt  = frt.(t)
	# plot(rt)|>display
	rt
end

## =============================================================================
# test RandomWalk
@testset "latent Rt" begin

	@model function test_rt(Y, θ, predict, ::Type{TV} = Vector{Float64}) where TV
		@unpack invlink, num_rt_steps, num_time_steps = θ
		R0         ~ truncated(Normal(2., .1), 1.5, 2.5)
		R1         ~ truncated(Normal(.5, .1), .3, .7)
		σ_rt       ~ truncated(Normal(0.1, .08), 0, .25)
	    latent_Rt  ~ National.RandomWalk(num_rt_steps, σ_rt, invlink(R1))

		Rt = TV(undef, num_time_steps)
		National.random_walks!(Rt, θ, predict, latent_Rt, R0, σ_rt)
		Y ~ MvNormal(Rt, 1e-5)
		return Rt
	end

	m = test_rt(rt, θ, false)
	@test length(m()) == θ.num_time_steps
	mle = optimize(m, MLE(), LBFGS(), Optim.Options(iterations = 1000, allow_f_increases = true))
	@test mle.optim_result.ls_success

	estimated_Rt = let #generated quantity
		v = mle.values
		latent_Rt = [v[Symbol("latent_Rt[$i]")] for i in 1:θ.num_rt_steps]
		Rt = Vector{Float64}(undef, θ.num_time_steps)
		National.random_walks!(Rt, θ, false, latent_Rt, v[:R0], v[:σ_rt])
		Rt
	end
	# plot!(estimated_Rt)
	@test isapprox(rt, estimated_Rt, atol=0.1)
end
## ============================================================================
# test latent infections
y = 100.
newly_infected       = Vector{Float64}(undef, θ.num_time_steps)
cumulative_infected  = Vector{Float64}(undef, θ.num_time_steps)
effective_Rt         = Vector{Float64}(undef, θ.num_time_steps)
National.infections!(newly_infected, cumulative_infected, effective_Rt, θ, y, rt)

@testset "latent infections with given Rt" begin
	@model function test_latent_infections(Y, θ, Rt, predict, ::Type{TV} = Vector{Float64}) where TV
		@unpack num_time_steps = θ

		############ infection dynamics
		τ  ~ Exponential(100/3)
		T  = typeof(τ)
		y  ~ truncated(Exponential(τ),T(0),T(1000))

		newly_infected       = TV(undef, num_time_steps)
		cumulative_infected  = TV(undef, num_time_steps)
		effective_Rt         = TV(undef, num_time_steps)

		National.infections!(newly_infected, cumulative_infected, effective_Rt, θ, y, Rt)
		Y ~ MvNormal(newly_infected, 1e-1)
		return newly_infected
	end

	m = test_latent_infections(newly_infected, θ, rt, false)
	@test length(m()) == θ.num_time_steps
	mle = optimize(m, MLE(), LBFGS(), Optim.Options(iterations = 1000, allow_f_increases = true))
	@test mle.optim_result.ls_success

	estimated_newly_infected  = let
		y = mle.values[:y]
		newly_infected       = Vector{Float64}(undef, θ.num_time_steps)
		cumulative_infected  = Vector{Float64}(undef, θ.num_time_steps)
		effective_Rt         = Vector{Float64}(undef, θ.num_time_steps)
		National.infections!(newly_infected, cumulative_infected, effective_Rt, θ, y, rt)
		newly_infected
	end
	# p = plot(newly_infected)
	# plot!(estimated_newly_infected)
	# @test all( estimated_newly_infected .≈ newly_infected )
	@test isapprox(estimated_newly_infected, newly_infected, atol=0.1)
end

## ============================================================================
# test latent infections #2
@testset "latent infections without Rt" begin
	@model function test_latent_infections_v2(Y, θ, predict, ::Type{TV} = Vector{Float64}) where TV
		@unpack invlink, num_rt_steps, num_time_steps = θ

		############# time varying reproduction number
	    R0         ~ truncated(Normal(2., .1), 1.5, 2.5)
		R1         ~ truncated(Normal(.5, .1), .3, .7)
		σ_rt       ~ truncated(Normal(0.1, .08), 0, .25)
	    latent_Rt  ~ National.RandomWalk(num_rt_steps, σ_rt, invlink(R1))

		Rt = TV(undef, num_time_steps)
		National.random_walks!(Rt, θ, predict, latent_Rt, R0, σ_rt)

		############ infection dynamics
		# τ  ~ Exponential(100/3)
		# T  = typeof(τ)
		y  ~ InverseGamma2(100, .1)#truncated(Normal(500, 200),T(10),T(1000))

		newly_infected       = TV(undef, num_time_steps)
		cumulative_infected  = TV(undef, num_time_steps)
		effective_Rt         = TV(undef, num_time_steps)

		National.infections!(newly_infected, cumulative_infected, effective_Rt, θ, y, Rt)
		Y ~ MvNormal(newly_infected, 1e-5)
		return newly_infected
	end

	m = test_latent_infections_v2(newly_infected, θ, false)
	@test length(m()) == θ.num_time_steps
	mle = optimize(m, MLE(), LBFGS(), Optim.Options(iterations = 1000, allow_f_increases = true))
	@test mle.optim_result.ls_success

	estimated_newly_infected, estimated_Rt  = let
		v = mle.values
		latent_Rt = [v[Symbol("latent_Rt[$i]")] for i in 1:θ.num_rt_steps]
		Rt = Vector{Float64}(undef, θ.num_time_steps)
		National.random_walks!(Rt, θ, false, latent_Rt, v[:R0], v[:σ_rt])

		y = v[:y]
		newly_infected       = Vector{Float64}(undef, θ.num_time_steps)
		cumulative_infected  = Vector{Float64}(undef, θ.num_time_steps)
		effective_Rt         = Vector{Float64}(undef, θ.num_time_steps)
		National.infections!(newly_infected, cumulative_infected, effective_Rt, θ, y, Rt)
		newly_infected, Rt
	end

	# plot(rt)
	# plot!(estimated_Rt)
	@test isapprox(rt, estimated_Rt, atol=0.1)

	# p = plot(newly_infected)
	# plot!(estimated_newly_infected)
	# @test all( estimated_newly_infected .≈ newly_infected )
	@test isapprox(estimated_newly_infected, newly_infected, atol=0.1)
end
