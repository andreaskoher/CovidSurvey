## ============================================================
@model function model_hospit(
    θ,     # [Bool] if `false`, will only compute what's needed to `observe` but not more
    predict,
	::Type{TV} = Vector{Float64},
	::Type{V}  = Float64;
) where {TV, V}

	@unpack num_observations, num_total_days, num_regions, num_rt_steps, invlink = θ
    # If we don't want to predict the future, we only need to compute up-to time-step `num_obs_countries[m]`
    num_time_steps = predict ? num_total_days : num_observations

	############# 2.) time varying reproduction number

	R0s        ~ filldist(truncated(Normal(3., 1.), 1., 5.), num_regions)
	R1s        ~ filldist(truncated(Normal(.8, .1), .5, 1.1), num_regions)
	σ_rt       ~ truncated(Normal(0.1, .05), 0, .25)
	latent_Rts ~ arraydist( RandomWalk.(num_rt_steps, Ref(σ_rt), invlink.(R1s)) )

	Rts = TV[TV(undef, num_time_steps[m]) for m in 1:num_regions]

	random_walks!(Rts, θ, predict, latent_Rts, R0s)

	############ 3.) infection dynamics
	τ  ~ Exponential(1 / 0.03) # `Exponential` has inverse parameterization of the one in Stan
	T  = typeof(τ)
	ys ~ filldist(truncated(Exponential(τ),T(0),T(1000)), num_regions)

	newly_infecteds      = TV[TV(undef, num_time_steps[m]) for m in 1:num_regions]
	cumulative_infecteds = TV[TV(undef, num_time_steps[m]) for m in 1:num_regions]
	# effective_Rts        = TV[TV(undef, num_time_steps[m]) for m in 1:num_regions]

	infections!(newly_infecteds, cumulative_infecteds, θ, τ, ys, Rts)
	# infections!(newly_infecteds, cumulative_infecteds, effective_Rts, θ, τ, ys, Rts)
	########### 4.) derive observables
    μ_i2h ~ truncated(Normal(12., 1.), 9, 16)
	ihr   ~ truncated(Normal(1.8/100,0.5/100),1/100,5/100)

	expected_daily_hospits = TV[TV(undef, num_time_steps[m]) for m in 1:num_regions]

	hospitalizations!(expected_daily_hospits, θ, μ_i2h, ihr, newly_infecteds)

	########### 4.) compare model to observations
	## 4.1) observe hospitalizations
	ϕ_h   ~ truncated(Normal(50, 10), 20, Inf)

	ℓ  = zero(V)
	ℓ += observe_hospitalizations(ℓ, θ, expected_daily_hospits, ϕ_h)

	Turing.@addlogprob! ℓ

	return (
		newly_infecteds = newly_infecteds,
		expected_daily_hospits = expected_daily_hospits,
		Rts = Rts,
		# effective_Rt = effective_Rts
	)
end
