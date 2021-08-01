using ..CovidSurvey: RandomWalk
@model function model_cases(
    θ,     # [Bool] if `false`, will only compute what's needed to `observe` but not more
    predict,
	::Type{TV} = Vector{Float64},
	::Type{V}  = Float64;
) where {TV, V}

	@unpack num_observations, num_total_days, num_regions, num_rt_steps, invlink, hospitmodel, casemodel, seromodel = θ
    # If we don't want to predict the future, we only need to compute up-to time-step `num_obs_countries[m]`
    num_time_steps = predict ? num_total_days : num_observations

	############# 2.) time varying reproduction number

	κ          ~ truncated(Normal(0, 1.), 0, Inf)
	R0s        ~ filldist(truncated(Normal(3., κ), 1., 5.), num_regions)
	R1s        ~ filldist(truncated(Normal(.8, .1), .5, 1.1), num_regions)
	σ_rt       ~ truncated(Normal(0.1, .05), 0, .25)
	latent_Rts ~ arraydist( RandomWalk.(num_rt_steps, Ref(σ_rt), invlink.(R1s)) )

	Rts = TV[TV(undef, num_time_steps[m]) for m in 1:num_regions]

	random_walks!(Rts, θ, predict, latent_Rts, R0s)

	############ 3.) infection dynamics
	τ  ~ Exponential(1 / 0.02) # `Exponential` has inverse parameterization of the one in Stan
	T  = typeof(τ)
	ys ~ filldist(truncated(Exponential(τ),T(0),T(1000)), num_regions)

	newly_infecteds      = TV[TV(undef, num_time_steps[m]) for m in 1:num_regions]
	cumulative_infecteds = TV[TV(undef, num_time_steps[m]) for m in 1:num_regions]
	# effective_Rts        = TV[TV(undef, num_time_steps[m]) for m in 1:num_regions]

	infections!(newly_infecteds, cumulative_infecteds, θ, τ, ys, Rts)
	# infections!(newly_infecteds, cumulative_infecteds, effective_Rts, θ, τ, ys, Rts)
	########### 4.) derive observables
    μ_i2h ~ truncated(Normal(9., 1.), 7, 11)
	μ_i2c ~ truncated(Normal(2., 1.), 0, 5)
	ihr   ~ truncated(Normal(1.8/100,0.5/100),.1/100,5/100)
	iar   ~ Beta(1,10)
	ϕ_h   ~ truncated(Normal(50, 10), 20, Inf)
	ϕ_c   ~ truncated(InverseGamma2(80, .5), 10, Inf)

	### day of week effect
	weekdayeffect_simplex ~ Dirichlet([7,7,7,7,7,3,1])
	weekdayeffect = TV(undef, 7)
	weekdayeffect!(weekdayeffect, weekdayeffect_simplex)

	### holiday effect
	holidayeffect ~ Beta(1,1)

	########### 3.) observation model
	expected_daily_hospits = TV[TV(undef, num_time_steps[m]) for m in 1:num_regions]
	expected_daily_cases   = TV[TV(undef, num_time_steps[m]) for m in 1:num_regions]

	hospit_observation_model = SimpleObsModel1(hospitmodel, μ_i2h, ihr, ϕ_h, expected_daily_hospits)
	# cases_observation_model = SimpleObsModel1(casemodel, μ_i2c, iar, ϕ_c, expected_daily_cases)
	cases_observation_model = WeekdayHolidayObsModel1(
        casemodel, μ_i2c, iar, ϕ_c, expected_daily_cases, weekdayeffect, holidayeffect
    )
	sero_observation_model   = SimpleSeroObsModel(seromodel, cumulative_infecteds)

	expected!(hospit_observation_model, newly_infecteds)
	expected!(cases_observation_model, newly_infecteds)

	########### 4.) compare model to observations
	## 4.1) observe hospitalizations

	ℓ  = zero(V)
	ℓ += logpdf(hospit_observation_model, θ.hospits)
	ℓ += logpdf(cases_observation_model, θ.cases)
	ℓ += logpdf(sero_observation_model, θ.sero)
	Turing.@addlogprob! ℓ

	return (
		# newly_infecteds = newly_infecteds,
		expected_daily_hospits = expected_daily_hospits,
		expected_daily_cases = expected_daily_cases,
		expected_seropos = cumulative_infecteds,
		Rts = Rts,
		# effective_Rt = effective_Rts
	)
end

## ============================================================
@model function model_hospit(
    θ,     # [Bool] if `false`, will only compute what's needed to `observe` but not more
    predict,
	::Type{TV} = Vector{Float64},
	::Type{V}  = Float64;
) where {TV, V}

	@unpack num_observations, num_total_days, num_regions, num_rt_steps, invlink, hospitmodel, seromodel = θ
    # If we don't want to predict the future, we only need to compute up-to time-step `num_obs_countries[m]`
    num_time_steps = predict ? num_total_days : num_observations

	############# 2.) time varying reproduction number

	κ          ~ truncated(Normal(0, 1.), 0, Inf)
	R0s        ~ filldist(truncated(Normal(3., κ), 1., 5.), num_regions)
	R1s        ~ filldist(truncated(Normal(.8, .1), .5, 1.1), num_regions)
	σ_rt       ~ truncated(Normal(0.1, .05), 0, .25)
	latent_Rts ~ arraydist( RandomWalk.(num_rt_steps, Ref(σ_rt), invlink.(R1s)) )

	Rts = TV[TV(undef, num_time_steps[m]) for m in 1:num_regions]

	random_walks!(Rts, θ, predict, latent_Rts, R0s)

	############ 3.) infection dynamics
	τ  ~ Exponential(1 / 0.02) # `Exponential` has inverse parameterization of the one in Stan
	T  = typeof(τ)
	ys ~ filldist(truncated(Exponential(τ),T(0),T(1000)), num_regions)

	newly_infecteds      = TV[TV(undef, num_time_steps[m]) for m in 1:num_regions]
	cumulative_infecteds = TV[TV(undef, num_time_steps[m]) for m in 1:num_regions]
	# effective_Rts        = TV[TV(undef, num_time_steps[m]) for m in 1:num_regions]

	infections!(newly_infecteds, cumulative_infecteds, θ, τ, ys, Rts)
	# infections!(newly_infecteds, cumulative_infecteds, effective_Rts, θ, τ, ys, Rts)
	########### 4.) derive observables
    μ_i2h ~ truncated(Normal(9., 1.), 7, 11)
	ihr   ~ truncated(Normal(1.8/100,0.5/100),.1/100,5/100)
	ϕ_h   ~ truncated(Normal(50, 10), 20, Inf)

	expected_daily_hospits   = TV[TV(undef, num_time_steps[m]) for m in 1:num_regions]
	hospit_observation_model = SimpleObsModel1(hospitmodel, μ_i2h, ihr, ϕ_h, expected_daily_hospits)
	sero_observation_model   = SimpleSeroObsModel(seromodel, cumulative_infecteds)

	expected!(hospit_observation_model, newly_infecteds)

	########### 4.) compare model to observations
	## 4.1) observe hospitalizations

	ℓ  = zero(V)
	ℓ += logpdf(hospit_observation_model, θ.hospits)
	ℓ += logpdf(sero_observation_model, θ.sero)
	Turing.@addlogprob! ℓ

	return (
		newly_infecteds = newly_infecteds,
		expected_daily_hospits = expected_daily_hospits,
		expected_seropos = cumulative_infecteds,
		Rts = Rts,
		# effective_Rt = effective_Rts
	)
end

## ============================================================
@model function model_deaths(
    θ,     # [Bool] if `false`, will only compute what's needed to `observe` but not more
    predict,
	::Type{TV} = Vector{Float64},
	::Type{V}  = Float64;
) where {TV, V}

	@unpack num_observations, num_total_days, num_regions, num_rt_steps, deathmodel, invlink, seromodel = θ
    # If we don't want to predict the future, we only need to compute up-to time-step `num_obs_countries[m]`
    num_time_steps = predict ? num_total_days : num_observations

	############# 2.) time varying reproduction number

	κ          ~ truncated(Normal(0, 1.), 0, Inf)
	R0s        ~ filldist(truncated(Normal(3., κ), 1., 5.), num_regions)
	R1s        ~ filldist(truncated(Normal(.8, .1), .5, 1.1), num_regions)
	σ_rt       ~ truncated(Normal(0.1, .05), 0, .25)
	latent_Rts ~ arraydist( RandomWalk.(num_rt_steps, Ref(σ_rt), invlink.(R1s)) )

	Rts = TV[TV(undef, num_time_steps[m]) for m in 1:num_regions]

	random_walks!(Rts, θ, predict, latent_Rts, R0s)

	############ 3.) infection dynamics
	τ  ~ Exponential(1 / 0.02) # `Exponential` has inverse parameterization of the one in Stan
	T  = typeof(τ)
	ys ~ filldist(truncated(Exponential(τ),T(0),T(1000)), num_regions)

	newly_infecteds      = TV[TV(undef, num_time_steps[m]) for m in 1:num_regions]
	cumulative_infecteds = TV[TV(undef, num_time_steps[m]) for m in 1:num_regions]
	# effective_Rts        = TV[TV(undef, num_time_steps[m]) for m in 1:num_regions]

	infections!(newly_infecteds, cumulative_infecteds, θ, τ, ys, Rts)
	# infections!(newly_infecteds, cumulative_infecteds, effective_Rts, θ, τ, ys, Rts)
	########### 4.) derive observables
    μ_i2d ~ truncated(Normal(20., 2.), 17, 25)
	idr   ~ truncated(Normal(6/1000, 3/1000), 1/1000, 10/1000)
	ϕ_d   ~ truncated(Normal(40, 10), 1, Inf)

	expected_daily_deaths    = TV[TV(undef, num_time_steps[m]) for m in 1:num_regions]
	deaths_observation_model = SimpleObsModel1(deathmodel, μ_i2d, idr, ϕ_d, expected_daily_deaths)
	sero_observation_model   = SimpleSeroObsModel(seromodel, cumulative_infecteds)
	expected!(deaths_observation_model, newly_infecteds)
	########### 4.) compare model to observations
	## 4.1) observe hospitalizations

	ℓ  = zero(V)
	ℓ += logpdf(deaths_observation_model, θ.deaths)
	ℓ += logpdf(sero_observation_model, θ.sero)
	Turing.@addlogprob! ℓ

	return (
		newly_infecteds = newly_infecteds,
		expected_daily_deaths = expected_daily_deaths,
		expected_seropos = cumulative_infecteds,
		Rts = Rts,
		# effective_Rt = effective_Rts
	)
end

## ============================================================
@model function model_cases(
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
	τ  ~ Exponential(1 / 0.02) # `Exponential` has inverse parameterization of the one in Stan
	T  = typeof(τ)
	ys ~ filldist(truncated(Exponential(τ),T(0),T(1000)), num_regions)

	newly_infecteds      = TV[TV(undef, num_time_steps[m]) for m in 1:num_regions]
	cumulative_infecteds = TV[TV(undef, num_time_steps[m]) for m in 1:num_regions]
	# effective_Rts        = TV[TV(undef, num_time_steps[m]) for m in 1:num_regions]

	infections!(newly_infecteds, cumulative_infecteds, θ, τ, ys, Rts)
	# infections!(newly_infecteds, cumulative_infecteds, effective_Rts, θ, τ, ys, Rts)
	########### 4.) derive observables
	μ_i2d ~ truncated(Normal(20., 1.), 18, 22)
	μ_i2c ~ truncated(Normal(2., 1.), 0, 5)
	iar   ~ Beta(1,10)
	ifr   ~ truncated(Normal(6/1000, 3/1000), 2/1000, 10/1000)

	expected_daily_deaths = TV[TV(undef, num_time_steps[m]) for m in 1:num_regions]
	expected_daily_cases  = TV[TV(undef, num_time_steps[m]) for m in 1:num_regions]

	deaths!(expected_daily_deaths, θ, μ_i2d, ifr, newly_infecteds)
	cases!(expected_daily_cases,  θ, μ_i2c, iar, newly_infecteds)

	########### 4.) compare model to observations
	## 4.1) observe
	ϕ_c  ~ truncated(InverseGamma2(80, .2), 30, Inf)
	ϕ_d   ~ truncated(Normal(40, 10), 10, Inf)

	ℓ  = zero(V)
	ℓ += observe_cases(θ, expected_daily_cases, ϕ_c)
	ℓ += observe_deaths(θ, expected_daily_deaths, ϕ_d)

	Turing.@addlogprob! ℓ

	return (
		newly_infecteds = newly_infecteds,
		expected_daily_deaths = expected_daily_deaths,
		expected_daily_cases = expected_daily_cases,
		Rts = Rts,
		# effective_Rt = effective_Rts
	)
end
