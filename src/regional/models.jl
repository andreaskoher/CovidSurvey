using ..CovidSurvey: RandomWalk
@model function model_cases(
    θ,     # [Bool] if `false`, will only compute what's needed to `observe` but not more
    predict,
	vectorize = true,
	::Type{TV} = Vector{Float64},
	::Type{V}  = Float64;
) where {TV, V}

	@unpack num_observations, num_total_days, num_regions, num_rt_steps, invlink, hospitmodel, casemodel, seromodel, num_covariates, include_early_dynamic = θ
    # If we don't want to predict the future, we only need to compute up-to time-step `num_obs_countries[m]`
    num_time_steps = predict ? num_total_days : num_observations

	############# 2.) time varying reproduction number

	# κ          ~ truncated(Normal(0, 1.), 0, Inf)
	# R0s        ~ filldist(truncated(Normal(2., κ), 1., 5.), num_regions)
	# σ_rt       ~ truncated(Normal(0.15, .05), 0, .25)
	R0         ~ truncated(Normal(3.0, 0.5), 1., 4.5)
	R0s        ~ filldist(truncated(Normal(R0, 0.2), 1., 4.5), num_regions)
	σ_rt       ~ truncated(Normal(0.15*2, .05*2), 0, .25*3)

	init_rw = if include_early_dynamic
			R0s
		else
			R1  ~ truncated(Normal(0.8, 0.1), 0., 1.2)
			R1s ~ filldist(truncated(Normal(R1, .1), 0., 1.2), num_regions)
	end

	latent_Rts_z = [TV(undef, num_rt_steps[m]) for m in 1:num_regions]
	latent_Rts = [TV(undef, num_rt_steps[m]) for m in 1:num_regions]
	for m in 1:num_regions
		latent_Rts_z[m] ~ RandomWalk(num_rt_steps[m])
		latent_Rts[m] = latent_Rts_z[m] * σ_rt .+ invlink(init_rw[m])
	end

	Rts = TV[TV(undef, num_time_steps[m]) for m in 1:num_regions]
	if num_covariates > 0
		grouped_effect ~ filldist( Laplace(0,0.2), num_covariates)
		effect_std ~ filldist( GammaMeanCv(0.1, 0.5), num_covariates)
		effects_z ~ filldist( MvNormal( num_covariates, 1.), num_regions)
		effects = [ effects_z[:,m] .* effect_std .+ grouped_effect for m in 1:num_regions]

		# grouped_effect ~ filldist( Exponential(0.2), num_covariates)
		# effect_std ~ filldist( GammaMeanCv(0.1, 0.5), num_covariates)
		# pooled_effects_distr_raw = Normal.( grouped_effect, effect_std )
		# pooled_effects_distr = arraydist( truncated.( pooled_effects_distr_raw, zero(V), V(Inf) ) )
		# effects ~ filldist( pooled_effects_distr, num_regions )

		if θ.semiparametric
			semiparametric!(Rts, θ, predict, latent_Rts, R0s, σ_rt, effects)
		else
			mixed!(Rts, θ, predict, latent_Rts, R0s, σ_rt, effects)
		end
	else
		random_walks!(Rts, θ, predict, latent_Rts, R0s, σ_rt)
	end

	############ 3.) infection dynamics
	τ  ~ Exponential(100/5) # `Exponential` has inverse parameterization of the one in Stan
	T  = typeof(τ)
	ys ~ filldist(truncated(Normal(τ, 30),T(0),T(500)), num_regions)

	newly_infecteds      = TV[TV(undef, num_time_steps[m]) for m in 1:num_regions]
	cumulative_infecteds = TV[TV(undef, num_time_steps[m]) for m in 1:num_regions]
	# effective_Rts        = TV[TV(undef, num_time_steps[m]) for m in 1:num_regions]

	infections!(newly_infecteds, cumulative_infecteds, θ, τ, ys, Rts)
	# infections!(newly_infecteds, cumulative_infecteds, effective_Rts, θ, τ, ys, Rts)
	########### 4.) derive observables
    μ_i2h ~ truncated(Normal(9.5, .5), 7, 15)
	μ_i2c ~ truncated(Normal(2., .5), 0, 5)
	ihr   ~ truncated(Normal(1.8/100,0.5/100), 1/100, 5/100)
	iar   ~ Beta(1,10)
	ϕ_h   ~ truncated(Normal(50, 10), 20, Inf)
	ϕ_c   ~ truncated(InverseGamma2(90, .5), 10, Inf)

	### day of week effect
	weekdayeffect_simplex ~ Dirichlet([7,7,7,7,7,3,1])
	weekdayeffect = TV(undef, 7)
	weekdayeffect!(weekdayeffect, weekdayeffect_simplex)

	### holiday effect
	holidayeffect ~ Beta(1,1)

	########### 3.) observation model
	expected_daily_hospits = TV[TV(undef, num_time_steps[m]) for m in 1:num_regions]
	expected_daily_cases   = TV[TV(undef, num_time_steps[m]) for m in 1:num_regions]

	hospit_observation_model = SimpleObsModel(hospitmodel, μ_i2h, ihr, ϕ_h, expected_daily_hospits, vectorize)
	# cases_observation_model = SimpleObsModel(casemodel, μ_i2c, iar, ϕ_c, expected_daily_cases)
	cases_observation_model = WeekdayHolidayObsModel(
        casemodel, μ_i2c, iar, ϕ_c, expected_daily_cases, weekdayeffect, holidayeffect, vectorize
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

# ============================================================
@model function model_hospit(
    θ,     # [Bool] if `false`, will only compute what's needed to `observe` but not more
	::Type{TV} = Vector{Float64},
	::Type{V}  = Float64;
    predict = false,
	vectorize = true,
	context   = :inference
) where {TV, V}

	@unpack num_observations, num_total_days, num_regions, num_rt_steps, hospitmodel, invlink, seromodel, num_covariates, rwscale, include_early_dynamic = θ
    # If we don't want to predict the future, we only need to compute up-to time-step `num_obs_countries[m]`
    num_time_steps = predict ? num_total_days : num_observations

	############# 2.) time varying reproduction number

	R0_std     ~ truncated(Normal(0, .5), 0, Inf)
	R0s        ~ filldist(truncated(Normal(3.5, R0_std), 1., 5.), num_regions)
	# R0         ~ truncated(Normal(3.0, 0.5), 0., 6.)
	# R0s        ~ filldist(truncated(Normal(R0, 0.2), 0., 6.), num_regions) #CovidSurvey.CustomMvNormal1(R0, 0.2, 1., 4.5, num_regions) #filldist(truncated(Normal(R0, 0.2), 1., 4.5), num_regions)
	σ_rt       ~ truncated(Normal(0.15*rwscale, .05*rwscale), 0, .4*rwscale)

	init_rw = if include_early_dynamic
			R0s
		else
			# R1  ~ truncated(Normal(0.8, 0.1), 0., 2.)
			# R1s ~ filldist(truncated(Normal(R1, 0.1), 0., 2.), num_regions)

			R1_std    ~ truncated(Normal(0, .1), 0, Inf)
			R1s       ~ filldist(truncated(Normal(0.8, R1_std), 0., 1.5), num_regions)
			R1s
	end

	latent_Rts = if include_early_dynamic
		latent_Rts_z = [TV(undef, num_rt_steps[m]) for m in 1:num_regions]
		latent_Rts = [TV(undef, num_rt_steps[m]) for m in 1:num_regions]
		for m in 1:num_regions
			latent_Rts_z[m] ~ CovidSurvey.RandomWalk(num_rt_steps[m])
			@. latent_Rts[m] = latent_Rts_z[m] * σ_rt + invlink(init_rw[m])
		end
		latent_Rts
	else
		n = first( num_rt_steps )
		latent_Rts_z ~ filldist( CovidSurvey.RandomWalk( n ), num_regions)
		rescale( latent_Rts_z, σ_rt, invlink.(init_rw) )
	end
	# latent_Rts_z = [TV(undef, num_rt_steps[m]) for m in 1:num_regions]
	# latent_Rts = [TV(undef, num_rt_steps[m]) for m in 1:num_regions]
	# for m in 1:num_regions
	# 	latent_Rts_z[m] ~ RandomWalk(num_rt_steps[m])
	# 	latent_Rts[m] = latent_Rts_z[m] * σ_rt .+ invlink(init_rw[m])
	# end

	Rts = TV[TV(undef, num_time_steps[m]) for m in 1:num_regions]
	if num_covariates > 0
		grouped_effects ~ filldist( Laplace(0,0.2), num_covariates)
		effect_std ~ filldist( GammaMeanCv(0.1, 0.5), num_covariates)
		effects_z ~ filldist( MvNormal( num_covariates, 1.), num_regions)
		effects = [ effects_z[:,m] .* effect_std .+ grouped_effects for m in 1:num_regions]

		# grouped_effect ~ filldist( Exponential(0.2), num_covariates)
		# effect_std ~ filldist( GammaMeanCv(0.1, 0.5), num_covariates)
		# pooled_effects_distr_raw = Normal.( grouped_effect, effect_std )
		# pooled_effects_distr = arraydist( truncated.( pooled_effects_distr_raw, zero(V), V(Inf) ) )
		# effects ~ filldist( pooled_effects_distr, num_regions )

		if θ.semiparametric
			semiparametric!(Rts, θ, predict, latent_Rts, R0s, σ_rt, effects)
		else
			mixed!(Rts, θ, predict, latent_Rts, R0s, σ_rt, effects)
		end
	else
		random_walks!(Rts, θ, predict, latent_Rts, R0s, σ_rt)
	end

	############ 3.) infection dynamics
	τ  ~ Exponential(20) # `Exponential` has inverse parameterization of the one in Stan
	T  = typeof(τ)
	# ys ~ filldist(truncated(Normal(τ, 30),T(1),T(500)), num_regions)
	ys ~ filldist(Exponential(τ), num_regions)

	newly_infecteds      = TV[TV(undef, num_time_steps[m]) for m in 1:num_regions]
	cumulative_infecteds = TV[TV(undef, num_time_steps[m]) for m in 1:num_regions]
	# effective_Rts        = TV[TV(undef, num_time_steps[m]) for m in 1:num_regions]

	infections!(newly_infecteds, cumulative_infecteds, θ, ys, Rts)
	# infections!(newly_infecteds, cumulative_infecteds, effective_Rts, θ, τ, ys, Rts)
	########### 4.) derive observables
	μ_i2h ~ truncated(Normal(11.7, .5), 8, 14)
	σ_i2h ~ truncated(Normal(6.5,0.5), 1, 14)
	ihr   ~ truncated(Normal(1.8/100,0.5/100), 0.5/100, 5/100) #GammaMeanCv(0.025, 0.1)
	ϕ_h   ~ truncated(Normal(50, 10), 20, Inf)

	expected_daily_hospit    = TV[TV(undef, num_time_steps[m]) for m in 1:num_regions]
	hospit_observation_model = SimpleObsModel2(hospitmodel, μ_i2h, σ_i2h, ihr, ϕ_h, expected_daily_hospit, vectorize)
	# hospit_observation_model = SimpleObsModel(hospitmodel, μ_i2h, ihr, ϕ_h, expected_daily_hospit, vectorize)
	sero_observation_model   = SimpleSeroObsModel(seromodel, cumulative_infecteds)
	expected!(hospit_observation_model, newly_infecteds)
	########### 4.) compare model to observations
	## 4.1) observe hospitalizations

	if context == :inference
		ℓ  = zero(V)
		ℓ += logpdf(hospit_observation_model, θ.hospits)
		ℓ += logpdf(sero_observation_model, θ.sero)
		Turing.@addlogprob! ℓ
		return (
			newly_infecteds = newly_infecteds,
			expected_daily_hospits = expected_daily_hospit,
			expected_seropos = cumulative_infecteds,
			Rts = Rts,
		)
	else
		if num_covariates > 0
			return (
				newly_infecteds = newly_infecteds,
				expected_daily_hospits = expected_daily_hospit,
				expected_seropos = cumulative_infecteds,
				Rts = Rts,
				predicted_daily_hospits = prediction(hospit_observation_model), #effects = effects,
				grouped_effects = [grouped_effects for m in 1:num_regions],
				effects = effects
			)
		else
			return (
				newly_infecteds = newly_infecteds,
				expected_daily_hospits = expected_daily_hospit,
				expected_seropos = cumulative_infecteds,
				Rts = Rts,
				predicted_daily_hospits = prediction(hospit_observation_model) #effects = effects,
				#effective_Rt = effective_Rts
			)
		end
	end
end

# @model function model_hospit(
#     θ,     # [Bool] if `false`, will only compute what's needed to `observe` but not more
# 	::Type{TV} = Vector{Float64},
# 	::Type{V}  = Float64;
#     predict = false,
# 	vectorize = true,
# 	context = :inference #:prediction, :loglikelihood
# ) where {TV, V}
#
# 	@unpack num_observations, num_total_days, num_regions, num_rt_steps, invlink, hospitmodel, seromodel, num_covariates, include_early_dynamic = θ
#     # If we don't want to predict the future, we only need to compute up-to time-step `num_obs_countries[m]`
#     num_time_steps = predict ? num_total_days : num_observations
#
# 	############# 2.) time varying reproduction number
#
# 	# κ          ~ truncated(Normal(0, 1.), 0, Inf)
# 	# R0s        ~ filldist(truncated(Normal(2., κ), 1., 5.), num_regions)
# 	# σ_rt       ~ truncated(Normal(0.15, .05), 0, .25)
# 	# R0         ~ truncated(Normal(3.0, 0.5), 1., 4.5)
# 	# R0_std     ~ GammaMeanCv(0.2, 0.5)
# 	# R0s        ~ filldist( truncated( Normal( R0, R0_std ), V(1.), V(4.5)), num_regions)
# 	R0s        ~ filldist(truncated(Normal(3., 1.), 1., 5.), num_regions)
# 	σ_rt       ~ truncated(Normal(0.15*2, .05*2), 0, .25*3)
#
# 	init_rw = if include_early_dynamic
# 			R0s
# 		else
# 			R1s ~ filldist(truncated(Normal(.8, .1), .5, 1.1), num_regions)
# 	end
#
# 	latent_Rts_z = [TV(undef, num_rt_steps[m]) for m in 1:num_regions]
# 	latent_Rts = [TV(undef, num_rt_steps[m]) for m in 1:num_regions]
# 	for m in 1:num_regions
# 		latent_Rts_z[m] ~ RandomWalk(num_rt_steps[m])
# 		latent_Rts[m] = latent_Rts_z[m] * σ_rt .+ invlink(init_rw[m])
# 	end
#
# 	Rts = TV[TV(undef, num_time_steps[m]) for m in 1:num_regions]
# 	if num_covariates > 0
# 		grouped_effects ~ filldist( Laplace(0,0.2), num_covariates)
# 		effect_std ~ filldist( GammaMeanCv(0.1, 0.5), num_covariates)
# 		effects_z ~ filldist( MvNormal( num_covariates, 1.), num_regions)
# 		effects = [ effects_z[:,m] .* effect_std .+ grouped_effects for m in 1:num_regions]
#
# 		# grouped_effect ~ filldist( Exponential(0.2), num_covariates)
# 		# effect_std ~ filldist( GammaMeanCv(0.1, 0.5), num_covariates)
# 		# pooled_effects_distr_raw = Normal.( grouped_effect, effect_std )
# 		# pooled_effects_distr = arraydist( truncated.( pooled_effects_distr_raw, zero(V), V(Inf) ) )
# 		# effects ~ filldist( pooled_effects_distr, num_regions )
#
# 		if θ.semiparametric
# 			semiparametric!(Rts, θ, predict, latent_Rts, R0s, σ_rt, effects)
# 		else
# 			mixed!(Rts, θ, predict, latent_Rts, R0s, σ_rt, effects)
# 		end
# 	else
# 		random_walks!(Rts, θ, predict, latent_Rts, R0s, σ_rt)
# 	end
#
#
# 	############ 3.) infection dynamics
# 	# y  ~ InverseGamma2(120, 0.2) # `Exponential` has inverse parameterization of the one in Stan
# 	# y_std   ~ InverseGamma2(10, 0.5)
# 	# ys ~ filldist(truncated(Normal(y, 10), 1., 500), num_regions)
# 	#ys      = y_z * y_std .+ y_mean
# 	τ  ~ Exponential(30.) # `Exponential` has inverse parameterization of the one in Stan
# 	T  = typeof(τ)
# 	ys ~ filldist(truncated(Exponential(τ),T(0),T(1000)), num_regions)
#
#
# 	newly_infecteds      = TV[TV(undef, num_time_steps[m]) for m in 1:num_regions]
# 	cumulative_infecteds = TV[TV(undef, num_time_steps[m]) for m in 1:num_regions]
# 	# effective_Rts        = TV[TV(undef, num_time_steps[m]) for m in 1:num_regions]
#
# 	infections!(newly_infecteds, cumulative_infecteds, θ, ys, Rts)
# 	# infections!(newly_infecteds, cumulative_infecteds, effective_Rts, θ, τ, ys, Rts)
# 	########### 4.) derive observables
#     μ_i2h ~ truncated(Normal(11., .5), 8, 14)
# 	σ_i2h ~ truncated(Normal(5.41,1), 1, 14)
# 	ihr   ~ truncated(Normal(1.8/100,0.5/100), 1/100, 5/100)
# 	ϕ_h   ~ truncated(Normal(50, 10), 20, Inf)
#
# 	expected_daily_hospits   = TV[TV(undef, num_time_steps[m]) for m in 1:num_regions]
# 	hospit_observation_model = SimpleObsModel2(hospitmodel, μ_i2h, σ_i2h, ihr, ϕ_h, expected_daily_hospits, vectorize)
# 	sero_observation_model   = SimpleSeroObsModel(seromodel, cumulative_infecteds)
#
# 	expected!(hospit_observation_model, newly_infecteds)
#
# 	########### 4.) compare model to observations
# 	## 4.1) observe hospitalizations
# 	ℓ  = zero(V)
# 	ℓ += logpdf(hospit_observation_model, θ.hospits)
# 	ℓ += logpdf(sero_observation_model, θ.sero)
# 	Turing.@addlogprob! ℓ
#
# 	return (
# 		newly_infecteds = newly_infecteds,
# 		expected_daily_hospits = expected_daily_hospits,
# 		expected_seropos = cumulative_infecteds,
# 		Rts = Rts,
# 		#predicted_daily_hospits = predicted_daily_hospits,
# 		# effective_Rt = effective_Rts
# 	)
# 	# if context == :inference
# 	# 	ℓ  = zero(V)
# 	# 	ℓ += logpdf(hospit_observation_model, θ.hospits)
# 	# 	ℓ += logpdf(sero_observation_model, θ.sero)
# 	# 	Turing.@addlogprob! ℓ
# 	#
# 	# elseif context == :loglikelihood
# 	# 	hospit_loglikelihoods = loglikelihoods(hospit_observation_model, θ.hospits)
# 	# 	sero_loglikelihoods = loglikelihoods(sero_observation_model, θ.sero)
# 	# 	return (;
# 	# 		hospit_loglikelihoods,
# 	# 		sero_loglikelihoods
# 	# 	)
# 	#
# 	# else #prediction
# 	# 	predicted_daily_hospits = prediction(hospit_observation_model)
# 	#
# 	# 	if num_covariates == 0
# 	# 		return (
# 	# 			newly_infecteds = newly_infecteds,
# 	# 			expected_daily_hospits = expected_daily_hospits,
# 	# 			expected_seropos = cumulative_infecteds,
# 	# 			Rts = Rts,
# 	# 			predicted_daily_hospits = predicted_daily_hospits,
# 	# 			# effective_Rt = effective_Rts
# 	# 		)
# 	# 	else
# 	# 		return (
# 	# 			newly_infecteds = newly_infecteds,
# 	# 			expected_daily_hospits = expected_daily_hospits,
# 	# 			expected_seropos = cumulative_infecteds,
# 	# 			Rts = Rts,
# 	# 			predicted_daily_hospits = predicted_daily_hospits,
# 	# 			grouped_effects = [grouped_effects for m in 1:num_regions],
# 	# 			effects = effects
# 	# 			#effective_Rt = effective_Rts
# 	# 		)
# 	# 	end
# 	# end
# end
#
#
# @model function model_hospit2(
#     θ,     # [Bool] if `false`, will only compute what's needed to `observe` but not more
# 	hospits,
# 	::Type{TV} = Vector{Float64},
# 	::Type{V}  = Float64;
#     predict = false,
# 	vectorize = true,
# 	context = :inference #:prediction, :loglikelihood
# ) where {TV, V}
#
# 	@unpack num_observations, num_total_days, num_regions, num_rt_steps, invlink, hospitmodel, seromodel, num_covariates, include_early_dynamic = θ
#     # If we don't want to predict the future, we only need to compute up-to time-step `num_obs_countries[m]`
#     num_time_steps = predict ? num_total_days : num_observations
#
# 	############# 2.) time varying reproduction number
#
# 	# κ          ~ truncated(Normal(0, 1.), 0, Inf)
# 	# R0s        ~ filldist(truncated(Normal(2., κ), 1., 5.), num_regions)
# 	# σ_rt       ~ truncated(Normal(0.15, .05), 0, .25)
# 	R0         ~ truncated(Normal(3.0, 0.5), 1., 4.5)
# 	R0s        ~ filldist(truncated(Normal(R0, 0.2), 1., 4.5), num_regions)
# 	σ_rt       ~ truncated(Normal(0.15*2, .05*2), 0, .25*3)
#
# 	init_rw = if include_early_dynamic
# 			R0s
# 		else
# 			R1s ~ filldist(truncated(Normal(.8, .1), .5, 1.1), num_regions)
# 	end
#
# 	latent_Rts_z = [TV(undef, num_rt_steps[m]) for m in 1:num_regions]
# 	latent_Rts = [TV(undef, num_rt_steps[m]) for m in 1:num_regions]
# 	for m in 1:num_regions
# 		latent_Rts_z[m] ~ RandomWalk(num_rt_steps[m])
# 		latent_Rts[m] = latent_Rts_z[m] * σ_rt .+ invlink(init_rw[m])
# 	end
#
# 	Rts = TV[TV(undef, num_time_steps[m]) for m in 1:num_regions]
# 	if num_covariates > 0
# 		grouped_effects ~ filldist( Laplace(0,0.2), num_covariates)
# 		effect_std ~ filldist( GammaMeanCv(0.1, 0.5), num_covariates)
# 		effects_z ~ filldist( MvNormal( num_covariates, 1.), num_regions)
# 		effects = [ effects_z[:,m] .* effect_std .+ grouped_effects for m in 1:num_regions]
#
# 		# grouped_effect ~ filldist( Exponential(0.2), num_covariates)
# 		# effect_std ~ filldist( GammaMeanCv(0.1, 0.5), num_covariates)
# 		# pooled_effects_distr_raw = Normal.( grouped_effect, effect_std )
# 		# pooled_effects_distr = arraydist( truncated.( pooled_effects_distr_raw, zero(V), V(Inf) ) )
# 		# effects ~ filldist( pooled_effects_distr, num_regions )
#
# 		if θ.semiparametric
# 			semiparametric!(Rts, θ, predict, latent_Rts, R0s, σ_rt, effects)
# 		else
# 			mixed!(Rts, θ, predict, latent_Rts, R0s, σ_rt, effects)
# 		end
# 	else
# 		random_walks!(Rts, θ, predict, latent_Rts, R0s, σ_rt)
# 	end
#
#
# 	############ 3.) infection dynamics
# 	τ  ~ Exponential(100 / 5) # `Exponential` has inverse parameterization of the one in Stan
# 	T  = typeof(τ)
# 	ys ~ filldist(truncated(Normal(τ, 30),T(0),T(500)), num_regions)
#
# 	newly_infecteds      = TV[TV(undef, num_time_steps[m]) for m in 1:num_regions]
# 	cumulative_infecteds = TV[TV(undef, num_time_steps[m]) for m in 1:num_regions]
# 	# effective_Rts        = TV[TV(undef, num_time_steps[m]) for m in 1:num_regions]
#
# 	infections!(newly_infecteds, cumulative_infecteds, θ, τ, ys, Rts)
# 	# infections!(newly_infecteds, cumulative_infecteds, effective_Rts, θ, τ, ys, Rts)
# 	########### 4.) derive observables
#     μ_i2h ~ truncated(Normal(11., .5), 8, 14)
# 	σ_i2h ~ truncated(Normal(5.41,1), 1, 14)
# 	ihr   ~ truncated(Normal(1.8/100,0.5/100), 1/100, 5/100)
# 	ϕ_h   ~ truncated(Normal(50, 10), 20, Inf)
#
# 	expected_daily_hospits   = TV[TV(undef, num_time_steps[m]) for m in 1:num_regions]
# 	hospit_observation_model = SimpleObsModel2(hospitmodel, μ_i2h, σ_i2h, ihr, ϕ_h, expected_daily_hospits, vectorize)
# 	sero_observation_model   = SimpleSeroObsModel(seromodel, cumulative_infecteds)
#
# 	expected!(hospit_observation_model, newly_infecteds)
#
# 	########### 4.) compare model to observations
# 	## 4.1) observe hospitalizations
#
# 	for m in 1:num_regions
# 		start = hospitmodel.starts[m]
# 		stop  = hospitmodel.stops[m]
# 		ts    = start:stop
# 		μs = expected_daily_hospits[m][ts]
# 		ys = hospits[m][ts]
#
# 		_out_of_bounds(μs, populations[m]) && (return T(Inf))
# 		if vectorize
# 			ys ~ arraydist(NegativeBinomial2.(μs, Ref(ϕ_h)))
# 		else
# 			for j in eachindex(ys)
# 				ys[j] ~ NegativeBinomial2(μs[j], ϕ_h)
# 			end
# 		end
# 	end
#
# 	if num_covariates == 0
# 		return (
# 			newly_infecteds = newly_infecteds,
# 			expected_daily_hospits = expected_daily_hospits,
# 			expected_seropos = cumulative_infecteds,
# 			Rts = Rts,
# 			# effective_Rt = effective_Rts
# 		)
# 	else
# 		return (
# 			newly_infecteds = newly_infecteds,
# 			expected_daily_hospits = expected_daily_hospits,
# 			expected_seropos = cumulative_infecteds,
# 			Rts = Rts,
# 			#grouped_effects = grouped_effects,
# 			#effects = effects
# 			# effective_Rt = effective_Rts
# 		)
# 	end
# end
# ============================================================
@model function model_deaths(
    θ,     # [Bool] if `false`, will only compute what's needed to `observe` but not more
	::Type{TV} = Vector{Float64},
	::Type{V}  = Float64;
	predict = false,
	vectorize = true,
	context   = :inference
) where {TV, V}

	@unpack num_observations, num_total_days, num_regions, num_rt_steps, deathmodel, invlink, seromodel, num_covariates, include_early_dynamic = θ
    # If we don't want to predict the future, we only need to compute up-to time-step `num_obs_countries[m]`
    num_time_steps = predict ? num_total_days : num_observations

	############# 2.) time varying reproduction number

	# κ          ~ truncated(Normal(0, 1.), 0, Inf)
	# R0s        ~ filldist(truncated(Normal(2., κ), 1., 5.), num_regions)
	# σ_rt       ~ truncated(Normal(0.15, .05), 0, .25)
	R0         ~ truncated(Normal(3.0, 0.5), 1., 4.5)
	R0s        ~ filldist(truncated(Normal(R0, 0.2), 1., 4.5), num_regions)
	σ_rt       ~ truncated(Normal(0.15*2, .05*2), 0, .25*3)

	init_rw = if include_early_dynamic
			R0s
		else
			R1s ~ filldist(truncated(Normal(.8, .1), .5, 1.1), num_regions)
	end

	latent_Rts_z = [TV(undef, num_rt_steps[m]) for m in 1:num_regions]
	latent_Rts = [TV(undef, num_rt_steps[m]) for m in 1:num_regions]
	for m in 1:num_regions
		latent_Rts_z[m] ~ RandomWalk(num_rt_steps[m])
		latent_Rts[m] = latent_Rts_z[m] * σ_rt .+ invlink(init_rw[m])
	end

	Rts = TV[TV(undef, num_time_steps[m]) for m in 1:num_regions]
	if num_covariates > 0
		grouped_effect ~ filldist( Laplace(0,0.2), num_covariates)
		effect_std ~ filldist( GammaMeanCv(0.1, 0.5), num_covariates)
		effects_z ~ filldist( MvNormal( num_covariates, 1.), num_regions)
		effects = [ effects_z[:,m] .* effect_std .+ grouped_effect for m in 1:num_regions]

		# grouped_effect ~ filldist( Exponential(0.2), num_covariates)
		# effect_std ~ filldist( GammaMeanCv(0.1, 0.5), num_covariates)
		# pooled_effects_distr_raw = Normal.( grouped_effect, effect_std )
		# pooled_effects_distr = arraydist( truncated.( pooled_effects_distr_raw, zero(V), V(Inf) ) )
		# effects ~ filldist( pooled_effects_distr, num_regions )

		if θ.semiparametric
			semiparametric!(Rts, θ, predict, latent_Rts, R0s, σ_rt, effects)
		else
			mixed!(Rts, θ, predict, latent_Rts, R0s, σ_rt, effects)
		end
	else
		random_walks!(Rts, θ, predict, latent_Rts, R0s, σ_rt)
	end

	############ 3.) infection dynamics
	τ  ~ Exponential(100 / 5) # `Exponential` has inverse parameterization of the one in Stan
	T  = typeof(τ)
	ys ~ filldist(truncated(Normal(τ, 30),T(0),T(500)), num_regions)

	newly_infecteds      = TV[TV(undef, num_time_steps[m]) for m in 1:num_regions]
	cumulative_infecteds = TV[TV(undef, num_time_steps[m]) for m in 1:num_regions]
	# effective_Rts        = TV[TV(undef, num_time_steps[m]) for m in 1:num_regions]

	infections!(newly_infecteds, cumulative_infecteds, θ, ys, Rts)
	# infections!(newly_infecteds, cumulative_infecteds, effective_Rts, θ, τ, ys, Rts)
	########### 4.) derive observables
    μ_i2d ~ truncated(Normal(20., .5), 10, 22)
	idr   ~ truncated(Normal(6/1000, 3/1000), 1/1000, 10/1000)
	ϕ_d   ~ truncated(Normal(50, 10), 20, Inf)

	expected_daily_deaths    = TV[TV(undef, num_time_steps[m]) for m in 1:num_regions]
	deaths_observation_model = SimpleObsModel(deathmodel, μ_i2d, idr, ϕ_d, expected_daily_deaths, vectorize)
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
		#predicted_daily_deaths = prediction(deaths_observation_model) #effects = effects,
		#effective_Rt = effective_Rts
	)
end
