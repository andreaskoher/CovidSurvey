@model function model_deaths(
    θ,     # [Bool] if `false`, will only compute what's needed to `observe` but not more
	::Type{TV} = Vector{Float64},
	::Type{V}  = Float64;
	predict = false,
	vectorize = true,
	context   = :inference
) where {TV, V}

    @unpack num_observations, num_total_days, num_regions, num_rt_steps, deathmodel, init_infected, num_covariates, invlink, rwscale, include_early_dynamic = θ
      # If we don't want to predict the future, we only need to compute up-to time-step `num_obs_countries[m]`
    num_time_steps = num_observations

    ############# 2.) time varying reproduction number

    R0_std     ~ truncated(Normal(.5, .2), 0.01, Inf)
    R0s        ~ filldist(truncated(Normal(0.8, R0_std), 0., 4.5), num_regions)
    σ_rt       ~ truncated(Normal(0.3*rwscale, .02*rwscale), 0, .5*rwscale)

    n = first( num_rt_steps )
    latent_Rts_z ~ filldist( CovidSurvey.RandomWalk( n ), num_regions)
    latent_Rts = CovidSurvey.rescale( latent_Rts_z, σ_rt, invlink.(R0s) )


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
    	    CovidSurvey.semiparametric!(Rts, θ, predict, latent_Rts, R0s, σ_rt, effects)
    	else
            CovidSurvey.mixed!(Rts, θ, predict, latent_Rts, R0s, σ_rt, effects)
    	end
    else
        CovidSurvey.random_walks!(Rts, θ, predict, latent_Rts, R0s, σ_rt)
    end

    ############ 3.) infection dynamics
    # τ  ~ Exponential(100.) # `Exponential` has inverse parameterization of the one in Stan
    ys ~ arraydist(Exponential.(init_infected))
    # ys = fill(100, num_regions)

    newly_infecteds      = TV[TV(undef, num_time_steps[m]) for m in 1:num_regions]
    CovidSurvey.infections!(newly_infecteds, θ, ys, Rts)

    ########### 4.) derive observables
    μ_i2d ~ truncated(Normal(21.82, 1.01), 18, 24)
    σ_i2d ~ truncated(Normal(4.26, 5.18), 1, 14)
    # idr   = 0.8 / 100
    idr   ~ truncated(Normal(8/1000, 1/1000), 1/1000, 15/1000)
    ϕ_d   ~ truncated(Normal(50, 10), 20, Inf)



    ### day of week effect
    weekdayeffect_simplex ~ filldist(Dirichlet([7,7,7,7,7,3,1]), num_regions)
    weekdayeffect = TV[TV(undef, 7) for m in 1:num_regions]
    CovidSurvey.weekdayeffect!(weekdayeffect, weekdayeffect_simplex)

    ### holiday effect
    holidayeffect ~ filldist(Beta(1,1), num_regions)

    expected_daily_deaths    = TV[TV(undef, num_time_steps[m]) for m in 1:num_regions]
    deaths_observation_model = CovidSurvey.WeekdayHolidayObsModel2(
        deathmodel, μ_i2d, σ_i2d, idr, ϕ_d, expected_daily_deaths, weekdayeffect, holidayeffect, vectorize
    )

    # deaths_observation_model = CovidSurvey.SimpleObsModel2(deathmodel, μ_i2d, σ_i2d, idr, ϕ_d, expected_daily_deaths, vectorize)
    CovidSurvey.expected!(deaths_observation_model, newly_infecteds)

    ########### 4.) compare model to observations
    ## 4.1) observe deaths

    if context == :inference
		ℓ = CovidSurvey.logpdf(deaths_observation_model, θ.deaths)
		Turing.@addlogprob! ℓ
		return (
			newly_infecteds = newly_infecteds,
			expected_daily_deaths = expected_daily_deaths,
			Rts = Rts,
		)
	else
		if num_covariates > 0
			return (
				newly_infecteds = newly_infecteds,
				expected_daily_deaths = expected_daily_deaths,
				Rts = Rts,
				predicted_daily_deaths = CovidSurvey.prediction(deaths_observation_model), #effects = effects,
				grouped_effects = [grouped_effects for m in 1:num_regions],
				effects = effects
			)
		else
			return (
				newly_infecteds = newly_infecteds,
				expected_daily_deaths = expected_daily_deaths,
				Rts = Rts,
				predicted_daily_deaths = CovidSurvey.prediction(deaths_observation_model) #effects = effects,
				#effective_Rt = effective_Rts
			)
		end
	end
end
