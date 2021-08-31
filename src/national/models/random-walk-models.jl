@model function model_v2(
    θ,
    predict,
	::Type{TV} = Vector{Float64},
	::Type{V}  = Float64;
) where {TV, V}

	@unpack num_observations, num_total_days, num_rt_steps, num_iar_steps, invlink, deathmodel, casemodel = θ
    # If we don't want to predict the future, we only need to compute up-to time-step `num_obs_countries[m]`
    num_time_steps   = predict ? num_total_days : num_observations
	############# 1.) time varying reproduction number

	R0         ~ truncated(Normal(3., 1.), 1.5, 5.)
	R1         ~ truncated(Normal(.8, .1), .5, 1.1)
	σ_rt       ~ truncated(Normal(0.15, .05), 0, .25)
	latent_Rt  ~ RandomWalk(num_rt_steps, σ_rt, invlink(R1))

	Rt = TV(undef, num_time_steps)

	random_walks!(Rt, θ, predict, latent_Rt, R0, σ_rt)

	############# 2.) time varying infection ascertainment rate
	σ_iar      ~ truncated(Normal(0.05, .03), 0, .15)
	iar0       ~ Beta(1,10)
	latent_iar ~ RandomWalk(num_iar_steps, σ_iar, logit(iar0))

	iar = TV(undef, num_time_steps)

	iar!(iar, θ, predict, latent_iar, iar0, σ_iar)

	############ 3.) infection dynamics
	τ  ~ Exponential(1 / 0.03) # `Exponential` has inverse parameterization of the one in Stan
	T  = typeof(τ)
	y  ~ truncated(Exponential(τ),T(1),T(1000))

	newly_infected       = TV(undef, num_time_steps)
	cumulative_infected  = TV(undef, num_time_steps)
	effective_Rt         = TV(undef, num_time_steps)

	# infections!(newly_infected, cumulative_infected, θ, τ, y, Rt)
	infections!(newly_infected, cumulative_infected, effective_Rt, θ, y, Rt)
	########### 4.) derive observables
	μ_i2d ~ truncated(Normal(20., 2.), 10, 25)
	μ_i2c ~ truncated(Normal(4.5, 1.), 2, 10)
	ifr   ~ truncated(Normal(6/1000, 3/1000), 2/1000, 10/1000)
	# ihr   ~ truncated(Normal(1/100,1/100),.1/100,5/100)

	### day of week effect
	weekdayeffect_simplex ~ Dirichlet([7,7,7,7,7,3,1])
	weekdayeffect = TV(undef, 7)
	weekdayeffect!(weekdayeffect, weekdayeffect_simplex)

	### holiday effect
	holidayeffect ~ Beta(1,1)

	########### 3.) observation model
	expected_daily_deaths = TV(undef, num_time_steps)
	expected_daily_cases  = TV(undef, num_time_steps)

	ϕ_c  ~ truncated(Normal(50, 10), 30, Inf)
	ϕ_d  ~ truncated(Normal(40, 10), 10, Inf)

    cases_observation_model = WeekdayHolidayObsModel(
        casemodel, μ_i2c, iar, ϕ_c, expected_daily_cases, weekdayeffect, holidayeffect
    )
    deaths_observation_model = SimpleObsModel(deathmodel, μ_i2d, ifr, ϕ_d, expected_daily_deaths)


	expected!(deaths_observation_model, newly_infected)
	expected!(cases_observation_model, newly_infected)

	# ## 4.3) seroprevalence study
	# serodelay = 16 # 14d self deferral + 16d seroconversion (see Erikstrup et al. https://doi.org/10.1093/cid/ciaa849)
	# expected_seropositive[1:serodelay]    .= zero(V)
	# expected_seropositive[serodelay+1:end] = cumulative_infected[1:end-serodelay]

	# select_seroprev = 2:4
	# seroprev_σ ~ arraydist( InverseGamma2.(seroprev_std[ select_seroprev ], Ref(.1)) )
    # seroprev_μ = expected_seropositive[ seroprev_idx[ select_seroprev ] ]
	# seroprev_mean[select_seroprev] ~ MvNormal(seroprev_μ, seroprev_σ) # select only early observations


	########### 4.) compare model to observations

	ℓ  = zero(V)
    ℓ += logpdf(cases_observation_model, θ.cases)
    ℓ += logpdf(deaths_observation_model, θ.deaths)
	Turing.@addlogprob! ℓ

	return (
		newly_infected        = newly_infected,
		expected_daily_deaths = expected_daily_deaths,
		expected_daily_cases  = expected_daily_cases,
		Rt                    = Rt,
		effective_Rt          = effective_Rt,
		cumulative_infected   = cumulative_infected,
		iar                   = iar
	)
end

# ================================================================================================
#                   Cases + Hospit + Deaths
# =================================================================================================
@model function model_v3(
    θ,
    predict,
	::Type{TV} = Vector{Float64},
	::Type{V}  = Float64;
) where {TV, V}

	@unpack num_observations, num_total_days, num_rt_steps, num_iar_steps, invlink, deathmodel, casemodel, hospitmodel = θ
    # If we don't want to predict the future, we only need to compute up-to time-step `num_obs_countries[m]`
    num_time_steps   = predict ? num_total_days : num_observations
	############# 1.) time varying reproduction number

	R0         ~ truncated(Normal(3., 1.), 1.5, 5.)
	R1         ~ truncated(Normal(.8, .1), .5, 1.1)
	σ_rt       ~ truncated(Normal(0.15, .05), 0, .25)
	latent_Rt  ~ RandomWalk(num_rt_steps, σ_rt, invlink(R1))

	Rt = TV(undef, num_time_steps)

	random_walks!(Rt, θ, predict, latent_Rt, R0, σ_rt)

	############# 2.) time varying infection ascertainment rate
	σ_iar      ~ truncated(Normal(0.05, .03), 0, .15)
	iar0       ~ Beta(1,10)
	latent_iar ~ RandomWalk(num_iar_steps, σ_iar, logit(iar0))

	iar = TV(undef, num_time_steps)

	iar!(iar, θ, predict, latent_iar, iar0, σ_iar)

	############ 3.) infection dynamics
	τ  ~ Exponential(1 / 0.03) # `Exponential` has inverse parameterization of the one in Stan
	T  = typeof(τ)
	y  ~ truncated(Exponential(τ),T(1),T(1000))

	newly_infected       = TV(undef, num_time_steps)
	cumulative_infected  = TV(undef, num_time_steps)
	effective_Rt         = TV(undef, num_time_steps)

	# infections!(newly_infected, cumulative_infected, θ, τ, y, Rt)
	infections!(newly_infected, cumulative_infected, effective_Rt, θ, y, Rt)
	########### 4.) derive observables
	μ_i2c ~ truncated(Normal(4.5, 1.), 2, 10)
	μ_i2h ~ truncated(Normal(12., 3.), 9, 16)
	μ_i2d ~ truncated(Normal(20., 2.), 10, 25)
	ihr   ~ truncated(Normal(1/100,1/100),.1/100,5/100)
	ifr   ~ truncated(Normal(6/1000, 3/1000), 2/1000, 10/1000)

	### day of week effect
	weekdayeffect_simplex ~ Dirichlet([7,7,7,7,7,3,1])
	weekdayeffect = TV(undef, 7)
	weekdayeffect!(weekdayeffect, weekdayeffect_simplex)

	### holiday effect
	holidayeffect ~ Beta(1,1)

	########### 3.) observation model
	expected_daily_cases  = TV(undef, num_time_steps)
	expected_daily_hospit = TV(undef, num_time_steps)
	expected_daily_deaths = TV(undef, num_time_steps)

	ϕ_c  ~ truncated(Normal(50, 10), 30, Inf)
	ϕ_h  ~ truncated(Normal(50, 10), 20, Inf)
	ϕ_d  ~ truncated(Normal(40, 10), 10, Inf)

    cases_observation_model = WeekdayHolidayObsModel(
        casemodel, μ_i2c, iar, ϕ_c, expected_daily_cases, weekdayeffect, holidayeffect
    )
	hospit_observation_model = SimpleObsModel(hospitmodel, μ_i2h, ihr, ϕ_h, expected_daily_hospit)
    deaths_observation_model = SimpleObsModel(deathmodel, μ_i2d, ifr, ϕ_d, expected_daily_deaths)

	expected!(cases_observation_model, newly_infected)
	expected!(hospit_observation_model, newly_infected)
	expected!(deaths_observation_model, newly_infected)

	# ## 4.3) seroprevalence study
	# serodelay = 16 # 14d self deferral + 16d seroconversion (see Erikstrup et al. https://doi.org/10.1093/cid/ciaa849)
	# expected_seropositive[1:serodelay]    .= zero(V)
	# expected_seropositive[serodelay+1:end] = cumulative_infected[1:end-serodelay]

	# select_seroprev = 2:4
	# seroprev_σ ~ arraydist( InverseGamma2.(seroprev_std[ select_seroprev ], Ref(.1)) )
    # seroprev_μ = expected_seropositive[ seroprev_idx[ select_seroprev ] ]
	# seroprev_mean[select_seroprev] ~ MvNormal(seroprev_μ, seroprev_σ) # select only early observations


	########### 4.) compare model to observations

	ℓ  = zero(V)
    ℓ += logpdf(cases_observation_model, θ.cases)
    ℓ += logpdf(hospit_observation_model, θ.hospit)
	ℓ += logpdf(deaths_observation_model, θ.deaths)
	Turing.@addlogprob! ℓ

	return (
		newly_infected        = newly_infected,
		expected_daily_cases  = expected_daily_cases,
		expected_daily_hospit = expected_daily_hospit,
		expected_daily_deaths = expected_daily_deaths,
		Rt                    = Rt,
		effective_Rt          = effective_Rt,
		cumulative_infected   = cumulative_infected,
		iar                   = iar
	)
end
# =============================================================================
#                      model_cases
# ==============================================================================
@model function model_cases(
    θ,
    predict,
	::Type{TV} = Vector{Float64},
	::Type{V}  = Float64;
) where {TV, V}

	@unpack num_observations, num_total_days, num_rt_steps, invlink, deathmodel, casemodel, seromodel, num_covariates = θ
    # If we don't want to predict the future, we only need to compute up-to time-step `num_obs_countries[m]`
    num_time_steps   = predict ? num_total_days : num_observations
	num_time_steps_2 = predict ? num_total_days : deathmodel.stop
	############# 2.) time varying reproduction number

	R0         ~ truncated(Normal(3., 1.), 1.5, 5.)
	R1         ~ truncated(Normal(.8, .1), .5, 1.1)
	σ_rt       ~ truncated(Normal(0.15, .05), 0, .25)
	latent_Rt  ~ RandomWalk(num_rt_steps, σ_rt, invlink(R1))

	Rt = TV(undef, num_time_steps)

	if num_covariates > 0
		effect ~ filldist(Exponential(0.2), num_covariates)
		if θ.semiparametric
			semiparametric!(Rt, θ, predict, latent_Rt, R0, σ_rt, effect)
		else
			mixed!(Rt, θ, predict, latent_Rt, R0, σ_rt, effect)
		end
	else
		random_walks!(Rt, θ, predict, latent_Rt, R0, σ_rt)
	end

	############ 3.) infection dynamics
	τ  ~ Exponential(1 / 0.03) # `Exponential` has inverse parameterization of the one in Stan
	T  = typeof(τ)
	y  ~ truncated(Exponential(τ),T(1),T(1000))

	newly_infected       = TV(undef, num_time_steps)
	cumulative_infected  = TV(undef, num_time_steps)
	effective_Rt         = TV(undef, num_time_steps)

	# infections!(newly_infected, cumulative_infected, θ, τ, y, Rt)
	infections!(newly_infected, cumulative_infected, effective_Rt, θ, y, Rt)
	########### 4.) derive observables
	#1
	# μ_i2d ~ truncated(Normal(20., 4.), 10, 30)
	# μ_i2c ~ truncated(Normal(4.5, 1.), 2, 10)
	#2
	μ_i2d ~ truncated(Normal(20., 2.), 10, 25)
	μ_i2c ~ truncated(Normal(2., 1.), 0, 5)
	iar   ~ Beta(1,10)
	ifr   ~ truncated(Normal(6/1000, 3/1000), 2/1000, 10/1000)
	# ihr   ~ truncated(Normal(1/100,1/100),.1/100,5/100)

	### day of week effect
	weekdayeffect_simplex ~ Dirichlet([7,7,7,7,7,3,1])
	weekdayeffect = TV(undef, 7)
	weekdayeffect!(weekdayeffect, weekdayeffect_simplex)

	### holiday effect
	holidayeffect ~ Beta(1,1)

	########### 3.) observation model
	expected_daily_deaths = TV(undef, num_time_steps_2)
	expected_daily_cases  = TV(undef, num_time_steps)

	ϕ_c  ~ truncated(InverseGamma2(80, .2), 30, Inf)
	ϕ_d  ~ truncated(Normal(40, 10), 10, Inf)

    cases_observation_model = WeekdayHolidayObsModel(
        casemodel, μ_i2c, iar, ϕ_c, expected_daily_cases, weekdayeffect, holidayeffect
    )
    deaths_observation_model = SimpleObsModel(deathmodel, μ_i2d, ifr, ϕ_d, expected_daily_deaths)
	sero_observation_model   = SimpleSeroObsModel2(seromodel, cumulative_infected)

	expected!(cases_observation_model, newly_infected)
	expected!(deaths_observation_model, newly_infected)

	########### 4.) compare model to observations

	ℓ  = zero(V)
    ℓ += logpdf(cases_observation_model, θ.cases)
    ℓ += logpdf(deaths_observation_model, θ.deaths)
	ℓ += logpdf(sero_observation_model, θ.sero)

	Turing.@addlogprob! ℓ

	return (
		newly_infected = newly_infected,
		expected_daily_deaths = expected_daily_deaths,
		expected_daily_cases = expected_daily_cases,
		Rt = Rt,
		effective_Rt = effective_Rt,
		expected_seropos = cumulative_infected
	)
end

# =============================================================================
#                      model_hospit
# ==============================================================================
@model function model_hospit(
    θ,
    predict,
	::Type{TV} = Vector{Float64},
	::Type{V}  = Float64;
) where {TV, V}

	@unpack num_observations, num_total_days, num_rt_steps, invlink, hospitmodel, seromodel, num_covariates = θ
    # If we don't want to predict the future, we only need to compute up-to time-step `num_obs_countries[m]`
    num_time_steps = predict ? num_total_days : num_observations

	############# 2.) time varying reproduction number
	R0         ~ truncated(Normal(3., 1.), 1.5, 5.)
	R1         ~ truncated(Normal(.8, .1), .5, 1.1)
	σ_rt       ~ truncated(Normal(0.15, .05), 0, .25)
	latent_Rt  ~ RandomWalk(num_rt_steps, σ_rt, invlink(R1))
	Rt = TV(undef, num_time_steps)
	if num_covariates > 0
		effect ~ filldist(Exponential(0.2), num_covariates)
		if θ.semiparametric
			semiparametric!(Rt, θ, predict, latent_Rt, R0, σ_rt, effect)
		else
			mixed!(Rt, θ, predict, latent_Rt, R0, σ_rt, effect)
		end
	else
		random_walks!(Rt, θ, predict, latent_Rt, R0, σ_rt)
	end

	############ 3.) infection dynamics
	τ  ~ Exponential(1 / 0.03) # `Exponential` has inverse parameterization of the one in Stan
	T  = typeof(τ)
	y  ~ truncated(Exponential(τ),T(1),T(1000))

	newly_infected       = TV(undef, num_time_steps)
	cumulative_infected  = TV(undef, num_time_steps)
	effective_Rt         = TV(undef, num_time_steps)

	# infections!(newly_infected, cumulative_infected, θ, τ, y, Rt)
	infections!(newly_infected, cumulative_infected, effective_Rt, θ, y, Rt)

	########### 4.) derive observables
	#1
    # μ ~ truncated(Normal(12., 1.), 9, 16)
	#2
	μ ~ truncated(Normal(9., 1.), 7, 15)
	α ~ truncated(Normal(1.8/100,0.5/100),1/100,5/100)
	ϕ ~ truncated(Normal(50, 10), 20, Inf)
	# σ_s = seromodel.std


	expected = TV(undef, num_time_steps)
	hospit_observation_model = SimpleObsModel(hospitmodel, μ, α, ϕ, expected)
	sero_observation_model   = SimpleSeroObsModel2(seromodel, cumulative_infected)

	expected!(hospit_observation_model, newly_infected)

	########### 4.) compare model to observations
	## 4.1) observe hospitalizations
	ℓ  = zero(V)
	ℓ += logpdf(hospit_observation_model, θ.hospit)
	ℓ += logpdf(sero_observation_model, θ.sero)
	Turing.@addlogprob! ℓ

	return (
		newly_infected = newly_infected,
		expected_daily_hospit = expected,
		Rt = Rt,
		effective_Rt = effective_Rt,
		expected_seropos = cumulative_infected
	)
end

# =============================================================================
#                      model_deaths
# ==============================================================================
@model function model_deaths(
    θ,
    predict,
	::Type{TV} = Vector{Float64},
	::Type{V}  = Float64;
) where {TV, V}

	@unpack num_observations, num_total_days, num_rt_steps, invlink, deathmodel, seromodel, num_covariates = θ
    # If we don't want to predict the future, we only need to compute up-to time-step `num_obs_countries[m]`
    num_time_steps = predict ? num_total_days : num_observations

	############# 2.) time varying reproduction number
	R0         ~ truncated(Normal(3., 1.), 1.5, 5.)
	R1         ~ truncated(Normal(.8, .1), .5, 1.1)
	σ_rt       ~ truncated(Normal(0.15, .05), 0, .25)
	latent_Rt  ~ RandomWalk(num_rt_steps, σ_rt, invlink(R1))

	Rt = TV(undef, num_time_steps)

	if num_covariates > 0
		effect ~ filldist(Exponential(0.2), num_covariates)
		if θ.semiparametric
			semiparametric!(Rt, θ, predict, latent_Rt, R0, σ_rt, effect)
		else
			mixed!(Rt, θ, predict, latent_Rt, R0, σ_rt, effect)
		end
	else
		random_walks!(Rt, θ, predict, latent_Rt, R0, σ_rt)
	end

	############ 3.) infection dynamics
	τ  ~ Exponential(1 / 0.03) # `Exponential` has inverse parameterization of the one in Stan
	T  = typeof(τ)
	y  ~ truncated(Exponential(τ),T(1),T(1000))

	newly_infected       = TV(undef, num_time_steps)
	cumulative_infected  = TV(undef, num_time_steps)
	effective_Rt         = TV(undef, num_time_steps)

	# infections!(newly_infected, cumulative_infected, θ, τ, y, Rt)
	infections!(newly_infected, cumulative_infected, effective_Rt, θ, y, Rt)

	########### 4.) derive observables
	μ ~ truncated(Normal(20., 2.), 10, 25)
	α ~ truncated(Normal(6/1000, 3/1000), 1/1000, 10/1000)
	ϕ ~ truncated(Normal(40, 10), 10, Inf)

	expected = TV(undef, num_time_steps)
	deaths_observation_model = SimpleObsModel(deathmodel, μ, α, ϕ, expected)
	sero_observation_model   = SimpleSeroObsModel2(seromodel, cumulative_infected)
	expected!(deaths_observation_model, newly_infected)

	########### 4.) compare model to observations
	## 4.1) observe deaths
	ℓ  = zero(V)
	ℓ += logpdf(deaths_observation_model, θ.deaths)
	ℓ += logpdf(sero_observation_model, θ.sero)
	Turing.@addlogprob! ℓ

	return (
		newly_infected = newly_infected,
		expected_daily_deaths = expected,
		Rt = Rt,
		effective_Rt = effective_Rt,
		expected_seropos = cumulative_infected
	)
end
