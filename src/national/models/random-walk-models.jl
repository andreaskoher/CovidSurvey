@model function model_v2(
	num_impute,        # [Int] num. of days for which to impute infections
    num_total_days,    # [Int] days of observed data + num. of days to forecast
    cases,            # [AbstractVector{<:AbstractVector{<:Int}}] reported infected
    deaths,            # [AbstractVector{<:AbstractVector{<:Int}}] reported deaths; rows indexed by i > N contain -1 and should be ignored
    π,                 # [AbstractVector{<:AbstractVector{<:Real}}] h * s
    π2,                 # [AbstractVector{<:AbstractVector{<:Real}}] h * s
    epidemic_start,    # [AbstractVector{<:Int}]
    population,        # [AbstractVector{<:Real}]
    serial_intervals,  # [AbstractVector{<:Real}] fixed pre-calculated serial interval (SI) using empirical data from Neil
	num_iar_steps,         # [Int] number of weeks (21)
    iar_index,        # [Vector{Array{Int64,1}}] Macro region index for each state
    lockdown,         # [Int] number of weeks (21)
    num_case_obs,       # [Int] 21 = max_date ("2020-05-11") - testing_date (2020-05-11) see publication
	weekday,
	holiday,
	π3,                 # [AbstractVector{<:AbstractVector{<:Real}}] h * s
    hospit,
	seroprev_mean,
    seroprev_std,
    seroprev_idx,
	π4,
    predict=false,     # [Bool] if `false`, will only compute what's needed to `observe` but not more
    ::Type{TV} = Vector{Float64},
	::Type{V}  = Float64;
	invlink = log,#KLogistic(V(3))
	link    = exp,#KLogit(V(k))
) where {TV, V}
    ############ 0.) some definitions
	num_obs        = length(cases)
    num_Rt_steps   = num_obs - lockdown
	num_sero       = length(seroprev_mean)
    cases_start    = num_obs-num_case_obs
	# If we don't want to predict the future, we only need to compute up-to time-step `num_obs_states[m]`
    num_time_steps = predict ? num_total_days : num_obs

    ############# 2.) time varying reproduction number
	R0        ~ truncated(Normal(3., 1.), 1., 5.)
	R1        ~ truncated(Normal(.8, .1), .5, 1.1)
	σ_rt      ~ truncated(Normal(0.05, .03), 0, .3)
	latent_Rt ~ RandomWalk(num_Rt_steps, σ_rt, invlink(R1))

    Rt = TV(undef, num_time_steps)
    Rt[1:lockdown] .= R0
	Rt[lockdown+1:num_obs] = link.(latent_Rt)
	if predict #if prediction
		num_pred_steps = num_total_days - num_obs
		latent_Rt_pred ~ RandomWalk(num_pred_steps, σ_rt, latent_Rt[end])
		Rt[num_obs+1:num_total_days] = link.(latent_Rt_pred)
	end

    ############ 3.) infection dynamics
	## 3.1) initialize serial interval
	num_si   = 15
	si       = serial_intervals
	# μ_si   ~ truncated(Normal(5.06, 0.33), 0, Inf)
	# σ_si   = 2.11
	# si     = pdf.( Ref(GammaMeanStd(μ_si, σ_si)), 1:num_si )
	# si    /= sum(si)

	## 3.2) seeding
	τ ~ Exponential(1 / 0.03) # `Exponential` has inverse parameterization of the one in Stan
	T = typeof(τ)
    y ~ truncated(Exponential(τ),T(0),T(1000))

	newly_infected      = TV(undef, num_time_steps)
	cumulative_infected = TV(undef, num_time_steps)
	newly_infected[1]   = y
	cumulative_infected[1] = zero(V)
	for t in 2:num_impute
		newly_infected[t] = y
		cumulative_infected[t] = cumulative_infected[t-1] + y
	end

    ## 3.3) run disease dynamics
	for t = (num_impute + 1):num_time_steps
		# Update cumulatively infected
		cumulative_infected[t] = cumulative_infected[t-1] + newly_infected[t - 1]
		# Adjusts for portion of population that is susceptible
		St = max(population - cumulative_infected[t], 0) / population
		# effective number of infectious individuals
		effectively_infectious = sum(newly_infected[τ] * si[t - τ] for τ = (t - 1):-1:max(t-num_si,1))
		# number of new infections (unobserved)
		newly_infected[t] = St * Rt[t] * effectively_infectious
	end

	########### 4.) derive observables
	## 4.1) initialize delay distributions
	num_i2c = 40
	num_i2h = 40
	num_i2d = 60

	μ_i2c ~ truncated(Normal(4.5, 1.), 3, 7)
    μ_i2h ~ truncated(Normal(12., 3.), 9, 16)
    μ_i2d ~ truncated(Normal(20., 4.), 10, 30)

	ϕ_i2c = 5.41
	ϕ_i2h = 5.41
	ϕ_i2d = 14.26

	i2c = pdf.( Ref(NegativeBinomial2(μ_i2c, ϕ_i2c)), 1:num_i2c )
	i2h = pdf.( Ref(NegativeBinomial2(μ_i2h, ϕ_i2h)), 1:num_i2h )
	i2d = pdf.( Ref(NegativeBinomial2(μ_i2d, ϕ_i2d)), 1:num_i2d )

	i2c /= sum(i2c)
	i2h /= sum(i2h)
	i2d /= sum(i2d)

	### day of week effect
	weekdayeffect_simplex ~ Dirichlet([7,7,7,7,7,3,1])
	num_days = 7
	weekdayeffect = TV(undef, num_days)
	for i in 1:num_days
		weekdayeffect[i] = weekdayeffect_simplex[i] * num_days
	end

	### holiday effect
	holidayeffect ~ Beta(1,1)

	############# 3.) time varying infection ascertainment rate
	σ_iar      ~ truncated(Normal(0.05, .03), 0, .15)
	iar0       ~ Beta(1,10)
	latent_iar ~ RandomWalk(num_iar_steps, σ_iar, logit(iar0))
	iar        = logistic.( latent_iar[iar_index[1:num_time_steps]] )

    ### 4.2) derive observables from infections
	ifr ~ truncated(Normal(6/1000, 3/1000), 1/1000, 10/1000)
	ihr ~ truncated(Normal(1/100,1/100),.1/100,5/100)

    expected_daily_cases     = TV(undef, num_time_steps)
    expected_daily_hospit    = TV(undef, num_time_steps)
    expected_daily_deaths    = TV(undef, num_time_steps)
	expected_seropositive    = TV(undef, num_time_steps)

	expected_daily_cases[1]  = 1e-15 * newly_infected[1]
    expected_daily_hospit[1] = 1e-15 * newly_infected[1]
    expected_daily_deaths[1] = 1e-15 * newly_infected[1]
	for t = 2:num_time_steps
		periodiceffect = (1 - holidayeffect * holiday[t]) * weekdayeffect[weekday[t]]
		expected_daily_cases[t]  = iar[t] * sum(newly_infected[τ] * i2c[t - τ] for τ = (t - 1):-1:max(t-num_i2c,1)) * periodiceffect
        expected_daily_hospit[t] = ihr * sum(newly_infected[τ] * i2h[t - τ] for τ = (t - 1):-1:max(t-num_i2h,1))
        expected_daily_deaths[t] = ifr * sum(newly_infected[τ] * i2d[t - τ] for τ = (t - 1):-1:max(t-num_i2d,1))
	end

	serodelay = 16 # 14d self deferral + 16d seroconversion (see Erikstrup et al. https://doi.org/10.1093/cid/ciaa849)
	expected_seropositive[1:serodelay]    .= zero(V)
	expected_seropositive[serodelay+1:end] = cumulative_infected[1:end-serodelay]

	########### 4.) compare model to observations
	## 4.1) observe cases
	if num_case_obs > 0
		ϕ_c  ~ truncated(Normal(25, 10), 10, Inf)
		ts_c = (num_obs-num_case_obs):num_obs
		μs_c = expected_daily_cases[ts_c]
		!all( 0 .< μs_c .< population ) && ( @error "expected_daily_cases < 0"; Turing.@addlogprob! -Inf; return)
		# (ϕ_c <= 0) && (@error "ϕ2 <= 0"; Turing.@addlogprob! -Inf; return)
		cases[ts_c] ~ arraydist(NegativeBinomial2.(μs_c, ϕ_c))
	end

	# 4.2) observe hospitalizations
	# ϕ_h   ~ truncated(Normal(25, 10), 10, 60)
	# ts_h  = epidemic_start:num_obs
	# μs_h  = expected_daily_hospit[ts_h]
	# !all( 0 .< μs_h .< population ) && (@error "expected_daily_hospit"; Turing.@addlogprob! -Inf; return)
	# # (ϕ_h <= 0) && (@error "ϕ <= 0"; Turing.@addlogprob! -Inf; return)
	# hospit[ts_h] ~ arraydist(NegativeBinomial2.(μs_h, ϕ_h))

	## 4.3) observe deaths
	ϕ_d   ~ truncated(Normal(20, 10), 0, Inf)
    ts_d  = epidemic_start:num_obs
	μs_d  = expected_daily_deaths[ts_d]
	!all( 0 .< μs_d .< population ) && (@error "expected_daily_deaths"; Turing.@addlogprob! -Inf; return)
	# (ϕ_d <= 0) && (@error "ϕ <= 0"; Turing.@addlogprob! -Inf; return)
	deaths[ts_d] ~ arraydist(NegativeBinomial2.(μs_d, ϕ_d))

    ## 4.3) seroprevalence study
    select_seroprev = 2:4
	seroprev_σ ~ arraydist( InverseGamma2.(seroprev_std[ select_seroprev ], Ref(.1)) )
    seroprev_μ = expected_seropositive[ seroprev_idx[ select_seroprev ] ]
	seroprev_mean[select_seroprev] ~ MvNormal(seroprev_μ, seroprev_σ) # select only early observations


    return (
		expected_daily_cases  = expected_daily_cases,
        expected_daily_deaths = expected_daily_deaths,
        expected_daily_hospit = expected_daily_hospit,
        Rt = Rt,
		expected_seropositive = expected_seropositive,
		iar = iar,
    )
end

## ================================================================================================
#                   Cases + Hospit + Deaths
# =================================================================================================

@model function model_v3(
	num_impute,        # [Int] num. of days for which to impute infections
    num_total_days,    # [Int] days of observed data + num. of days to forecast
    cases,            # [AbstractVector{<:AbstractVector{<:Int}}] reported infected
    deaths,            # [AbstractVector{<:AbstractVector{<:Int}}] reported deaths; rows indexed by i > N contain -1 and should be ignored
    π,                 # [AbstractVector{<:AbstractVector{<:Real}}] h * s
    π2,                 # [AbstractVector{<:AbstractVector{<:Real}}] h * s
    epidemic_start,    # [AbstractVector{<:Int}]
    population,        # [AbstractVector{<:Real}]
    serial_intervals,  # [AbstractVector{<:Real}] fixed pre-calculated serial interval (SI) using empirical data from Neil
	num_iar_steps,         # [Int] number of weeks (21)
    iar_index,        # [Vector{Array{Int64,1}}] Macro region index for each state
    lockdown,         # [Int] number of weeks (21)
    num_case_obs,       # [Int] 21 = max_date ("2020-05-11") - testing_date (2020-05-11) see publication
	weekday,
	holiday,
	π3,                 # [AbstractVector{<:AbstractVector{<:Real}}] h * s
    hospit,
	seroprev_mean,
    seroprev_std,
    seroprev_idx,
	π4,
    predict=false,     # [Bool] if `false`, will only compute what's needed to `observe` but not more
    ::Type{TV} = Vector{Float64},
	::Type{V}  = Float64;
	invlink = log,#KLogistic(V(3))
	link    = exp,#KLogit(V(k))
) where {TV, V}
    ############ 0.) some definitions
	num_obs        = length(cases)
    num_Rt_steps   = num_obs - lockdown
	num_sero       = length(seroprev_mean)
    cases_start    = num_obs-num_case_obs
	# If we don't want to predict the future, we only need to compute up-to time-step `num_obs_states[m]`
    num_time_steps = predict ? num_total_days : num_obs

    ############# 2.) time varying reproduction number
	R0        ~ truncated(Normal(3., 1.), 1., 5.)
	R1        ~ truncated(Normal(.8, .1), .5, 1.1)
	σ_rt      ~ truncated(Normal(0.05, .03), 0, .3)
	latent_Rt ~ RandomWalk(num_Rt_steps, σ_rt, invlink(R1))

    Rt = TV(undef, num_time_steps)
    Rt[1:lockdown] .= R0
	Rt[lockdown+1:num_obs] = link.(latent_Rt)
	if predict #if prediction
		num_pred_steps = num_total_days - num_obs
		latent_Rt_pred ~ RandomWalk(num_pred_steps, σ_rt, latent_Rt[end])
		Rt[num_obs+1:num_total_days] = link.(latent_Rt_pred)
	end

    ############ 3.) infection dynamics
	## 3.1) initialize serial interval
	num_si   = 15
	si       = serial_intervals
	# μ_si   ~ truncated(Normal(5.06, 0.33), 0, Inf)
	# σ_si   = 2.11
	# si     = pdf.( Ref(GammaMeanStd(μ_si, σ_si)), 1:num_si )
	# si    /= sum(si)

	## 3.2) seeding
	τ ~ Exponential(1 / 0.03) # `Exponential` has inverse parameterization of the one in Stan
	T = typeof(τ)
    y ~ truncated(Exponential(τ),T(0),T(1000))

	newly_infected      = TV(undef, num_time_steps)
	cumulative_infected = TV(undef, num_time_steps)
	newly_infected[1]   = y
	cumulative_infected[1] = zero(V)
	for t in 2:num_impute
		newly_infected[t] = y
		cumulative_infected[t] = cumulative_infected[t-1] + y
	end

    ## 3.3) run disease dynamics
	for t = (num_impute + 1):num_time_steps
		# Update cumulatively infected
		cumulative_infected[t] = cumulative_infected[t-1] + newly_infected[t - 1]
		# Adjusts for portion of population that is susceptible
		St = max(population - cumulative_infected[t], 0) / population
		# effective number of infectious individuals
		effectively_infectious = sum(newly_infected[τ] * si[t - τ] for τ = (t - 1):-1:max(t-num_si,1))
		# number of new infections (unobserved)
		newly_infected[t] = St * Rt[t] * effectively_infectious
	end

	########### 4.) derive observables
	## 4.1) initialize delay distributions
	num_i2c = 40
	num_i2h = 40
	num_i2d = 60

	μ_i2c ~ truncated(Normal(4.5, 1.), 3, 7)
    μ_i2h ~ truncated(Normal(12., 3.), 9, 16)
    μ_i2d ~ truncated(Normal(20., 4.), 10, 30)

	ϕ_i2c = 5.41
	ϕ_i2h = 5.41
	ϕ_i2d = 14.26

	i2c = pdf.( Ref(NegativeBinomial2(μ_i2c, ϕ_i2c)), 1:num_i2c )
	i2h = pdf.( Ref(NegativeBinomial2(μ_i2h, ϕ_i2h)), 1:num_i2h )
	i2d = pdf.( Ref(NegativeBinomial2(μ_i2d, ϕ_i2d)), 1:num_i2d )

	i2c /= sum(i2c)
	i2h /= sum(i2h)
	i2d /= sum(i2d)

	### day of week effect
	weekdayeffect_simplex ~ Dirichlet([7,7,7,7,7,3,1])
	num_days = 7
	weekdayeffect = TV(undef, num_days)
	for i in 1:num_days
		weekdayeffect[i] = weekdayeffect_simplex[i] * num_days
	end

	### holiday effect
	holidayeffect ~ Beta(1,1)

	############# 3.) time varying infection ascertainment rate
	σ_iar      ~ truncated(Normal(0.05, .03), 0, .15)
	iar0       ~ Beta(1,10)
	latent_iar ~ RandomWalk(num_iar_steps, σ_iar, logit(iar0))
	iar        = logistic.( latent_iar[iar_index[1:num_time_steps]] )

    ### 4.2) derive observables from infections
	ifr ~ truncated(Normal(6/1000, 3/1000), 1/1000, 10/1000)
	ihr ~ truncated(Normal(1/100,1/100),.1/100,5/100)

    expected_daily_cases     = TV(undef, num_time_steps)
    expected_daily_hospit    = TV(undef, num_time_steps)
    expected_daily_deaths    = TV(undef, num_time_steps)
	expected_seropositive    = TV(undef, num_time_steps)

	expected_daily_cases[1]  = 1e-15 * newly_infected[1]
    expected_daily_hospit[1] = 1e-15 * newly_infected[1]
    expected_daily_deaths[1] = 1e-15 * newly_infected[1]
	for t = 2:num_time_steps
		periodiceffect = (1 - holidayeffect * holiday[t]) * weekdayeffect[weekday[t]]
		expected_daily_cases[t]  = iar[t] * sum(newly_infected[τ] * i2c[t - τ] for τ = (t - 1):-1:max(t-num_i2c,1)) * periodiceffect
        expected_daily_hospit[t] = ihr * sum(newly_infected[τ] * i2h[t - τ] for τ = (t - 1):-1:max(t-num_i2h,1))
        expected_daily_deaths[t] = ifr * sum(newly_infected[τ] * i2d[t - τ] for τ = (t - 1):-1:max(t-num_i2d,1))
	end

	serodelay = 16 # 14d self deferral + 16d seroconversion (see Erikstrup et al. https://doi.org/10.1093/cid/ciaa849)
	expected_seropositive[1:serodelay]    .= zero(V)
	expected_seropositive[serodelay+1:end] = cumulative_infected[1:end-serodelay]

	########### 4.) compare model to observations
	## 4.1) observe cases
	if num_case_obs > 0
		ϕ_c  ~ truncated(Normal(25, 10), 10, Inf)
		ts_c = (num_obs-num_case_obs):num_obs
		μs_c = expected_daily_cases[ts_c]
		!all( 0 .< μs_c .< population ) && ( @error "expected_daily_cases < 0"; Turing.@addlogprob! -Inf; return)
		# (ϕ_c <= 0) && (@error "ϕ2 <= 0"; Turing.@addlogprob! -Inf; return)
		cases[ts_c] ~ arraydist(NegativeBinomial2.(μs_c, ϕ_c))
	end

	# 4.2) observe hospitalizations
	ϕ_h   ~ truncated(Normal(25, 10), 10, 60)
	ts_h  = epidemic_start:num_obs
	μs_h  = expected_daily_hospit[ts_h]
	!all( 0 .< μs_h .< population ) && (@error "expected_daily_hospit"; Turing.@addlogprob! -Inf; return)
	# (ϕ_h <= 0) && (@error "ϕ <= 0"; Turing.@addlogprob! -Inf; return)
	hospit[ts_h] ~ arraydist(NegativeBinomial2.(μs_h, ϕ_h))

	## 4.3) observe deaths
	ϕ_d   ~ truncated(Normal(20, 10), 0, Inf)
    ts_d  = epidemic_start:num_obs
	μs_d  = expected_daily_deaths[ts_d]
	!all( 0 .< μs_d .< population ) && (@error "expected_daily_deaths"; Turing.@addlogprob! -Inf; return)
	# (ϕ_d <= 0) && (@error "ϕ <= 0"; Turing.@addlogprob! -Inf; return)
	deaths[ts_d] ~ arraydist(NegativeBinomial2.(μs_d, ϕ_d))

    ## 4.3) seroprevalence study
    select_seroprev = 2:4
	seroprev_σ ~ arraydist( InverseGamma2.(seroprev_std[ select_seroprev ], Ref(.1)) )
    seroprev_μ = expected_seropositive[ seroprev_idx[ select_seroprev ] ]
	seroprev_mean[select_seroprev] ~ MvNormal(seroprev_μ, seroprev_σ) # select only early observations


    return (
		expected_daily_cases  = expected_daily_cases,
        expected_daily_deaths = expected_daily_deaths,
        expected_daily_hospit = expected_daily_hospit,
        Rt = Rt,
		expected_seropositive = expected_seropositive,
		iar = iar,
    )
end

## =============================================================================
#                      model_cases
# ==============================================================================
@model function model_cases(
    num_impute,        # [Int] num. of days for which to impute infections
    num_total_days,    # [Int] days of observed data + num. of days to forecast
    cases,            # [AbstractVector{<:AbstractVector{<:Int}}] reported infected
    deaths,            # [AbstractVector{<:AbstractVector{<:Int}}] reported deaths; rows indexed by i > N contain -1 and should be ignored
    π,                 # [AbstractVector{<:AbstractVector{<:Real}}] h * s
    π2,                 # [AbstractVector{<:AbstractVector{<:Real}}] h * s
    epidemic_start,    # [AbstractVector{<:Int}]
    population,        # [AbstractVector{<:Real}]
    serial_intervals,  # [AbstractVector{<:Real}] fixed pre-calculated serial interval (SI) using empirical data from Neil
	num_iar_steps,         # [Int] number of weeks (21)
    iar_index,        # [Vector{Array{Int64,1}}] Macro region index for each state
    lockdown,         # [Int] number of weeks (21)
    num_case_obs,       # [Int] 21 = max_date ("2020-05-11") - testing_date (2020-05-11) see publication
	weekday,
	holiday,
	π3,                 # [AbstractVector{<:AbstractVector{<:Real}}] h * s
    hospit,
	seroprev_mean,
    seroprev_std,
    seroprev_idx,
	π4,
    predict=false,     # [Bool] if `false`, will only compute what's needed to `observe` but not more
    ::Type{TV} = Vector{Float64},
	::Type{V}  = Float64;
	invlink = log,#KLogistic(V(3))
	link    = exp,#KLogit(V(k))
) where {TV, V}
    ############ 0.) some definitions
	num_obs        = length(cases)
    num_Rt_steps   = num_obs - lockdown
	num_sero       = length(seroprev_mean)
    cases_start    = num_obs-num_case_obs
	# If we don't want to predict the future, we only need to compute up-to time-step `num_obs_states[m]`
    num_time_steps = predict ? num_total_days : num_obs

    ############# 2.) time varying reproduction number
	R0        ~ truncated(Normal(3., 1.), 1., 5.)
	R1        ~ truncated(Normal(.8, .1), .5, 1.1)
	σ_rt      ~ truncated(Normal(0.05, .03), 0, .3)
	latent_Rt ~ RandomWalk(num_Rt_steps, σ_rt, invlink(R1))

    Rt = TV(undef, num_time_steps)
    Rt[1:lockdown] .= R0
	Rt[lockdown+1:num_obs] = link.(latent_Rt)
	if predict #if prediction
		num_pred_steps = num_total_days - num_obs
		latent_Rt_pred ~ RandomWalk(num_pred_steps, σ_rt, latent_Rt[end])
		Rt[num_obs+1:num_total_days] = link.(latent_Rt_pred)
	end

    ############ 3.) infection dynamics
	## 3.1) initialize serial interval
	num_si   = 15
	si       = serial_intervals
	# μ_si   ~ truncated(Normal(5.06, 0.33), 0, Inf)
	# σ_si   = 2.11
	# si     = pdf.( Ref(GammaMeanStd(μ_si, σ_si)), 1:num_si )
	# si    /= sum(si)

	## 3.2) seeding
	τ ~ Exponential(1 / 0.03) # `Exponential` has inverse parameterization of the one in Stan
	T = typeof(τ)
    y ~ truncated(Exponential(τ),T(0),T(1000))

	newly_infected      = TV(undef, num_time_steps)
	cumulative_infected = TV(undef, num_time_steps)
	newly_infected[1]   = y
	cumulative_infected[1] = zero(V)
	for t in 2:num_impute
		newly_infected[t] = y
		cumulative_infected[t] = cumulative_infected[t-1] + y
	end

    ## 3.3) run disease dynamics
	for t = (num_impute + 1):num_time_steps
		# Update cumulatively infected
		cumulative_infected[t] = cumulative_infected[t-1] + newly_infected[t - 1]
		# Adjusts for portion of population that is susceptible
		St = max(population - cumulative_infected[t], 0) / population
		# effective number of infectious individuals
		effectively_infectious = sum(newly_infected[τ] * si[t - τ] for τ = (t - 1):-1:max(t-num_si,1))
		# number of new infections (unobserved)
		newly_infected[t] = St * Rt[t] * effectively_infectious
	end

	########### 4.) derive observables
	## 4.1) initialize delay distributions
	num_i2c = 40
	num_i2h = 40
	num_i2d = 60

	μ_i2c ~ truncated(Normal(4.5, 1.), 3, 7)
    μ_i2h ~ truncated(Normal(12., 3.), 9, 16)
    μ_i2d ~ truncated(Normal(20., 4.), 10, 30)

	ϕ_i2c = 5.41
	ϕ_i2h = 5.41
	ϕ_i2d = 14.26

	i2c = pdf.( Ref(NegativeBinomial2(μ_i2c, ϕ_i2c)), 1:num_i2c )
	i2h = pdf.( Ref(NegativeBinomial2(μ_i2h, ϕ_i2h)), 1:num_i2h )
	i2d = pdf.( Ref(NegativeBinomial2(μ_i2d, ϕ_i2d)), 1:num_i2d )

	i2c /= sum(i2c)
	i2h /= sum(i2h)
	i2d /= sum(i2d)

	### day of week effect
	weekdayeffect_simplex ~ Dirichlet([7,7,7,7,7,3,1])
	num_days = 7
	weekdayeffect = TV(undef, num_days)
	for i in 1:num_days
		weekdayeffect[i] = weekdayeffect_simplex[i] * num_days
	end

	### holiday effect
	holidayeffect ~ Beta(1,1)

    ### 4.2) derive observables from infections
	ifr ~ truncated(Normal(6/1000, 3/1000), 1/1000, 10/1000)
	ihr ~ truncated(Normal(1/100,1/100),.1/100,5/100)
    iar ~ Beta(1,10)

    expected_daily_cases     = TV(undef, num_time_steps)
    expected_daily_hospit    = TV(undef, num_time_steps)
    expected_daily_deaths    = TV(undef, num_time_steps)
	expected_seropositive    = TV(undef, num_time_steps)

	expected_daily_cases[1]  = 1e-15 * newly_infected[1]
    expected_daily_hospit[1] = 1e-15 * newly_infected[1]
    expected_daily_deaths[1] = 1e-15 * newly_infected[1]
	for t = 2:num_time_steps
		periodiceffect = (1 - holidayeffect * holiday[t]) * weekdayeffect[weekday[t]]
		expected_daily_cases[t]  = iar * sum(newly_infected[τ] * i2c[t - τ] for τ = (t - 1):-1:max(t-num_i2c,1)) * periodiceffect
        expected_daily_hospit[t] = ihr * sum(newly_infected[τ] * i2h[t - τ] for τ = (t - 1):-1:max(t-num_i2h,1))
        expected_daily_deaths[t] = ifr * sum(newly_infected[τ] * i2d[t - τ] for τ = (t - 1):-1:max(t-num_i2d,1))
	end

	serodelay = 16 # 14d self deferral + 16d seroconversion (see Erikstrup et al. https://doi.org/10.1093/cid/ciaa849)
	expected_seropositive[1:serodelay]    .= zero(V)
	expected_seropositive[serodelay+1:end] = cumulative_infected[1:end-serodelay]

	########### 4.) compare model to observations
	## 4.1) observe cases
	ϕ_c  ~ truncated(Normal(25, 10), 0, Inf)
	ts_c = cases_start:num_obs
	μs_c = expected_daily_cases[ts_c]
	!all( 0 .< μs_c .< population ) && ( @error "expected_daily_cases < 0"; Turing.@addlogprob! -Inf; return)
	cases[ts_c] ~ arraydist(NegativeBinomial2.(μs_c, ϕ_c))

    ## 4.3) observe deaths
	ϕ_d   ~ truncated(Normal(20, 10), 0, Inf)
    ts_d  = epidemic_start:cases_start-1
	μs_d  = expected_daily_deaths[ts_d]
	!all( 0 .< μs_d .< population ) && (@error "expected_daily_deaths"; Turing.@addlogprob! -Inf; return)
	deaths[ts_d] ~ arraydist(NegativeBinomial2.(μs_d, ϕ_d))

    ## 4.3) seroprevalence study
    select_seroprev = 2:4
	seroprev_σ ~ arraydist( InverseGamma2.(seroprev_std[ select_seroprev ], Ref(.1)) )
    seroprev_μ = expected_seropositive[ seroprev_idx[ select_seroprev ] ]
	seroprev_mean[select_seroprev] ~ MvNormal(seroprev_μ, seroprev_σ) # select only early observations


    return (
		expected_daily_cases  = expected_daily_cases,
        expected_daily_deaths = expected_daily_deaths,
        expected_daily_hospit = expected_daily_hospit,
        Rt = Rt,
		expected_seropositive = expected_seropositive,
		iar = iar,
    )
end
## =============================================================================
#                      model_hospit
# ==============================================================================
## ============================================================
@model function model_hospit(
    θ,
    predict,
	::Type{TV} = Vector{Float64},
	::Type{V}  = Float64;
) where {TV, V}

	@unpack num_observations, num_total_days, num_rt_steps, invlink = θ
    # If we don't want to predict the future, we only need to compute up-to time-step `num_obs_countries[m]`
    num_time_steps = predict ? num_total_days : num_observations

	############# 2.) time varying reproduction number

	R0         ~ truncated(Normal(3., 1.), 1., 5.)
	R1         ~ truncated(Normal(.8, .1), .5, 1.1)
	σ_rt       ~ truncated(Normal(0.1, .05), 0, .25)
	latent_Rt  ~ RandomWalk(num_rt_steps, σ_rt, invlink(R1))

	Rt = TV(undef, num_time_steps)

	random_walks!(Rt, θ, predict, latent_Rt, R0)

	############ 3.) infection dynamics
	τ  ~ Exponential(1 / 0.03) # `Exponential` has inverse parameterization of the one in Stan
	T  = typeof(τ)
	y  ~ truncated(Exponential(τ),T(0),T(1000))

	newly_infected       = TV(undef, num_time_steps)
	cumulative_infected  = TV(undef, num_time_steps)
	# effective_Rt         = TV(undef, num_time_steps)

	infections!(newly_infected, cumulative_infected, θ, τ, y, Rt)
	# infections!(newly_infected, cumulative_infected, effective_Rt, θ, τ, ys, Rts)
	########### 4.) derive observables
    μ_i2h ~ truncated(Normal(12., 1.), 9, 16)
	ihr   ~ truncated(Normal(1.8/100,0.5/100),1/100,5/100)

	expected_daily_hospit = TV(undef, num_time_steps)

	hospitalizations!(expected_daily_hospit, θ, μ_i2h, ihr, newly_infected)

	########### 4.) compare model to observations
	## 4.1) observe hospitalizations
	ϕ_h   ~ truncated(Normal(50, 10), 20, Inf)

	ℓ  = zero(V)
	ℓ += observe_hospitalizations(ℓ, θ, expected_daily_hospit, ϕ_h)

	Turing.@addlogprob! ℓ

	return (
		newly_infected = newly_infected,
		expected_daily_hospit = expected_daily_hospit,
		Rt = Rt,
		# effective_Rt = effective_Rts
	)
end

# @model function model_hospit(
# 	num_impute,        # [Int] num. of days for which to impute infections
#     num_total_days,    # [Int] days of observed data + num. of days to forecast
#     cases,            # [AbstractVector{<:AbstractVector{<:Int}}] reported infected
#     deaths,            # [AbstractVector{<:AbstractVector{<:Int}}] reported deaths; rows indexed by i > N contain -1 and should be ignored
#     π,                 # [AbstractVector{<:AbstractVector{<:Real}}] h * s
#     π2,                 # [AbstractVector{<:AbstractVector{<:Real}}] h * s
#     epidemic_start,    # [AbstractVector{<:Int}]
#     population,        # [AbstractVector{<:Real}]
#     serial_intervals,  # [AbstractVector{<:Real}] fixed pre-calculated serial interval (SI) using empirical data from Neil
# 	num_iar_steps,         # [Int] number of weeks (21)
#     iar_index,        # [Vector{Array{Int64,1}}] Macro region index for each state
#     lockdown,         # [Int] number of weeks (21)
#     num_case_obs,       # [Int] 21 = max_date ("2020-05-11") - testing_date (2020-05-11) see publication
# 	weekday,
# 	holiday,
# 	π3,                 # [AbstractVector{<:AbstractVector{<:Real}}] h * s
#     hospit,
# 	seroprev_mean,
#     seroprev_std,
#     seroprev_idx,
# 	π4,
#     predict=false,     # [Bool] if `false`, will only compute what's needed to `observe` but not more
#     ::Type{TV} = Vector{Float64},
# 	::Type{V}  = Float64;
# 	invlink = log,#KLogistic(V(3))
# 	link    = exp,#KLogit(V(k))
# ) where {TV, V}
#     ############ 0.) some definitions
# 	num_obs        = length(cases)
#     num_Rt_steps   = num_obs - lockdown
# 	num_sero       = length(seroprev_mean)
#     cases_start    = num_obs-num_case_obs
# 	# If we don't want to predict the future, we only need to compute up-to time-step `num_obs_states[m]`
#     num_time_steps = predict ? num_total_days : num_obs
#
#     ############# 2.) time varying reproduction number
# 	R0        ~ truncated(Normal(3., 1.), 1., 5.)
# 	R1        ~ truncated(Normal(.8, .1), .5, 1.1)
# 	σ_rt      ~ truncated(Normal(0.05, .03), 0, .3)
# 	latent_Rt ~ RandomWalk(num_Rt_steps, σ_rt, invlink(R1))
#
#     Rt = TV(undef, num_time_steps)
#     Rt[1:lockdown] .= R0
# 	Rt[lockdown+1:num_obs] = link.(latent_Rt)
# 	if predict #if prediction
# 		num_pred_steps = num_total_days - num_obs
# 		latent_Rt_pred ~ RandomWalk(num_pred_steps, σ_rt, latent_Rt[end])
# 		Rt[num_obs+1:num_total_days] = link.(latent_Rt_pred)
# 	end
#
#     ############ 3.) infection dynamics
# 	## 3.1) initialize serial interval
# 	num_si   = 15
# 	si       = serial_intervals
# 	# μ_si   ~ truncated(Normal(5.06, 0.33), 0, Inf)
# 	# σ_si   = 2.11
# 	# si     = pdf.( Ref(GammaMeanStd(μ_si, σ_si)), 1:num_si )
# 	# si    /= sum(si)
#
# 	## 3.2) seeding
# 	τ ~ Exponential(1 / 0.03) # `Exponential` has inverse parameterization of the one in Stan
# 	T = typeof(τ)
#     y ~ truncated(Exponential(τ),T(0),T(1000))
#
# 	newly_infected      = TV(undef, num_time_steps)
# 	cumulative_infected = TV(undef, num_time_steps)
# 	newly_infected[1]   = y
# 	cumulative_infected[1] = zero(V)
# 	for t in 2:num_impute
# 		newly_infected[t] = y
# 		cumulative_infected[t] = cumulative_infected[t-1] + y
# 	end
#
#     ## 3.3) run disease dynamics
# 	for t = (num_impute + 1):num_time_steps
# 		# Update cumulatively infected
# 		cumulative_infected[t] = cumulative_infected[t-1] + newly_infected[t - 1]
# 		# Adjusts for portion of population that is susceptible
# 		St = max(population - cumulative_infected[t], 0) / population
# 		# effective number of infectious individuals
# 		effectively_infectious = sum(newly_infected[τ] * si[t - τ] for τ = (t - 1):-1:max(t-num_si,1))
# 		# number of new infections (unobserved)
# 		newly_infected[t] = St * Rt[t] * effectively_infectious
# 	end
#
# 	########### 4.) derive observables
# 	## 4.1) initialize delay distributions
# 	num_i2c = 40
# 	num_i2h = 40
# 	num_i2d = 60
#
# 	μ_i2c ~ truncated(Normal(4.5, 1.), 3, 7)
#     μ_i2h ~ truncated(Normal(12., 3.), 9, 16)
#     μ_i2d ~ truncated(Normal(20., 4.), 10, 30)
#
# 	ϕ_i2c = 5.41
# 	ϕ_i2h = 5.41
# 	ϕ_i2d = 14.26
#
# 	i2c = pdf.( Ref(NegativeBinomial2(μ_i2c, ϕ_i2c)), 1:num_i2c )
# 	i2h = pdf.( Ref(NegativeBinomial2(μ_i2h, ϕ_i2h)), 1:num_i2h )
# 	i2d = pdf.( Ref(NegativeBinomial2(μ_i2d, ϕ_i2d)), 1:num_i2d )
#
# 	i2c /= sum(i2c)
# 	i2h /= sum(i2h)
# 	i2d /= sum(i2d)
#
# 	### day of week effect
# 	weekdayeffect_simplex ~ Dirichlet([7,7,7,7,7,3,1])
# 	num_days = 7
# 	weekdayeffect = TV(undef, num_days)
# 	for i in 1:num_days
# 		weekdayeffect[i] = weekdayeffect_simplex[i] * num_days
# 	end
#
# 	### holiday effect
# 	holidayeffect ~ Beta(1,1)
#
#     ### 4.2) derive observables from infections
# 	ifr ~ truncated(Normal(6/1000, 3/1000), 1/1000, 10/1000)
# 	ihr ~ truncated(Normal(1/100,1/100),.1/100,5/100)
#     iar ~ Beta(1,10)
#
#     expected_daily_cases     = TV(undef, num_time_steps)
#     expected_daily_hospit    = TV(undef, num_time_steps)
#     expected_daily_deaths    = TV(undef, num_time_steps)
# 	expected_seropositive    = TV(undef, num_time_steps)
#
# 	expected_daily_cases[1]  = 1e-15 * newly_infected[1]
#     expected_daily_hospit[1] = 1e-15 * newly_infected[1]
#     expected_daily_deaths[1] = 1e-15 * newly_infected[1]
# 	for t = 2:num_time_steps
# 		periodiceffect = (1 - holidayeffect * holiday[t]) * weekdayeffect[weekday[t]]
# 		expected_daily_cases[t]  = iar * sum(newly_infected[τ] * i2c[t - τ] for τ = (t - 1):-1:max(t-num_i2c,1)) * periodiceffect
#         expected_daily_hospit[t] = ihr * sum(newly_infected[τ] * i2h[t - τ] for τ = (t - 1):-1:max(t-num_i2h,1))
#         expected_daily_deaths[t] = ifr * sum(newly_infected[τ] * i2d[t - τ] for τ = (t - 1):-1:max(t-num_i2d,1))
# 	end
#
# 	serodelay = 16 # 14d self deferral + 16d seroconversion (see Erikstrup et al. https://doi.org/10.1093/cid/ciaa849)
# 	expected_seropositive[1:serodelay]    .= zero(V)
# 	expected_seropositive[serodelay+1:end] = cumulative_infected[1:end-serodelay]
#
# 	########### 4.) compare model to observations
# 	## 4.1) observe hospitalizations
# 	ϕ_h   ~ truncated(Normal(25, 10), 0, Inf)
# 	ts_h  = epidemic_start:num_obs
# 	μs_h  = expected_daily_hospit[ts_h]
# 	!all( 0 .< μs_h .< population ) && (@error "expected_daily_hospit"; Turing.@addlogprob! -Inf; return)
# 	hospit[ts_h] ~ arraydist(NegativeBinomial2.(μs_h, ϕ_h))
#
#     ## 4.3) seroprevalence study
#     select_seroprev = 2:4
# 	seroprev_σ ~ arraydist( InverseGamma2.(seroprev_std[ select_seroprev ], Ref(.1)) )
#     seroprev_μ = expected_seropositive[ seroprev_idx[ select_seroprev ] ]
# 	seroprev_mean[select_seroprev] ~ MvNormal(seroprev_μ, seroprev_σ) # select only early observations
#
#
#     return (
# 		expected_daily_cases  = expected_daily_cases,
#         expected_daily_deaths = expected_daily_deaths,
#         expected_daily_hospit = expected_daily_hospit,
#         Rt = Rt,
# 		expected_seropositive = expected_seropositive,
# 		iar = iar,
#     )
# end
## =============================================================================
#                      model_deaths
# ==============================================================================
@model function model_deaths(
	num_impute,        # [Int] num. of days for which to impute infections
    num_total_days,    # [Int] days of observed data + num. of days to forecast
    cases,            # [AbstractVector{<:AbstractVector{<:Int}}] reported infected
    deaths,            # [AbstractVector{<:AbstractVector{<:Int}}] reported deaths; rows indexed by i > N contain -1 and should be ignored
    π,                 # [AbstractVector{<:AbstractVector{<:Real}}] h * s
    π2,                 # [AbstractVector{<:AbstractVector{<:Real}}] h * s
    epidemic_start,    # [AbstractVector{<:Int}]
    population,        # [AbstractVector{<:Real}]
    serial_intervals,  # [AbstractVector{<:Real}] fixed pre-calculated serial interval (SI) using empirical data from Neil
	num_iar_steps,         # [Int] number of weeks (21)
    iar_index,        # [Vector{Array{Int64,1}}] Macro region index for each state
    lockdown,         # [Int] number of weeks (21)
    num_case_obs,       # [Int] 21 = max_date ("2020-05-11") - testing_date (2020-05-11) see publication
	weekday,
	holiday,
	π3,                 # [AbstractVector{<:AbstractVector{<:Real}}] h * s
    hospit,
	seroprev_mean,
    seroprev_std,
    seroprev_idx,
	π4,
    predict=false,     # [Bool] if `false`, will only compute what's needed to `observe` but not more
    ::Type{TV} = Vector{Float64},
	::Type{V}  = Float64;
	invlink = log,#KLogistic(V(3))
	link    = exp,#KLogit(V(k))
) where {TV, V}
    ############ 0.) some definitions
	num_obs        = length(cases)
    num_Rt_steps   = num_obs - lockdown
	num_sero       = length(seroprev_mean)
    cases_start    = num_obs-num_case_obs
	# If we don't want to predict the future, we only need to compute up-to time-step `num_obs_states[m]`
    num_time_steps = predict ? num_total_days : num_obs

    ############# 2.) time varying reproduction number
	R0        ~ truncated(Normal(3., 1.), 1., 5.)
	R1        ~ truncated(Normal(.8, .1), .5, 1.1)
	σ_rt      ~ truncated(Normal(0.05, .03), 0, .3)
	latent_Rt ~ RandomWalk(num_Rt_steps, σ_rt, invlink(R1))

    Rt = TV(undef, num_time_steps)
    Rt[1:lockdown] .= R0
	Rt[lockdown+1:num_obs] = link.(latent_Rt)
	if predict #if prediction
		num_pred_steps = num_total_days - num_obs
		latent_Rt_pred ~ RandomWalk(num_pred_steps, σ_rt, latent_Rt[end])
		Rt[num_obs+1:num_total_days] = link.(latent_Rt_pred)
	end

    ############ 3.) infection dynamics
	## 3.1) initialize serial interval
	num_si   = 15
	si       = serial_intervals
	# μ_si   ~ truncated(Normal(5.06, 0.33), 0, Inf)
	# σ_si   = 2.11
	# si     = pdf.( Ref(GammaMeanStd(μ_si, σ_si)), 1:num_si )
	# si    /= sum(si)

	## 3.2) seeding
	τ ~ Exponential(1 / 0.03) # `Exponential` has inverse parameterization of the one in Stan
	T = typeof(τ)
    y ~ truncated(Exponential(τ),T(0),T(1000))

	newly_infected      = TV(undef, num_time_steps)
	cumulative_infected = TV(undef, num_time_steps)
	newly_infected[1]   = y
	cumulative_infected[1] = zero(V)
	for t in 2:num_impute
		newly_infected[t] = y
		cumulative_infected[t] = cumulative_infected[t-1] + y
	end

    ## 3.3) run disease dynamics
	for t = (num_impute + 1):num_time_steps
		# Update cumulatively infected
		cumulative_infected[t] = cumulative_infected[t-1] + newly_infected[t - 1]
		# Adjusts for portion of population that is susceptible
		St = max(population - cumulative_infected[t], 0) / population
		# effective number of infectious individuals
		effectively_infectious = sum(newly_infected[τ] * si[t - τ] for τ = (t - 1):-1:max(t-num_si,1))
		# number of new infections (unobserved)
		newly_infected[t] = St * Rt[t] * effectively_infectious
	end

	########### 4.) derive observables
	## 4.1) initialize delay distributions
	num_i2c = 40
	num_i2h = 40
	num_i2d = 60

	μ_i2c ~ truncated(Normal(4.5, 1.), 3, 7)
    μ_i2h ~ truncated(Normal(12., 3.), 9, 16)
    μ_i2d ~ truncated(Normal(20., 4.), 10, 30)

	ϕ_i2c = 5.41
	ϕ_i2h = 5.41
	ϕ_i2d = 14.26

	i2c = pdf.( Ref(NegativeBinomial2(μ_i2c, ϕ_i2c)), 1:num_i2c )
	i2h = pdf.( Ref(NegativeBinomial2(μ_i2h, ϕ_i2h)), 1:num_i2h )
	i2d = pdf.( Ref(NegativeBinomial2(μ_i2d, ϕ_i2d)), 1:num_i2d )

	i2c /= sum(i2c)
	i2h /= sum(i2h)
	i2d /= sum(i2d)

	### day of week effect
	weekdayeffect_simplex ~ Dirichlet([7,7,7,7,7,3,1])
	num_days = 7
	weekdayeffect = TV(undef, num_days)
	for i in 1:num_days
		weekdayeffect[i] = weekdayeffect_simplex[i] * num_days
	end

	### holiday effect
	holidayeffect ~ Beta(1,1)

    ### 4.2) derive observables from infections
	ifr ~ truncated(Normal(6/1000, 3/1000), 1/1000, 10/1000)
	ihr ~ truncated(Normal(1/100,1/100),.1/100,5/100)
    iar ~ Beta(1,10)

    expected_daily_cases     = TV(undef, num_time_steps)
    expected_daily_hospit    = TV(undef, num_time_steps)
    expected_daily_deaths    = TV(undef, num_time_steps)
	expected_seropositive    = TV(undef, num_time_steps)

	expected_daily_cases[1]  = 1e-15 * newly_infected[1]
    expected_daily_hospit[1] = 1e-15 * newly_infected[1]
    expected_daily_deaths[1] = 1e-15 * newly_infected[1]
	for t = 2:num_time_steps
		periodiceffect = (1 - holidayeffect * holiday[t]) * weekdayeffect[weekday[t]]
		expected_daily_cases[t]  = iar * sum(newly_infected[τ] * i2c[t - τ] for τ = (t - 1):-1:max(t-num_i2c,1)) * periodiceffect
        expected_daily_hospit[t] = ihr * sum(newly_infected[τ] * i2h[t - τ] for τ = (t - 1):-1:max(t-num_i2h,1))
        expected_daily_deaths[t] = ifr * sum(newly_infected[τ] * i2d[t - τ] for τ = (t - 1):-1:max(t-num_i2d,1))
	end

	serodelay = 16 # 14d self deferral + 16d seroconversion (see Erikstrup et al. https://doi.org/10.1093/cid/ciaa849)
	expected_seropositive[1:serodelay]    .= zero(V)
	expected_seropositive[serodelay+1:end] = cumulative_infected[1:end-serodelay]

	########### 4.) compare model to observations
	## 4.1) observe deaths
	ϕ_d   ~ truncated(Normal(20, 10), 0, Inf)
    ts_d  = epidemic_start:num_obs
	μs_d  = expected_daily_deaths[ts_d]
	!all( 0 .< μs_d .< population ) && (@error "expected_daily_deaths"; Turing.@addlogprob! -Inf; return)
	deaths[ts_d] ~ arraydist(NegativeBinomial2.(μs_d, ϕ_d))

    ## 4.3) seroprevalence study
    select_seroprev = 2:4
	seroprev_σ ~ arraydist( InverseGamma2.(seroprev_std[ select_seroprev ], Ref(.1)) )
    seroprev_μ = expected_seropositive[ seroprev_idx[ select_seroprev ] ]
	seroprev_mean[select_seroprev] ~ MvNormal(seroprev_μ, seroprev_σ) # select only early observations


    return (
		expected_daily_cases  = expected_daily_cases,
        expected_daily_deaths = expected_daily_deaths,
        expected_daily_hospit = expected_daily_hospit,
        Rt = Rt,
		expected_seropositive = expected_seropositive,
		iar = iar,
    )
end
