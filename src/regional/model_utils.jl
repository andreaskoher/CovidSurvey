struct WeekdayHolidayObsModel{T,M,A,P,E,W,H,V} <: CovidSurvey.ObservationsModel
    ps::T
    μ::M
    α::A
    ϕ::P
	expecteds::E
    weekdayeffect::W
    holidayeffect::H
	vectorize::V
end

struct SimpleObsModel{T,M,A,P,E,V} <: CovidSurvey.ObservationsModel
    ps::T
    μ::M
    α::A
    ϕ::P
	expecteds::E
	vectorize::V
end

struct SimpleObsModel2{T,M,S,A,P,E,V} <: CovidSurvey.ObservationsModel
    ps::T
    μ::M
	σ::S
    α::A
    ϕ::P
	expecteds::E
	vectorize::V
end

rescale(xs, σ, μs) = @. xs * σ + μs'
# ============================================================================
# random walk model
function random_walks!(Rts, θ, predict, latent_Rts::AbstractVector, R0s, σ_rt)
	@unpack num_regions, rt_step_indices, lockdown_indices, num_observations, link, include_early_dynamic = θ #include_early_dynamic

	for m in 1:num_regions
		rt_step_index = rt_step_indices[m]
		Rt            = Rts[m]
		lockdown      = lockdown_indices[m]
		num_obs       = num_observations[m]
		latent_Rt     = latent_Rts[m]#latent_Rts[:,m] #include_early_dynamic ? latent_Rts[m] : latent_Rts[:,m]
		R0            = R0s[m]

		if include_early_dynamic
			Rt[1:num_obs] = link.(latent_Rt[rt_step_index])
		else
			Rt[1:lockdown] .= R0
			Rt[lockdown+1:num_obs] = link.(latent_Rt[rt_step_index])
		end
		if predict
			Rt[num_obs+1:end] .= Rt[num_obs]
		end
	end
end

function random_walks!(Rts, θ, predict, latent_Rts::AbstractMatrix, R0s, σ_rt)
	@unpack num_regions, rt_step_indices, lockdown_indices, num_observations, link, include_early_dynamic = θ

	for m in 1:num_regions
		rt_step_index = rt_step_indices[m]
		Rt            = Rts[m]
		lockdown      = lockdown_indices[m]
		num_obs       = num_observations[m]
		latent_Rt     = latent_Rts[:,m] #include_early_dynamic ? latent_Rts[m] : latent_Rts[:,m]
		R0            = R0s[m]

		if include_early_dynamic
			Rt[1:num_obs] = link.(latent_Rt[rt_step_index])
		else
			Rt[1:lockdown] .= R0
			Rt[lockdown+1:num_obs] = link.(latent_Rt[rt_step_index])
		end
		if predict
			Rt[num_obs+1:end] .= Rt[num_obs]
		end
	end
end

# function random_walks!(Rts, θ, predict, latent_Rts, R0s)
# 	@unpack num_regions, rt_step_indices, lockdown_indices, num_observations, link = θ
#
# 	for m in 1:num_regions
# 		rt_step_index = rt_step_indices[m]
# 		Rt            = Rts[m]
# 		lockdown      = lockdown_indices[m]
# 		num_obs       = num_observations[m]
# 		latent_Rt     = latent_Rts[:,m]
# 		R0            = R0s[m]
#
# 		Rt[1:lockdown] .= R0
# 		Rt[lockdown+1:num_obs] = link.(latent_Rt[rt_step_index])
# 		if predict
# 			Rt[num_obs+1:end] .= Rt[num_obs]
# 		end
# 	end
# end

# ============================================================================
# semi-parametric model
function semiparametric!(Rts, θ, predict, latent_Rts, R0s, σ_rt, effects)
	@unpack num_regions, lockdown_indices, num_observations, link, rt_step_indices, covariates, covariates_starts, num_total_days, include_early_dynamic= θ #rt_step_indices

	for m in 1:num_regions
		Rt            = Rts[m]
		lockdown      = lockdown_indices[m]
		rt_step_index = rt_step_indices[m]
		num_obs       = num_observations[m]
		latent_Rt     = include_early_dynamic ? latent_Rts[m] : latent_Rts[:,m] #latent_Rts[m]
		R0            = R0s[m]
		num_total     = num_total_days[m]
		covariate     = covariates[m]
		effect        = effects[m]
		covariates_start = covariates_starts[m]

		if include_early_dynamic
			Rt[1:covariates_start-1] = link.(latent_Rt[rt_step_index])
		else
			Rt[1:lockdown] .= R0
			Rt[lockdown+1:covariates_start-1] = link.(latent_Rt[rt_step_index])
		end
		Rt[covariates_start:num_obs] = link.(
			latent_Rt[end] .+ covariate * effect
		)
		if predict
			Rt[num_obs+1:num_total] .= Rt[num_obs]
			# @unpack num_total_days, invlink = θ
			# n  = num_total_days - num_observations
			# n < 1 && return nothing
			# rw = RandomWalk(n, σ_rt, invlink(Rt[num_observations]))
			# Rt[num_observations+1:end] = link.( rand( rw ) )
		end
	end
	return nothing
end

# ============================================================================
# semi-parametric model
function mixed!(Rts, θ, predict, latent_Rts, R0s, σ_rt, effects)
	@unpack (
		num_regions,
		lockdown_indices,
		num_observations,
		link,
		rt_step_indices,
		covariates,
		covariates_starts,
		num_total_days,
		include_early_dynamic
	) = θ #rt_step_indices

	for m in 1:num_regions
		Rt               = Rts[m]
		lockdown_index   = lockdown_indices[m]
		rt_step_index    = rt_step_indices[m]
		num_obs          = num_observations[m]
		latent_Rt        = include_early_dynamic ? latent_Rts[m] : latent_Rts[:,m]#latent_Rts[m]
		R0               = R0s[m]
		num_total        = num_total_days[m]
		covariate        = covariates[m]
		effect           = effects[m]
		covariates_start = covariates_starts[m]
		predictor        = covariate * effect

		if include_early_dynamic
			for i in 1:covariates_start-1
				Rt[i] = link(latent_Rt[rt_step_index[i]])
			end
			for (j,i) in enumerate(covariates_start:num_obs)
				Rt[i] = link(latent_Rt[rt_step_index[i]] + predictor[j])
			end
		else
			Rt[1:lockdown_index] .= R0
			for (j,i) in enumerate(lockdown_index+1:covariates_start-1)
				Rt[i] = link(latent_Rt[rt_step_index[j]])
			end
			n = covariates_start - lockdown_index - 1
			for (j,i) in enumerate(covariates_start:num_obs)
				Rt[i] = link(latent_Rt[rt_step_index[j+n]] + predictor[j])
			end
		end
		if predict
			Rt[num_obs+1:num_total] .= Rt[num_obs]
			# @unpack num_total_days, invlink = θ
			# n  = num_total_days - num_observations
			# n < 1 && return nothing
			# rw = RandomWalk(n, σ_rt, invlink(Rt[num_observations]))
			# Rt[num_observations+1:end] = link.( rand( rw ) )
		end
	end
	return nothing
end

# ============================================================================
# infection model
function infections!(newly_infecteds, cumulative_infecteds, θ, ys, Rts)
	@unpack num_regions, populations = θ
	for m in 1:num_regions
		regionaldata = (
			Rt                     = Rts[m],
			population             = populations[m],
			y                      = ys[m]
		)

		initepidemic!(newly_infecteds[m], cumulative_infecteds[m], θ, regionaldata)
		runepidemic!(newly_infecteds[m], cumulative_infecteds[m], θ, regionaldata)
	end
end

function initepidemic!(newly_infected, cumulative_infected, effective_Rt, θ, regionaldata)
	@unpack num_impute = θ
	@unpack y, Rt, population = regionaldata
	newly_infected[1]      = y
	cumulative_infected[1] = zero( eltype(cumulative_infected) )
	effective_Rt[1]        = Rt[1]
	for t in 2:num_impute
		newly_infected[t] = y
		cumulative_infected[t] = cumulative_infected[t-1] + y
		St = max(population - cumulative_infected[t], 0) / population
		effective_Rt[t] = St * Rt[t]
	end
end

function initepidemic!(newly_infected, cumulative_infected, θ, regionaldata)
	@unpack num_impute = θ
	@unpack y, population = regionaldata
	newly_infected[1]      = y
	cumulative_infected[1] = zero( eltype(cumulative_infected) )
	for t in 2:num_impute
		newly_infected[t] = y
		cumulative_infected[t] = cumulative_infected[t-1] + y
		# St = max(population - cumulative_infected[t], 0) / population
	end
end

function runepidemic!(newly_infected, cumulative_infected, effective_Rts, θ, regionaldata)
	@unpack num_impute, serial_interval, num_si = θ
	@unpack population, Rt = regionaldata
	num_time_step = length(newly_infected)

	for t = (num_impute + 1):num_time_step
		# Update cumulatively infected
		C = cumulative_infected[t] = cumulative_infected[t-1] + newly_infected[t - 1]
		susceptible = max(population - C, 0)

		# Adjusts for portion of population that is susceptible
		#St = max(population - cumulative_infected[t], 0) / population
		# effective number of infectious individuals
		effectively_infectious = sum(newly_infected[τ] * serial_interval[t - τ] for τ = (t - 1):-1:max(t-num_si,1))

		unadjusted_new_infections = effectively_infectious * Rt[t]

		# number of new infections (unobserved)
		# newly_infected[t] = susceptible * (1 - exp(- unadjusted_new_infections / population))
		newly_infected[t] = susceptible * (1 - exp(- unadjusted_new_infections / population))
		# newly_infected[t] = effective_Rt[t] * effectively_infectious

		effective_Rt[t] = susceptible / population * Rt[t]
	end
end

function runepidemic!(newly_infected, cumulative_infected, θ, regionaldata)
	@unpack num_impute, serial_interval, num_si = θ
	@unpack population, Rt = regionaldata
	num_time_step = length(newly_infected)

	for t = (num_impute + 1):num_time_step
		# Update cumulatively infected
		C = cumulative_infected[t] = cumulative_infected[t-1] + newly_infected[t - 1]
		susceptible = max(population - C, 0)

		# Adjusts for portion of population that is susceptible
		effectively_infectious = sum(newly_infected[τ] * serial_interval[t - τ] for τ = (t - 1):-1:max(t-num_si,1))

		unadjusted_new_infections = effectively_infectious * Rt[t]

		# number of new infections (unobserved)
		# newly_infected[t] = susceptible * (1 - exp(- unadjusted_new_infections / population))
		newly_infected[t] = susceptible * (1 - exp(- unadjusted_new_infections / population))

	end
	return nothing
end

# ============================================================================
# unbound infection model
function infections!(newly_infecteds, θ, ys, Rts)
	@unpack num_regions = θ
	for m in 1:num_regions
		regionaldata = (
			Rt                     = Rts[m],
			y                      = ys[m]
		)

		initepidemic!(newly_infecteds[m], θ, regionaldata)
		runepidemic!(newly_infecteds[m], θ, regionaldata)
	end
end

function initepidemic!(newly_infected, θ, regionaldata)
	@unpack num_impute = θ
	@unpack y, Rt = regionaldata
	newly_infected[1] = y
	for t in 2:num_impute
		newly_infected[t] = y
	end
end

function runepidemic!(newly_infected, θ, regionaldata)
	@unpack num_impute, serial_interval, num_si = θ
	@unpack Rt = regionaldata
	num_time_step = length(newly_infected)

	for t = (num_impute + 1):num_time_step
		effectively_infectious = sum(newly_infected[τ] * serial_interval[t - τ] for τ = (t - 1):-1:max(t-num_si,1))
		newly_infected[t] = effectively_infectious * Rt[t]
	end
	return nothing
end

# ============================================================================
# observations

function infection2observation(obsmodel::SimpleObsModel)
	@unpack ps, μ = obsmodel
	@unpack delay_dispersion, delay_length, delay_dist = ps
	i2o = pdf.( Ref(delay_dist(μ, delay_dispersion)), 1:delay_length )
	i2o /= sum(i2o)
	return i2o
end

function infection2observation(obsmodel::SimpleObsModel2)
	@unpack ps, μ, σ = obsmodel
	@unpack  delay_length, delay_dist = ps
	i2o = pdf.( Ref(delay_dist(μ, σ)), 1:delay_length )
	i2o /= sum(i2o)
	return i2o
end

function expected!(obsmodel::Union{SimpleObsModel, SimpleObsModel2}, newly_infecteds)
	@unpack ps, α, expecteds = obsmodel
	@unpack num_regions, delay_length = ps

	i2o = infection2observation(obsmodel)

	for m in 1:num_regions

		expected       = expecteds[m]
		newly_infected = newly_infecteds[m]
		num_time_step  = length(expected)

		expected[1] = 1e-15*newly_infected[1]
		for t = 2:num_time_step
			expected[t] = α * sum(newly_infected[τ] * i2o[t - τ] for τ = (t - 1):-1:max(t-delay_length,1))
			# expected[t] = α * sum(newly_infected[τ] * i2o[t - τ] for τ = (t - 1):-1:max(t-delay_length,1))
		end
	end
	return nothing
end

function expected!(obsmodel::WeekdayHolidayObsModel, newly_infecteds)
	@unpack ps, α, expecteds, holidayeffect, weekdayeffect = obsmodel
	@unpack num_regions, delay_length, holidays, weekdays = ps

	i2o = infection2observation(obsmodel)

	for m in 1:num_regions

		expected       = expecteds[m]
		newly_infected = newly_infecteds[m]
		num_time_step  = length(expected)
		holiday        = holidays[m]
		weekday        = weekdays[m]

		expected[1] = 1e-15*newly_infected[1]
		for t = 2:num_time_step
			weekday_holiday_effect = (1 - holidayeffect * holiday[t]) * weekdayeffect[weekday[t]]
			expected[t] = α * weekday_holiday_effect * sum(newly_infected[τ] * i2o[t - τ] for τ = (t - 1):-1:max(t-delay_length,1))
		end
	end
	return nothing
end

function _fastinference(ys, μs, ϕ)
	dist = arraydist(NegativeBinomial2.(μs, Ref(ϕ)))
	return logpdf(dist, ys)
end

function _slowinference(ys, μs, ϕ)
	ℓ = zero(typeof(ϕ))
	for (μ,y) in zip(μs,ys)
		dist = NegativeBinomial2(μ, ϕ)
		ℓ   += logpdf(dist, y)
	end
	return ℓ
end

function _prediction(ys, μs, ϕ)
	ℓ = zero(typeof(ϕ))
	for μ in μs
		dist = NegativeBinomial2(μ, ϕ)
		y    = rand(dist)
		ℓ   += logpdf(dist, y)
	end
	return ℓ
end

function Turing.logpdf(obsmodel::CovidSurvey.ObservationsModel, observed)
	@unpack ps, ϕ, expecteds, vectorize = obsmodel
	@unpack populations, starts, stops, num_regions = ps

	# _logpdf = if vectorize
	# 	_fastinference
	# else
	# 	if eltype(ϕ) <: Real
	# 		_prediction
	# 	else
	# 		_slowinference
	# 	end
	# end
	_logpdf = _fastinference

	T = typeof(ϕ)
	ℓ = zero(T)
	for m in 1:num_regions
		ts    = starts[m]:stops[m]
		μs = expecteds[m][ts]
		ys = observed[m][ts]

		#_out_of_bounds(μs, populations[m]) && (@warn "out of bounds"; return T(Inf))
		ℓ += _logpdf(ys, μs, ϕ)
	end
	return ℓ
end

_out_of_bounds(x, upper_lim) = !all(@. 0. < x < upper_lim )

# ============================================================================
# pointwise_loglikelihoods
function Turing.pointwise_loglikelihoods(obsmodel::CovidSurvey.ObservationsModel, observed)
	@unpack ps, ϕ, expecteds, vectorize = obsmodel
	@unpack populations, starts, stops, num_regions = ps

	logliks = Vector{Float64}[]
	for m in 1:num_regions
		ts    = starts[m]:stops[m]
		μs = expecteds[m][ts]
		ys = observed[m][ts]

		loglik = [ logpdf(NegativeBinomial2(μ, ϕ), y) for (μ,y) in zip(μs,ys) ]
		push!(logliks, loglik)
	end
	return logliks
end

# ============================================================================
# predictions
function prediction(obsmodel::CovidSurvey.ObservationsModel)
	@unpack ps, ϕ, expecteds = obsmodel
	@unpack starts, stops, num_regions = ps

	ys = Vector{Float64}[]
	for m in 1:num_regions
		μs = expecteds[m]
		μs[μs .< 1e-2] .= 1e-2
		y = [ rand(NegativeBinomial2(μ, ϕ)) for μ in μs ]
		push!(ys, y)
	end
	return ys
end

# ============================================================================
# weekday effect

function weekdayeffect!(weekdayeffect, weekdayeffect_simplex)
	num_days = 7
	for i in eachindex(weekdayeffect)
		@inbounds weekdayeffect[i] = weekdayeffect_simplex[i] * num_days
	end
end

# ============================================================================
# observe sero positive

struct SimpleSeroObsModel{T,E}# <: CovidSurvey.ObservationsModel
    ps::T
	expecteds::E
end

_unrealistic(ŷ,y) = any( ŷ .> 20*y )
# _unrealistic(y,ŷ) = any( ŷ .> 1.5*y )
function Turing.logpdf(obsmodel::SimpleSeroObsModel, observeds)
	@unpack ps, expecteds = obsmodel
	@unpack indexs, delay, populations, std, num_regions = ps

	T = eltype(expecteds[1])
	ℓ = zero(T)
	for m in 1:num_regions
		index      = indexs[m]
		expected   = expecteds[m]
		population = populations[m]
		observed   = observeds[m]

		μ    = expected[index .- delay] / population
		dist = MvNormal(μ, std) # select only early observations
		#_unrealistic(μ, observed) && (return T(Inf))
		ℓ += logpdf(dist, observed)
	end
	return ℓ
end
