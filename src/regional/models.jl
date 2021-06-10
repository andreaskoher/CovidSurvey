using DistributionsAD

struct RandomWalk{Tn, Ts, Tx} <: ContinuousMultivariateDistribution
  n::Tn
  s::Ts
  x0::Tx
end

Distributions.rand(rng::AbstractRNG, d::RandomWalk{Tn, Ts, Tx}) where {Tn, Ts, Tx} = begin
  x = Vector{Tx}(undef, d.n)
  x[1] = d.x0
  for i in 2:d.n
    x[i] = x[i-1] + rand(Normal(0, d.s))
  end
  return x
end

Distributions.logpdf(d::RandomWalk{Tn, Ts, Tx}, x::AbstractVector{T}) where {Tn, Ts, Tx, T} =
    logpdf( MvNormal( d.n-1, d.s ), diff(x) )# + logpdf( Normal( zero(Ts), x[1] ) )

Bijectors.bijector(d::RandomWalk) = Identity{1}()

Bijectors.bijector( ::DistributionsAD.VectorOfMultivariate{Continuous, RandomWalk{Tn, Ts, Tx}, Vector{RandomWalk{Tn, Ts, Tx}}} ) where {Tn, Ts, Tx} = Identity{2}()


Base.length(d::RandomWalk) = d.n
##
function random_walks!(Rts, θ, predict, latent_Rts, R0s)
	@unpack num_regions, rt_step_indices, lockdown_indices, num_observations, link = θ

	for m in 1:num_regions
		rt_step_index = rt_step_indices[m]
		Rt            = Rts[m]
		lockdown      = lockdown_indices[m]
		num_obs       = num_observations[m]
		latent_Rt     = latent_Rts[:,m]
		R0            = R0s[m]

		Rt[1:lockdown] .= R0
		Rt[lockdown+1:num_obs] = link.(latent_Rt)
		if predict
			Rt[num_obs+1:end] = Rt[num_obs]
		end
	end
end

function infections!(newly_infecteds, cumulative_infecteds, θ, τ, ys, Rts)
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
	return nothing
end

function initepidemic!(newly_infected, cumulative_infected, θ, regionaldata)
	@unpack num_impute = θ
	@unpack y = regionaldata
	newly_infected[1]      = y
	cumulative_infected[1] = zero( eltype(cumulative_infected) )

	for t in 2:num_impute
		newly_infected[t] = y
		cumulative_infected[t] = cumulative_infected[t-1] + y
	end
end

function runepidemic!(newly_infected, cumulative_infected, θ, regionaldata)
	@unpack num_impute, serial_interval, num_si = θ
	@unpack population, Rt = regionaldata
	num_time_step = length(newly_infected)

	for t = (num_impute + 1):num_time_step
		# Update cumulatively infected
		cumulative_infected[t] = cumulative_infected[t-1] + newly_infected[t - 1]
		# Adjusts for portion of population that is susceptible
		St = max(population - cumulative_infected[t], 0) / population
		# effective number of infectious individuals
		effectively_infectious = sum(newly_infected[τ] * serial_interval[t - τ] for τ = (t - 1):-1:max(t-num_si,1))
		# number of new infections (unobserved)
		newly_infected[t] = St * Rt[t] * effectively_infectious
	end
	return nothing
end

function infection2hospit(θ, μ_i2h)
	@unpack ϕ_i2h, num_i2h = θ
	i2h = pdf.( Ref(NegativeBinomial2(μ_i2h, ϕ_i2h)), 1:num_i2h )
	i2h /= sum(i2h)
	return i2h
end

function hospitalizations!(expected_daily_hospits, θ, μ_i2h, ihr, newly_infecteds)
	@unpack num_regions, num_i2h = θ

	i2h = infection2hospit(θ, μ_i2h)

	for m in 1:num_regions

		expected_daily_hospit    = expected_daily_hospits[m]
		newly_infected           = newly_infecteds[m]
		num_time_step            = length(expected_daily_hospit)

		expected_daily_hospit[1] = 1e-15 * newly_infected[1]
		for t = 2:num_time_step
			expected_daily_hospit[t] = ihr * sum(newly_infected[τ] * i2h[t - τ] for τ = (t - 1):-1:max(t-num_i2h,1))
		end
	end
	return nothing
end

function observe_hospitalizations(ℓ, θ, expected_daily_hospits, ϕ_h)
	@unpack num_regions, populations, num_observations, hospits, epidemic_start = θ

	T = typeof(ℓ)
	for m in 1:num_regions
		population            = populations[m]
		expected_daily_hospit = expected_daily_hospits[m]
		num_obs               = num_observations[m]
		hospit                = hospits[m]

		ts_h  = epidemic_start:num_obs
		μs_h  = expected_daily_hospit[ts_h]
		!all( 0 .< μs_h .< population ) && (@error "expected_daily_hospit"; return T(Inf))
		dist  = arraydist(NegativeBinomial2.(μs_h, ϕ_h))
		ℓ += logpdf(dist, hospit[ts_h])
	end
	return ℓ
end
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
	σ_rt       ~ truncated(Normal(0.05, .03), 0, .3)
	latent_Rts ~ arraydist( RandomWalk.(num_rt_steps, Ref(σ_rt), invlink.(R1s)) )

	Rts = TV[TV(undef, num_time_steps[m]) for m in 1:num_regions]

	random_walks!(Rts, θ, predict, latent_Rts, R0s)

	############ 3.) infection dynamics
	τ  ~ Exponential(1 / 0.03) # `Exponential` has inverse parameterization of the one in Stan
	T  = typeof(τ)
	ys ~ filldist(truncated(Exponential(τ),T(0),T(1000)), num_regions)

	newly_infecteds      = TV[TV(undef, num_time_steps[m]) for m in 1:num_regions]
	cumulative_infecteds = TV[TV(undef, num_time_steps[m]) for m in 1:num_regions]

	infections!(newly_infecteds, cumulative_infecteds, θ, τ, ys, Rts)

	########### 4.) derive observables
    μ_i2h ~ truncated(Normal(12., 3.), 9, 16)
	ihr   ~ truncated(Normal(1/100,1/100),.1/100,5/100)

	expected_daily_hospits = TV[TV(undef, num_time_steps[m]) for m in 1:num_regions]

	hospitalizations!(expected_daily_hospits, θ, μ_i2h, ihr, newly_infecteds)

	########### 4.) compare model to observations
	## 4.1) observe hospitalizations
	ϕ_h   ~ truncated(Normal(25, 10), 0, Inf)
	ℓ = zero(V)
	ℓ += observe_hospitalizations(ℓ, θ, expected_daily_hospits, ϕ_h)
	Turing.@addlogprob! ℓ

	return (
		expected_daily_hospits = expected_daily_hospits,
		Rts = Rts
	)
end


@model function model_v2(
    num_impute,        # [Int] num. of days for which to impute infections
    num_total_days,    # [Int] days of observed data + num. of days to forecast
    cases,             # [AbstractVector{<:AbstractVector{<:Int}}] reported cases
    deaths,            # [AbstractVector{<:AbstractVector{<:Int}}] reported deaths; rows indexed by i > N contain -1 and should be ignored
    π,                 # [AbstractVector{<:AbstractVector{<:Real}}] h * s
    covariates,        # [Vector{<:AbstractMatrix}]
    epidemic_start,    # [AbstractVector{<:Int}]
    population,        # [AbstractVector{<:Real}]
    serial_intervals,  # [AbstractVector{<:Real}] fixed pre-calculated serial interval (SI) using empirical data from Neil
    lockdown_index,    # [Int] the index for the `lockdown` covariate in `covariates`
    predict=false,     # [Bool] if `false`, will only compute what's needed to `observe` but not more
    ::Type{TV} = Vector{Float64}
) where {TV}
    # `covariates` should be of length `num_countries` and each entry correspond to a matrix of size `(num_total_days, num_covariates)`
    num_covariates = size(covariates[1], 2)
    num_countries = length(cases)
    num_obs_countries = length.(cases)

    # If we don't want to predict the future, we only need to compute up-to time-step `num_obs_countries[m]`
    last_time_steps = predict ? fill(num_total_days, num_countries) : num_obs_countries

    # Latent variables
    τ ~ Exponential(1 / 0.03) # `Exponential` has inverse parameterization of the one in Stan
    y ~ filldist(Exponential(τ), num_countries)
    ϕ ~ truncated(Normal(0, 5), 0, Inf)
    κ ~ truncated(Normal(0, 0.5), 0, Inf)
    μ ~ filldist(truncated(Normal(3.28, κ), 0, Inf), num_countries)

    α_hier ~ filldist(Gamma(.1667, 1), num_covariates)
    α = α_hier .- log(1.05) / 6.

    ifr_noise ~ filldist(truncated(Normal(1., 0.1), 0, Inf), num_countries)

    # lockdown-related
    γ ~ truncated(Normal(0, 0.2), 0, Inf)
    lockdown ~ filldist(Normal(0, γ), num_countries)

    # Initialization of some quantities
    expected_daily_cases = TV[TV(undef, last_time_steps[m]) for m in 1:num_countries]
    cases_pred = TV[TV(undef, last_time_steps[m]) for m in 1:num_countries]
    expected_daily_deaths = TV[TV(undef, last_time_steps[m]) for m in 1:num_countries]
    Rt = TV[TV(undef, last_time_steps[m]) for m in 1:num_countries]
    Rt_adj = TV[TV(undef, last_time_steps[m]) for m in 1:num_countries]

    # Loops over countries and perform independent computations for each country
    # since this model does not include any notion of migration across borders.
    # => might has well wrap it in a `@threads` to perform the computation in parallel.
    @threads for m = 1:num_countries
        # Country-specific parameters
        ifr_noise_m = ifr_noise[m]
        π_m = π[m]
        pop_m = population[m]
        expected_daily_cases_m = expected_daily_cases[m]
        cases_pred_m = cases_pred[m]
        expected_daily_deaths_m = expected_daily_deaths[m]
        Rt_m = Rt[m]
        Rt_adj_m = Rt_adj[m]

        last_time_step = last_time_steps[m]

        # Imputation of `num_impute` days
        expected_daily_cases_m[1:num_impute] .= y[m]
        cases_pred_m[1] = zero(cases_pred_m[1])
        cases_pred_m[2:num_impute] .= cumsum(expected_daily_cases_m[1:num_impute - 1])

        xs = covariates[m][1:last_time_step, :] # extract covariates for the wanted time-steps and country `m`
        Rt_m .= μ[m] * exp.(xs * (-α) + (- lockdown[m]) * xs[:, lockdown_index])

        # Adjusts for portion of pop that are susceptible
        Rt_adj_m[1:num_impute] .= (max.(pop_m .- cases_pred_m[1:num_impute], zero(cases_pred_m[1])) ./ pop_m) .* Rt_m[1:num_impute]

        for t = (num_impute + 1):last_time_step
            # Update cumulative cases
            cases_pred_m[t] = cases_pred_m[t - 1] + expected_daily_cases_m[t - 1]

            # Adjusts for portion of pop that are susceptible
            Rt_adj_m[t] = (max(pop_m - cases_pred_m[t], zero(cases_pred_m[t])) / pop_m) * Rt_m[t]

            expected_daily_cases_m[t] = Rt_adj_m[t] * sum(expected_daily_cases_m[τ] * serial_intervals[t - τ] for τ = 1:(t - 1))
        end

        expected_daily_deaths_m[1] = 1e-15 * expected_daily_cases_m[1]
        for t = 2:last_time_step
            expected_daily_deaths_m[t] = ifr_noise_m * sum(expected_daily_cases_m[τ] * π_m[t - τ] for τ = 1:(t - 1))
        end
    end

    # Observe
    # Doing observations in parallel provides a small speedup
    @threads for m = 1:num_countries
        # Extract the estimated expected daily deaths for country `m`
        expected_daily_deaths_m = expected_daily_deaths[m]
        # Extract time-steps for which we have observations
        ts = epidemic_start[m]:num_obs_countries[m]
        # Observe!
        deaths[m][ts] ~ arraydist(NegativeBinomial2.(expected_daily_deaths_m[ts], ϕ))
    end
    return (
        expected_daily_cases = expected_daily_cases,
        expected_daily_deaths = expected_daily_deaths,
        Rt = Rt,
        Rt_adjusted = Rt_adj
    )
end
