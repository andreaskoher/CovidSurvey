# This is the most up-to-date one
@model function model_v2(
    num_impute,        # [Int] num. of days for which to impute infections
    num_total_days,    # [Int] days of observed data + num. of days to forecast
    cases,             # [AbstractVector{<:AbstractVector{<:Int}}] reported cases
    deaths,            # [AbstractVector{<:AbstractVector{<:Int}}] reported deaths; rows indexed by i > N contain -1 and should be ignored
    π,                 # [AbstractVector{<:AbstractVector{<:Real}}] h * s
    π2,
    epidemic_start,    # [AbstractVector{<:Int}]
    population,        # [AbstractVector{<:Real}]
    serial_intervals,  # [AbstractVector{<:Real}] fixed pre-calculated serial interval (SI) using empirical data from Neil
    num_iar_steps,         # [Int] number of weeks (21)
    iar_index,        # [Vector{Array{Int64,1}}] Macro region index for each state
    lockdown,         # [Int] number of weeks (21)
    num_case_obs,       # [Int] 21 = max_date ("2020-05-11") - testing_date (2020-05-11) see publication
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

    # `covariates` should be of length `num_countries` and each entry correspond to a matrix of size `(num_total_days, num_covariates)`
    num_countries = length(cases)
    num_obs_countries = length.(cases)

    # If we don't want to predict the future, we only need to compute up-to time-step `num_obs_countries[m]`
    last_time_steps = predict ? fill(num_total_days, num_countries) : num_obs_countries

    # Latent variables
	τ         ~ Exponential(1 / 0.03) # `Exponential` has inverse parameterization of the one in Stan
	T = typeof(τ)
    y         ~ filldist(truncated(Exponential(τ),T(0),T(1000)), num_countries)
    ϕ         ~ truncated(Normal(0, 5), 0, Inf)
	ϕ2        ~ truncated(Normal(25, 5), 10, 40)
    ϕ3        ~ truncated(Normal(25, 5), 10, 40)
	R0        ~ truncated(Normal(3.6, .8), 2, 5.)
	R1        ~ truncated(Normal(.8, .1), .5, 1.1)
    ifr       ~ truncated(Normal(6/1000, 3/1000), 3/1000, 10/1000)
	ihr       ~ truncated(Normal(1/100,1/100),.1/100,5/100)
    iar0      ~ Beta(10,100) #Beta(20,25)
	σ_rt      ~ truncated(Normal(0.05, .03), 0, .15)
	σ_iar     ~ truncated(Normal(0.05, .03), 0, .15)

	seroprev_σ ~ arraydist(InverseGamma2.(seroprev_std, Ref(.1)))

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
