struct RandomWalk{Tn, Ts, Tx} <: ContinuousMultivariateDistribution
  n::Tn
  s::Ts
  x0::Tx
end
# Distributions.rand(rng::AbstractRNG, d::Markov3{Tn, Ts}) where {Tn, Ts} = pushfirst(cumsum(rand(Normal(0, d.s), d.n-1)), zero(Ts))
Distributions.rand(rng::AbstractRNG, d::RandomWalk{Tn, Ts, Tx}) where {Tn, Ts, Tx} = begin
  x = Vector{Tx}(undef, d.n)
  x[1] = d.x0
  for i in 2:d.n
    x[i] = x[i-1] + rand(Normal(0, d.s))
  end
  return x
end
Distributions.logpdf(d::RandomWalk{Tn, Ts}, x::AbstractVector{T}) where {Tn, Ts, T} =
    logpdf( MvNormal( d.n-1, d.s ), diff(x) )# + logpdf( Normal( zero(Ts), x[1] ) )
Bijectors.bijector(d::RandomWalk) = Identity{1}()
## =============================================================================
#                      include
# ==============================================================================
include("models/parametric_model.jl")
include("models/random-walk-models.jl")

## =============================================================================
#                      model_constant_iar_contacts
# ==============================================================================
@model function model_constant_iar_contacts(
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
	covariates,
	covariates_start,
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
	num_si = length(serial_intervals)
	num_π = length(π)
	num_π2 = length(π2)
	num_π3 = length(π3)
	num_obs = length(cases)
	num_Rt_steps = num_obs - lockdown
    num_obs = length(cases)
	num_Rt_steps = num_obs - lockdown
	num_covariates = size(covariates, 2)
	# If we don't want to predict the future, we only need to compute up-to time-step `num_obs_states[m]`
    num_time_steps = predict ? num_total_days : num_obs

    # Latent variables
    τ         ~ Exponential(1 / 0.03) # `Exponential` has inverse parameterization of the one in Stan
	T = typeof(τ)
    y         ~ truncated(Exponential(τ),T(0),T(1000))
	ϕ         ~ truncated(Normal(0, 5), 0, Inf)
    ϕ2        ~ truncated(Normal(25, 5), 10, 40)
    ϕ3        ~ truncated(Normal(25, 5), 10, 40)
	R0        ~ truncated(Normal(3.6, .8), 2, 4.5)
	R1        ~ truncated(Normal(.8, .1), .5, 1.1)
    ifr       ~ truncated(Normal(6/1000, 3/1000), 3/1000, 10/1000)
	ihr       ~ truncated(Normal(1/100,1/100),.1/100,5/100)
    iar       ~ Beta(1,10) #Beta(20,25)
	σ_rt      ~ truncated(Normal(0.05, .03), 0, .15)
	seroprev_σ ~ arraydist(InverseGamma2.(seroprev_std, Ref(.1)))
	# seroprev_μ ~ MvNormal(seroprev_mean, seroprev_mean*.01)
	α         ~ filldist(Beta(1, 5), num_covariates)

	!isfinite(R0) && (@error "R0"; Turing.@addlogprob! -Inf; return) #DEBUG
	population < num_impute*y && (@error "population < num_impute*y"; Turing.@addlogprob! -Inf; return) #DEBUG
	y<0 && (@error "y<0"; Turing.@addlogprob! -Inf; return) #DEBUG

    infected              = TV(undef, num_time_steps)
    expected_daily_cases  = TV(undef, num_time_steps)
    expected_daily_hospit = TV(undef, num_time_steps)
	expected_seropositive = TV(undef, num_time_steps)
    cumcases              = TV(undef, num_time_steps)
    expected_daily_deaths = TV(undef, num_time_steps)
    latent_Rt             = TV(undef, num_time_steps-lockdown)
	Rt                    = TV(undef, num_time_steps)
    Rt_adj                = TV(undef, num_time_steps)

    # Imputation of `num_impute` days
    infected[1:num_impute] .= y
    cumcases[1] = zero(V)
    cumcases[2:num_impute] .= cumsum(infected[1:num_impute - 1])

	latent_Rt[1] = invlink(R1)
	for t in 2:num_Rt_steps
		latent_Rt[t] ~ Normal(latent_Rt[t-1], σ_rt)
	end
	latent_Rt[num_Rt_steps+1:end] .= latent_Rt[num_Rt_steps]
	latent_Rt[covariates_start-lockdown:end] .= latent_Rt[covariates_start-lockdown:end] .+ covariates[1:num_time_steps-covariates_start+1, :] * α #
	Rt[1:lockdown] .= R0
	Rt[lockdown+1:num_time_steps] = link.(latent_Rt)


	# Adjusts for portion of pop that are susceptible
    @. Rt_adj[1:num_impute] = (max(population - cumcases[1:num_impute], zero(V)) / population) * Rt[1:num_impute]
    for t = (num_impute + 1):num_time_steps
        # Update cumulative infected
        cumcases[t] = cumcases[t - 1] + infected[t - 1]
        # Adjusts for portion of pop that are susceptible
        susceptible = (max(population - cumcases[t], zero(V)) / population)
        Rt_adj[t] = susceptible * Rt[t]
        infected[t] = Rt_adj[t] * sum(infected[τ] * serial_intervals[t - τ] for τ = (t - 1):-1:max(t-num_si,1)) #max(t-50, 1)
    end
	!all( 0 .< infected .< population ) && (@error "infected<0"; Turing.@addlogprob! -Inf; return)

    expected_daily_deaths[1] = 1e-15 * infected[1]
	expected_daily_hospit[1] = 1e-15 * infected[1]
	expected_daily_cases[1]  = 1e-15 * infected[1]
	expected_seropositive[1] = 1e-15 * infected[1]
	for t = 2:num_time_steps
		expected_daily_cases[t]  = sum(infected[τ] * π2[t - τ] for τ = (t - 1):-1:max(t-num_π2,1)) #max(t-30, 1)
		expected_daily_deaths[t] = ifr * sum(infected[τ] * π[t - τ] for τ = (t - 1):-1:max(t-num_π,1)) #max(t-100, 1)
        expected_daily_hospit[t] = ihr * sum(infected[τ] * π3[t - τ] for τ = (t - 1):-1:max(t-num_π3,1)) #max(t-50, 1)
		expected_seropositive[t] = sum( infected[τ] * π4[t - τ] for τ = 1:t-1 )
    end
	observed_daily_cases = iar * expected_daily_cases
	# Extract time-steps for which we have observations

	ts = epidemic_start:num_obs
	# μs  = expected_daily_deaths[ts]
	# σ2s = @. μs + μs^2 / ϕ
    # deaths[ts] ~ arraydist(NegativeBinomial3.(μs, σ2s))
	μs  = expected_daily_hospit[ts]
	!all( 0 .< μs .< population ) && (@error "expected_daily_hospit"; Turing.@addlogprob! -Inf; return)
	(ϕ3 <= 0) && (@error "ϕ3 <= 0"; Turing.@addlogprob! -Inf; return)
	hospit[ts] ~ arraydist(NegativeBinomial2.(μs, ϕ3))
    if num_case_obs > 0
		ts = (num_obs-num_case_obs):num_obs
		μs  = observed_daily_cases[ts]
		!all( 0 .< μs .< population ) && ( @error "observed_daily_cases < 0"; Turing.@addlogprob! -Inf; return)
		(ϕ2 <= 0) && (@error "ϕ2 <= 0"; Turing.@addlogprob! -Inf; return)
        cases[ts] ~ arraydist(NegativeBinomial2.(μs, ϕ2))
    end

    # seroprevalence study
	seroprev_mean ~ MvNormal(expected_seropositive[seroprev_idx], seroprev_σ)
	# seroprev_μ ~ MvNormal(cumcases[seroprev_idx], seroprev_σ)

    return (
		observed_daily_cases  = observed_daily_cases,
        expected_daily_deaths = expected_daily_deaths,
        expected_daily_hospit = expected_daily_hospit,
        Rt = Rt,
		expected_seropositive = expected_seropositive,
		iar = iar,
    )
end
## =============================================================================
#                      model_constant_iar
# ==============================================================================
@model function model_constant_iar(
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
	num_si = length(serial_intervals)
	num_π = length(π)
	num_π2 = length(π2)
	num_π3 = length(π3)
	num_obs = length(cases)
	num_Rt_steps = num_obs - lockdown
    num_obs = length(cases)
	num_Rt_steps = num_obs - lockdown
	# If we don't want to predict the future, we only need to compute up-to time-step `num_obs_states[m]`
    num_time_steps = predict ? num_total_days : num_obs

    # Latent variables
    τ         ~ Exponential(1 / 0.03) # `Exponential` has inverse parameterization of the one in Stan
	T = typeof(τ)
    y         ~ truncated(Exponential(τ),T(0),T(1000))
	ϕ         ~ truncated(Normal(0, 5), 0, Inf)
    ϕ2        ~ truncated(Normal(25, 5), 10, 40)
    ϕ3        ~ truncated(Normal(25, 5), 10, 40)
	R0        ~ truncated(Normal(3.6, .8), 2, 4.5)
	R1        ~ truncated(Normal(.8, .1), .5, 1.1)
    ifr       ~ truncated(Normal(6/1000, 3/1000), 3/1000, 10/1000)
	ihr       ~ truncated(Normal(1/100,1/100),.1/100,5/100)
    iar       ~ Beta(1,10) #Beta(20,25)
	σ_rt      ~ truncated(Normal(0.05, .03), 0, .15)
	seroprev_σ ~ arraydist(InverseGamma2.(seroprev_std, Ref(.1)))
	# seroprev_μ ~ MvNormal(seroprev_mean, seroprev_mean*.01)

	!isfinite(R0) && (@error "R0"; Turing.@addlogprob! -Inf; return) #DEBUG
	population < num_impute * y && (@error "population < num_impute*y"; Turing.@addlogprob! -Inf; return) #DEBUG
	y<0 && (@error "y<0"; Turing.@addlogprob! -Inf; return) #DEBUG

    infected              = TV(undef, num_time_steps)
    expected_daily_cases  = TV(undef, num_time_steps)
    expected_daily_hospit = TV(undef, num_time_steps)
	expected_seropositive = TV(undef, num_time_steps)
    cumcases              = TV(undef, num_time_steps)
    expected_daily_deaths = TV(undef, num_time_steps)
    latent_Rt             = TV(undef, num_Rt_steps)
	Rt                    = TV(undef, num_time_steps)
    Rt_adj                = TV(undef, num_time_steps)

    # Imputation of `num_impute` days
    infected[1:num_impute] .= y
    cumcases[1] = zero(V)
    cumcases[2:num_impute] .= cumsum(infected[1:num_impute - 1])

	latent_Rt[1] = invlink(R1)
	for t in 2:num_Rt_steps
		latent_Rt[t] ~ Normal(latent_Rt[t-1], σ_rt)
	end

	Rt[1:lockdown] .= R0
	Rt[lockdown+1:num_obs] = link.(latent_Rt)
	if predict #if prediction
		Rt[num_obs+1:num_total_days] .= Rt[num_obs]
	end

	# Adjusts for portion of pop that are susceptible
    @. Rt_adj[1:num_impute] = (max(population - cumcases[1:num_impute], zero(V)) / population) * Rt[1:num_impute]
    for t = (num_impute + 1):num_time_steps
        # Update cumulative infected
        cumcases[t] = cumcases[t - 1] + infected[t - 1]
        # Adjusts for portion of pop that are susceptible
        susceptible = (max(population - cumcases[t], zero(V)) / population)
        Rt_adj[t] = susceptible * Rt[t]
        infected[t] = Rt_adj[t] * sum(infected[τ] * serial_intervals[t - τ] for τ = (t - 1):-1:max(t-num_si,1)) #max(t-50, 1)
    end
	!all( 0 .< infected .< population ) && (@error "infected<0"; Turing.@addlogprob! -Inf; return)

    expected_daily_deaths[1] = 1e-15 * infected[1]
	expected_daily_hospit[1] = 1e-15 * infected[1]
	expected_daily_cases[1]  = 1e-15 * infected[1]
	expected_seropositive[1] = 1e-15 * infected[1]
	for t = 2:num_time_steps
		expected_daily_cases[t]  = sum(infected[τ] * π2[t - τ] for τ = (t - 1):-1:max(t-num_π2,1)) #max(t-30, 1)
		expected_daily_deaths[t] = ifr * sum(infected[τ] * π[t - τ] for τ = (t - 1):-1:max(t-num_π,1)) #max(t-100, 1)
        expected_daily_hospit[t] = ihr * sum(infected[τ] * π3[t - τ] for τ = (t - 1):-1:max(t-num_π3,1)) #max(t-50, 1)
		expected_seropositive[t] = sum( infected[τ] * π4[t - τ] for τ = 1:t-1 )
    end
	observed_daily_cases = iar * expected_daily_cases
	# Extract time-steps for which we have observations

	ts = epidemic_start: num_obs
	# μs  = expected_daily_deaths[ts]
	# σ2s = @. μs + μs^2 / ϕ
    # deaths[ts] ~ arraydist(NegativeBinomial3.(μs, σ2s))
	μs  = expected_daily_hospit[ts]
	!all( 0 .< μs .< population ) && (@error "expected_daily_hospit"; Turing.@addlogprob! -Inf; return)
	(ϕ3 <= 0) && (@error "ϕ3 <= 0"; Turing.@addlogprob! -Inf; return)
	hospit[ts] ~ arraydist(NegativeBinomial2.(μs, ϕ3))
    if num_case_obs > 0
		ts = (num_obs-num_case_obs):num_obs
		μs  = observed_daily_cases[ts]
		!all( 0 .< μs .< population ) && ( @error "observed_daily_cases < 0"; Turing.@addlogprob! -Inf; return)
		(ϕ2 <= 0) && (@error "ϕ2 <= 0"; Turing.@addlogprob! -Inf; return)
        cases[ts] ~ arraydist(NegativeBinomial2.(μs, ϕ2))
    end

    # seroprevalence study
	seroprev_mean ~ MvNormal(expected_seropositive[seroprev_idx], seroprev_σ)
	# seroprev_μ ~ MvNormal(cumcases[seroprev_idx], seroprev_σ)

    return (
		observed_daily_cases  = observed_daily_cases,
        expected_daily_deaths = expected_daily_deaths,
        expected_daily_hospit = expected_daily_hospit,
        Rt = Rt,
		expected_seropositive = expected_seropositive,
		iar = iar,
    )
end
## =============================================================================
#                      model_contacts_v3
# ==============================================================================
@model function model_contacts_v3(
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
	covariates,
	covariates_start,
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
	num_si = length(serial_intervals)
	num_π = length(π)
	num_π2 = length(π2)
	num_π3 = length(π3)
	num_obs = length(cases)
	num_Rt_steps = num_obs - lockdown
    num_obs = length(cases)
	num_Rt_steps = num_obs - lockdown
	num_covariates = size(covariates, 2)
	# If we don't want to predict the future, we only need to compute up-to time-step `num_obs_states[m]`
    num_time_steps = predict ? num_total_days : num_obs

    # Latent variables
    τ         ~ Exponential(1 / 0.03) # `Exponential` has inverse parameterization of the one in Stan
	T = typeof(τ)
    y         ~ truncated(Exponential(τ),T(0),T(1000))
	ϕ         ~ truncated(Normal(0, 5), 0, Inf)
    ϕ2        ~ truncated(Normal(25, 5), 10, 40)
    ϕ3        ~ truncated(Normal(25, 5), 10, 40)
	R0        ~ truncated(Normal(3.6, .8), 2, 4.5)
	R1        ~ truncated(Normal(.8, .1), .5, 1.1)
    ifr       ~ truncated(Normal(6/1000, 3/1000), 3/1000, 10/1000)
	ihr       ~ truncated(Normal(1/100,1/100),.1/100,5/100)
    iar       ~ Beta(1,10) #Beta(20,25)
	σ_rt      ~ truncated(Normal(0.05, .03), 0, .15)
	seroprev_σ ~ arraydist(InverseGamma2.(seroprev_std, Ref(.1)))
	# seroprev_μ ~ MvNormal(seroprev_mean, seroprev_mean*.01)
	α         ~ filldist(Beta(1, 5), num_covariates)

	!isfinite(R0) && (@error "R0"; Turing.@addlogprob! -Inf; return) #DEBUG
	population < num_impute*y && (@error "population < num_impute*y"; Turing.@addlogprob! -Inf; return) #DEBUG
	y<0 && (@error "y<0"; Turing.@addlogprob! -Inf; return) #DEBUG

    infected              = TV(undef, num_time_steps)
    expected_daily_cases  = TV(undef, num_time_steps)
    expected_daily_hospit = TV(undef, num_time_steps)
	expected_seropositive = TV(undef, num_time_steps)
    cumcases              = TV(undef, num_time_steps)
    expected_daily_deaths = TV(undef, num_time_steps)
    latent_Rt             = TV(undef, num_time_steps-lockdown)
	Rt                    = TV(undef, num_time_steps)
    Rt_adj                = TV(undef, num_time_steps)

    # Imputation of `num_impute` days
    infected[1:num_impute] .= y
    cumcases[1] = zero(V)
    cumcases[2:num_impute] .= cumsum(infected[1:num_impute - 1])

	latent_Rt[1] = invlink(R1)
	for t in 2:num_Rt_steps
		latent_Rt[t] ~ Normal(latent_Rt[t-1], σ_rt)
	end
	latent_Rt[num_Rt_steps+1:end] .= latent_Rt[num_Rt_steps]
	latent_Rt[covariates_start-lockdown:end] .= latent_Rt[covariates_start-lockdown:end] .+ covariates[1:num_time_steps-covariates_start+1, :] * α #
	Rt[1:lockdown] .= R0
	Rt[lockdown+1:num_time_steps] = link.(latent_Rt)


	# Adjusts for portion of pop that are susceptible
    @. Rt_adj[1:num_impute] = (max(population - cumcases[1:num_impute], zero(V)) / population) * Rt[1:num_impute]
    for t = (num_impute + 1):num_time_steps
        # Update cumulative infected
        cumcases[t] = cumcases[t - 1] + infected[t - 1]
        # Adjusts for portion of pop that are susceptible
        susceptible = (max(population - cumcases[t], zero(V)) / population)
        Rt_adj[t] = susceptible * Rt[t]
        infected[t] = Rt_adj[t] * sum(infected[τ] * serial_intervals[t - τ] for τ = (t - 1):-1:max(t-num_si,1)) #max(t-50, 1)
    end
	!all( 0 .< infected .< population ) && (@error "infected<0"; Turing.@addlogprob! -Inf; return)

    expected_daily_deaths[1] = 1e-15 * infected[1]
	expected_daily_hospit[1] = 1e-15 * infected[1]
	expected_daily_cases[1]  = 1e-15 * infected[1]
	expected_seropositive[1] = 1e-15 * infected[1]
	for t = 2:num_time_steps
		expected_daily_cases[t]  = sum(infected[τ] * π2[t - τ] for τ = (t - 1):-1:max(t-num_π2,1)) #max(t-30, 1)
		expected_daily_deaths[t] = ifr * sum(infected[τ] * π[t - τ] for τ = (t - 1):-1:max(t-num_π,1)) #max(t-100, 1)
        expected_daily_hospit[t] = ihr * sum(infected[τ] * π3[t - τ] for τ = (t - 1):-1:max(t-num_π3,1)) #max(t-50, 1)
		expected_seropositive[t] = sum( infected[τ] * π4[t - τ] for τ = 1:t-1 )
    end
	observed_daily_cases = iar * expected_daily_cases
	# Extract time-steps for which we have observations

	ts = epidemic_start:(num_obs-num_case_obs) #DEBUG JUST FOR TESTING: REPLACE WITH num_obs !
	μs  = expected_daily_deaths[ts]
	!all( 0 .< μs .< population ) && (@error "expected_daily_deaths"; Turing.@addlogprob! -Inf; return)
	(ϕ <= 0) && (@error "ϕ <= 0"; Turing.@addlogprob! -Inf; return)
	deaths[ts] ~ arraydist(NegativeBinomial2.(μs, ϕ))
	# μs  = expected_daily_hospit[ts]
	# !all( 0 .< μs .< population ) && (@error "expected_daily_hospit"; Turing.@addlogprob! -Inf; return)
	# (ϕ3 <= 0) && (@error "ϕ3 <= 0"; Turing.@addlogprob! -Inf; return)
	# hospit[ts] ~ arraydist(NegativeBinomial2.(μs, ϕ3))
    if num_case_obs > 0
		ts = (num_obs-num_case_obs):num_obs
		μs  = observed_daily_cases[ts]
		!all( 0 .< μs .< population ) && ( @error "observed_daily_cases < 0"; Turing.@addlogprob! -Inf; return)
		(ϕ2 <= 0) && (@error "ϕ2 <= 0"; Turing.@addlogprob! -Inf; return)
        cases[ts] ~ arraydist(NegativeBinomial2.(μs, ϕ2))
    end

    # seroprevalence study
	seroprev_mean[1:4] ~ MvNormal(expected_seropositive[seroprev_idx[1:4]], seroprev_σ[1:4])
	# seroprev_μ ~ MvNormal(cumcases[seroprev_idx], seroprev_σ)

    return (
		observed_daily_cases  = observed_daily_cases,
        expected_daily_deaths = expected_daily_deaths,
        expected_daily_hospit = expected_daily_hospit,
        Rt = Rt,
		expected_seropositive = expected_seropositive,
		iar = iar,
    )
end
## =============================================================================
#                      model_contacts_v4
# ==============================================================================
@model function model_contacts_v4(
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
	covariates,
	covariates_start,
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
	num_si = length(serial_intervals)
	num_π = length(π)
	num_π2 = length(π2)
	num_π3 = length(π3)
	num_obs = length(cases)
	num_Rt_steps = num_obs - lockdown
    num_obs = length(cases)
	num_Rt_steps = num_obs - lockdown
	num_covariates = size(covariates, 2)
	# If we don't want to predict the future, we only need to compute up-to time-step `num_obs_states[m]`
    num_time_steps = predict ? num_total_days : num_obs

    # Latent variables
    τ         ~ Exponential(1 / 0.03) # `Exponential` has inverse parameterization of the one in Stan
	T = typeof(τ)
    y         ~ truncated(Exponential(τ),T(0),T(1000))
	ϕ         ~ truncated(Normal(0, 5), 0, Inf)
    ϕ2        ~ truncated(Normal(25, 5), 10, 40)
    ϕ3        ~ truncated(Normal(25, 5), 10, 40)
	R0        ~ truncated(Normal(3.6, .8), 2, 4.5)
	R1        ~ truncated(Normal(.8, .1), .5, 1.1)
    ifr       ~ truncated(Normal(6/1000, 3/1000), 3/1000, 10/1000)
	ihr       ~ truncated(Normal(1/100,1/100),.1/100,5/100)
    iar0      ~ Beta(10,100) #Beta(20,25)
	σ_rt      ~ truncated(Normal(0.05, .03), 0, .15)
	σ_iar     ~ truncated(Normal(0.05, .03), 0, .15)
	seroprev_σ ~ arraydist(InverseGamma2.(seroprev_std, Ref(.1)))
	# seroprev_μ ~ MvNormal(seroprev_mean, seroprev_mean*.01)
	α         ~ filldist(Beta(1, 5), num_covariates)

	!isfinite(R0) && (@error "R0"; Turing.@addlogprob! -Inf; return) #DEBUG
	population < num_impute*y && (@error "population < num_impute*y"; Turing.@addlogprob! -Inf; return) #DEBUG
	y<0 && (@error "y<0"; Turing.@addlogprob! -Inf; return) #DEBUG

    infected              = TV(undef, num_time_steps)
    expected_daily_cases  = TV(undef, num_time_steps)
    expected_daily_hospit = TV(undef, num_time_steps)
	expected_seropositive = TV(undef, num_time_steps)
    cumcases              = TV(undef, num_time_steps)
    expected_daily_deaths = TV(undef, num_time_steps)
    latent_Rt             = TV(undef, num_time_steps-lockdown)
	Rt                    = TV(undef, num_time_steps)
    Rt_adj                = TV(undef, num_time_steps)

    # Imputation of `num_impute` days
    infected[1:num_impute] .= y
    cumcases[1] = zero(V)
    cumcases[2:num_impute] .= cumsum(infected[1:num_impute - 1])

	latent_Rt[1] = invlink(R1)
	for t in 2:num_Rt_steps
		latent_Rt[t] ~ Normal(latent_Rt[t-1], σ_rt)
	end
	latent_Rt[num_Rt_steps+1:end] .= latent_Rt[num_Rt_steps]
	latent_Rt[covariates_start-lockdown:end] .= latent_Rt[covariates_start-lockdown:end] .+ covariates[1:num_time_steps-covariates_start+1, :] * α #
	Rt[1:lockdown] .= R0
	Rt[lockdown+1:num_time_steps] = link.(latent_Rt)


	# Adjusts for portion of pop that are susceptible
    @. Rt_adj[1:num_impute] = (max(population - cumcases[1:num_impute], zero(V)) / population) * Rt[1:num_impute]
    for t = (num_impute + 1):num_time_steps
        # Update cumulative infected
        cumcases[t] = cumcases[t - 1] + infected[t - 1]
        # Adjusts for portion of pop that are susceptible
        susceptible = (max(population - cumcases[t], zero(V)) / population)
        Rt_adj[t] = susceptible * Rt[t]
        infected[t] = Rt_adj[t] * sum(infected[τ] * serial_intervals[t - τ] for τ = (t - 1):-1:max(t-num_si,1)) #max(t-50, 1)
    end
	!all( 0 .< infected .< population ) && (@error "infected<0"; Turing.@addlogprob! -Inf; return)

	latent_iar = TV(undef, num_iar_steps)
	latent_iar[1] = logit(iar0)
	for i in 2:num_iar_steps
		latent_iar[i] ~ Normal(latent_iar[i-1], σ_iar)
	end
	iar = logistic.( latent_iar[iar_index[1:num_time_steps]] )

    expected_daily_deaths[1] = 1e-15 * infected[1]
	expected_daily_hospit[1] = 1e-15 * infected[1]
	expected_daily_cases[1]  = 1e-15 * infected[1]
	expected_seropositive[1] = 1e-15 * infected[1]
	for t = 2:num_time_steps
		expected_daily_cases[t]  = sum(infected[τ] * π2[t - τ] for τ = (t - 1):-1:max(t-num_π2,1)) #max(t-30, 1)
		expected_daily_deaths[t] = ifr * sum(infected[τ] * π[t - τ] for τ = (t - 1):-1:max(t-num_π,1)) #max(t-100, 1)
        expected_daily_hospit[t] = ihr * sum(infected[τ] * π3[t - τ] for τ = (t - 1):-1:max(t-num_π3,1)) #max(t-50, 1)
		expected_seropositive[t] = sum( infected[τ] * π4[t - τ] for τ = 1:t-1 )
    end
	observed_daily_cases = iar .* expected_daily_cases
	# Extract time-steps for which we have observations

	ts = epidemic_start:(num_obs-num_case_obs) #DEBUG JUST FOR TESTING: REPLACE WITH num_obs !
	μs  = expected_daily_deaths[ts]
	!all( 0 .< μs .< population ) && (@error "expected_daily_deaths"; Turing.@addlogprob! -Inf; return)
	(ϕ <= 0) && (@error "ϕ <= 0"; Turing.@addlogprob! -Inf; return)
	deaths[ts] ~ arraydist(NegativeBinomial2.(μs, ϕ))
	# μs  = expected_daily_hospit[ts]
	# !all( 0 .< μs .< population ) && (@error "expected_daily_hospit"; Turing.@addlogprob! -Inf; return)
	# (ϕ3 <= 0) && (@error "ϕ3 <= 0"; Turing.@addlogprob! -Inf; return)
	# hospit[ts] ~ arraydist(NegativeBinomial2.(μs, ϕ3))
    if num_case_obs > 0
		ts = (num_obs-num_case_obs):num_obs
		μs  = observed_daily_cases[ts]
		!all( 0 .< μs .< population ) && ( @error "observed_daily_cases < 0"; Turing.@addlogprob! -Inf; return)
		(ϕ2 <= 0) && (@error "ϕ2 <= 0"; Turing.@addlogprob! -Inf; return)
        cases[ts] ~ arraydist(NegativeBinomial2.(μs, ϕ2))
    end

    # seroprevalence study
	seroprev_mean ~ MvNormal(expected_seropositive[seroprev_idx], seroprev_σ)
	# seroprev_μ ~ MvNormal(cumcases[seroprev_idx], seroprev_σ)

    return (
		observed_daily_cases  = observed_daily_cases,
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
	num_obs = length(cases)
	num_Rt_steps = num_obs - lockdown
	num_sero = length(seroprev_mean)
	# If we don't want to predict the future, we only need to compute up-to time-step `num_obs_states[m]`
    num_time_steps = predict ? num_total_days : num_obs

    ############## 1.) Priors
    τ         ~ Exponential(1 / 0.03) # `Exponential` has inverse parameterization of the one in Stan
	T = typeof(τ)
    y         ~ truncated(Exponential(τ),T(0),T(1000))
    ϕ_c       ~ truncated(Normal(25, 5), 10, 40)
    ϕ_h       ~ truncated(Normal(25, 5), 10, 40)
    ϕ_d       ~ truncated(Normal(0, 5), 0, Inf)
	R0        ~ truncated(Normal(3.6, .8), 2, 4.5)
	R1        ~ truncated(Normal(.8, .1), .5, 1.1)
    ifr       ~ truncated(Normal(6/1000, 3/1000), 3/1000, 10/1000)
	ihr       ~ truncated(Normal(1/100,1/100),.1/100,5/100)
    iar       ~ Beta(1,10) #Beta(20,25)
	σ_rt       ~ truncated(Normal(0.05, .03), 0, .15)
	seroprev_σ ~ arraydist(InverseGamma2.(seroprev_std, Ref(.1)))
	latent_Rt  ~ RandomWalk(num_Rt_steps, σ_rt, invlink(R1))

    μ_si       ~ truncated(Normal(5.06, 0.33), 0, Inf)
    μ_i2c      ~ truncated(Normal(11., 2.), 0, Inf)
    μ_i2h      ~ truncated(Normal(12., 3.), 0, Inf)
    μ_i2d      ~ truncated(Normal(22., 4.), 0, Inf)

    ############# 2.) time varying reproduction number
    Rt = TV(undef, num_time_steps)
    Rt[1:lockdown] .= R0
	Rt[lockdown+1:num_obs] = link.(latent_Rt)
	if predict #if prediction
		Rt[num_obs+1:num_total_days] .= Rt[num_obs]
	end

    ############ 3.) infection dynamics
	## 3.1) initialize serial interval
	num_si = 15
	σ_si   = 2.11
	si     = pdf.( Ref(GammaMeanStd(μ_si, σ_si)), 1:num_si )
	si    /= sum(si)

	## 3.2) seeding
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

	ϕ_i2c = 5.41
	ϕ_i2h = 5.41
	ϕ_i2d = 14.26

	i2c = pdf.( Ref(NegativeBinomial2(μ_i2c, ϕ_i2c)), 1:num_i2c )
	i2h = pdf.( Ref(NegativeBinomial2(μ_i2h, ϕ_i2h)), 1:num_i2h )
	i2d = pdf.( Ref(NegativeBinomial2(μ_i2d, ϕ_i2d)), 1:num_i2d )

	i2c /= sum(i2c)
	i2h /= sum(i2h)
	i2d /= sum(i2d)

    ### 4.2) derive observables from infections
    expected_daily_cases     = TV(undef, num_time_steps)
    expected_daily_hospit    = TV(undef, num_time_steps)
    expected_daily_deaths    = TV(undef, num_time_steps)
	expected_daily_cases[1]  = 1e-15 * newly_infected[1]
    expected_daily_hospit[1] = 1e-15 * newly_infected[1]
    expected_daily_deaths[1] = 1e-15 * newly_infected[1]
	for t = 2:num_time_steps
		expected_daily_cases[t]  = iar * sum(newly_infected[τ] * i2c[t - τ] for τ = (t - 1):-1:max(t-num_i2c,1))
        expected_daily_hospit[t] = ihr * sum(newly_infected[τ] * i2h[t - τ] for τ = (t - 1):-1:max(t-num_i2h,1))
        expected_daily_deaths[t] = ifr * sum(newly_infected[τ] * i2d[t - τ] for τ = (t - 1):-1:max(t-num_i2d,1))
	end

	serodelay = 14
	expected_seropositive = cumulative_infected[ seroprev_idx .- serodelay ]

    ########### 4.) compare model to observations
    ## 4.1) observe deaths
    ts_d  = epidemic_start:(num_obs-num_case_obs)
	μs_d  = expected_daily_deaths[ts_d]
	!all( 0 .< μs_d .< population ) && (@error "expected_daily_deaths"; Turing.@addlogprob! -Inf; return)
	(ϕ_d <= 0) && (@error "ϕ <= 0"; Turing.@addlogprob! -Inf; return)
	deaths[ts_d] ~ arraydist(NegativeBinomial2.(μs_d, ϕ_d))

	## 4.2) observe cases
	ts_c = (num_obs-num_case_obs):num_obs
	μs_c  = expected_daily_cases[ts_c]
	!all( 0 .< μs_c .< population ) && ( @error "expected_daily_cases < 0"; Turing.@addlogprob! -Inf; return)
	(ϕ_c <= 0) && (@error "ϕ2 <= 0"; Turing.@addlogprob! -Inf; return)
    cases[ts_c] ~ arraydist(NegativeBinomial2.(μs_c, ϕ_c))

    ## 4.3) seroprevalence study
	seroprev_mean ~ MvNormal(expected_seropositive, seroprev_σ)

    return (
		expected_daily_cases  = expected_daily_cases,
        expected_daily_deaths = expected_daily_deaths,
        expected_daily_hospit = expected_daily_hospit,
        Rt = Rt,
		cumulative_infected = cumulative_infected,
		iar = iar,
    )
end
## =============================================================================
#                      model_hospit
# ==============================================================================
@model function model_hospit(
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
	num_obs = length(cases)
	num_Rt_steps = num_obs - lockdown
	num_sero = length(seroprev_mean)
	# If we don't want to predict the future, we only need to compute up-to time-step `num_obs_states[m]`
    num_time_steps = predict ? num_total_days : num_obs

    ############## 1.) Priors
    τ         ~ Exponential(1 / 0.03) # `Exponential` has inverse parameterization of the one in Stan
	T = typeof(τ)
    y         ~ truncated(Exponential(τ),T(0),T(1000))
    ϕ_c       ~ truncated(Normal(25, 5), 10, 40)
    ϕ_h       ~ truncated(Normal(25, 5), 10, 40)
    ϕ_d       ~ truncated(Normal(0, 5), 0, Inf)
	R0        ~ truncated(Normal(3.6, .8), 2, 4.5)
	R1        ~ truncated(Normal(.8, .1), .5, 1.1)
    ifr       ~ truncated(Normal(6/1000, 3/1000), 3/1000, 10/1000)
	ihr       ~ truncated(Normal(1/100,1/100),.1/100,5/100)
    iar       ~ Beta(1,10) #Beta(20,25)
	σ_rt       ~ truncated(Normal(0.05, .03), 0, .15)
	seroprev_σ ~ arraydist(InverseGamma2.(seroprev_std, Ref(.1)))
	latent_Rt  ~ RandomWalk(num_Rt_steps, σ_rt, invlink(R1))

    μ_si       ~ truncated(Normal(5.06, 0.33), 0, Inf)
    μ_i2c      ~ truncated(Normal(11., 2.), 0, Inf)
    μ_i2h      ~ truncated(Normal(12., 3.), 0, Inf)
    μ_i2d      ~ truncated(Normal(22., 4.), 0, Inf)

    ############# 2.) time varying reproduction number
    Rt = TV(undef, num_time_steps)
    Rt[1:lockdown] .= R0
	Rt[lockdown+1:num_obs] = link.(latent_Rt)
	if predict #if prediction
		Rt[num_obs+1:num_total_days] .= Rt[num_obs]
	end

    ############ 3.) infection dynamics
	## 3.1) initialize serial interval
	num_si = 15
	σ_si   = 2.11
	si     = pdf.( Ref(GammaMeanStd(μ_si, σ_si)), 1:num_si )
	si    /= sum(si)

	## 3.2) seeding
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

	ϕ_i2c = 5.41
	ϕ_i2h = 5.41
	ϕ_i2d = 14.26

	i2c = pdf.( Ref(NegativeBinomial2(μ_i2c, ϕ_i2c)), 1:num_i2c )
	i2h = pdf.( Ref(NegativeBinomial2(μ_i2h, ϕ_i2h)), 1:num_i2h )
	i2d = pdf.( Ref(NegativeBinomial2(μ_i2d, ϕ_i2d)), 1:num_i2d )

	i2c /= sum(i2c)
	i2h /= sum(i2h)
	i2d /= sum(i2d)

    ### 4.2) derive observables from infections
    expected_daily_cases     = TV(undef, num_time_steps)
    expected_daily_hospit    = TV(undef, num_time_steps)
    expected_daily_deaths    = TV(undef, num_time_steps)
	expected_daily_cases[1]  = 1e-15 * newly_infected[1]
    expected_daily_hospit[1] = 1e-15 * newly_infected[1]
    expected_daily_deaths[1] = 1e-15 * newly_infected[1]
	for t = 2:num_time_steps
		expected_daily_cases[t]  = iar * sum(newly_infected[τ] * i2c[t - τ] for τ = (t - 1):-1:max(t-num_i2c,1))
        expected_daily_hospit[t] = ihr * sum(newly_infected[τ] * i2h[t - τ] for τ = (t - 1):-1:max(t-num_i2h,1))
        expected_daily_deaths[t] = ifr * sum(newly_infected[τ] * i2d[t - τ] for τ = (t - 1):-1:max(t-num_i2d,1))
	end

	serodelay = 14
	expected_seropositive = cumulative_infected[ seroprev_idx .- serodelay ]

    ########### 4.) compare model to observations
    ## 4.1) observe hospitalizations
	ts_h = epidemic_start:num_obs
	μs_h  = expected_daily_hospit[ts_h]
	!all( 0 .< μs_h .< population ) && ( @error "expected_daily_hospit < 0"; Turing.@addlogprob! -Inf; return)
	(ϕ_h <= 0) && (@error "ϕ_h <= 0"; Turing.@addlogprob! -Inf; return)
    hospit[ts_h] ~ arraydist(NegativeBinomial2.(μs_h, ϕ_h))

    ## 4.3) seroprevalence study
	seroprev_mean ~ MvNormal(expected_seropositive, seroprev_σ)

    return (
		expected_daily_cases  = expected_daily_cases,
        expected_daily_deaths = expected_daily_deaths,
        expected_daily_hospit = expected_daily_hospit,
        Rt = Rt,
		cumulative_infected = cumulative_infected,
		iar = iar,
    )
end
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
	num_obs = length(cases)
	num_Rt_steps = num_obs - lockdown
	num_sero = length(seroprev_mean)
	# If we don't want to predict the future, we only need to compute up-to time-step `num_obs_states[m]`
    num_time_steps = predict ? num_total_days : num_obs

    ############## 1.) Priors
    τ         ~ Exponential(1 / 0.03) # `Exponential` has inverse parameterization of the one in Stan
	T = typeof(τ)
    y         ~ truncated(Exponential(τ),T(0),T(1000))
    ϕ_c       ~ truncated(Normal(25, 5), 10, 40)
    ϕ_h       ~ truncated(Normal(25, 5), 10, 40)
    ϕ_d       ~ truncated(Normal(0, 5), 0, Inf)
	R0        ~ truncated(Normal(3.6, .8), 2, 4.5)
	R1        ~ truncated(Normal(.8, .1), .5, 1.1)
    ifr       ~ truncated(Normal(6/1000, 3/1000), 3/1000, 10/1000)
	ihr       ~ truncated(Normal(1/100,1/100),.1/100,5/100)
    iar       ~ Beta(1,10) #Beta(20,25)
	σ_rt       ~ truncated(Normal(0.05, .03), 0, .15)
	seroprev_σ ~ arraydist(InverseGamma2.(seroprev_std, Ref(.1)))
	latent_Rt  ~ RandomWalk(num_Rt_steps, σ_rt, invlink(R1))

    μ_si       ~ truncated(Normal(5.06, 0.33), 0, Inf)
    μ_i2c      ~ truncated(Normal(11., 2.), 0, Inf)
    μ_i2h      ~ truncated(Normal(12., 3.), 0, Inf)
    μ_i2d      ~ truncated(Normal(22., 4.), 0, Inf)

    ############# 2.) time varying reproduction number
    Rt = TV(undef, num_time_steps)
    Rt[1:lockdown] .= R0
	Rt[lockdown+1:num_obs] = link.(latent_Rt)
	if predict #if prediction
		Rt[num_obs+1:num_total_days] .= Rt[num_obs]
	end

    ############ 3.) infection dynamics
	## 3.1) initialize serial interval
	num_si = 15
	σ_si   = 2.11
	si     = pdf.( Ref(GammaMeanStd(μ_si, σ_si)), 1:num_si )
	si    /= sum(si)

	## 3.2) seeding
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

	ϕ_i2c = 5.41
	ϕ_i2h = 5.41
	ϕ_i2d = 14.26

	i2c = pdf.( Ref(NegativeBinomial2(μ_i2c, ϕ_i2c)), 1:num_i2c )
	i2h = pdf.( Ref(NegativeBinomial2(μ_i2h, ϕ_i2h)), 1:num_i2h )
	i2d = pdf.( Ref(NegativeBinomial2(μ_i2d, ϕ_i2d)), 1:num_i2d )

	i2c /= sum(i2c)
	i2h /= sum(i2h)
	i2d /= sum(i2d)

    ### 4.2) derive observables from infections
    expected_daily_cases     = TV(undef, num_time_steps)
    expected_daily_hospit    = TV(undef, num_time_steps)
    expected_daily_deaths    = TV(undef, num_time_steps)
	expected_seropositive    = TV(undef, num_time_steps)

	expected_daily_cases[1]  = 1e-15 * newly_infected[1]
    expected_daily_hospit[1] = 1e-15 * newly_infected[1]
    expected_daily_deaths[1] = 1e-15 * newly_infected[1]
	for t = 2:num_time_steps
		expected_daily_cases[t]  = iar * sum(newly_infected[τ] * i2c[t - τ] for τ = (t - 1):-1:max(t-num_i2c,1))
        expected_daily_hospit[t] = ihr * sum(newly_infected[τ] * i2h[t - τ] for τ = (t - 1):-1:max(t-num_i2h,1))
        expected_daily_deaths[t] = ifr * sum(newly_infected[τ] * i2d[t - τ] for τ = (t - 1):-1:max(t-num_i2d,1))
	end

	serodelay = 30 # 14d self deferral + 16d seroconversion (see Erikstrup et al. https://doi.org/10.1093/cid/ciaa849)
	expected_seropositive[1:serodelay]    .= zero(V)
	expected_seropositive[serodelay+1:end] = cumulative_infected[1:end-serodelay]

    ########### 4.) compare model to observations
    ## 4.1) observe hospitalizations
	ts_d = epidemic_start:num_obs
	μs_d  = expected_daily_deaths[ts_d]
	!all( 0 .< μs_d .< population ) && ( @error "expected_daily_deaths < 0"; Turing.@addlogprob! -Inf; return)
	(ϕ_d <= 0) && (@error "ϕ_d <= 0"; Turing.@addlogprob! -Inf; return)
    deaths[ts_d] ~ arraydist(NegativeBinomial2.(μs_d, ϕ_d))

    ## 4.3) seroprevalence study
	seroprev_mean[1:4] ~ MvNormal(expected_seropositive[seroprev_idx[1:4]], seroprev_σ[1:4]) # select only early observations

    return (
		expected_daily_cases  = expected_daily_cases,
        expected_daily_deaths = expected_daily_deaths,
        expected_daily_hospit = expected_daily_hospit,
        Rt = Rt,
		cumulative_infected = cumulative_infected,
		iar = iar,
    )
end
## =============================================================================
#                      model_contacts_fitted
# ==============================================================================
@model function model_contacts_fitted(
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
    covariates,
	covariates_start,
	π3,                 # [AbstractVector{<:AbstractVector{<:Real}}] h * s
    hospit,
	seroprev_mean,
    seroprev_std,
    seroprev_idx,
	priors,
    predict=false,     # [Bool] if `false`, will only compute what's needed to `observe` but not more
    ::Type{TV} = Vector{Float64},
	::Type{V}  = Float64;
	invlink = log,#KLogistic(V(3))
	link    = exp,#KLogit(V(k))
) where {TV, V}
	num_si = length(serial_intervals)
	num_π = length(π)
	num_π2 = length(π2)
	num_π3 = length(π3)
	num_obs = length(cases)
	num_Rt_steps = num_obs - lockdown
    num_obs = length(cases)
	num_covariates = size(covariates, 2)
	num_Rt_steps = num_obs - lockdown
    # If we don't want to predict the future, we only need to compute up-to time-step `num_obs_states[m]`
    num_time_steps = predict ? num_total_days : num_obs

    # Latent variables
    y         ~ truncated( priors[:y], 0, 1000)
    ϕ2        ~ truncated( priors[:ϕ2], 0, Inf)
    ϕ3        ~ truncated( priors[:ϕ3], 0, Inf)
	R0        ~ truncated( priors[:R0], 2, 4.5)
	R1        ~ truncated( priors[:R1], .5, 1.1)
    ifr       ~ truncated( priors[:ifr], 3/1000, 10/1000)
	ihr       ~ truncated( priors[:ihr],.1/100,5/100)
    iar0      ~ truncated( priors[:iar0], 0, 1)
	α         ~ filldist(Beta(1, 100), num_covariates)
	# α         = zeros(V, num_covariates)
	seroprev_σ ~ arraydist( priors[:seroprev_σ] )

    infected              = TV(undef, num_time_steps)
    expected_daily_cases  = TV(undef, num_time_steps)
    expected_daily_hospit = TV(undef, num_time_steps)
    cumcases              = TV(undef, num_time_steps)
    expected_daily_deaths = TV(undef, num_time_steps)
	latent_Rt             = TV(undef, num_Rt_steps)
	Rt                    = TV(undef, num_time_steps)
    Rt_adj                = TV(undef, num_time_steps)
	latent_iar             = TV(undef, num_iar_steps)

    # Imputation of `num_impute` days
    infected[1:num_impute] .= y
    cumcases[1] = zero(V)
    cumcases[2:num_impute] .= cumsum(infected[1:num_impute - 1])

	# hidden markov model for Rt
	latent_Rt[1] = invlink(R1)
	latent_Rt[2:num_Rt_steps] ~ arraydist(priors[:logRt])
	Rt[1:lockdown] .= R0
	Rt[lockdown+1:num_obs] = link.(latent_Rt)
	if predict #if prediction
		Rt[num_obs+1:num_total_days] .= Rt[num_obs]
	end

	# Adjusts for portion of pop that are susceptible
	@. Rt_adj[1:num_impute] = (max(population - cumcases[1:num_impute], zero(V)) / population) * Rt[1:num_impute]

    for t = (num_impute + 1):num_time_steps
        # Update cumulative infected
        cumcases[t] = cumcases[t - 1] + infected[t - 1]
        # Adjusts for portion of pop that are susceptible
        susceptible = (max(population - cumcases[t], zero(V)) / population)
        Rt_adj[t] = susceptible * Rt[t]
        infected[t] = Rt_adj[t] * sum(infected[τ] * serial_intervals[t - τ] for τ = (t - 1):-1:max(t-num_si,1)) #max(t-50, 1)
    end
	any( infected .< 0 ) && (i = findfirst(<(0), infected); @error "infected<0 at $(i)"; Turing.@addlogprob! -Inf; return)
	any( infected .> population ) && (i = findfirst(>(population), infected); @error "infected>population at $(i)"; Turing.@addlogprob! -Inf; return)
	any( .! isfinite.(infected) ) && (i = findfirst(x->!isfinite(x), infected); @error "infected is not finite at $(i)"; Turing.@addlogprob! -Inf; return)

	latent_iar[1] = logit(iar0)
	latent_iar[2:num_iar_steps] ~ arraydist( priors[:latent_iar] )
	iar = logistic.( latent_iar[ iar_index[1:num_time_steps] ] )

    expected_daily_deaths[1] = 1e-15 * infected[1]
	expected_daily_hospit[1] = 1e-15 * infected[1]
	expected_daily_cases[1] = infected[1]
    for t = 2:num_time_steps
		expected_daily_deaths[t] = ifr * sum(infected[τ] * π[t - τ] for τ = (t - 1):-1:max(t-num_π,1)) #max(t-100, 1)
        expected_daily_cases[t]  = sum(infected[τ] * π2[t - τ] for τ = (t - 1):-1:max(t-num_π2,1)) #max(t-30, 1)
        expected_daily_hospit[t] = ihr * sum(infected[τ] * π3[t - τ] for τ = (t - 1):-1:max(t-num_π3,1)) #max(t-50, 1)
    end
	observed_daily_cases = iar .* expected_daily_cases
	# Extract time-steps for which we have observations

	ts = epidemic_start:num_obs
	μs  = expected_daily_hospit[ts]
	!all( 0 .< μs .< population ) && (@error "expected_daily_hospit"; Turing.@addlogprob! -Inf; return)
	(ϕ3 <= 0) && (@error "ϕ3 <= 0"; Turing.@addlogprob! -Inf; return)
	hospit[ts] ~ arraydist(NegativeBinomial2.(μs, ϕ3))
    if num_case_obs > 0
		ts = (num_obs-num_case_obs):num_obs
		μs  = observed_daily_cases[ts]
		!all( 0 .< μs .< population ) && (@error "observed_daily_cases"; Turing.@addlogprob! -Inf; return)
		(ϕ3 <= 0) && (@error "ϕ2 <= 0"; Turing.@addlogprob! -Inf; return)
		cases[ts] ~ arraydist(NegativeBinomial2.(μs, ϕ2))
	end

    # seroprevalence study
	seroprev_mean ~ MvNormal(cumcases[seroprev_idx], seroprev_σ)

    return (
		observed_daily_cases  = observed_daily_cases,
        expected_daily_deaths = expected_daily_deaths,
        expected_daily_hospit = expected_daily_hospit,
        Rt = Rt,
		cumulative_cases      = cumcases,
		iar = iar
    )
end
# ## =============================================================================
# #                      model_contacts
# # ==============================================================================
# @model function model_contacts(
#     num_impute,        # [Int] num. of days for which to impute infections
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
# 	covariates,
# 	covariates_start,
# 	π3,                 # [AbstractVector{<:AbstractVector{<:Real}}] h * s
#     hospit,
# 	seroprev_mean,
#     seroprev_std,
#     seroprev_idx,
#     predict=false,     # [Bool] if `false`, will only compute what's needed to `observe` but not more
#     ::Type{TV} = Vector{Float64},
# 	::Type{V}  = Float64;
# 	invlink = log,#KLogit(V(3))
# 	link    = exp,#KLogistic(V(3))
# ) where {TV, V}
#     num_obs = length(cases)
# 	num_covariates = size(covariates, 2)
# 	num_Rt_steps = num_obs - lockdown
# 	# If we don't want to predict the future, we only need to compute up-to time-step `num_obs_states[m]`
#     num_time_steps = predict ? num_total_days : num_obs
#
#     # Latent variables
#     τ         ~ Exponential(1 / 0.03) # `Exponential` has inverse parameterization of the one in Stan
# 	T = typeof(τ)
#     y         ~ truncated(Exponential(τ),T(0),T(1000))
# 	ϕ         ~ truncated(Normal(0, 5), 0, Inf)
#     ϕ2        ~ truncated(Normal(25, 5), 10, 40)
#     ϕ3        ~ truncated(Normal(25, 5), 10, 40)
# 	R0        ~ truncated(Normal(3.6, .8), 2, 5.)
# 	R1        ~ truncated(Normal(.8, .1), .5, 1.1)
#     ifr       ~ truncated(Normal(6/1000, 3/1000), 3/1000, 10/1000)
# 	ihr       ~ truncated(Normal(1/100,1/100),.1/100,5/100)
#     iar0      ~ Beta(10,100) #Beta(20,25)
# 	σ_rt      ~ truncated(Normal(0.05, .03), 0, .15)
# 	σ_iar     ~ InverseGamma2(0.1, .2)
# 	# σ_rt      = .05
# 	# σ_iar     = .05
# 	seroprev_σ ~ arraydist(InverseGamma2.(seroprev_std, Ref(.001)))
# 	# seroprev_μ ~ MvNormal(seroprev_mean, seroprev_mean*.01)
# 	α         ~ filldist(Beta(1, 100), num_covariates)
#
# 	!isfinite(R0) && (@error "R0"; Turing.@addlogprob! -Inf; return) #DEBUG
# 	population < num_impute*y && (@error "population < num_impute*y"; Turing.@addlogprob! -Inf; return) #DEBUG
# 	y<0 && (@error "y<0"; Turing.@addlogprob! -Inf; return) #DEBUG
#
#     infected              = TV(undef, num_time_steps)
#     expected_daily_cases  = TV(undef, num_time_steps)
#     expected_daily_hospit = TV(undef, num_time_steps)
#     cumcases              = TV(undef, num_time_steps)
#     expected_daily_deaths = TV(undef, num_time_steps)
# 	latent_Rt             = TV(undef, num_Rt_steps)
# 	Rt                    = TV(undef, num_time_steps)
#     Rt_adj                = TV(undef, num_time_steps)
#
#     # Imputation of `num_impute` days
#     infected[1:num_impute] .= y
#     cumcases[1] = zero(V)
#     cumcases[2:num_impute] .= cumsum(infected[1:num_impute - 1])
#
# 	latent_Rt[1] = invlink(R1)
# 	for t in 2:num_Rt_steps
# 		latent_Rt[t] ~ Normal(latent_Rt[t-1], σ_rt)
# 	end
# 	latent_Rt[covariates_start-lockdown:num_Rt_steps] .= latent_Rt[covariates_start-lockdown:num_Rt_steps] .+ covariates[1:num_obs-covariates_start+1, :] * α
#
# 	Rt[1:lockdown] .= R0
# 	Rt[lockdown+1:num_obs] = link.(latent_Rt)
# 	if predict #if prediction
# 		Rt[num_obs+1:num_total_days] .= Rt[num_obs]
# 	end
# 	# effect1 = 1 .+ α[1] * covariates[1:num_obs-covariates_start+1,1]
# 	# effect2 = 1 .+ α[2] * covariates[1:num_obs-covariates_start+1,2]
# 	# effect3 = 1 .+ α[3] * covariates[1:num_obs-covariates_start+1,3]
# 	# effect4 = 1 .+ α[4] * covariates[1:num_obs-covariates_start+1,4]
# 	# @. Rt[covariates_start:num_obs] *= effect1 + effect2 + effect3 + effect4
#
# 	# Adjusts for portion of pop that are susceptible
#     @. Rt_adj[1:num_impute] = (max(population - cumcases[1:num_impute], zero(V)) / population) * Rt[1:num_impute]
#     for t = (num_impute + 1):num_time_steps
#         # Update cumulative infected
#         cumcases[t] = cumcases[t - 1] + infected[t - 1]
#         # Adjusts for portion of pop that are susceptible
#         susceptible = (max(population - cumcases[t], zero(V)) / population)
#         Rt_adj[t] = susceptible * Rt[t]
#         infected[t] = Rt_adj[t] * sum(infected[τ] * serial_intervals[t - τ] for τ = 1:t-1) #max(t-50, 1)
#     end
# 	any( infected .< 0 ) && (i = findfirst(<(0), infected); @error "infected<0 at $(i)"; Turing.@addlogprob! -Inf; return)
# 	any( .! isfinite.(infected) ) && (i = findfirst(x->!isfinite(x), infected); @error "infected is not finite at $(i)"; Turing.@addlogprob! -Inf; return)
# 	any( infected .> population ) && (i = findfirst(>(population), infected); @error "infected>population at $(i)"; Turing.@addlogprob! -Inf; return)
# 	#
# 	latent_iar = TV(undef, num_iar_steps)
# 	latent_iar[1] = logit(iar0)
# 	for i in 2:num_iar_steps
# 		latent_iar[i] ~ Normal(latent_iar[i-1], σ_iar)
# 	end
# 	iar = logistic.( latent_iar[iar_index[1:num_time_steps]] )
#
#     expected_daily_deaths[1] = 1e-15 * infected[1]
# 	expected_daily_hospit[1] = 1e-15 * infected[1]
# 	expected_daily_cases[1] = infected[1]
#     for t = 2:num_time_steps
# 		expected_daily_deaths[t] = ifr * sum(infected[τ] * π[t - τ] for τ = 1:t-1) #max(t-100, 1)
#         expected_daily_cases[t]  = sum(infected[τ] * π2[t - τ] for τ = 1:t-1) #max(t-30, 1)
#         expected_daily_hospit[t] = ihr * sum(infected[τ] * π3[t - τ] for τ = 1:t-1) #max(t-50, 1)
#     end
# 	observed_daily_cases = iar .* expected_daily_cases
# 	# Extract time-steps for which we have observations
#
# 	ts = epidemic_start:num_obs
# 	# μs  = expected_daily_deaths[ts]
# 	# σ2s = @. μs + μs^2 / ϕ
#     # deaths[ts] ~ arraydist(NegativeBinomial3.(μs, σ2s))
# 	μs  = expected_daily_hospit[ts]
# 	!all( 0 .< μs .< population ) && (@error "expected_daily_hospit"; Turing.@addlogprob! -Inf; return)
# 	(ϕ3 <= 0) && (@error "ϕ3 <= 0"; Turing.@addlogprob! -Inf; return)
# 	hospit[ts] ~ arraydist(NegativeBinomial2.(μs, ϕ3))
#     if num_case_obs > 0
# 		ts = (num_obs-num_case_obs):num_obs
# 		μs  = observed_daily_cases[ts]
# 		!all( 0 .< μs .< population ) && (@error "observed_daily_cases"; Turing.@addlogprob! -Inf; return)
# 		(ϕ2 <= 0) && (@error "ϕ2 <= 0"; Turing.@addlogprob! -Inf; return)
#         cases[ts] ~ arraydist(NegativeBinomial2.(μs, ϕ2))
#     end
#
#     # seroprevalence study
# 	seroprev_mean ~ MvNormal(cumcases[seroprev_idx], seroprev_σ)
# 	# seroprev_μ ~ MvNormal(cumcases[seroprev_idx], seroprev_σ)
#
#     return (
# 		observed_daily_cases  = observed_daily_cases,
#         expected_daily_deaths = expected_daily_deaths,
#         expected_daily_hospit = expected_daily_hospit,
#         Rt = Rt,
# 		cumulative_cases      = cumcases,
# 		iar = iar
#     )
# end
## =============================================================================
#                      model_contacts_v2
# ==============================================================================
@model function model_contacts_v2(
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
	covariates,
	covariates_start,
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
	num_si = length(serial_intervals)
	num_π = length(π)
	num_π2 = length(π2)
	num_π3 = length(π3)
	num_obs = length(cases)
	num_Rt_steps = num_obs - lockdown
    num_obs = length(cases)
	num_Rt_steps = num_obs - lockdown
	num_covariates = size(covariates, 2)
	# If we don't want to predict the future, we only need to compute up-to time-step `num_obs_states[m]`
    num_time_steps = predict ? num_total_days : num_obs

    # Latent variables
    τ         ~ Exponential(1 / 0.03) # `Exponential` has inverse parameterization of the one in Stan
	T = typeof(τ)
    y         ~ truncated(Exponential(τ),T(0),T(1000))
	ϕ         ~ truncated(Normal(0, 5), 0, Inf)
    ϕ2        ~ truncated(Normal(25, 5), 10, 40)
    ϕ3        ~ truncated(Normal(25, 5), 10, 40)
	R0        ~ truncated(Normal(3.6, .8), 2, 4.5)
	R1        ~ truncated(Normal(.8, .1), .5, 1.1)
    ifr       ~ truncated(Normal(6/1000, 3/1000), 3/1000, 10/1000)
	ihr       ~ truncated(Normal(1/100,1/100),.1/100,5/100)
    iar0      ~ Beta(10,100) #Beta(20,25)
	σ_rt      ~ truncated(Normal(0.05, .03), 0, .15)
	σ_iar     ~ truncated(Normal(0.05, .03), 0, .15)
	# σ_rt      = .05
	# σ_iar     = .05
	seroprev_σ ~ arraydist(InverseGamma2.(seroprev_std, Ref(.1)))
	# seroprev_μ ~ MvNormal(seroprev_mean, seroprev_mean*.01)
	α         ~ filldist(Beta(1, 5), num_covariates)

	!isfinite(R0) && (@error "R0"; Turing.@addlogprob! -Inf; return) #DEBUG
	population < num_impute*y && (@error "population < num_impute*y"; Turing.@addlogprob! -Inf; return) #DEBUG
	y<0 && (@error "y<0"; Turing.@addlogprob! -Inf; return) #DEBUG

    infected              = TV(undef, num_time_steps)
    expected_daily_cases  = TV(undef, num_time_steps)
    expected_daily_hospit = TV(undef, num_time_steps)
	expected_seropositive = TV(undef, num_time_steps)
    cumcases              = TV(undef, num_time_steps)
    expected_daily_deaths = TV(undef, num_time_steps)
    latent_Rt             = TV(undef, num_Rt_steps)
	Rt                    = TV(undef, num_time_steps)
    Rt_adj                = TV(undef, num_time_steps)

    # Imputation of `num_impute` days
    infected[1:num_impute] .= y
    cumcases[1] = zero(V)
    cumcases[2:num_impute] .= cumsum(infected[1:num_impute - 1])

	latent_Rt[1] = invlink(R1)
	for t in 2:num_Rt_steps
		latent_Rt[t] ~ Normal(latent_Rt[t-1], σ_rt)
	end
	latent_Rt[covariates_start-lockdown:num_Rt_steps] .= latent_Rt[covariates_start-lockdown:num_Rt_steps] .+ covariates[1:num_obs-covariates_start+1, :] * α #

	Rt[1:lockdown] .= R0
	Rt[lockdown+1:num_obs] = link.(latent_Rt)
	if predict #if prediction
		Rt[num_obs+1:num_total_days] .= Rt[num_obs]
	end

	# Adjusts for portion of pop that are susceptible
    @. Rt_adj[1:num_impute] = (max(population - cumcases[1:num_impute], zero(V)) / population) * Rt[1:num_impute]
    for t = (num_impute + 1):num_time_steps
        # Update cumulative infected
        cumcases[t] = cumcases[t - 1] + infected[t - 1]
        # Adjusts for portion of pop that are susceptible
        susceptible = (max(population - cumcases[t], zero(V)) / population)
        Rt_adj[t] = susceptible * Rt[t]
        infected[t] = Rt_adj[t] * sum(infected[τ] * serial_intervals[t - τ] for τ = (t - 1):-1:max(t-num_si,1)) #max(t-50, 1)
    end
	!all( 0 .< infected .< population ) && (@error "infected<0"; Turing.@addlogprob! -Inf; return)

	#
	latent_iar = TV(undef, num_iar_steps)
	latent_iar[1] = logit(iar0)
	for i in 2:num_iar_steps
		latent_iar[i] ~ Normal(latent_iar[i-1], σ_iar)
	end
	iar = logistic.( latent_iar[iar_index[1:num_time_steps]] )

    expected_daily_deaths[1] = 1e-15 * infected[1]
	expected_daily_hospit[1] = 1e-15 * infected[1]
	expected_daily_cases[1]  = 1e-15 * infected[1]
	expected_seropositive[1] = 1e-15 * infected[1]
	for t = 2:num_time_steps
		expected_daily_cases[t]  = sum(infected[τ] * π2[t - τ] for τ = (t - 1):-1:max(t-num_π2,1)) #max(t-30, 1)
		expected_daily_deaths[t] = ifr * sum(infected[τ] * π[t - τ] for τ = (t - 1):-1:max(t-num_π,1)) #max(t-100, 1)
        expected_daily_hospit[t] = ihr * sum(infected[τ] * π3[t - τ] for τ = (t - 1):-1:max(t-num_π3,1)) #max(t-50, 1)
		expected_seropositive[t] = sum( infected[τ] * π4[t - τ] for τ = 1:t-1 )
    end
	observed_daily_cases = iar .* expected_daily_cases
	# Extract time-steps for which we have observations

	ts = epidemic_start:num_obs
	# μs  = expected_daily_deaths[ts]
	# σ2s = @. μs + μs^2 / ϕ
    # deaths[ts] ~ arraydist(NegativeBinomial3.(μs, σ2s))
	μs  = expected_daily_hospit[ts]
	!all( 0 .< μs .< population ) && (@error "expected_daily_hospit"; Turing.@addlogprob! -Inf; return)
	(ϕ3 <= 0) && (@error "ϕ3 <= 0"; Turing.@addlogprob! -Inf; return)
	hospit[ts] ~ arraydist(NegativeBinomial2.(μs, ϕ3))
    if num_case_obs > 0
		ts = (num_obs-num_case_obs):num_obs
		μs  = observed_daily_cases[ts]
		!all( 0 .< μs .< population ) && ( @error "observed_daily_cases < 0"; Turing.@addlogprob! -Inf; return)
		(ϕ2 <= 0) && (@error "ϕ2 <= 0"; Turing.@addlogprob! -Inf; return)
        cases[ts] ~ arraydist(NegativeBinomial2.(μs, ϕ2))
    end

    # seroprevalence study
	seroprev_mean ~ MvNormal(expected_seropositive[seroprev_idx], seroprev_σ)
	# seroprev_μ ~ MvNormal(cumcases[seroprev_idx], seroprev_σ)

    return (
		observed_daily_cases  = observed_daily_cases,
        expected_daily_deaths = expected_daily_deaths,
        expected_daily_hospit = expected_daily_hospit,
        Rt = Rt,
		expected_seropositive = expected_seropositive,
		iar = iar,
    )
end
## =============================================================================
#                      model_contacts_shifted
# ==============================================================================
@model function model_contacts_shifted(
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
	covariates,
	covariates_start,
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
	num_si = length(serial_intervals)
	num_π = length(π)
	num_π2 = length(π2)
	num_π3 = length(π3)
	num_obs = length(cases)
	num_Rt_steps = num_obs - lockdown
    num_obs = length(cases)
	num_Rt_steps = num_obs - lockdown
	num_covariates = size(covariates, 2)
	# If we don't want to predict the future, we only need to compute up-to time-step `num_obs_states[m]`
    num_time_steps = predict ? num_total_days : num_obs

    # Latent variables
    τ         ~ Exponential(1 / 0.03) # `Exponential` has inverse parameterization of the one in Stan
	T = typeof(τ)
    y         ~ truncated(Exponential(τ),T(0),T(1000))
	ϕ         ~ truncated(Normal(0, 5), 0, Inf)
    ϕ2        ~ truncated(Normal(25, 5), 10, 40)
    ϕ3        ~ truncated(Normal(25, 5), 10, 40)
	R0        ~ truncated(Normal(3.6, .8), 2, 4.5)
	R1        ~ truncated(Normal(.8, .1), .5, 1.1)
    ifr       ~ truncated(Normal(6/1000, 3/1000), 3/1000, 10/1000)
	ihr       ~ truncated(Normal(1/100,1/100),.1/100,5/100)
    iar0      ~ Beta(10,100) #Beta(20,25)
	σ_rt      ~ truncated(Normal(0.05, .03), 0, .15)
	σ_iar     ~ truncated(Normal(0.05, .03), 0, .15)
	# σ_rt      = .05
	# σ_iar     = .05
	seroprev_σ ~ arraydist(InverseGamma2.(seroprev_std, Ref(.1)))
	# seroprev_μ ~ MvNormal(seroprev_mean, seroprev_mean*.01)
	α         ~ filldist(Beta(1, 5), num_covariates)

	!isfinite(R0) && (@error "R0"; Turing.@addlogprob! -Inf; return) #DEBUG
	population < num_impute*y && (@error "population < num_impute*y"; Turing.@addlogprob! -Inf; return) #DEBUG
	y<0 && (@error "y<0"; Turing.@addlogprob! -Inf; return) #DEBUG

    infected              = TV(undef, num_time_steps)
    expected_daily_cases  = TV(undef, num_time_steps)
    expected_daily_hospit = TV(undef, num_time_steps)
	expected_seropositive = TV(undef, num_time_steps)
    cumcases              = TV(undef, num_time_steps)
    expected_daily_deaths = TV(undef, num_time_steps)
    latent_Rt             = TV(undef, num_time_steps-lockdown)
	Rt                    = TV(undef, num_time_steps)
    Rt_adj                = TV(undef, num_time_steps)

    # Imputation of `num_impute` days
    infected[1:num_impute] .= y
    cumcases[1] = zero(V)
    cumcases[2:num_impute] .= cumsum(infected[1:num_impute - 1])

	latent_Rt[1] = invlink(R1)
	for t in 2:num_Rt_steps
		latent_Rt[t] ~ Normal(latent_Rt[t-1], σ_rt)
	end
	latent_Rt[num_Rt_steps+1:end] .= latent_Rt[num_Rt_steps]
	latent_Rt[covariates_start-lockdown:end] .= latent_Rt[covariates_start-lockdown:end] .+ covariates[1:num_time_steps-covariates_start+1, :] * α #
	Rt[1:lockdown] .= R0
	Rt[lockdown+1:num_time_steps] = link.(latent_Rt)

	# Adjusts for portion of pop that are susceptible
    @. Rt_adj[1:num_impute] = (max(population - cumcases[1:num_impute], zero(V)) / population) * Rt[1:num_impute]
    for t = (num_impute + 1):num_time_steps
        # Update cumulative infected
        cumcases[t] = cumcases[t - 1] + infected[t - 1]
        # Adjusts for portion of pop that are susceptible
        susceptible = (max(population - cumcases[t], zero(V)) / population)
        Rt_adj[t] = susceptible * Rt[t]
        infected[t] = Rt_adj[t] * sum(infected[τ] * serial_intervals[t - τ] for τ = (t - 1):-1:max(t-num_si,1)) #max(t-50, 1)
    end
	!all( 0 .< infected .< population ) && (@error "infected<0"; Turing.@addlogprob! -Inf; return)

	#
	latent_iar = TV(undef, num_iar_steps)
	latent_iar[1] = logit(iar0)
	for i in 2:num_iar_steps
		latent_iar[i] ~ Normal(latent_iar[i-1], σ_iar)
	end
	iar = logistic.( latent_iar[iar_index[1:num_time_steps]] )

    expected_daily_deaths[1] = 1e-15 * infected[1]
	expected_daily_hospit[1] = 1e-15 * infected[1]
	expected_daily_cases[1]  = 1e-15 * infected[1]
	expected_seropositive[1] = 1e-15 * infected[1]
	for t = 2:num_time_steps
		expected_daily_cases[t]  = sum(infected[τ] * π2[t - τ] for τ = (t - 1):-1:max(t-num_π2,1)) #max(t-30, 1)
		expected_daily_deaths[t] = ifr * sum(infected[τ] * π[t - τ] for τ = (t - 1):-1:max(t-num_π,1)) #max(t-100, 1)
        expected_daily_hospit[t] = ihr * sum(infected[τ] * π3[t - τ] for τ = (t - 1):-1:max(t-num_π3,1)) #max(t-50, 1)
		expected_seropositive[t] = sum( infected[τ] * π4[t - τ] for τ = 1:t-1 )
    end
	observed_daily_cases = iar .* expected_daily_cases
	# Extract time-steps for which we have observations

	ts = epidemic_start:num_obs
	# μs  = expected_daily_deaths[ts]
	# σ2s = @. μs + μs^2 / ϕ
    # deaths[ts] ~ arraydist(NegativeBinomial3.(μs, σ2s))
	μs  = expected_daily_hospit[ts]
	!all( 0 .< μs .< population ) && (@error "expected_daily_hospit"; Turing.@addlogprob! -Inf; return)
	(ϕ3 <= 0) && (@error "ϕ3 <= 0"; Turing.@addlogprob! -Inf; return)
	hospit[ts] ~ arraydist(NegativeBinomial2.(μs, ϕ3))
    if num_case_obs > 0
		ts = (num_obs-num_case_obs):num_obs
		μs  = observed_daily_cases[ts]
		!all( 0 .< μs .< population ) && ( @error "observed_daily_cases < 0"; Turing.@addlogprob! -Inf; return)
		(ϕ2 <= 0) && (@error "ϕ2 <= 0"; Turing.@addlogprob! -Inf; return)
        cases[ts] ~ arraydist(NegativeBinomial2.(μs, ϕ2))
    end

    # seroprevalence study
	seroprev_mean ~ MvNormal(expected_seropositive[seroprev_idx], seroprev_σ)
	# seroprev_μ ~ MvNormal(cumcases[seroprev_idx], seroprev_σ)

    return (
		observed_daily_cases  = observed_daily_cases,
        expected_daily_deaths = expected_daily_deaths,
        expected_daily_hospit = expected_daily_hospit,
        Rt = Rt,
		expected_seropositive = expected_seropositive,
		iar = iar,
    )
end

## =============================================================================
#                      model_contacts_v2
# ==============================================================================
@model function model_contacts_test(
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
	covariates,
	covariates_start,
	π3,                 # [AbstractVector{<:AbstractVector{<:Real}}] h * s
    hospit,
	seroprev_mean,
    seroprev_std,
    seroprev_idx,
	π4,
	priors,
    predict=false,     # [Bool] if `false`, will only compute what's needed to `observe` but not more
    ::Type{TV} = Vector{Float64},
	::Type{V}  = Float64;
	invlink = log,#KLogistic(V(3))
	link    = exp,#KLogit(V(k))
) where {TV, V}
	num_si = length(serial_intervals)
	num_π = length(π)
	num_π2 = length(π2)
	num_π3 = length(π3)
	num_obs = length(cases)
	num_Rt_steps = num_obs - lockdown
    num_obs = length(cases)
	num_Rt_steps = num_obs - lockdown
	num_covariates = size(covariates, 2)
	num_latent_Rt = covariates_start-lockdown-1
	# If we don't want to predict the future, we only need to compute up-to time-step `num_obs_states[m]`
    num_time_steps = predict ? num_total_days : num_obs

    # Latent variables
	y         ~ truncated( priors[:y], 0, 1000)
    ϕ2        ~ truncated( priors[:ϕ2], 0, Inf)
    ϕ3        ~ truncated( priors[:ϕ3], 0, Inf)
	R0        ~ truncated( priors[:R0], 2, 4.5)
	R1        ~ truncated( priors[:R1], .5, 1.1)
    ifr       ~ truncated( priors[:ifr], 3/1000, 10/1000)
	ihr       ~ truncated( priors[:ihr],.1/100,5/100)
    iar0      ~ truncated( priors[:iar0], 0, 1)
	seroprev_σ~ arraydist( priors[:seroprev_σ] )

    infected              = TV(undef, num_time_steps)
    expected_daily_cases  = TV(undef, num_time_steps)
    expected_daily_hospit = TV(undef, num_time_steps)
	expected_seropositive = TV(undef, num_time_steps)
    cumcases              = TV(undef, num_time_steps)
    expected_daily_deaths = TV(undef, num_time_steps)
    latent_Rt             = TV(undef, num_latent_Rt)
	Rt                    = TV(undef, num_time_steps)
    Rt_adj                = TV(undef, num_time_steps)

    # Imputation of `num_impute` days
    infected[1:num_impute] .= y
    cumcases[1] = zero(V)
    cumcases[2:num_impute] .= cumsum(infected[1:num_impute - 1])

	num_latent_Rt = covariates_start-lockdown-1
	latent_Rt[1] = invlink(R1)
	latent_Rt[2:num_latent_Rt] ~ arraydist(priors[:logRt][1:num_latent_Rt-1])
	Rt[1:lockdown] .= R0
	Rt[lockdown+1:lockdown+num_latent_Rt] = link.(latent_Rt)
	Rt[lockdown+num_latent_Rt:num_obs] ~ arraydist(priors[:Rt])
	if predict #if prediction
		Rt[num_obs+1:num_total_days] .= Rt[num_obs]
	end

	# Adjusts for portion of pop that are susceptible
    @. Rt_adj[1:num_impute] = (max(population - cumcases[1:num_impute], zero(V)) / population) * Rt[1:num_impute]
    for t = (num_impute + 1):num_time_steps
        # Update cumulative infected
        cumcases[t] = cumcases[t - 1] + infected[t - 1]
        # Adjusts for portion of pop that are susceptible
        susceptible = (max(population - cumcases[t], zero(V)) / population)
        Rt_adj[t] = susceptible * Rt[t]
        infected[t] = Rt_adj[t] * sum(infected[τ] * serial_intervals[t - τ] for τ = (t - 1):-1:max(t-num_si,1)) #max(t-50, 1)
    end

	latent_iar[1] = logit(iar0)
	latent_iar[2:num_iar_steps] ~ arraydist( priors[:latent_iar] )
	iar = logistic.( latent_iar[ iar_index[1:num_time_steps] ] )

    expected_daily_deaths[1] = 1e-15 * infected[1]
	expected_daily_hospit[1] = 1e-15 * infected[1]
	expected_daily_cases[1]  = 1e-15 * infected[1]
	expected_seropositive[1] = 1e-15 * infected[1]
	for t = 2:num_time_steps
		expected_daily_cases[t]  = sum(infected[τ] * π2[t - τ] for τ = (t - 1):-1:max(t-num_π2,1)) #max(t-30, 1)
		expected_daily_deaths[t] = ifr * sum(infected[τ] * π[t - τ] for τ = (t - 1):-1:max(t-num_π,1)) #max(t-100, 1)
        expected_daily_hospit[t] = ihr * sum(infected[τ] * π3[t - τ] for τ = (t - 1):-1:max(t-num_π3,1)) #max(t-50, 1)
		expected_seropositive[t] = sum( infected[τ] * π4[t - τ] for τ = 1:t-1 )
    end
	observed_daily_cases = iar .* expected_daily_cases
	# Extract time-steps for which we have observations

	ts = epidemic_start:num_obs
	μs  = expected_daily_hospit[ts]
	hospit[ts] ~ arraydist(NegativeBinomial2.(μs, ϕ3))
    if num_case_obs > 0
		ts = (num_obs-num_case_obs):num_obs
		μs  = observed_daily_cases[ts]
	    cases[ts] ~ arraydist(NegativeBinomial2.(μs, ϕ2))
    end

    # seroprevalence study
	# seroprev_mean ~ MvNormal(expected_seropositive[seroprev_idx], seroprev_σ)
	# seroprev_μ ~ MvNormal(cumcases[seroprev_idx], seroprev_σ)

    return (
		observed_daily_cases  = observed_daily_cases,
        expected_daily_deaths = expected_daily_deaths,
        expected_daily_hospit = expected_daily_hospit,
        Rt = Rt,
		expected_seropositive = expected_seropositive,
		iar = iar,
    )
end
#---------------------------------------------------------------------------------------
# GP for Rt

struct MvNormalGP{T<:AbstractGPs.FiniteGP} <: AbstractMvNormal
    f::T
end
AbstractGPs.rand(rng::Random.AbstractRNG, f::MvNormalGP) = rand(rng, f.f)
AbstractGPs.logpdf(f::MvNormalGP, y::AbstractVector{<:Real}) = logpdf(f.f, y)

kernel(s::Real, l::Real) =
    s^2 * KernelFunctions.transform(Matern32Kernel(), 1/l)

@model function model_gp(
    num_impute,        # [Int] num. of days for which to impute infections
    num_total_days,    # [Int] days of observed data + num. of days to forecast
    cases,            # [AbstractVector{<:AbstractVector{<:Int}}] reported infected
    deaths,            # [AbstractVector{<:AbstractVector{<:Int}}] reported deaths; rows indexed by i > N contain -1 and should be ignored
    π,                 # [AbstractVector{<:AbstractVector{<:Real}}] h * s
    π2,                 # [AbstractVector{<:AbstractVector{<:Real}}] h * s
    epidemic_start,    # [AbstractVector{<:Int}]
    population,        # [AbstractVector{<:Real}]
    serial_intervals,  # [AbstractVector{<:Real}] fixed pre-calculated serial interval (SI) using empirical data from Neil
    lockdown,         # [Int] number of weeks (21)
    num_case_obs,       # [Int] 21 = max_date ("2020-05-11") - testing_date (2020-05-11) see publication
    π3,                 # [AbstractVector{<:AbstractVector{<:Real}}] h * s
    hospit,
	seroprev_mean,
    seroprev_std,
    seroprev_idx,
    predict=false,     # [Bool] if `false`, will only compute what's needed to `observe` but not more
    ::Type{TV} = Vector{Float64},
	::Type{V}  = Float64
) where {TV, V}
	num_si = length(serial_intervals)
	num_π = length(π)
	num_π2 = length(π2)
	num_π3 = length(π3)
	num_obs = length(cases)
	num_Rt_steps = num_obs - lockdown
    num_obs = length(cases)
    # If we don't want to predict the future, we only need to compute up-to time-step `num_obs_states[m]`
    num_time_steps = predict ? num_total_days : num_obs

    # Latent variables
    # τ         ~ Exponential(1 / 0.03) # `Exponential` has inverse parameterization of the one in Stan
	# T = typeof(τ)
    # y         ~ truncated(Exponential(τ),T(0),T(1000))
	y         ~ truncated(Exponential(100),0.,1000.)
	ϕ         ~ truncated(Normal(0, 5), 0, Inf)
    ϕ2        ~ truncated(Normal(25, 5), 10, 40)
    ϕ3        ~ truncated(Normal(25, 5), 10, 40)
	R0        ~ truncated(Normal(3.28, .4), 2, 4)
	R1        ~ truncated(Normal(.8, .2), .5, 1.2)
    ifr_noise ~ truncated(Normal(1., 0.2), 0, Inf)
	ihr       ~ truncated(Normal(2/100,2/100),.1/100,6/100)
    iar0      ~ Beta(10,40) #Beta(20,25)
	# σ_rt      ~ InverseGamma2(0.2, .3)
	σ_rt      = .1
	σ_iar     = .05
	# seroprev_σ ~ arraydist(InverseGamma2.(seroprev_std, Ref(.01)))
	# seroprev_μ ~ MvNormal(seroprev_mean, seroprev_mean*.01)

	!isfinite(R0) && (@error "R0"; Turing.@addlogprob! -Inf; return) #DEBUG
	population < num_impute*y && (@error "population < num_impute*y"; Turing.@addlogprob! -Inf; return) #DEBUG
	y<0 && (@error "y<0"; Turing.@addlogprob! -Inf; return) #DEBUG

    infected              = TV(undef, num_time_steps)
    expected_daily_cases  = TV(undef, num_time_steps)
    expected_daily_hospit = TV(undef, num_time_steps)
    cumcases              = TV(undef, num_time_steps)
    expected_daily_deaths = TV(undef, num_time_steps)
    logRt                 = TV(undef, num_time_steps)
    Rt_adj                = TV(undef, num_time_steps)
	latent_iar             = TV(undef, num_time_steps)

    # Imputation of `num_impute` days
    infected[1:num_impute] .= y
    cumcases[1] = zero(V)
    cumcases[2:num_impute] .= cumsum(infected[1:num_impute - 1])

	gp = GP(kernel(.3, 20.))
	f = to_sde(gp, SArrayStorage(Float64))
	X = RegularSpacing(0.0, 1., num_obs - lockdown)
    slow_Rt_effect ~ MvNormalGP( f(X) )

	logRt[1:lockdown] .= log(R0)
	logRt[lockdown+1:num_obs] = slow_Rt_effect .- slow_Rt_effect[1] .+ log(R1)
	if predict #if prediction
		logRt[num_obs+1:num_total_days] .= logRt[num_obs]
	end

	# Adjusts for portion of pop that are susceptible
    @. Rt_adj[1:num_impute] = (max(population - cumcases[1:num_impute], zero(V)) / population) * exp(logRt[1:num_impute])
    for t = (num_impute + 1):num_time_steps
        # Update cumulative infected
        cumcases[t] = cumcases[t - 1] + infected[t - 1]
        # Adjusts for portion of pop that are susceptible
        susceptible = (max(population - cumcases[t], zero(V)) / population)
        Rt_adj[t] = susceptible * exp(logRt[t])
        infected[t] = Rt_adj[t] * sum(infected[τ] * serial_intervals[t - τ] for τ = (t - 1):-1:max(t-num_si,1)) #max(t-50, 1)
    end
	any( infected .< 0 ) && (i = findfirst(<(0), infected); @error "infected<0 at $(i)"; Turing.@addlogprob! -Inf; return)
	any( .! isfinite.(infected) ) && (i = findfirst(x->!isfinite(x), infected); @error "infected is not finite at $(i)"; Turing.@addlogprob! -Inf; return)
	any( infected .> population ) && (i = findfirst(>(population), infected); @error "infected>population at $(i)"; Turing.@addlogprob! -Inf; return)

	cases_start = num_obs - num_case_obs + 1
	latent_iar[1:cases_start-1] .= logit(iar0)
	if num_case_obs > 0
		gp = GP(kernel(.3, 20.))
		f = to_sde(gp, SArrayStorage(Float64))
		X = RegularSpacing(0.0, 1., num_case_obs)
		slow_IAR_effect ~ MvNormalGP( f(X) )
		latent_iar[cases_start:num_obs] = slow_IAR_effect .- slow_IAR_effect[1] .+ logit(iar0)
	end
	if predict #if prediction
		latent_iar[num_obs+1:num_total_days] .= latent_iar[num_obs]
	end
	iar = logistic.(latent_iar)

    expected_daily_deaths[1] = 1e-15 * infected[1]
	expected_daily_hospit[1] = 1e-15 * infected[1]
	expected_daily_cases[1] = infected[1]
    for t = 2:num_time_steps
		expected_daily_deaths[t] = ifr_noise * sum(infected[τ] * π[t - τ] for τ = (t - 1):-1:max(t-num_π,1)) #max(t-100, 1)
        expected_daily_cases[t]  = sum(infected[τ] * π2[t - τ] for τ = (t - 1):-1:max(t-num_π2,1)) #max(t-30, 1)
        expected_daily_hospit[t] = ihr * sum(infected[τ] * π3[t - τ] for τ = (t - 1):-1:max(t-num_π3,1)) #max(t-50, 1)
    end
	observed_daily_cases = iar .* expected_daily_cases
	# Extract time-steps for which we have observations

	ts = epidemic_start:num_obs
	# μs  = expected_daily_deaths[ts]
	# σ2s = @. μs + μs^2 / ϕ
    # deaths[ts] ~ arraydist(NegativeBinomial3.(μs, σ2s))
	μs  = expected_daily_hospit[ts]
	σ2s = @. μs + μs^2 / ϕ3
	hospit[ts] ~ arraydist(NegativeBinomial3.(μs, σ2s))
    if num_case_obs > 0
		ts = (num_obs-num_case_obs):num_obs
		μs  = observed_daily_cases[ts]
		σ2s = @. μs + μs^2 / ϕ2
        cases[ts] ~ arraydist(NegativeBinomial3.(μs, σ2s))
    end

    # seroprevalence study
	seroprev_mean ~ MvNormal(cumcases[seroprev_idx], seroprev_std)
	# seroprev_μ ~ MvNormal(cumcases[seroprev_idx], seroprev_σ)

    return (
		observed_daily_cases  = observed_daily_cases,
        expected_daily_deaths = expected_daily_deaths,
        expected_daily_hospit = expected_daily_hospit,
        Rt = exp.(logRt),
		cumulative_cases      = cumcases,
		iar = iar
    )
end
