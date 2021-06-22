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
# ============================================================================
# random walk model
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
		Rt[lockdown+1:num_obs] = link.(latent_Rt[rt_step_index])
		if predict
			Rt[num_obs+1:end] .= Rt[num_obs]
		end
	end
end

# ============================================================================
# infection model
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
		St = max(population - cumulative_infected[t], 0) / population
	end
end

function runepidemic!(newly_infected, cumulative_infected, effective_Rts, θ, regionaldata)
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

		effective_Rt[t] = St * Rt[t]
		# number of new infections (unobserved)
		newly_infected[t] = effective_Rt[t] * effectively_infectious
	end
	return nothing
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

# ============================================================================
# observations
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

# ============================================================================
# post-processing
function groupregions(gq::NamedTuple)
    ks = keys(gq)
    vs = values(gq)
    n  = length(first(first(gq)))
    d  = Dict()
    for (k,v) in zip(ks,vs)
        d[k] = [[v[j][i] for j in eachindex(v)] for i in 1:n]
    end
    (; d... )
end

function posterior(model, turing_data, chain)
    m_pred = model(turing_data, true)
    gq = Turing.generated_quantities(m_pred, chain)
    gp = vectup2tupvec(reshape(gq, length(gq)))
    groupregions(gp)
end
# ===========================================================================
name2model = Dict(
    "hospit" => Regional.model_hospit,
)
