function firefox(p; fname="tmp.html", fdir=normpath(homedir(), ".tmp"))
    # mkpath(dir)
    filepath = normpath(fdir, fname)
    savefig(p, filepath)
    run(`firefox $(filepath)`, wait=false)
    return p
end

struct WeekdayHolidayObsModel{T,M,A,P,E,W,H,V} <: ObservationsModel
    ps::T
    μ::M
    α::A
    ϕ::P
	expecteds::E
    weekdayeffect::W
    holidayeffect::H
	vectorize::V
end

struct SimpleObsModel{T,M,A,P,E,V} <: ObservationsModel
    ps::T
    μ::M
    α::A
    ϕ::P
	expecteds::E
	vectorize::V
end

struct SimpleObsModel2{T,M,S,A,P,E,V} <: ObservationsModel
    ps::T
    μ::M
	σ::S
    α::A
    ϕ::P
	expecteds::E
	vectorize::V
end
# ============================================================================
# random walk model
# function random_walks!(Rts, θ, predict, latent_Rts, R0s, σ_rt)
# 	@unpack num_regions, rt_step_indices, lockdown_indices, num_observations, link, include_early_dynamic = θ
#
# 	for m in 1:num_regions
# 		rt_step_index = rt_step_indices[m]
# 		Rt            = Rts[m]
# 		lockdown      = lockdown_indices[m]
# 		num_obs       = num_observations[m]
# 		latent_Rt     = latent_Rts[m]
# 		R0            = R0s[m]
#
# 		if include_early_dynamic
# 			Rt[1:num_obs] = link.(latent_Rt[rt_step_index])
# 		else
# 			Rt[1:lockdown] .= R0
# 			Rt[lockdown+1:num_obs] = link.(latent_Rt[rt_step_index])
# 		end
# 		if predict
# 			Rt[num_obs+1:end] .= Rt[num_obs]
# 		end
# 	end
# end

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
# semi-parametric model
function semiparametric!(Rts, θ, predict, latent_Rts, R0s, σ_rt, effects)
	@unpack num_regions, lockdown_indices, num_observations, link, rt_step_indices, covariates, covariates_starts, num_total_days, include_early_dynamic= θ #rt_step_indices

	for m in 1:num_regions
		Rt            = Rts[m]
		lockdown      = lockdown_indices[m]
		rt_step_index = rt_step_indices[m]
		num_obs       = num_observations[m]
		latent_Rt     = latent_Rts[m]
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
		lockdown         = lockdown_indices[m]
		rt_step_index    = rt_step_indices[m]
		num_obs          = num_observations[m]
		latent_Rt        = latent_Rts[m]
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
			Rt[1:lockdown] .= R0
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

		expected[1] = 1e-15 * newly_infected[1]
		for t = 2:num_time_step
			expected[t] = α * sum(newly_infected[τ] * i2o[t - τ] for τ = (t - 1):-1:max(t-delay_length,1))
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

		expected[1] = 1e-15 * newly_infected[1]
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

function Turing.logpdf(obsmodel::ObservationsModel, observed)
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

		_out_of_bounds(μs, populations[m]) && (@warn "out of bounds"; return T(Inf))
		ℓ += _logpdf(ys, μs, ϕ)
	end
	return ℓ
end

_out_of_bounds(x, upper_lim) = !all(@. 0. < x < upper_lim )

# ============================================================================
# pointwise_loglikelihoods
function Turing.pointwise_loglikelihoods(obsmodel::ObservationsModel, observed)
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
function prediction(obsmodel::ObservationsModel)
	@unpack ps, ϕ, expecteds = obsmodel
	@unpack starts, stops, num_regions = ps

	ys = Vector{Float64}[]
	for m in 1:num_regions
		μs = expecteds[m]
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

struct SimpleSeroObsModel{T,E}# <: ObservationsModel
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

# ============================================================================
# post-processing
function save_results(fdir, ps, data, chain)
	ignores = [k for (k,v) in pairs(ps) if ((v isa String) && isempty(v))]
	mkpath(fdir)

	@info "Saving at: $fdir"
    let
        dic = Dict( zip( keys(ps), values(ps) ) )
        fname = normpath(fdir, savename("PARAMS", ps, "csv"; ignores) )
        safesave( fname, DataFrame( dic ) )
        bson( normpath(fdir, "params.bson") ,  dic )
	end
	let
        fname = normpath( fdir, savename("DATA", ps, "bson"; ignores) )
        bson( fname, Dict("data"=>data) )
	end

    fname = normpath( fdir, savename("CHAIN", ps, "jls"; ignores) )
    safesave( fname, chain )
    fname
end

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
    m_pred = model(turing_data, predict = true, context=:prediction)
    gq = Turing.generated_quantities(m_pred, chain)
    gp = vectup2tupvec(reshape(gq, length(gq)))
    groupregions(gp)
end

function rtstats(Rts, dates; startdate=Date("2020-05-15"), stopdate=nothing)
    ib = findfirst(==(startdate), dates)
    ie = isnothing(stopdate) ? length(dates) : findfirst(==(Date(stopdate)), dates)
    Rt_array = hcat(Rts...)[ib:ie,:]
    qs = [quantile(v, [0.025, 0.25, 0.5, 0.75, 0.975]) for v in eachrow(Rt_array)]
    llq, lq, mq, uq, uuq = (eachrow(hcat(qs...))..., )

    return DataFrame((; dates = dates[ib:ie], llq, lq, mq, uq, uuq))
end

# ============================================================================
# post-processing

struct PostProcessing2{F,P,I,C,D,M}
	fdir::F
	ps::P
	ignores::I
	chain::C
	data::D
	model::M
end

function PostProcessing2(fdir, ps, ignores, exclude, fname)
	chain = let
	    chain = read(fname, Chains)
	    not(chain, exclude)
	end
	data = let
		fn_data  = replace(fname, "CHAIN_"=>"DATA_")
		fn_data  = replace(fn_data, ".jls"=>".bson")
		BSON.load(fn_data)["data"]
	end

	model = Regional.name2model[ps.model]
	PostProcessing2(fdir, ps, ignores, chain, data, model)
end

function not(c::Chains, exclude::AbstractVector=[])
    isempty(exclude) && return c
    n = size(c,3)
    s = filter(x->x ∉ exclude, 1:n)
    return c[:,:,s]
end

function parse_fname(fname; warmup = nothing)
	ignores = []
	fdir_raw, parsed_args, _ = parse_savename(fname, parsetypes = (Int, Float64, Bool))
	!isnothing(warmup) && (parsed_args["warmup"] = warmup)
	!("preds" in keys(parsed_args)) && (parsed_args["preds"] = nothing; push!(ignores, :preds))
	!("prefix" in keys(parsed_args)) && (parsed_args["prefix"] = ""; push!(ignores, :prefix))
    !("cases" in keys(parsed_args)) && (parsed_args["cases"] = "2020-06-01"; push!(ignores, :cases))

	ps = NamedTuple(Symbol(k)=>v for (k,v) in pairs(parsed_args))

	folders = split(fdir_raw, "/")
	i = findfirst(==("CovidSurvey"), folders)
	fdir = normpath(homedir(),"dev/CovidSurvey", folders[i+1:end-1]...)
	@info "save post-processing in $fdir"
	return (; fdir, ps, ignores)
end

function savechain(p::PostProcessing2)
	@unpack fdir, ps, ignores, chain = p
    fname = normpath( fdir, savename(ps.prefix*"CHAIN", ps, "jls"; ignores) )
    safesave( fname, chain )
end

function plot_chains(p::PostProcessing2; plot_results = false)
	@unpack fdir, ps, ignores, chain = p
	n = filter( x->!occursin(r"latent_Rts", x), String.(names(chain)))
    p = plot(chain[n]);
    fname = normpath( fdir, savename(ps.prefix*"FIG-CHAINSPLOT", ps, "html"; ignores) )
    savefig(p, fname )
    plot_results && (display(p); run(`firefox $(fname)`, wait=false))
	return nothing
end

function plot_means(p::PostProcessing2; plot_results=false)
	@unpack fdir, ps, ignores, chain = p
	n = filter( x->!occursin(r"latent_Rts", x), String.(names(chain)))
	p = meanplot(chain[n]);
	fname = normpath( fdir, savename(ps.prefix*"FIG-MEANPLOT", ps, "html"; ignores) )
	savefig(p, fname )
	plot_results && (display(p); run(`firefox $(fname)`, wait=false))
	return nothing
end

function skip_warmup(p::PostProcessing2)
	@unpack fdir, ps, ignores, chain, data, model = p
	chain = chain[ps.warmup+1:end,:,:]
	PostProcessing2(fdir, ps, ignores, chain, data, model)
end

function diagnostics(p::PostProcessing2, plot_results = false)
	@unpack fdir, ps, ignores, chain = p

	diagnostics = describe(chain) |> first |> DataFrame

	highlighters = (
		Highlighter((data, i, j) -> (j == 7) && abs.(data[i, 7]) > 1.1, bold = true, foreground = :red),
	)
	pretty_table(diagnostics; highlighters, crop=:none)

	highlighters = (
		HTMLHighlighter((data, i, j) -> (j == 7) && abs.(data[i, 7]) > 1.1, HTMLDecoration(color = "red")),
	)
	table = pretty_table(String, diagnostics; highlighters, backend = Val(:html) )
	fname = normpath( fdir, savename(ps.prefix*"DIAGNOSTICS", ps, "html"; ignores) )
	write( fname, table)
	plot_results && (display(p); run(`firefox $(fname)`, wait=false))
	println("median Rhat: $(median( diagnostics[:,7] ))")

	if ps.chains > 1
        @info "gelman diagnostics"
        diagnostics = gelmandiag(chain) |> DataFrame

		highlighters = (
			Highlighter((data, i, j) -> (j == 2) && data[i, 2] > 1.2, bold = true, foreground = :red),
		)
		pretty_table(diagnostics; highlighters, crop=:none)

		highlighters = (
			HTMLHighlighter((data, i, j) -> (j == 2) && data[i, 2] > 1.2, HTMLDecoration(color = "red")),
		)
		table = pretty_table(String, diagnostics; backend = Val(:html), highlighters )
		fname = normpath( fdir, savename(ps.prefix*"GELMANDIAG", ps, "html"; ignores) )
		write( fname, table)
		plot_results && (display(p); run(`firefox $(fname)`, wait=false))
		println("max gelmandiag: $(maximum( diagnostics[:,2] ))")
		println("median gelmandiag: $(median( diagnostics[:,2] ))")
    end
end

function generate_posterior(p::PostProcessing2)
	@unpack fdir, ps, ignores, chain, model, data = p
	chains_params = Turing.MCMCChains.get_sections(chain, :parameters)
	generated_posterior = Regional.posterior(model, data.turing_data, chains_params)
    fname = normpath( fdir, savename(ps.prefix*"GENERATED-QUANTITIES", ps, "bson"; ignores) )
    dic = Dict( zip( keys(generated_posterior), values(generated_posterior) ) )
    bson( fname ,  dic )
	return generated_posterior
end

function plot_regions(p, gp; plot_results = false)
	@unpack fdir, ps, ignores, chain, data = p
	for r in Regional.regions
        recipe = Regional.RegionPlottingRecipe(data, gp, r)
        p = plot(recipe)
        fname = normpath( fdir, savename(ps.prefix*"FIG-PREDICTION-$(uppercase(r))", ps, "html"; ignores) )
        savefig( p, fname )
        plot_results && (display(p); run(`firefox $(fname)`, wait=false))
    end
end

function plot_rt(p, gp; plot_results = false)
	@unpack fdir, ps, ignores, chain, data = p
    recipe = Regional.RtsPlottingRecipe(data, gp)
    p = plot(recipe)
    fname = normpath( fdir, savename(ps.prefix*"FIG-RT", ps, "html"; ignores) )
    savefig( p, fname )
    plot_results && (display(p); run(`firefox $(fname)`, wait=false))
	return nothing
end

function save_rt(p, gp)
	@unpack fdir, ps, ignores, chain, data = p
	for (i,r) in enumerate(Regional.regions)
        df = Regional.rtstats(
            gp.Rts[i],
            data.dates[i];
            startdate=Date("2020-05-15"),
            stopdate=data.observations_end
        )
        fname = normpath( fdir, savename(ps.prefix*"Rt-$(uppercase(r))", ps, "csv"; ignores) )
        save(fname, df)
    end
end

function predictive(turing_data)
    turing_data_dct = convert(Dict, turing_data)
    turing_data_dct = convert(Dict{Symbol, Any}, turing_data_dct)
    turing_data_dct[:cases]   = [similar(h, Missing) for h in turing_data_dct[:cases]]
    turing_data_dct[:hospits] = [similar(h, Missing) for h in turing_data_dct[:hospits]]
    turing_data_dct[:deaths]  = [similar(h, Missing) for h in turing_data_dct[:deaths]]
    return namedtuple(turing_data_dct)
end

function ArviZ.plot_autocorr(p::PostProcessing2; plot_results = false)
	@unpack fdir, ps, ignores, chain = p
	c = MCMCChains.get_sections(chain, :parameters)
	n = filter(
	    x-> !occursin("latent_Rts", x) &&
	        !occursin("effect", x)     &&
	        !occursin("ys", x)     &&
	        !occursin("R0", x), String.(names(c))
	)
	ArviZ.plot_autocorr(c; var_names=n);
	fname = normpath( fdir, savename(ps.prefix*"FIG-AUTOCORR", ps, "png"; ignores) )
	plt.gcf().savefig(fname)
	plot_results && (display(p); run(`firefox $(fname)`, wait=false))
	return nothing
end

function ArviZ.plot_pair(p::PostProcessing2; plot_results = false)
	@unpack fdir, ps, ignores, chain = p
	c = MCMCChains.get_sections(chain, :parameters)
	n = filter(
	    x-> !occursin("latent_Rts", x) &&
	        !occursin("effect", x)     &&
	        !occursin("ys", x)     &&
	        !occursin("R0", x), String.(names(c))
	)
	idata = from_mcmcchains(
		chain;
		#coords=Dict("school" => schools),
		#dims=Dict("y" => ["school"], "σ" => ["school"], "θ" => ["school"]),
		library="Turing",
	)
	plot_pair(
	    idata;
	    #coords=Dict("school" => ["Choate", "Deerfield", "Phillips Andover"]),
	    divergences=true,
	);
	fname = normpath( fdir, savename(ps.prefix*"FIG-PAIRS", ps, "png"; ignores) )
	plt.gcf().savefig(fname)
	plot_results && (display(p); run(`firefox $(fname)`, wait=false))
	return nothing
end

# =============================================================================
sampled_effects(gp, predictor, region) = getindex.(gp.effects[region], predictor)
sampled_effects(gp, predictor) = getindex.(gp.grouped_effects |> first, predictor)

function relative_Rt_change(x, pp::Regional.PostProcessing2, args...; R0 = 1)
    invf = pp.data.turing_data.invlink
    f    = pp.data.turing_data.link
    @assert R0 == 1 #otherwise check regional_effect_estimation.jl

    @_ sampled_effects(args...) |>
        f.(invf(R0) .+ __ * x) |>
        @. ( __ / R0 - R0 ) * 100
end

function get_effect_sizes(x, pp, args...)
    if x == 0
        return sampled_effects(args...)
    else
        return relative_Rt_change(x, pp, args...)
    end
end

function effect_quantiles(args...)
    v     = get_effect_sizes(args...)
    ll,uu = hpdi(v; alpha=0.05)
    l,u   = hpdi(v; alpha=0.5)
    m     = median(v)
    return ll, l, m, u, uu
end

function effect_quantiles(pp::Regional.PostProcessing2, gp; effect_on_Rt=0)
    data = []
    for (i,p) in enumerate(pp.data.predictors)
        for (j,r) in enumerate(Regional.regions)
            q = effect_quantiles(effect_on_Rt, pp, gp, i, j)
            push!(data, [q..., p, r])
        end
        r = "grouped"
        q = effect_quantiles(effect_on_Rt, pp, gp, i)
        push!(data, [q..., p, r])
    end

    DataFrame(
        (median=m, lower95=ll, lower50=l, upper50=u, upper95=uu, predictor=p, region=r)
        for (ll, l, m, u, uu, p, r) in data
    )
end

# ==========================================================================
Base.getindex(df::DataFrame, col::Union{AbstractString, Integer, Symbol}) =
	df[!, col] |> only

function plot_effect!(p, effects; addlabel = false, ylabel = :region)
    label1 = addlabel ? "95% HPDI" : nothing
    label2 = addlabel ? "50% HPDI" : nothing
    label3 = addlabel ? "median" : nothing

    ll, l, m, u, uu, n = @_ effects |>
        getindex.(Ref(__), [:lower95, :lower50, :median, :upper50, :upper95, ylabel])

    plot!(p, [ll, uu],[n,n], lw=2, c=:midnightblue, lab=label1)
    plot!(p, [l,u],[n,n], lw=6, c=:midnightblue, lab=label2)
    scatter!(p, [m], [n], mc=:white, lab=label3, msw=2, lc=:black, ms=5)
    return p
end

function plot_regional_effect!(p, effects::DataFrame, region; kwargs...)
    @_ effects |>
        filter(_.region == region, __) |>
        plot_effect!(p, __; kwargs...)
end

function plot_regional_effects(i, num_predictors, effects::DataFrame; xlabel = "effect size")

    p = plot(;
          xlabel = i == num_predictors ? xlabel : ""
        , legend = i == 1 ? :outertopright : nothing
        , title = effects.predictor |> first
        , bottom_margin = i == num_predictors ? 0mm : 3mm
        , top_margin = i == 1 ? 3mm : 3mm
    )

    for (j,region) in enumerate(Regional.regions)
        plot_regional_effect!(
            p, effects, region;
            addlabel = (i == 1) && (j == 1)
        )
    end

    plot_regional_effect!(p, effects, "grouped")
    return p
end

function plot_regional_effects(effects::DataFrame; xlabel = "change in Rt [\\%]")
    ps = Vector{Plots.Plot}()
    predictors = unique(effects.predictor)
    num_predictors = length(predictors)
    for (i, predictor) in enumerate(predictors)
        p = @_ effects |>
            filter(_.predictor == predictor, __) |>
            plot_regional_effects(i, num_predictors, __; xlabel)
        push!(ps, p)
    end
    plot(ps...,
        layout=(num_predictors,1),
        size=(500, num_predictors*250),
        link=:x, sharex=true
    )
end


function plot_grouped_effects(effects; xlabel)
    ps = Vector{Plots.Plot}()
    predictors = unique(effects.predictor)
    num_predictors = length(predictors)

    p = plot(;
          xlabel
        , legend = :outertopright
        , title = "grouped effects"
    )

    for (i, predictor) in enumerate(predictors)
        @_ effects |>
            filter(_.predictor == predictor, __) |>
            filter(_.region == "grouped", __) |>
            Regional.plot_effect!(p, __; addlabel = i == 1, ylabel = :predictor)
    end
	return p
end

function plot_effects(pp::PostProcessing2, gp; plot_results = true, grouped = false, effect_on_Rt = 0.)
	@unpack fdir, ps, ignores, chain = pp
	effects = effect_quantiles(pp, gp)

	suffix = "-EFFECTS"
	suffix = grouped ? suffix *= "-GROUPED" : suffix *= "-REGIONAL"
	suffix = effect_on_Rt == 0 ? suffix *= "-ABSOLUTE" :  suffix *= "-RT=$(effect_on_Rt)"
	xlabel = effect_on_Rt == 0 ? "effect size" : "change in Rt [\\%]"
	plot_effects = grouped ? plot_grouped_effects : plot_regional_effects

	fname = normpath( fdir, savename(ps.prefix*"FIG"*suffix, ps, "png"; ignores) )
	p = plot_effects(effects; xlabel)
	savefig(p, fname )
	plot_results && (display(p); run(`firefox $(fname)`, wait=false))

	fname = normpath( fdir, savename(ps.prefix*"TABLE"*suffix, ps, "csv"; ignores) )
	CSV.write(fname, effects)
	return nothing
end

# =============================================================================



# =============================================================================

function runtime(chain)
    return chain.info.stop_time - chain.info.start_time |>
		mean |>
        round |>
        Dates.Second |>
        Dates.CompoundPeriod |>
        Dates.canonicalize
end

function finish(p::PostProcessing2)
	@unpack chain = p
    println("number of divervences: $(sum( chain[:,:numerical_error,:]) )")
    println("median tree depth: $( median(chain[:,:tree_depth,:]) )")
	max_tree_depth = maximum(chain[:,:tree_depth,:])
	println("fraction of max tree depth = $(max_tree_depth): $(
	 	sum( chain[:,:tree_depth,:] .== max_tree_depth ) / size(chain, 1) / size(chain, 3) * 100
	) \\%")
    @show Regional.runtime(chain)
end

function postprocessing(fname; plot_results = false, exclude = [], warmup = nothing)
	## ==========================================================================
    @info "load data"
    fdir, ps, ignores = parse_fname(fname; warmup)
	p = PostProcessing2(fdir, ps, ignores, exclude, fname)
	savechain(p)
    ## ==========================================================================
    @info "plot chain"
    plot_chains(p; plot_results)
    ## ==========================================================================
    @info "meanplot"
    plot_means(p; plot_results)
    ## ==========================================================================
    @info "perform diagnostics"
	p = skip_warmup(p)
	diagnostics(p)
	## ==========================================================================
	@info "plot autocorr"
	plot_autocorr(p; plot_results)
	## ==========================================================================
	@info "plot pairs"
	plot_pair(p; plot_results)
	## ==========================================================================
    @info "make predictions"
    gp = generate_posterior(p)
    ## ==========================================================================
    @info "plot regions"
    plot_regions(p, gp; plot_results)
    ## ==========================================================================
    @info "plot rt"
    plot_rt(p, gp; plot_results)
	## ==========================================================================
	if !isempty(p.data.predictors)
		@info "plot predictors"
		pgfplotsx()
		default(titlefontsize = 20, legendfontsize = 18, labelfontsize = 18, guidefontsize = 18, tickfontsize = 12, framestyle = :zerolines, yminorgrid = true)

		Regional.plot_effects(p, gp; plot_results, grouped = false, effect_on_Rt = 0.)
		Regional.plot_effects(p, gp; plot_results, grouped = false, effect_on_Rt = 2.)
		Regional.plot_effects(p, gp; plot_results, grouped = false, effect_on_Rt = -0.5)

		Regional.plot_effects(p, gp; plot_results, grouped = true, effect_on_Rt = 0.)
		Regional.plot_effects(p, gp; plot_results, grouped = true, effect_on_Rt = 2.)
		Regional.plot_effects(p, gp; plot_results, grouped = true, effect_on_Rt = -0.5)
		plotlyjs()
		default()
	end
    # -----------------------------------------------------------------------------
    @info "store reproduction number"
    save_rt(p, gp)
	finish(p)
	return nothing
end

##
# @info "plot hospitalizations"
# let
#     recipe = Regional.HospitsPlottingRecipe(data, generated_posterior)
#     p = plot(recipe)
#     fname = normpath( fdir, savename(ps.prefix*"FIG-HOSPIT", ps, "html") )
#     savefig( p, fname )
# end


#-----------------------------------------------------------------------------
# @info "plot prior vs. posterior"
# pchain = sample(m, Prior(), ps.steps + ps.warmup; progress=true)[ps.warmup+1:end,:,:]
# n = filter( x->!occursin(r"\[", x), String.(names(pchain)))
# p = density(chainscat(pchain[n], chain[n]))
# savefig(p, projectdir("figures/tmp", fname*"_PRIORvsPOSTERIOR.html") )
# run(`firefox $(projectdir("figures/tmp", fname*"_PRIORvsPOSTERIOR.html"))`, wait=false)

# ===========================================================================
name2model = Dict(
    "hospit" => Regional.model_hospit,
	"deaths" => Regional.model_deaths,
	"cases"  => Regional.model_cases,
)


# NOTE OUTDATED

# # ============================================================================
# function infection2death(θ, μ_i2d)
# 	@unpack ϕ_i2d, num_i2d = θ
# 	i2d = pdf.( Ref(NegativeBinomial2(μ_i2d, ϕ_i2d)), 1:num_i2d )
# 	i2d /= sum(i2d)
# 	return i2d
# end

# function observe_hospitalizations(θ, expected_daily_hospits, ϕ_h)
# 	@unpack num_regions, populations, num_observations, hospits, epidemic_start = θ
#
# 	ℓ = 0.
# 	T = typeof(ℓ)
# 	for m in 1:num_regions
# 		population            = populations[m]
# 		expected_daily_hospit = expected_daily_hospits[m]
# 		num_obs               = num_observations[m]
# 		hospit                = hospits[m]
#
# 		ts_h  = epidemic_start:num_obs
# 		μs_h  = expected_daily_hospit[ts_h]
# 		!all( 0 .< μs_h .< population ) && (@error "expected_daily_hospit"; return T(Inf))
# 		dist  = arraydist(NegativeBinomial2.(μs_h, ϕ_h))
# 		ℓ += logpdf(dist, hospit[ts_h])
# 	end
# 	return ℓ
# end
#
# function deaths!(expected_daily_deaths, θ, μ_i2d, idr, newly_infecteds)
# 	@unpack num_regions, num_i2d = θ
#
# 	i2d = infection2death(θ, μ_i2d)
#
# 	for m in 1:num_regions
#
# 		expected_daily_death     = expected_daily_deaths[m]
# 		newly_infected           = newly_infecteds[m]
# 		num_time_step            = length(expected_daily_death)
#
# 		expected_daily_death[1] = 1e-15 * newly_infected[1]
# 		for t = 2:num_time_step
# 			expected_daily_death[t] = idr * sum(newly_infected[τ] * i2d[t - τ] for τ = (t - 1):-1:max(t-num_i2d,1))
# 		end
# 	end
# 	return nothing
# end
#
# function observe_deaths(θ, expected_daily_deaths, ϕ_d)
# 	@unpack num_regions, populations, num_observations, deaths, epidemic_start = θ
#
# 	ℓ = 0.
# 	T = typeof(ℓ)
# 	for m in 1:num_regions
# 		population            = populations[m]
# 		expected_daily_death  = expected_daily_deaths[m]
# 		num_obs               = num_observations[m]
# 		death                 = deaths[m]
#
# 		ts_d  = epidemic_start:num_obs
# 		μs_d  = expected_daily_death[ts_d]
# 		!all( 0 .< μs_d .< population ) && (@error "expected_daily_death"; return T(Inf))
# 		dist  = arraydist(NegativeBinomial2.(μs_d, ϕ_d))
# 		ℓ += logpdf(dist, death[ts_d])
# 	end
# 	return ℓ
# end
#
# # ============================================================================
# function infection2cases(θ, μ_i2c)
# 	@unpack ϕ_i2c, num_i2c = θ
# 	i2c = pdf.( Ref(NegativeBinomial2(μ_i2c, ϕ_i2c)), 1:num_i2c )
# 	i2c /= sum(i2c)
# 	return i2c
# end
#
# function cases!(expected_daily_cases, θ, μ_i2c, iar, newly_infecteds)
# 	@unpack num_regions, num_i2c = θ
#
# 	i2c = infection2cases(θ, μ_i2c)
#
# 	for m in 1:num_regions
#
# 		expected_daily_case     = expected_daily_cases[m]
# 		newly_infected           = newly_infecteds[m]
# 		num_time_step            = length(expected_daily_case)
#
# 		expected_daily_case[1] = 1e-15 * newly_infected[1]
# 		for t = 2:num_time_step
# 			expected_daily_case[t] = iar * sum(newly_infected[τ] * i2c[t - τ] for τ = (t - 1):-1:max(t-num_i2c,1))
# 		end
# 	end
# 	return nothing
# end
#
# function observe_cases(θ, expected_daily_cases, ϕ_c)
# 	@unpack num_regions, populations, num_observations, cases, cases_start_indices = θ
#
# 	ℓ = 0.
# 	T = typeof(ℓ)
# 	for m in 1:num_regions
# 		population            = populations[m]
# 		expected_daily_case   = expected_daily_cases[m]
# 		num_obs               = num_observations[m]
# 		case                  = cases[m]
# 		cases_start_index     = cases_start_indices[m]
#
# 		ts_c  = cases_start_index:num_obs
# 		μs_c  = expected_daily_cases[ts_d]
# 		!all( 0 .< μs_c .< population ) && (@error "expected_daily_cases"; return T(Inf))
# 		dist  = arraydist(NegativeBinomial2.(μs_c, ϕ_c))
# 		ℓ += logpdf(dist, case[ts_c])
# 	end
# 	return ℓ
# end
#
# # QUESTION : use general observation model as in national approach?
# # function expected!(obsmodel::SimpleObsModel, newly_infecteds)
# # 	@unpack θ, μ, α, expecteds = obsmodel
# # 	@unpack num_regions, delay_length = θ
# #
# # 	i2o = inf2obs(θ, μ)
# #
# # 	for m in 1:num_regions
# #
# # 		expected       = expecteds[m]
# # 		newly_infected = newly_infecteds[m]
# # 		num_time_step  = length(expected)
# #
# # 		@inbounds expected[1] = 1e-15 * newly_infected[1]
# # 		for t = 2:delay_length+1
# # 			ᾱ = selectvalue(α, t)
# # 			_expected!(expected, t, ᾱ, newly_infected, i2o, 1)
# # 		end
# # 		for t = delay_length+2:num_time_step
# # 			ᾱ = selectvalue(α, t)
# # 			_expected!(expected, t, ᾱ, newly_infected, i2o, t-delay_length)
# # 		end
# # 	end
# # end
# #
# # function inf2obs(param, μ)
# # 	@unpack delay_dispersion, delay_length, delay_dist = param
# # 	i2o = pdf.( Ref(delay_dist(μ, delay_dispersion)), 1:delay_length )
# # 	i2o /= sum(i2o)
# # 	return i2o
# # end
# #
# # _expected!(expected, t, α, args...) =
# # @inbounds expected[t] = α * infectious(t, args...)
# #
# # infectious(t, newly_infected, i2o, stop) =
# #     sum(newly_infected[τ] * i2o[t - τ] for τ = (t - 1):-1:stop)
