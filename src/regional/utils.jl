struct WeekdayHolidayObsModel1{T,M,A,P,E,W,H} <: ObservationsModel
    ps::T
    μ::M
    α::A
    ϕ::P
	expecteds::E
    weekdayeffect::W
    holidayeffect::H
end

struct SimpleObsModel1{T,M,A,P,E} <: ObservationsModel
    ps::T
    μ::M
    α::A
    ϕ::P
	expecteds::E
end

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

function infection2observation(ps, μ)
	@unpack delay_dispersion, delay_length, delay_dist = ps
	i2o = pdf.( Ref(delay_dist(μ, delay_dispersion)), 1:delay_length )
	i2o /= sum(i2o)
	return i2o
end

function expected!(obsmodel::SimpleObsModel1, newly_infecteds)
	@unpack ps, μ, α, expecteds = obsmodel
	@unpack num_regions, delay_length = ps

	i2o = infection2observation(ps, μ)

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

function expected!(obsmodel::WeekdayHolidayObsModel1, newly_infecteds)
	@unpack ps, μ, α, expecteds, holidayeffect, weekdayeffect = obsmodel
	@unpack num_regions, delay_length, holidays, weekdays = ps

	i2o = infection2observation(ps, μ)

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

function Turing.logpdf(obsmodel::ObservationsModel, observed)
	@unpack ps, ϕ, expecteds = obsmodel
	@unpack populations, starts, stops, num_regions = ps

	T     = typeof(ϕ)
	ℓ = 0.
	for m in 1:num_regions
		ts    = starts[m]:stops[m]
		μs = expecteds[m][ts]
		ys = observed[m][ts]
		_out_of_bounds(μs, populations[m]) && (return T(Inf))
		dist = arraydist(NegativeBinomial2.(μs, ϕ))
		ℓ += logpdf(dist, ys)
	end
	return ℓ
end

_out_of_bounds(x, upper_lim) = !all(@. 0 < x < upper_lim )

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

_unrealistic(ŷ,y) = any( ŷ .> 4*y )
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
		_unrealistic(μ, observed) && (return T(Inf))
		ℓ += logpdf(dist, observed)
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

struct PostProcessing1{F,P,I,C,D,M}
	fdir::F
	ps::P
	ignores::I
	chain::C
	data::D
	model::M
end

function PostProcessing1(fdir, ps, ignores, exclude)
	chain = let
	    fname = normpath( fdir, savename(ps.prefix*"CHAIN", ps, "jls"; ignores) )
	    chain = read(fname, Chains)
	    not(chain, exclude)
	end
	data = let
		fname = normpath( fdir, savename(ps.prefix*"DATA", ps, "bson"; ignores) )
		BSON.load(fname)["data"]
	end

	model = Regional.name2model[ps.model]
	PostProcessing1(fdir, ps, ignores, chain, data, model)
end

function not(c::Chains, exclude::AbstractVector=[])
    isempty(exclude) && return c
    n = size(c,3)
    s = filter(x->x ∉ exclude, 1:n)
    return c[:,:,s]
end

function parse_fname(fname)
	ignores = []
	fdir_raw, parsed_args, _ = parse_savename(fname, parsetypes = (Int, Float64, Bool))
	!("preds" in keys(parsed_args)) && (parsed_args["preds"] = nothing; push!(ignores, :preds))
	!("prefix" in keys(parsed_args)) && (parsed_args["prefix"] = ""; push!(ignores, :prefix))
	ps = NamedTuple(Symbol(k)=>v for (k,v) in pairs(parsed_args))

	folders = split(fdir_raw, "/")
	fdir = normpath(homedir(),"dev/CovidSurvey/", folders[end-2], folders[end-1])
	return (; fdir, ps, ignores)
end

function savechain(p::PostProcessing1)
	@unpack fdir, ps, ignores, chain = p
    fname = normpath( fdir, savename(ps.prefix*"CHAIN", ps, "jls"; ignores) )
    safesave( fname, chain )
end

function plot_chains(p::PostProcessing1; plot_results = false)
	@unpack fdir, ps, ignores, chain = p
	n = filter( x->!occursin(r"latent_Rts", x), String.(names(chain)))
    p = plot(chain[n]);
    fname = normpath( fdir, savename(ps.prefix*"FIG-CHAINSPLOT", ps, "html"; ignores) )
    savefig(p, fname )
    plot_results && run(`firefox $(fname)`, wait=false)
	return nothing
end

function plot_means(p::PostProcessing1; plot_results=false)
	@unpack fdir, ps, ignores, chain = p
	n = filter( x->!occursin(r"latent_Rts", x), String.(names(chain)))
	p = meanplot(chain[n]);
	fname = normpath( fdir, savename(ps.prefix*"FIG-MEANPLOT", ps, "html"; ignores) )
	savefig(p, fname )
	plot_results && run(`firefox $(fname)`, wait=false)
	return nothing
end

function skip_warmup(p::PostProcessing1)
	@unpack fdir, ps, ignores, chain, data, model = p
	chain = chain[ps.warmup+1:end,:,:]
	PostProcessing1(fdir, ps, ignores, chain, data, model)
end

function diagnostics(p::PostProcessing1)
	@unpack fdir, ps, ignores, chain = p
	if ps.chains > 1
        @info "gelman diagnostics"
        diagnostics = gelmandiag(chain)
        fname = normpath( fdir, savename(ps.prefix*"GELMANDIAG", ps, "csv"; ignores) )
        safesave( fname, diagnostics)
        pretty_table(diagnostics; crop=:none)
    end
end

function generate_posterior(p::PostProcessing1)
	@unpack fdir, ps, ignores, chain, model, data = p
	generated_posterior = Regional.posterior(model, data.turing_data, chain)
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
        fname = normpath( fdir, savename(ps.prefix*"FIG-$(uppercase(r))", ps, "html"; ignores) )
        savefig( p, fname )
        plot_results && run(`firefox $(fname)`, wait=false)
    end
end

function plot_rt(p, gp; plot_results = false)
	@unpack fdir, ps, ignores, chain, data = p
    recipe = Regional.RtsPlottingRecipe(data, gp)
    p = plot(recipe)
    fname = normpath( fdir, savename(ps.prefix*"FIG-RT", ps, "html"; ignores) )
    savefig( p, fname )
    plot_results && run(`firefox $(fname)`, wait=false)
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

function postprocessing(fname; plot_results = false, exclude = [])
	## ==========================================================================
    @info "load data"
    fdir, ps, ignores = parse_fname(fname)
	p = PostProcessing1(fdir, ps, ignores, exclude)
	savechain(p)

    ## ==========================================================================
    @info "plot chain"
    plot_chains(p; plot_results)

    ## ==========================================================================
    @info "meanplot"
    plot_means(p; plot_results)
    ## ==========================================================================
    # perform diagnostics
	p = skip_warmup(p)
	diagnostics(p)
    ## ==========================================================================
    @info "make predictions"
    gp = generate_posterior(p)
    ## ==========================================================================
    @info "plot regions"
    plot_regions(p, gp; plot_results)
    ## ==========================================================================
    @info "plot rt"
    plot_rt(p, gp; plot_results)
    # -----------------------------------------------------------------------------
    @info "store reproduction number"
    save_rt(p, gp)
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
# # function expected!(obsmodel::SimpleObsModel1, newly_infecteds)
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
