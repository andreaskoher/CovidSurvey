function firefox(p; fname="tmp.html", fdir=normpath(homedir(), ".tmp"))
    # mkpath(dir)
    filepath = normpath(fdir, fname)
    savefig(p, filepath)
    run(`firefox $(filepath)`, wait=false)
    return p
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
	fname = normpath( fdir, savename(ps.prefix*"GENERATED-QUANTITIES", ps, "bson"; ignores) )
	if isfile(fname)
		return BSON.load(fname) |> NamedTuple
	else
		chains_params = Turing.MCMCChains.get_sections(chain, :parameters)
		generated_posterior = Regional.posterior(model, data.turing_data, chains_params)
	    fname = normpath( fdir, savename(ps.prefix*"GENERATED-QUANTITIES", ps, "bson"; ignores) )
	    dic = Dict( zip( keys(generated_posterior), values(generated_posterior) ) )
	    bson( fname ,  dic )
		return generated_posterior
	end
end

function plot_regions(p, gp; plot_results = false)
	@unpack fdir, ps, ignores, chain, data = p
	for r in data.regions
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
	for (i,r) in enumerate(data.regions)
        df = Regional.rtstats(
            gp.Rts[i],
            data.dates[i];
            startdate= first(data.dates[i]),
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
	        !occursin("R0", x) &&
			!occursin("R1", x), String.(names(c))
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
	        !occursin("R0", x) &&
			!occursin("R1", x), String.(names(c))
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
        f.(invf(float(R0)) .+ __ * x) |>
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
        for (j,r) in enumerate(pp.data.regions)
            q = effect_quantiles(effect_on_Rt, pp, gp, i, j)
            push!(data, [q..., p, r])
        end
        r = "grouped"
        q = effect_quantiles(effect_on_Rt, pp, gp, i)
        push!(data, [q..., p, r])
    end

	DataFrame(
		lower95 = getindex.(data, 1),
		lower50 = getindex.(data, 2),
	    median = getindex.(data, 3),
	    upper50 = getindex.(data, 4),
	    upper95 = getindex.(data, 5),
	    predictor = getindex.(data, 6),
	    region = getindex.(data, 7),
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

function plot_regional_effects(i, num_predictors, effects::DataFrame, regions = Regional.regions; xlabel = "effect size")

    p = plot(;
          xlabel = i == num_predictors ? xlabel : ""
        , legend = i == 1 ? :outertopright : nothing
        , title = effects.predictor |> first
        , bottom_margin = i == num_predictors ? 0mm : 3mm
        , top_margin = i == 1 ? 3mm : 3mm
    )

    for (j,region) in enumerate(regions)
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
            plot_regional_effects(i, num_predictors, __, effects.region; xlabel)
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
	effects = effect_quantiles(pp, gp; effect_on_Rt)

	suffix = "-EFFECTS"
	suffix = grouped ? suffix *= "-GROUPED" : suffix *= "-REGIONAL"
	suffix = effect_on_Rt == 0 ? suffix *= "-ABSOLUTE" :  suffix *= "-RT=$(effect_on_Rt)"
	xlabel = effect_on_Rt == 0 ? "effect size" : "change in Rt [\\%]"
	plot_effects = grouped ? plot_grouped_effects : plot_regional_effects

	fname = normpath( fdir, savename(ps.prefix*"FIG"*suffix, ps, "html"; ignores) )
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
	#savechain(p)
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
	@info "make predictions"
	gp = generate_posterior(p)
	## ==========================================================================
	# @info "plot autocorr"
	# plot_autocorr(p; plot_results)
	## ==========================================================================
	# @info "plot pairs"
	# plot_pair(p; plot_results)
    ## ==========================================================================
    @info "plot regions"
    plot_regions(p, gp; plot_results)
    ## ==========================================================================
    @info "plot rt"
    plot_rt(p, gp; plot_results)
	## ==========================================================================
	if p.data.turing_data.num_covariates > 0
		@info "plot predictors"
		# pgfplotsx()
		# default(titlefontsize = 20, legendfontsize = 18, labelfontsize = 18, guidefontsize = 18, tickfontsize = 12, framestyle = :zerolines, yminorgrid = true)

		Regional.plot_effects(p, gp; plot_results, grouped = false, effect_on_Rt = 0.)
		Regional.plot_effects(p, gp; plot_results, grouped = false, effect_on_Rt = 2.)
		Regional.plot_effects(p, gp; plot_results, grouped = false, effect_on_Rt = -0.5)

		Regional.plot_effects(p, gp; plot_results, grouped = true, effect_on_Rt = 0.)
		Regional.plot_effects(p, gp; plot_results, grouped = true, effect_on_Rt = 2.)
		Regional.plot_effects(p, gp; plot_results, grouped = true, effect_on_Rt = -0.5)
		# plotlyjs()
		# default()
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
	"intdeaths" => Regional.model_international_deaths,
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
