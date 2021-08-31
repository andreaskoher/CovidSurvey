function firefox(p; fname="tmp.html", fdir=normpath(homedir(), ".tmp"))
    # mkpath(dir)
    filepath = normpath(fdir, fname)
    savefig(p, filepath)
    run(`firefox $(filepath)`, wait=false)
    return p
end

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
    return fname
end

# ============================================================================
# post-processing

struct PostProcessing3{F,P,I,C,D,M}
	fdir::F
	ps::P
	ignores::I
	chain::C
	data::D
	model::M
end

function PostProcessing3(fdir, ps, ignores, exclude, fname)
	chain = let
	    chain = read(fname, Chains)
	    not(chain, exclude)
	end
	data = let
		fn_data  = replace(fname, "CHAIN_"=>"DATA_")
		fn_data  = replace(fn_data, ".jls"=>".bson")
		BSON.load(fn_data)["data"]
	end

	model = National.name2model[ps.model]
	PostProcessing3(fdir, ps, ignores, chain, data, model)
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
	ps = NamedTuple(Symbol(k)=>v for (k,v) in pairs(parsed_args))

	folders = split(fdir_raw, "/")
	i = findfirst(==("CovidSurvey"), folders)
	fdir = normpath(homedir(),"dev/CovidSurvey", folders[i+1:end-1]...)
	@info "save post-processing in $fdir"
	return (; fdir, ps, ignores)
end

function savechain(pp::PostProcessing3)
	@unpack fdir, ps, ignores, chain = pp
    fname = normpath( fdir, savename(ps.prefix*"CHAIN", ps, "jls"; ignores) )
    safesave( fname, chain )
end

function plot_chains(pp::PostProcessing3; plot_results = false)
	@unpack fdir, ps, ignores, chain = pp
	n = filter( x->!occursin(r"latent_Rt", x), String.(names(chain)))
    p = plot(chain[n]);
    fname = normpath( fdir, savename(ps.prefix*"FIG-CHAINSPLOT", ps, "html"; ignores) )
    savefig(p, fname )
    plot_results && (display(p); run(`firefox $(fname)`, wait=false))
	return nothing
end

function plot_means(pp::PostProcessing3; plot_results=false)
	@unpack fdir, ps, ignores, chain = pp
	n = filter( x->!occursin(r"latent_Rt", x), String.(names(chain)))
	p = meanplot(chain[n]);
	fname = normpath( fdir, savename(ps.prefix*"FIG-MEANPLOT", ps, "html"; ignores) )
	savefig(p, fname )
	plot_results && (display(p); run(`firefox $(fname)`, wait=false))
	return nothing
end

function skip_warmup(pp::PostProcessing3)
	@unpack fdir, ps, ignores, chain, data, model = pp
	chain = chain[ps.warmup+1:end,:,:]
	PostProcessing3(fdir, ps, ignores, chain, data, model)
end

function diagnostics(pp::PostProcessing3)
	@unpack fdir, ps, ignores, chain = pp
	if ps.chains > 1
        @info "gelman diagnostics"
        diagnostics = gelmandiag(chain)
        fname = normpath( fdir, savename(ps.prefix*"GELMANDIAG", ps, "csv"; ignores) )
        safesave( fname, diagnostics)
        pretty_table(diagnostics; crop=:none)
    end
end

function generate_posterior(pp::PostProcessing3)
	@unpack fdir, ps, ignores, chain, model, data = pp

	generated_posterior = let
	    m_pred = model(data.turing_data, true)
	    gq = Turing.generated_quantities(m_pred, chain)
	    vectup2tupvec(reshape(gq, length(gq)))
	end

    fname = normpath( fdir, savename(ps.prefix*"GENERATED-QUANTITIES", ps, "bson"; ignores) )
    dic = Dict( zip( keys(generated_posterior), values(generated_posterior) ) )
    bson( fname ,  dic )
	return generated_posterior
end

function plot_prediction(pp, gp; plot_results = false)
	@unpack fdir, ps, ignores, chain, data = pp
	p = National.plot(data, gp)
	fname = normpath( fdir, savename(ps.prefix*"FIG-PREDICTION", ps, "html"; ignores) )
    savefig( p, fname )
    plot_results && (display(p); run(`firefox $(fname)`, wait=false))
end

function save_rt(p, gp)
	@unpack fdir, ps, ignores, chain, data = p
    df = rtstats(
        gp.Rt,
        data.dates;
        startdate=Date("2020-05-15"),
        stopdate=data.observations_end
    )
    fname = normpath( fdir, savename(ps.prefix*"Rt", ps, "csv"; ignores) )
    save(fname, df)
end

function rtstats(Rt, dates; startdate=Date("2020-05-15"), stopdate=nothing)
    ib = findfirst(==(startdate), dates)
    ie = isnothing(stopdate) ? length(dates) : findfirst(==(Date(stopdate)), dates)
    Rt_array = hcat(Rt...)[ib:ie,:]
    qs = [quantile(v, [0.025, 0.25, 0.5, 0.75, 0.975]) for v in eachrow(Rt_array)]
    llq, lq, mq, uq, uuq = (eachrow(hcat(qs...))..., )

    return DataFrame((; dates = dates[ib:ie], llq, lq, mq, uq, uuq))
end

function save_generated_posterior(p, gp)
	@unpack fdir, ps, ignores, chain, data = p
    fname = normpath( fdir, savename(ps.prefix*"GENERATED-QUANTITIES", ps, "bson"; ignores) )
    dic = Dict( zip( keys(gp), values(gp) ) )
    bson( fname ,  dic )
end

function postprocessing(fname; plot_results = false, exclude = [], warmup = nothing)
	## ==========================================================================
    @info "load data"
    fdir, ps, ignores = parse_fname(fname; warmup)
	p = PostProcessing3(fdir, ps, ignores, exclude, fname)
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
    @info "save generated quantities"
	save_generated_posterior(p, gp)
    ## ==========================================================================
    @info "plot prediction"
    plot_prediction(p, gp; plot_results)
    # -----------------------------------------------------------------------------
    @info "store reproduction number"
    save_rt(p, gp)
	return nothing
end

name2model = Dict(
    "hospit" => National.model_hospit,
	"deaths" => National.model_deaths,
	"cases"  => National.model_cases,
)
