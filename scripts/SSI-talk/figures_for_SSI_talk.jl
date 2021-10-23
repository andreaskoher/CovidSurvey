using DrWatson
quickactivate(@__DIR__)
@show projectdir()
##
using CovidSurvey
using CSV
using DataFrames
using Dates
# using ArviZ
using PrettyTables
import PyPlot as plt
# using RCall
using Turing
using StatsPlots
using Plots.PlotMeasures
using Random
using StatsBase
using BSON
using ColorSchemes
using Measures
colors = ColorSchemes.tableau_10.colors

using Underscores
import Downloads
using RData

gr()
pgfplotsx()
## ==========================================================================
#          Introduction

fdir, ps, ignores = Regional.parse_fname(fname)

pp = let
	fname = "/home/ankoh/dev/CovidSurvey/reports/regional_prediction_end_of_peak/2021-09-07/CHAIN_chains=10_model=hospit_observ=2021-03-01_regional=true_rwstep=1_semipar=true_std=false_steps=500_warmup=2000.jls"
	chain = read(fname, Chains)

  data_params = (
        observationsend  = ps.observ
      , predictors        = ps.preds|> CovidSurvey.parse_predictors
      # , cases_start       = ps.cases
      # , hospitmodel       = Regional.HospitInit1(obs_stop="2020-07-01")
      , casemodel         = Regional.CaseInit(obs_start="2020-06-01")
      , seromodel         = Regional.SeroInit(delay=0, std=1.)
      , rwstep            = ps.rwstep
      , epidemicstart    = 20
      , numimpute        = 6
      , include_early_dynamic = false#ps.include
      , link              = KLogistic(4.5)
      , invlink           = KLogit(4.5)
      , predictive        = false
      , covariates_kwargs = Dict(
        :semiparametric => ps.semipar,
        :fname          => [projectdir("data/contacts/dk/", "mean_contact_rates_region=$(r).csv" ) for r in 1:Regional.nregions],#[projectdir("data", "mean_contact_rates_region=$r.csv" ) for r in 1:Regional.nregions],
        :shift          => -1,
        :startdate      => "2020-11-10", # >= 1000 cases / day
        :datecol        => "dates",
        :aggregation    => ps.rwstep,
        :standartize    => ps.std,
        :mobility       => [projectdir("data/mobility/mobility_region=$(Regional.regions[i]).csv") for i in 1:Regional.nregions]
        )
      )
  data = Regional.load_data(; data_params... )

	model = Regional.name2model[ps.model]
	Regional.PostProcessing2(fdir, ps, ignores, chain, data, model)
end



function Plots.plot!(p::Plots.Plot, r::Regional.ObservationPlottingRecipe1; plot_only_fit=false)
    o  = r.observed
    e  = r.expected
    pr = r.predicted
    ed = r.enddate
    sd = r.startdate

    if !plot_only_fit
        !isnothing(ed) && vline!(p, [ed], lab="end observations", lw=2, lc=:black, hover="$ed")
        !isnothing(ed) && vline!(p, [sd], lab="start observations", lw=2, lc=:black, hover="$sd", ls=:dash)
    end
    plot!(p, o.dates, o.values, α=0.5, lc=:match, lab="observed $(r.label)", c=:midnightblue, lw=4, ylab="cases")
    # plot_confidence_timeseries!(p, pr.dates, pr.values; label = "predicted $(r.label)", c=:midnightblue) #Dict(hover=>strdates)
    plot_confidence_timeseries!(p, e.dates, e.values; c=colors[2], label = "expected $(r.label)") #Dict(hover=>strdates)
    if plot_only_fit
        xlo = Dates.value.(sd)
        xup = Dates.value.(ed)
        yup = maximum( o.values[o.dates .< ed] ) * 1.1
        xlims!(p, xlo, xup)
        ylims!(p, 0, yup)
    end
    return p
end


p = let region = "capital"
    gp = Regional.generate_posterior(pp)
    @unpack fdir, ps, ignores, chain, data = pp

    p1 = let
        recipe = Regional.RtPlottingRecipe1(data, gp, region, "Rt")
        p = plot(title = "Reproduction Number", bottom_margin = 20mm)
        plot!(p, recipe; plot_only_fit=true)
    end

    p2 = let
        r = recipe = Regional.LatentInfectionsPlottingRecipe3(pp.data, gp, region, "infections")
        p = plot(title = "Latent Infections", bottom_margin = 20mm)
        plot!(p, recipe; plot_only_fit=true)
    end

    p3 = let
        recipe = Regional.ObservationPlottingRecipe1(pp.data, gp, region, "hospitalizations")
        p = plot(title = "Observed Hospitalizations")
        plot!(p, recipe; plot_only_fit=true)
    end

    plot(p1, p2, p3, layout=(3,1), size=(800, 3*250), sharex=true, link=:x, legend = nothing, dpi=900)
end

savefig(p, projectdir("figures/SSI-talk/intro.png"))


























## ==========================================================================
#          Qualitative Analysis
# ==========================================================================
function StatsBase.zscore(df::DataFrame)
    y = similar(df)
    for n in names(df)
        y[!,n] = zscore(df[:,n])
    end
    return y
end

function conform(args...; normalize = false, stopdate=nothing, startdate=nothing)
    s = isnothing(startdate) ? maximum( first.( select.(args, Ref(:date) ) ) ).date : Date(startdate)
    e = isnothing(stopdate) ? minimum( last.( select.(args, Ref(:date) ) ) ).date : Date(stopdate)
    x = s:Day(1):e
    y = filter.(Ref(:date => >=(s)), args)
    y = filter.(Ref(:date => <=(e)), y)
    normalize && ( y = zscore.( select.( y, Ref(Not(:date)) ) ) )
    (x, y...)
end


function load_regional_data(i; kwargs...)

    observables = Dict()
    Regional.observables!(observables)
    hospit = observables["hospit"]
    cases = observables["cases"]

    # fname = projectdir("reports/2021-08-01/Rt-$(uppercase(Regional.regions[i]))_chains=10_epidemicstart=10_folder=2021-08-01_model=cases_numimpute=6_rwstep=1_steps=100_warmup=500.csv")
    fname = projectdir("/home/ankoh/dev/CovidSurvey/reports/regional_prediction_end_of_peak/2021-09-07/Rt-$(uppercase(Regional.regions[i]))_chains=10_model=hospit_observ=2021-03-01_regional=true_rwstep=1_semipar=true_std=false_steps=500_warmup=2000.csv")
    rt = CSV.File(fname) |> DataFrame
    rename!(rt, "dates"=>"date")

    # fname = projectdir("data/raw_contact_rates_region=$(i)_aggregation=7.csv")
    fname = projectdir("data/contacts/dk/averaged_contact_rates_region=$i.csv")
    surv = CSV.File(fname) |> DataFrame
    "dates" in names(surv) && rename!(surv, "dates"=>"date")

    fname = projectdir("data/mobility/mobility_region=$(Regional.regions[i]).csv")
    mobil = CSV.File(fname) |> DataFrame

    return conform(cases, hospit, rt, surv, mobil; kwargs...)
end

function plot_comparison(date, cases, hospit, rt, surv, mobil; column=:country, plotactive = false)
    plots  = Vector{Plots.Plot}()
    ds = Date("2020-11-10")
    de = Date("2021-02-01")
    p = plot(; xaxis = true, title="observed hospitalizations", xticks=:native, yticks=:native, bottom_margin = 10mm)
    plotactive && vspan!([ds, de], label="active epidemic", α=0.2, fc=:midnightblue)
    #plot!(date, zscore(cases[:,column]), lab="cases", lw=2)
    plot!(date, hospit[:,column], lab="hospitalizations", lw=2, c=colors[1], legend=false)
    push!(plots, p)

    p = plot(; xaxis = true, title="reproduction number", xticks=:native, bottom_margin = 10mm,xformatter=_->"", legend=false)
    plot!(rt.date, rt.mq, ribbon=(rt.mq - rt.lq, rt.uq - rt.mq), c=colors[2], yticks=:native, lab="reproduction number", lw=2)
    plotactive && vspan!([ds, de], label="active epidemic", α=0.2, fc=:midnightblue)
    push!(plots, p)

    # p = plot(; xaxis = true, title="contact survey", xticks=:native, yticks=:native, bottom_margin = 0mm, legend = false)
    # for (i, c) in enumerate(["family", "colleagues", "friends"])#names(surv, Not(:date))
    #     plot!(surv.date, surv[:,c], lab=c, lw=2, c=colors[i+3])
    # end
    # plot!(surv.date, surv[:,"strangers"], lab="strangers", lw=2, c=colors[3])
    # plotactive && vspan!([ds, de], label="active epidemic", α=0.2, fc=:midnightblue)
    # push!(plots, p)

    p = plot(; xaxis = true, title="mobility", xticks=:native, yticks=:native, legend = :bottomleft)
    plot!(mobil.date, mobil.apple, lab="apple", lw=2, c=colors[5])
    plot!(mobil.date, mobil.telco, lab="telco", lw=2, c=colors[4])
    plot!(mobil.date, mobil.google, lab="google", lw=3, c=colors[3])
    mobil.telco
    plotactive && vspan!([ds, de], label="active epidemic", α=0.2, fc=:midnightblue)
    push!(plots, p)

    return plot(plots..., layout=(length(plots),1), size=(1200, length(plots)*250), sharex=true, link=:x, dpi=900)
end


i = 1
r = Regional.regions[i]
data  = load_regional_data(i; stopdate="2021-02-01")
p     = plot_comparison( data...; column=Regional.regions[i], plotactive=true)
savefig(p, projectdir("figures/SSI-talk/regionaldata_#4.png"))






















## ==========================================================================
#         International Data Qualitative Analysis
# ==========================================================================

function parsetime(sraw)
    svec = split(sraw, "-")
    if length(svec) == 3
        d,m,y = svec
        sclean = join([d,m,"20$y"], "-")
        return Date(sclean, "d-u-y")
    else
        d = sraw[1:2]
        m = sraw[3:5]
        y = sraw[6:9]
        sclean = join([d,m,y], "-")
        return Date(sclean, "d-u-y")
    end
end

function readcontacts( fname )
    @_ fname |>
        CSV.read(__, DataFrame) |>
        DataFrames.select(__,
            :dataRound_glb => :batch,
            :edate => ByRow(x->parsetime(x)) => :date,
            "country_string" => :region,
            "qNew_3_2_resp" => :family,
            "qNew_3_3_resp" => :colleagues,
            "qNew_3_4_resp" => :friends,
            "qNew_3_5_resp" => :strangers,
        ) |>
    sort(__, :date)
end

function loadrt(fname)
    data = RData.load(fname)
    dates      = data["time"]
    values     = @_ data["draws"] |>
                    transpose |>
                    Array |>
                    __[:, 2001 : end]
    recipe = Regional.RtPlottingRecipe1( nothing, nothing, nothing, (; dates, values), "Rt")
    return (; recipe, dates)
end

let
    ctype = "strangers"
    regions = ["Hungary", "UK", "France", "Sweden", "Denmark", "Germany", "Italy", "USA"]
    for region in regions
        # plot raw data

        # plot case counts
        epidemic = CSV.read(projectdir("data/epidata/epidata_region=$(region).csv"), DataFrame)
        is = findfirst(>(22), epidemic.cases_per_100000)
        ds = epidemic.date[is]
        de1 = epidemic.date[is:end][findfirst(<(10), epidemic.cases_per_100000[is:end])]
        de2 = epidemic.date[findfirst(>(0.05), epidemic.fraction_vaccinated)]
        de  = min(de1, de2)
        p1 = plot(title="observed cases")
        vspan!([ds, de], label="active epidemic", α=0.2, fc=:midnightblue)
        plot!(epidemic.date, epidemic.deaths, fc=colors[1], lc=colors[1], label="smoothed cases / 100_000", lw=4)#, hover=epidemic.date)
        #plot!(twinx(), epidemic.date, epidemic.cases, fc=colors[1], lc=colors[1], label="cases", lw=0.8, α=0.5)


		fname = projectdir("reports/stanfit/2021-10-05/sampled_rt_region=$(region).rds")
		rt = loadrt(fname)
		p2 = plot(title="Reproduction Number")
		plot!(p2, rt.recipe)
        vspan!([ds, de], label="active epidemic", α=0.2, fc=:midnightblue)

		# plot predictors
		smoothed = CSV.read(projectdir("data/contacts/int/smoothed_contact_rates_region=$(region).csv"), DataFrame)
        # suffix = region != "Denmark" ? "" : "_low_resolution"
		mobility = CSV.read(projectdir("data/mobility/mobility_country=$(region).csv"), DataFrame)
        p3 = plot(title="predictors")
        vspan!([ds, de], label="active epidemic", α=0.2, fc=:midnightblue)
        plot!(p3, smoothed.date, zscore(smoothed[:,ctype]); label="contact to $ctype", hover=smoothed.date, lw=4)
        plot!(p3, mobility.date, zscore(mobility.apple); label="apple", lw=1, hover=mobility.date)
        plot!(p3, mobility.date, zscore(mobility.google); label="google", lw=1, hover=mobility.date)

		p4 = plot(title="fraction of vaccinated")
		vspan!([ds, de], label="active epidemic", α=0.2, fc=:midnightblue)
		plot!(epidemic.date, epidemic.fraction_vaccinated * 100, fc=colors[1], lc=colors[1], label="vaccinated", lw=4)#, hover=epidemic.date)

        p = plot(p1,p2,p3, p4, layout=(4,1), size=(1200, 4*250), sharex=true, link=:x, xlim=(rt.dates[1],Date("2021-06-01")), legend=false, dpi=900)
        # firefox(p, fname="$region.html")
		savefig(p, projectdir("figures/SSI-talk/international_data_region=$region.png"))
    end
end
