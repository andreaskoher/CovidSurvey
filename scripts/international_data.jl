using DrWatson
quickactivate(@__DIR__)
@show projectdir()
##
using CSV
using DataFrames
using Dates
using Plots
using BSON
using Underscores
import Downloads
using ColorSchemes
using RData
using CovidSurvey
using StatsBase
colors = ColorSchemes.tableau_10.colors
# using Makie
# using GLMakie
# GLMakie.activate!()
plotlyjs()
##

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

function plot_rt(fname)
    p = plot(title="Reproduction Number")
    data = RData.load(fname)
    dates      = data["time"]
    values     = @_ data["draws"] |>
                    transpose |>
                    Array |>
                    __[:, 2001 : end]
    recipe = Regional.RtPlottingRecipe1( nothing, nothing, nothing, (; dates, values), "Rt")
    return plot!(p, recipe)
end

function firefox(p; fname="tmp.html", fdir=normpath(homedir(), ".tmp"))
    # mkpath(dir)
    filepath = normpath(fdir, fname)
    savefig(p, filepath)
    run(`firefox $(filepath)`, wait=false)
    return p
end

##
let
    ctype = "strangers"
    rawcontacts = readcontacts( projectdir("data/contacts/int/international_contacts.csv") )
    regions = unique(rawcontacts.region)
    for region in regions
        # plot raw data
        plts = Vector{Plots.Plot}()
        raw = filter(:region => ==(region), rawcontacts)
        p = scatter(raw.date, raw.strangers, mα=0.1, mc=colors[2], msc=:match; title="raw contacts to $ctype in $region", lab=nothing)
        push!(plts, p)
        # plotweeks!(p1, raw.date)

        # plot preprocessed data
        batched  = CSV.read(projectdir("data/contacts/int/mean_contact_rates_region=$(region)_aggregation=batch.csv"), DataFrame)
        smoothed = CSV.read(projectdir("data/contacts/int/smoothed_contact_rates_region=$(region).csv"), DataFrame)
        suffix = region != "Denmark" ? "" : "_low_resolution"

        p = plot(title="pre-processed contacts to $ctype in $region")
        scatter!(batched.date, batched[:,ctype], mα=1, mc=colors[1], msc=:match; lab=nothing)
        plot!(smoothed.date, smoothed[:,ctype*suffix]; label=false, ribbon=3smoothed[:,"$(ctype)_std"suffix], hover=smoothed.date, lw=4)
        # plotweeks!(p2, batched.date)
        push!(plts, p)

        # plot case counts
        epidemic = CSV.read(projectdir("data/epidata/epidata_region=$(region).csv"), DataFrame)
        is = findfirst(>(22), epidemic.cases_per_100000)
        ds = epidemic.date[is]
        de1 = epidemic.date[is:end][findfirst(<(10), epidemic.cases_per_100000[is:end])]
        de2 = epidemic.date[findfirst(>(0.05), epidemic.fraction_vaccinated)]
        de  = min(de1, de2)
        p = plot(title="observed cases")
        vspan!([ds, de], label="active epidemic", α=0.2, fc=:midnightblue)
        plot!(epidemic.date, epidemic.deaths, fc=colors[1], lc=colors[1], label="smoothed cases / 100_000", lw=4)#, hover=epidemic.date)
        #plot!(twinx(), epidemic.date, epidemic.cases, fc=colors[1], lc=colors[1], label="cases", lw=0.8, α=0.5)
        push!(plts, p)

        # plot predictors
        mobility = CSV.read(projectdir("data/mobility/mobility_country=$(region).csv"), DataFrame)
        p = plot(title="predictors")
        vspan!([ds, de], label="active epidemic", α=0.2, fc=:midnightblue)
        plot!(p, smoothed.date, zscore(smoothed[:,ctype]); label="contact to $ctype", hover=smoothed.date, lw=4)
        plot!(p, mobility.date, zscore(mobility.apple); label="apple", lw=1, hover=mobility.date)
        plot!(p, mobility.date, zscore(mobility.google); label="google", lw=1, hover=mobility.date)
        push!(plts, p)

        fname = projectdir("reports/stanfit/2021-10-05/sampled_rt_region=$(region).rds")
        if isfile(fname)
            p = plot_rt(fname)
            vspan!([ds, de], label="active epidemic", α=0.2, fc=:midnightblue)
            push!(plts, p)
        end

        p = plot(title="fraction of vaccinated")
        vspan!([ds, de], label="active epidemic", α=0.2, fc=:midnightblue)
        plot!(epidemic.date, epidemic.fraction_vaccinated * 100, fc=colors[1], lc=colors[1], label="vaccinated", lw=4)#, hover=epidemic.date)
        push!(plts, p)

        p = plot(plts..., layout=(length(plts),1), size=(1000, length(plts)*250), sharex=true, link=:x)
        firefox(p, fname="$region.html")
    end
end




















## =========================================================================
#         tests
pr = let
    data = RData.load(projectdir("reports/stanfit/2021-10-05/sampled_predictions.rds"))
    dates      = data["time"]
    values     = @_ data["draws"] |>
                transpose |>
                Array |>
                __[:, 2001 : end]
    (; dates, values )
end
o = ( dates = epidemic.date, values = epidemic.deaths )

p = plot()
plot!(p, o.dates, o.values, α=0.5, lc=:match, lab="observed", c=:midnightblue, lw=4, ylab="cases")
plot_confidence_timeseries!(p, pr.dates, pr.values; label = "predicted (95%CI)", c=:midnightblue) #Dict(hover=>strdates)
plot_confidence_timeseries!(p, e.dates, e.values; label = "predicted (50%CI)") #Dict(hover=>strdates)
plot!(epidemic.date, epidemic.deaths, fc=colors[1], lc=colors[1], label="deaths", lw=4)#, hover=epidemic.date)
##
# plot case counts
let
    epidemic = CSV.read(projectdir("data/epidata/epidata_region=$(region).csv"), DataFrame)
    is = findfirst(>(22), epidemic.cases_per_100000)
    ds = epidemic.date[is]
    de1 = epidemic.date[is:end][findfirst(<(10), epidemic.cases_per_100000[is:end])]
    de2 = epidemic.date[findfirst(>(0.05), epidemic.fraction_vaccinated)]
    de  = min(de1, de2)
    p1 = plot(title="observed deaths")
    vspan!([ds, de], label="active epidemic", α=0.2, fc=:midnightblue)
    plot!(epidemic.date, epidemic.deaths, fc=colors[1], lc=colors[1], label="deaths", lw=4)#, hover=epidemic.date)
    #plot!(twinx(), epidemic.date, epidemic.cases, fc=colors[1], lc=colors[1], label="cases", lw=0.8, α=0.5)

    p2 = plot(recipe)

    # plot predictors
    mobility = CSV.read(projectdir("data/mobility/mobility_country=$(region).csv"), DataFrame)
    p3 = plot(title="predictors")
    vspan!([ds, de], label="active epidemic", α=0.2, fc=:midnightblue)
    plot!(p3, smoothed.date, zscore(smoothed[:,ctype]); label="contact to $ctype", hover=smoothed.date, lw=4)
    plot!(p3, mobility.date, zscore(mobility.apple); label="apple", lw=1, hover=mobility.date)
    plot!(p3, mobility.date, zscore(mobility.google); label="google", lw=1, hover=mobility.date)

    p4 = plot(title="fraction of vaccinated")
    vspan!([ds, de], label="active epidemic", α=0.2, fc=:midnightblue)
    plot!(epidemic.date, epidemic.fraction_vaccinated * 100, fc=colors[1], lc=colors[1], label="vaccinated", lw=4)#, hover=epidemic.date)

    p = plot(p1, p2, p3, p4, layout=(4,1), size=(1000, 4*250), sharex=true, link=:x)
end
