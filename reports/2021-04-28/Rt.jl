using DrWatson
quickactivate(@__DIR__)
@show projectdir()
##
using CovidSurvey
using CSV
using DataFrames
using Dates
# using RCall
using Turing
using StatsPlots
using Plots.PlotMeasures
using Random
## ===========================================================================
# load data and model
# ============================================================================
# fname = projectdir("reports/2021-04-28/Rt_cases=2020-06-01_chains=10_hospit=true_model=contacts2_observ=2021-02-07_preds=CF,CC,CR,CS_seed=20000_sero=true_steps=2000_warmup=1000.csv")
# fname = projectdir("reports/2021-04-28/Rt_cases=2020-06-01_chains=10_hospit=true_model=v2_observ=2021-02-07_seed=10000_sero=true_steps=2000_warmup=1000.csv")
fname = projectdir("reports/2021-04-28/Rt_cases=2020-06-01_chains=10_hospit=true_model=contacts2_observ=2020-12-16_preds=CF,CC,CR,CS_seed=30000_sero=true_steps=1000_warmup=1000.csv")
# fname = projectdir("reports/2021-04-28-b/Rt_cases=2020-06-01_chains=10_hospit=true_model=contacts2_observ=2020-12-16_preds=CF,CC,CR,CS_seed=40000_sero=true_steps=2000_warmup=1000.csv")
# fname = projectdir("reports/2021-04-28-b/Rt_cases=2020-06-01_chains=10_hospit=true_model=contacts2_observ=2021-02-07_preds=CF,CC,CR,CS_seed=40000_sero=true_steps=2000_warmup=1000.csv")
rtp = load(fname) |> DataFrame
# fname = projectdir("reports/2021-04-28/Rt_cases=2020-06-01_chains=10_hospit=true_model=v2_observ=2021-02-07_seed=10000_sero=true_steps=2000_warmup=1000.csv")
# fname = projectdir("reports/constant_iar/Rt_cases=2020-06-01_chains=10_hospit=true_model=const_observ=2021-02-06_sero=true_steps=2000_warmup=1000.csv")
fname = projectdir("reports/2021-04-27/Rt_cases=2020-06-01_chains=10_hospit=true_model=v2_observ=2020-12-16_seed=10000_sero=true_steps=2000_warmup=1000.csv")
rt = load(fname) |> DataFrame
#------------------------------------------------------------------------------
# Rt_with_vs_without_predictors
Plots.scalefontsizes(1.3)
pgfplotsx()
p = let
    p = plot(legend=:bottomright, ylabel=L"R_t")
    plot!(rtp.date, rtp.mq, ribbon=(rtp.mq - rtp.llq, rtp.uuq - rtp.mq), lw=2, label="with predictors (contact data)")
    plot!(rt.date, rt.mq, ribbon=(rt.mq - rt.llq, rt.uuq - rt.mq), lw=2, label="without predictors")
    # xlims!((t[1],t[end]))
end
savefig(projectdir("figures/compare_Rt.png"))
#------------------------------------------------------------------------------
# model vs SSI
Plots.scalefontsizes(1.5)

function make_consistent(df1, df2; startdate=nothing, enddate=nothing)
    df1 = deepcopy(df1)
    df2 = deepcopy(df2)
    ts = isnothing(startdate) ? max(df1.date[1], df2.date[1]) : Date(startdate)
    te = isnothing(enddate) ? min(df1.date[end], df2.date[end]) : Date(enddate)
    df1 = df1[ findfirst(==(ts), df1.date) : findfirst(==(te), df1.date), :]
    df2 = df2[ findfirst(==(ts), df2.date) : findfirst(==(te), df2.date), :]
    @assert all(df1.date .== df2.date)
    return df1, df2
end

pgfplotsx()
p = let
    rt = projectdir("reports/test_single_observation_AlternativeDelays/Rt_cases=2020-06-01_chains=10_hospit=true_model=cases_observ=2021-02-06_sero=true_steps=2000_warmup=1000.csv") |> load |> DataFrame
    rt_ssi = projectdir("data/Rt_SSI.csv") |> load |> DataFrame
    rt, rt_ssi = make_consistent(rt, rt_ssi, enddate="2020-12-15")
    p = plot(legend=:topleft, ylabel=L"R_t", size=(800, 500))
    plot!(rt_ssi.date, rt_ssi.mq, ribbon=(rt_ssi.mq - rt_ssi.llq, rt_ssi.uuq - rt_ssi.mq), lw=2, label="SSI", c=:sienna)
    plot!(rt.date, rt.mq, ribbon=(rt.mq - rt.llq, rt.uuq - rt.mq), lw=2, label="model without predictors", c=:royalblue)
end
savefig(projectdir("figures/compare_Rt_model_without_predictors_vs_SSI.png"))
#------------------------------------------------------------------------------

#------------------------------------------------------------------------------
fnames = [
    # projectdir("reports/test_single_observation_AlternativeDelays/Rt_cases=2020-06-01_chains=10_hospit=true_model=cases_observ=2021-02-06_sero=true_steps=2000_warmup=1000.csv")
    # projectdir("reports/test_single_observation_AlternativeDelays/Rt_cases=2020-06-01_chains=10_hospit=true_model=hospit_observ=2021-02-06_sero=true_steps=2000_warmup=1000.csv")
    # projectdir("reports/test_single_observation_AlternativeDelays/Rt_cases=2020-06-01_chains=10_hospit=true_model=deaths_observ=2021-02-06_sero=true_steps=2000_warmup=1000.csv")

    # projectdir("reports/test_single_observation_BrauersParameters/Rt_cases=2020-06-01_chains=10_hospit=true_model=cases_observ=2021-03-25_sero=true_steps=2000_warmup=1000.csv")
    # projectdir("reports/test_single_observation_BrauersParameters/Rt_cases=2020-06-01_chains=10_hospit=true_model=hospit_observ=2021-03-25_sero=true_steps=2000_warmup=1000.csv")
    # projectdir("reports/test_single_observation_BrauersParameters/Rt_cases=2020-06-01_chains=10_hospit=true_model=deaths_observ=2021-03-25_sero=true_steps=2000_warmup=1000.csv")

    projectdir("reports/variable-delays/Rt_cases=2020-06-01_chains=10_hospit=true_model=cases_observ=2021-02-06_sero=true_steps=4000_warmup=1000.csv")
    projectdir("reports/variable-delays/Rt_cases=2020-06-01_chains=10_hospit=true_model=hospit_observ=2021-02-06_sero=true_steps=4000_warmup=1000.csv")
    projectdir("reports/variable-delays/Rt_cases=2020-06-01_chains=10_hospit=true_model=deaths_observ=2021-02-06_sero=true_steps=4000_warmup=1000.csv")
]
ns = ["cases", "hospit", "death"]

# pgfplotsx()
plotlyjs()
p = let
    # rt, rt_ssi = make_consistent(rt, rt_ssi, enddate="2020-12-15")
    p = plot(legend=:outertopright, ylabel="R_t", size=(1200, 600))
    for (fname, n) in zip(fnames, ns)
        rt = load(fname) |> DataFrame
        plot!(rt.date, rt.mq, ribbon=(rt.mq - rt.llq, rt.uuq - rt.mq), lw=2, label=n)
    end
    # xlims!((t[1],t[end]))
    rt = projectdir("data/Rt.csv") |> load |> DataFrame
    plot!(rt.date, rt[:,"Rt-SSI"], lw=2, label="SSI")
    # plot!(rt.date, rt[:,"Rt-SSI-shifted"], lw=2, label="SSI-shifted")
    # plot!(rt.date, rt[:,"Rt-observ-2021-02-07"], lw=2, label="observ-2021-02-07")
    survey = load( "/home/and/data/covidsurvey/smoothed_contacts.csv" ) |> DataFrame
    plot!(survey.date, survey.colleagues, ticks=:native, label="colleagues")
    plot!(survey.date, survey.family, ticks=:native, label="family")
    plot!(survey.date, survey.strangers, ticks=:native, label="strangers")
    plot!(survey.date, survey.friends, ticks=:native, label="friends")
    p
end;
savefig("/home/and/tmp/figures/Rt_vs_contacts_2021-06-04.html")
run(`firefox /home/and/tmp/figures/Rt_vs_contacts_2021-06-04.html`, wait=false)

#------------------------------------------------------------------------------
# Rt_vs_predictors
pgfplotsx()
cov = turing_data.covariates
p = let
    p = plot(legend=:topright)
    plot!(Rt_original.date, Rt_original.mq, ribbon=(Rt_original.mq - Rt_original.llq, Rt_original.uuq - Rt_original.mq), linewidth=2, label="Rt without predictors", c=:black)
    p2 = twinx()
    for (i, n) in enumerate(["colleagues","strangers","family","friends"])
        plot!(p2, t, cov[1:length(t),i], lw=2, label=n, grid=:off, xticks=:none)
    end
    xlims!((t[1],t[end]))
end
# savefig(p, projectdir("figures/", "Rt_vs_predictors.png") )
#------------------------------------------------------------------------------
# compare to SSI data to predictors
Rt_SSI = National.readsurvey("/home/and/data/covidsurvey/contacts.csv")
Rt_SSI = select( Rt_SSI, [:Rt,:uncertainty_lower,:uncertainty_upper, :date])
p = let
    p = plot(legend=:topright)
    plot!(Rt_SSI.date, Rt_SSI.Rt, ribbon=(Rt_SSI.Rt - Rt_SSI.uncertainty_lower, Rt_SSI.uncertainty_upper - Rt_SSI.Rt), linewidth=2, label="Rt (SSI) without predictors", c=:black)
    p2 = twinx()
    for (i, n) in enumerate(["colleagues","strangers","family","friends"])
        plot!(p2, t, cov[1:length(t),i], lw=2, label=n, grid=:off, xticks=:none)
    end
    xlims!((t[1],t[end]))
end
# savefig(p, projectdir("figures/", "Rt_from_SSI_vs_predictors.png") )
#------------------------------------------------------------------------------
# compare to SSI data to inferred Rt
p = let
    p = plot(legend=:topright)
    plot!(Rt_original.date, Rt_original.mq, ribbon=(Rt_original.mq - Rt_original.llq, Rt_original.uuq - Rt_original.mq), linewidth=2, label="Rt without predictors")
    plot!(Rt_SSI.date, Rt_SSI.Rt, ribbon=(Rt_SSI.Rt - Rt_SSI.uncertainty_lower, Rt_SSI.uncertainty_upper - Rt_SSI.Rt), linewidth=2, label="Rt (SSI)", c=:black)
    xlims!((t[1],t[end]))
end
# savefig(p, projectdir("figures/", "Rt_without_predictors_vs_Rt_from_SSI.png") )
#------------------------------------------------------------------------------
# compare to Rt from SSI or model to predictors
survey = National.readsurveys_raw()
p = let
    ps = []
    ts = survey.date
    for (i, n) in [(3,:family), (1,:colleagues), (4,:friends), (2,:strangers)] #"CC,CS,CF,CR"
        p = plot()
        plot!(
         #Rt_SSI.date, Rt_SSI.Rt, ribbon=(Rt_SSI.Rt - Rt_SSI.uncertainty_lower, Rt_SSI.uncertainty_upper - Rt_SSI.Rt),
         Rt_original.date, Rt_original.mq, ribbon=(Rt_original.mq - Rt_original.llq, Rt_original.uuq - Rt_original.mq),
         linewidth=2, label=nothing, c=:black, α=.5, ylabel="\$R_t\$",
         right_margin=20mm, bottom_margin=10mm)
        p2 = twinx()
        xs = survey[:, n]
        x = cov[1:length(t),i]
        National.surveynorm!(xs)
        plot!(p2, ts, xs / std(xs) * std(x), lw=2,
            label="$n (raw)", grid=:off, xticks=:none, c=:darkred,
            ylabel="contacts (normalized)",
            right_margin=20mm, bottom_margin=10mm,
            extra_kwargs=:subplot, legend=:outertop, legend_columns=-1)
        plot!(p2, t, x, lw=4, label="$n (smoothed)",
            grid=:off, xticks=:none, c=:darkorange,
            right_margin=20mm, bottom_margin=10mm,
            extra_kwargs=:subplot, legend=:outertop, legend_columns=-1)
        xlims!((t[1],t[end]))
        push!(ps, p)
    end
    #l = @layout [a{0.2w} b{0.2w}; c d]
    plot(ps..., layout=4, size=(1300, 1000))
end
# savefig(p, projectdir("figures/", "Rt_vs_predictors_smoothed_and_raw.png") )
