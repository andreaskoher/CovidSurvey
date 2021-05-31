using DrWatson
quickactivate(@__DIR__)
@show projectdir()
##
using CSV
using DataFrames
# using DataFramesMeta
using Dates
using StatsBase
using CovidSurvey
using StatsPlots
using LaTeXStrings

using KernelFunctions
using AbstractGPs
using Stheno

# Load standard packages from the Julia ecosystem
using Optim # Standard optimisation algorithms.
using ParameterHandling # Helper functionality for dealing with model parameters.
using Zygote # Algorithmic Differentiation
using Turing
using ReverseDiff
using Memoization
setadbackend(:reversediff)
Turing.setrdcache(true)

using ParameterHandling: value, flatten
plotlyjs()
#---------------------------------------------------------------------------
# utils
struct ELBO{X,Y,U}
    x::X
    y::Y
    u::U
end

function (g::ELBO)(θ)
    ps = unpack(θ)
    f  = build_gp(ps)
    fx = f(g.x, ps.v)
    fu = f(g.u, 1e-6)
    return -elbo(fx, g.y, fu) #-logpdf(f(X1, ps.v), y_friends)
end

struct NLML{X,Y}
    x::X
    y::Y
end

function (g::NLML)(θ)
    ps = unpack(θ)
    f  = build_gp(ps)
    fx = f(g.x, ps.v)
    return -logpdf(fx, g.y) #-logpdf(f(X1, ps.v), y_friends)
end

#----------------------------------------------------------------------------
# survey data
holidays = load( projectdir("data", "holidays.csv") ) |> DataFrame
Rt_SSI = National.readsurvey("/home/and/data/covidsurvey/contacts.csv")
Rt_SSI = Rt_SSI[:,[:Rt,:uncertainty_lower,:uncertainty_upper, :date]]
# survey = National.readsurvey("/home/and/data/covidsurvey/contacts_0425.csv")
# survey = National.readsurveys_raw()
survey = load( "/home/and/data/covidsurvey/smoothed_contacts.csv" ) |> DataFrame
survey[!,:weekend] = Float64.( dayofweek.(survey.date) .> 5 )
#Rt = load(projectdir("out", "Rt.csv"))|>DataFrame
Rt = load( projectdir("reports/2021-04-27/Rt_cases=2020-06-01_chains=10_hospit=true_model=v2_observ=2020-12-16_seed=10000_sero=true_steps=2000_warmup=1000.csv") ) |> DataFrame
t = survey.date
holidays = holidays[holidays.date .∈ Ref(t),:]
#----------------------------------------------------------------------------
# plot survey data
plot(t, Rt.mq, ribbon=(Rt.mq - Rt.llq, Rt.uuq - Rt.mq), linewidth=0, label="95% quantiles")
plot!(t, Rt.mq, ribbon=(Rt.mq - Rt.lq, Rt.uq - Rt.mq), linewidth=2, label="50% quantiles", hover=mq)
plot!(survey.date, survey.colleagues, ticks=:native, label="colleagues")
plot!(survey.date, survey.family, ticks=:native, label="family")
plot!(survey.date, survey.strangers, ticks=:native, label="strangers")
plot!(survey.date, survey.friends, ticks=:native, label="friends")
#--------------------------------------------------------------------
# time series decomposition
survey_col = :friends_2m_p95
d = survey[:, survey_col]
y = zscore(d)

params = (
    s1 = positive(1.),
    s2 = positive(1.),
    s3 = positive(1.),
    l1 = positive(20.),
    l2 = positive(1.),
    # l3 = positive(50.),
    v  = positive(0.1),
)

function build_gp(ps::NamedTuple)
    return @gppp let
        f₁ = GP( ps.s1 * Matern52Kernel() ∘ ScaleTransform(1/ps.l1) )
        f₂ = GP( ps.s2 * Matern12Kernel() ∘ ScaleTransform(1/ps.l2) )
        f₃ = GP( ps.s3 * PeriodicKernel() ∘ ScaleTransform(1/7)  )
        f₀ = f₁ + f₂ + f₃
    end
end

x_ = 1. : length(survey.date)
x = GPPPInput(:f₀, x_)

flat_initial_params, unflatten = flatten(params)
unpack(θ::Vector) = value(unflatten(θ))
##
# obj = ELBO(x, y, u)
obj = NLML(x,y)
obj(flat_initial_params)

opt = Optim.optimize(obj, flat_initial_params, NelderMead(), Optim.Options(show_trace=true))
# opt = Optim.optimize(obj, opt.minimizer, LBFGS(), Optim.Options(show_trace=true), autodiff=:forward)
flat_final_params = opt.minimizer
final_params = value(unflatten(flat_final_params))
##
ps = final_params
f  = build_gp(final_params)
fx = f(x, final_params.v)
fp = posterior(fx, y);
#--------------------------------------------------------------------
x1 = GPPPInput(:f₁, x_);
x2 = GPPPInput(:f₂, x_);
x3 = GPPPInput(:f₃, x_);
##
pgfplotsx()
# Plots.scalefontsizes(.5)
p = let
    p1 = let
        p = plot(t, y, c=:black, label="data");
        plot!(p, t, fp(x); color=:red, ls=:dash, label="fit")
        plot!(p, t, fp(x1); color=:red, label="slow trend")
        ih = [findfirst(==(h),t) for h in holidays.date]
        th = t[ih]
        yh = y[ih]
        scatter!(th, yh, label="holidays", mα=0, msa=1, hover=holidays.name)
        # annotate!([(Dates.value(th[i]) , yh[i], Plots.text(holidays.name[i], :top, 8) ) for (i,n) in enumerate(holidays.name)])
    end

    p2 = let
        p = plot()
        plot!(p, t, fp(x2); color=:orange, label="fast trend");
        hline!([0], label=nothing, c=:black, lw=2)
    end

    # p3 = let
    #     p = plot()
    #     plot!(p, t, fp(x3); color=:green, label="quasi-periodic trend");
    #     hline!([0], label=nothing, c=:black, lw=2)
    # end

    p4 = let
        df = DataFrame(
            :date => t,
            :weekly_effect => mean.( marginals( fp(x3) ) ),
            :day_of_week => dayofweek.(t)
        )
        gdf = groupby(df, :day_of_week, sort=true)
        cdf = combine(gdf, :weekly_effect => mean)

        p = plot()
        plot!(p, dayname.(cdf.day_of_week), cdf.weekly_effect_mean; color=:green, label="weekly effect", lw=2);
        hline!([0], label=nothing, c=:black, lw=2)
    end

    p5 = let
        ϵ = y-mean.( marginals( fp(x) ) )
        ih = [findfirst(==(h),t) for h in holidays.date]
        ϵh = ϵ[ih]
        p = plot()
        plot!(p, t, ϵ, label="residuals", c=:orange)
        scatter!(p, holidays.date, ϵh, label="holidays", mα=0, msa=1, hover=holidays.name)
        hline!([0], label=nothing, c=:black, lw=2)
    end
    plot(p1, p2, p4, p5, layout=(4,1), shapex=true, size=(1500, 1000))
end
savefig( p, projectdir("figures/", "survey_$(survey_col)_decomposition.png") )
run(`firefox $( projectdir("figures/", "survey_$(survey_col)_decomposition.png") )`, wait=false)
#---------------------------------------------------------------------
p = let
    m = marginals( fp(x1) )
    mf = mean.(m)
    sf = std.(m)
    sr = std(Rt.mq)
    smf = std(mf)

    mf .-= mf[1]
    mf .*= sr / smf
    mf .+= Rt.mq[1]
    sf .*= sr / smf

    p = plot(Rt.date, Rt.mq, ribbon=(Rt.mq - Rt.llq, Rt.uuq - Rt.mq), linewidth=0, label="R_t 95% quantiles", size=(1200, 800))
    plot!(Rt.date, Rt.mq, ribbon=(Rt.mq - Rt.lq, Rt.uq - Rt.mq), linewidth=2, label="R_t 50% quantiles", hover=Rt.mq)
    plot!(t, mf, ribbon=(sf,sf), label="contacts to $(survey_col) (smoothed trend)")
    #plot!(t, survey.Rt)
end
savefig( p, projectdir("figures/", "survey_$(survey_col)_vs_Rt.png") )
run(`firefox $( projectdir("figures/", "survey_$(survey_col)_vs_Rt.png") )`, wait=false)
#-----------------------------------------------------------------------------
smoothed[!, survey_col] = mean.( marginals( fp(x1) ) )
# smoothed2 = rename( smoothed, :family_2m_p95=>:family, :colleagues_2m_p95=>:colleagues, :strangers_2m_p95=>:strangers, :friends_2m_p95=>:friends)
# save( "/home/and/data/covidsurvey/smoothed_contacts_0425_2m.csv", smoothed)
# smoothed = DataFrame(:date => t)
##
plotlyjs()
smoothed = load( "/home/and/data/covidsurvey/smoothed_contacts.csv" ) |> DataFrame
contact = [:family, :colleagues, :friends, :strangers]
##
plotlyjs()
p = let
    p = let
            plot(legend=:outertopright, ticks=:native)
            plot!(Rt.date, Rt.mq, ribbon=(Rt.mq - Rt.lq, Rt.uq - Rt.mq), linewidth=2, label="R_t 50% quantiles")
            plot!(Rt_SSI.date, Rt_SSI.Rt, ribbon=(Rt_SSI.Rt - Rt_SSI.uncertainty_lower, Rt_SSI.uncertainty_upper - Rt_SSI.Rt), linewidth=2, label="R_t (SSI)", hover=Rt_SSI.Rt)
    end;

    ps = [p]
    for i in 1:4
        p = plot(legend=:outertopright)
        plot!(smoothed.date, smoothed[:, contact[i]], label="$(contact[i]) (smoothed)")
        y = zscore(survey[:, contact[i]])
        plot!(survey.date, y, label="$(contact[i]) (raw)")
        push!(ps, p)
    end;
    plot(ps..., layout=(5,1), size=(800, 1000), sharex=true, link=:x)
end
##
# p = let
pgfplotsx()
# rt = load( projectdir("reports/2021-04-27/Rt_cases=2020-06-01_chains=10_hospit=true_model=v2_observ=2020-12-16_seed=10000_sero=true_steps=2000_warmup=1000.csv") ) |> DataFrame
# rt = load( projectdir("reports/2021-04-28/Rt_cases=2020-06-01_chains=10_hospit=true_model=v2_observ=2021-02-07_seed=10000_sero=true_steps=2000_warmup=1000.csv") ) |> DataFrame
# rt = load(projectdir("out", "Rt.csv"))|>DataFrame
# rt = load(projectdir("data", "Rt_SSI.csv"))|>DataFrame
# rt = load(projectdir("reports/test_single_observation_AlternativeDelays/Rt_cases=2020-06-01_chains=10_hospit=true_model=cases_observ=2021-02-06_sero=true_steps=2000_warmup=1000.csv")) |> DataFrame
rt = load(projectdir("reports/test_single_observation_AlternativeDelays/Rt_cases=2020-06-01_chains=10_hospit=true_model=hospit_observ=2021-02-06_sero=true_steps=2000_warmup=1000.csv")) |> DataFrame


function plot_Rt_vs_contacts(c)
    p = plot(title="Rt (hospit) vs. $c")
    #plot!(Rt.date, Rt.mq, ribbon=(Rt.mq - Rt.lq, Rt.uq - Rt.mq), linewidth=2, label="R_t 50% quantiles")
    plot!(rt.date, rt.mq
        , ribbon=(rt.mq - rt.llq
        , rt.uuq - rt.mq)
        , lw=2
        , label=L"R_t"
        , hover=rt.mq
        , c=:black
        , ylabel = "reproduction rate"
        , legend=:topleft)
    y = zscore(survey[:, c])
    plot!(twinx()
        , survey.date .+ Day(0), y
        , label="Daily contacts"
        , lc=:darkorange
        , lw=4
        , ylabel="10-day lagged contacts (standardized)"
        , legend=:topright
        , xticks = :none)
    return p
end

p = let
    ps = []
    for c in [:family, :colleagues, :friends, :strangers]
        p = plot_Rt_vs_contacts(c)
        push!(ps, p)
    end
    plot(ps..., layout=(2,2), size=(1200, 1000))
end
##
savefig(p, projectdir("figures/" ,"Rt-hospit_vs_smoothed-contacts.png"))
##
function plot_Rt_vs_contacts(i)
    p = plot(title="$(contact[i])")
    #plot!(Rt.date, Rt.mq, ribbon=(Rt.mq - Rt.lq, Rt.uq - Rt.mq), linewidth=2, label="R_t 50% quantiles")
    plot!(Rt_SSI.date, Rt_SSI.Rt
        , ribbon=(Rt_SSI.Rt - Rt_SSI.uncertainty_lower
        , Rt_SSI.uncertainty_upper - Rt_SSI.Rt)
        , lw=2
        , label="R_t (SSI)"
        , hover=Rt_SSI.Rt
        , c=:black
        , ylabel = "reproduction rate"
        , legend=:topleft)
    # y = zscore(survey[:, contact[i]])
    y = smoothed[:, contact[i]]
    plot!(twinx()
        , smoothed.date .+ Day(10), y
        , label="smoothed daily contacts"
        , lc=:darkblue
        , ylabel="10-day lagged contacts (standardized)"
        , legend=:topright
        , xticks = :none)
    return p
end

p = let
    ps = []
    for i in 1:4
        p = plot_Rt_vs_contacts(i)
        push!(ps, p)
    end
    plot(ps..., layout=(2,2), size=(1200, 1000))
end

# savefig(p, projectdir("figures/" ,"decomposed_contacts_smoothed.png"))
##


ps = [p]
for i in 1:4
    p = plot(legend=:outertopright)
    plot!(smoothed.date, smoothed[:, contact[i]], label="$(contact[i]) (smoothed)")
    y = zscore(survey[:, contact[i]])
    plot!(survey.date, y, label="$(contact[i]) (raw)")
    push!(ps, p)
end;
plot(ps..., layout=(5,1), size=(800, 1000), sharex=true, link=:x)
# end
## =============================================================================
#                 fit survey data to Rt
# ==============================================================================
function getx(survey; suffix="", norm=true)
    xs = []
    cols = [
        Symbol("family"*suffix),
        Symbol("colleagues"*suffix),
        Symbol("friends"*suffix),
        Symbol("strangers"*suffix)
    ]
    for col in cols
        x = survey[:,col] .|> Float64
        norm && National.surveynorm!(x)
        push!(xs, x)
    end
    return hcat(xs...)
end

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

function getcov(survey, rt; suffix="")
    x = covariates(survey; suffix) .|> Float64
    t = survey.date
    y = rt.mq .|> Float64
    # lq = rt.llq
    # uq = rt.uuq
    # u = (y .- lq) / 1.65 #1.95
    # y = Rt.mq #Rt_SSI.Rt
    # lq = Rt.llq
    return (; t, x, y)
end

function gety_SSI(rt)
    lq = rt.uncertainty_lower
    uq = rt.uncertainty_upper
    y = rt.Rt
    u = max.(abs.(y .- lq), abs.(uq .- y)) / 1.65 #1.95
    t = rt.date
    return (; t, y, u, lq, uq)
end

function gety(rt)
    lq = rt.llq
    uq = rt.uuq
    y = rt.mq
    u = max.(abs.(y .- lq), abs.(uq .- y)) / 1.95
    t = rt.date
    return (; t, y, u, lq, uq)
end
## plot Rt vs Rt_SSI
p = let
    # rt = load( projectdir("reports/2021-04-28/Rt_cases=2020-06-01_chains=10_hospit=true_model=v2_observ=2021-02-07_seed=10000_sero=true_steps=2000_warmup=1000.csv") ) |> DataFrame
    rt2 = load( projectdir("reports/constant_iar/Rt_cases=2020-06-01_chains=10_hospit=true_model=const-iar_observ=2021-02-06_sero=true_steps=2000_warmup=1000.csv") ) |> DataFrame
    rt = load( projectdir("reports/const_iar_cases_only/Rt_cases=2020-06-01_chains=5_hospit=true_model=const-iar_observ=2021-02-06_sero=true_steps=5000_warmup=2000.csv")) |> DataFrame
    rt_SSI, rt = make_consistent(Rt_SSI, rt)
    plot(legend=:outertopright)
    plot!(rt.date, rt.mq, ribbon=(rt.mq - rt.lq, rt.uq - rt.mq), linewidth=2, label="R_t")
    plot!(rt_SSI.date, rt_SSI.Rt, ribbon=(rt_SSI.Rt - rt_SSI.uncertainty_lower, rt_SSI.uncertainty_upper - rt_SSI.Rt), linewidth=2, label="R_t (SSI)", hover=Rt_SSI.Rt)
    plot!(rt2.date, rt2.mq, ribbon=(rt2.mq - rt2.lq, rt2.uq - rt2.mq), linewidth=2, label="R2_t")
end
##
# Plots.scalefontsizes(1.3)
# pgfplotsx()
plotlyjs()
p = let
    rt = load( projectdir("data/Rt.csv") ) |> DataFrame
    y1 = rt[:,"Rt-observ-2021-02-07"]
    y2 = rt[:,"Rt-SSI-shifted"]
    t  = rt.date
    p = plot(legend=:bottomright, ylabel="R_t", xticks=:native)
    plot!(t, y1, lw=2, label="without predictors")
    plot!(t, y2, lw=2, label="SSI")
    # xlims!((t[1],t[end]))
end
savefig(projectdir("figures/compare_Rt_vs_Rt-SSI.html"))
##
survey_fname = "/home/and/data/covidsurvey/contacts_0425.csv"

suffix=""
d = National.readsurvey(survey_fname)
d = National.readsurveys()
fname = normpath( homedir(), "data/covidsurvey/smoothed_contacts_0425_2m.csv" )
fname = normpath( homedir(), "data/covidsurvey/smoothed_contacts.csv" )
# fname = projectdir("data", "Rt_SSI.csv")
# fname = projectdir("reports/test_single_observation_AlternativeDelays/Rt_cases=2020-06-01_chains=10_hospit=true_model=cases_observ=2021-02-06_sero=true_steps=2000_warmup=1000.csv")
d = CSV.File(fname)|>DataFrame
d.date += Day(12)


rt = load( projectdir("reports/2021-04-27/Rt_cases=2020-06-01_chains=10_hospit=true_model=v2_observ=2020-12-16_seed=10000_sero=true_steps=2000_warmup=1000.csv") ) |> DataFrame
rt = load( projectdir("reports/2021-04-28/Rt_cases=2020-06-01_chains=10_hospit=true_model=v2_observ=2021-02-07_seed=10000_sero=true_steps=2000_warmup=1000.csv") ) |> DataFrame
rt = load(projectdir("out", "Rt.csv"))|>DataFrame
rt = deepcopy(Rt_SSI)
rt = load(projectdir("data", "Rt_SSI.csv"))|>DataFrame
rt = load(
    projectdir("reports/test_single_observation_AlternativeDelays/Rt_cases=2020-06-01_chains=10_hospit=true_model=cases_observ=2021-02-06_sero=true_steps=2000_warmup=1000.csv")
) |> DataFrame
d, rt = make_consistent(d, rt, enddate="2020-12-16")
# x = d.mq
# National.surveynorm!(x)
# x = x[:,:]
x = getx(d; suffix)
t, y, u, lq, uq = gety(rt)
# t, y, u, lq, uq = gety_SSI(rt)

# survey = National.readsurveys()
# survey = National.readsurveys_raw()
# survey = National.readsurvey(survey_fname)

# masks_start = findfirst(==(Date("2020-10-29")), t)
# @model function linear_regression_masks(y, x, u, i, ::Type{TV} = Vector{Float64}) where TV
#     n, m = size(x)
#     α_with_mask  ~ filldist(Beta(1, 50), m)
#     α_without_mask  ~ filldist(Beta(1, 50), m)
#     σ2 ~ InverseGamma(2.,.1)
#     ŷ  = TV(undef, n)
#     ŷ[1:i]  .= log(y[1]) .+ x[1:i,:] * α_without_mask
#     ŷ[i+1:end] .= log(y[1]) .+ x[i+1:end,:] * α_with_mask
#     uy ~ MvNormal(y, u)
#     uy ~ MvNormal(exp.(ŷ), σ2)
#     return exp.(ŷ)
# end

@model function linear_regression(y, x, u; link=exp, invlink=log)
    n, m = size(x)
    # α ~ filldist(Beta(1, 5), m)
    α ~ filldist(InverseGamma(2.,1.), m)
    σ2 ~ InverseGamma(2.,.1)
    ŷ = invlink(y[1]) .+ x * α
    # uy ~ MvNormal(y, u)
    # uy ~ MvNormal(link.(ŷ), σ2)
    y ~ MvNormal(link.(ŷ), σ2)
    return link.(ŷ)
end


# m = linear_regression_masks(y, x, u, i; link=KLogistic(3.), invlink=KLogit(3.))
m = linear_regression(y, x, u; link=KLogistic(4.), invlink=KLogit(4.))
c = sample(m, NUTS(200, 0.6), 1200)
c = c[200+1:end,:,:]
ŷs = generated_quantities(m,c)
ŷs = hcat(ŷs...)
##


##
# pgfplotsx()
plotlyjs()
# plot(c)
p = let
    ŷ = mean(ŷs, dims=2)
    e = sum(abs2, y-ŷ) / length(y)
    p = plot(legend=:outertop)
    plot!(t, y
        , ribbon=(y - lq, uq - y)
        # , ribbon=(y - Rt.llq, Rt.uuq - y)
        , lw=2
        , label="R_t (SSI)"
        , c=:black
        , ylabel = "reproduction rate"
        # , legend=:topright
        , α=.4)
    CovidSurvey.plot_confidence_timeseries!(p, t, ŷs, label = "fit (MSE = $e)")
    # vline!([Date("2020-10-29")], label="masks introduced")
    p
end
# plot(c)
# savefig(p, projectdir("figures/", "linearregression_smoothed_survey_on_Rt.png") )
# savefig(p, projectdir("figures/", "linearregression_smoothed_survey_on_Rt.html") )
#---------------------------------------------------------------------------
# make new prior for Rt
# rt = deepcopy(Rt_SSI)
rt = load( projectdir("reports/2021-04-28/Rt_cases=2020-06-01_chains=10_hospit=true_model=v2_observ=2021-02-07_seed=10000_sero=true_steps=2000_warmup=1000.csv") ) |> DataFrame
d = National.readsurveys()
d, rt = make_consistent(d, rt)

# t, y, u, lq, uq = gety_SSI(rt)
t, y, u, lq, uq = gety(rt)

# prior = [Normal(μ, σ) for (μ, σ) in zip(y,u)]
save(projectdir("reports/test_contacts/RtPrior.csv"), DataFrame((; t, y, u)))

# prior = [fit(Normal{Float64}, ŷ) for ŷ in eachrow(ŷs)]
# plot(mean.(prior), ribbon=std.(prior))
# plot!(y, ribbon=(y - lq, uq - y), linewidth=2, label="R_t")
# plot!(mean(ŷs, dims=2), ribbon=std(ŷs, dims=2))
#---------------------------------------------------------------------------
# plot posterior effect size distribution vs. prior
using Plots.PlotMeasures
pgfplotsx()
p = let
    # [:family, :colleagues, :friends, :strangers]
    xs = c["α[1]"]|>vec
    p1 = histogram(xs, normalize=true, title="contacts to family", label=nothing, fc=:gray, lw=.1, xlabel="effect size", bottom_margin=3mm, right_margin=0mm);
    # plot!(Beta(1,1), xlims=(0,0.1), lw=2, label="prior", legend=:topright);
    vline!([mean(xs)], lw=2, label="mean = $(round(mean(xs), sigdigits=2))", c=:black, ls=:dash)
    xs = c["α[2]"]|>vec
    p2 = histogram(xs, normalize=true, title="contacts to colleagues", label=nothing, fc=:gray, lw=.1, xlabel="effect size", bottom_margin=3mm, right_margin=0mm);
    # plot!(Beta(1,1), xlims=(0,0.1), lw=2, label="prior", legend=:topright);
    vline!([mean(xs)], lw=2, label="mean = $(round(mean(xs), sigdigits=2))", c=:black, ls=:dash)
    xs = c["α[3]"]|>vec
    p3 = histogram(xs, normalize=true, title="contacts to friends", label=nothing, fc=:gray, lw=.1, xlabel="effect size", top_margin=3mm, right_margin=0mm);
    # plot!(Beta(1,1), xlims=(0,0.1), lw=2, label="prior", legend=:topright);
    vline!([mean(xs)], lw=2, label="mean = $(round(mean(xs), sigdigits=2))", c=:black, ls=:dash)
    xs = c["α[4]"]|>vec
    p4 = histogram(xs, normalize=true, title="contacts to strangers", label=nothing, fc=:gray, lw=.1, xlabel="effect size", top_margin=3mm, right_margin=0mm);
    # plot!(Beta(1,1), xlims=(0,0.1), lw=2, label="prior", legend=:topright);
    vline!([mean(xs)], lw=2, label="mean = $(round(mean(xs), sigdigits=2))", c=:black, ls=:dash)
    plot(p1,p2,p3,p4, layout=(2,2), legend=:topright, )
end
#----------------------------------------------------------------------------
using Plots.PlotMeasures
pgfplotsx()
p1 = let
    # [:family, :colleagues, :friends, :strangers]
    xs = c["α_without_mask[1]"]|>vec
    p1 = histogram(xs, normalize=true, title="contacts to family", label=nothing, fc=:gray, lw=.1, xlabel="effect size", bottom_margin=3mm, right_margin=0mm);
    # plot!(Beta(1,1), xlims=(0,0.1), lw=2, label="prior", legend=:topright);
    vline!([mean(xs)], lw=2, label="mean = $(round(mean(xs), sigdigits=2))", c=:black, ls=:dash)
    xs = c["α_without_mask[2]"]|>vec
    p2 = histogram(xs, normalize=true, title="contacts to colleagues", label=nothing, fc=:gray, lw=.1, xlabel="effect size", bottom_margin=3mm, right_margin=0mm);
    # plot!(Beta(1,1), xlims=(0,0.1), lw=2, label="prior", legend=:topright);
    vline!([mean(xs)], lw=2, label="mean = $(round(mean(xs), sigdigits=2))", c=:black, ls=:dash)
    xs = c["α_without_mask[3]"]|>vec
    p3 = histogram(xs, normalize=true, title="contacts to friends", label=nothing, fc=:gray, lw=.1, xlabel="effect size", top_margin=3mm, right_margin=0mm);
    # plot!(Beta(1,1), xlims=(0,0.1), lw=2, label="prior", legend=:topright);
    vline!([mean(xs)], lw=2, label="mean = $(round(mean(xs), sigdigits=2))", c=:black, ls=:dash)
    xs = c["α_without_mask[4]"]|>vec
    p4 = histogram(xs, normalize=true, title="contacts to strangers", label=nothing, fc=:gray, lw=.1, xlabel="effect size", top_margin=3mm, right_margin=0mm);
    # plot!(Beta(1,1), xlims=(0,0.1), lw=2, label="prior", legend=:topright);
    vline!([mean(xs)], lw=2, label="mean = $(round(mean(xs), sigdigits=2))", c=:black, ls=:dash)
    plot(p1,p2,p3,p4, layout=(2,2), legend=:topright, )
end

p2 = let
    # [:family, :colleagues, :friends, :strangers]
    xs = c["α_with_mask[1]"]|>vec
    p1 = histogram(xs, normalize=true, title="contacts to family", label=nothing, fc=:gray, lw=.1, xlabel="effect size", bottom_margin=3mm, right_margin=0mm);
    # plot!(Beta(1,1), xlims=(0,0.1), lw=2, label="prior", legend=:topright);
    vline!([mean(xs)], lw=2, label="mean = $(round(mean(xs), sigdigits=2))", c=:black, ls=:dash)
    xs = c["α_with_mask[2]"]|>vec
    p2 = histogram(xs, normalize=true, title="contacts to colleagues", label=nothing, fc=:gray, lw=.1, xlabel="effect size", bottom_margin=3mm, right_margin=0mm);
    # plot!(Beta(1,1), xlims=(0,0.1), lw=2, label="prior", legend=:topright);
    vline!([mean(xs)], lw=2, label="mean = $(round(mean(xs), sigdigits=2))", c=:black, ls=:dash)
    xs = c["α_with_mask[3]"]|>vec
    p3 = histogram(xs, normalize=true, title="contacts to friends", label=nothing, fc=:gray, lw=.1, xlabel="effect size", top_margin=3mm, right_margin=0mm);
    # plot!(Beta(1,1), xlims=(0,0.1), lw=2, label="prior", legend=:topright);
    vline!([mean(xs)], lw=2, label="mean = $(round(mean(xs), sigdigits=2))", c=:black, ls=:dash)
    xs = c["α_with_mask[4]"]|>vec
    p4 = histogram(xs, normalize=true, title="contacts to strangers", label=nothing, fc=:gray, lw=.1, xlabel="effect size", top_margin=3mm, right_margin=0mm);
    # plot!(Beta(1,1), xlims=(0,0.1), lw=2, label="prior", legend=:topright);
    vline!([mean(xs)], lw=2, label="mean = $(round(mean(xs), sigdigits=2))", c=:black, ls=:dash)
    plot(p1,p2,p3,p4, layout=(2,2), legend=:topright, )
end

plot(p1, p2, layout=(2,1), size=(1000, 1500))
# savefig(p, projectdir("figures/", "linearregression_smoothed_survey_on_Rt_effect_analysis.png") )
# savefig(p, projectdir("figures/", "linearregression_smoothed_survey_on_Rt_effect_analysis.html") )
##
survey_fname = "/home/and/data/covidsurvey/contacts_0425.csv"
survey = National.readsurvey(survey_fname)
#--------------------------------------------------------------------------
# variable day
@model function linear_regression(y, x, u; link=exp, invlink=log)
    n, m = size(x)
    # α ~ filldist(Beta(1, 5), m)
    α ~ filldist(InverseGamma(2.,1.), m)
    σ2 ~ InverseGamma(2.,.1)
    ŷ = invlink(y[1]) .+ x * α
    # uy ~ MvNormal(y, u)
    # uy ~ MvNormal(link.(ŷ), σ2)
    y ~ MvNormal(link.(ŷ), σ2)
    return link.(ŷ)
end

fname = normpath( homedir(), "data/covidsurvey/smoothed_contacts.csv" )
# fname = projectdir("data", "Rt_SSI.csv")
# fname = projectdir("reports/test_single_observation_AlternativeDelays/Rt_cases=2020-06-01_chains=10_hospit=true_model=cases_observ=2021-02-06_sero=true_steps=2000_warmup=1000.csv")
d = CSV.File(fname)|>DataFrame

function getdata(rt, d, lag)
    d  = deepcopy(d)
    rt = deepcopy(rt)
    d.date += Day(lag)
    d, rt = make_consistent(d, rt, enddate="2021-01-15")
    x = getx(d)
    t, y, u, lq, uq = gety(rt)
    # t, y, u, lq, uq = gety_SSI(rt)
    return t, x, y, u
end

# rt = load( projectdir("reports/2021-04-28/Rt_cases=2020-06-01_chains=10_hospit=true_model=v2_observ=2021-02-07_seed=10000_sero=true_steps=2000_warmup=1000.csv") ) |> DataFrame
# rt = load(
#     projectdir("reports/test_single_observation_AlternativeDelays/Rt_cases=2020-06-01_chains=10_hospit=true_model=cases_observ=2021-02-06_sero=true_steps=2000_warmup=1000.csv")
# ) |> DataFrame
rt = load(
    projectdir("data/Rt_SSI.csv")
) |> DataFrame
function delay_analysis(rt, d; lags=0:1:25)
    es = []
    ŷs = []
    ts = []
    ps = []
    @progress for l in lags
        t, x, y, u = getdata(rt, d, l)
        m = linear_regression(y, x, u; link=KLogistic(3.), invlink=KLogit(3.))
        c = sample(m, NUTS(200, 0.6), 1200)
        c = c[200+1:end,:,:]
        gq = generated_quantities(m,c)
        ŷ = mean(gq)
        e = sum(abs2, y-ŷ)/length(y)
        @show e
        p = plot(t, ŷ, title="$l: $e");
        plot!(rt.date, y);
        push!(ps, p)
        push!(ts, t)
        push!(es, e)
        push!(ŷs, ŷ)
    end
    (; ps, ts, es, ŷs)
end
lags = 0:1:20
res = delay_analysis(rt, d; lags)
# res_SSI = deepcopy(res)
##
pgfplotsx()
p = let
    p1 = plot(lags, res.es, xlabel="lag [days]", ylabel="mean squared error", label="mse after shift \\& fit", title="find time lag that minimizes error") #best lag = 11
    i = 12#argmin(res.es)
    vline!([i], label="best fit")
    p2 = plot(rt.date, rt.mq, label=L"R_t")
    plot!(res.ts[i], res.ŷs[i], label="best fit", ylabel=L"R_t")
    plot(p1, p2, layout=(2,1), size=(800, 500))
end
savefig(projectdir("figures/Rt_vs_predictors_delayed.png"))
## cross correlation analysis
let
    lags = 0:100
    rt = load( projectdir("reports/2021-04-28/Rt_cases=2020-06-01_chains=10_hospit=true_model=v2_observ=2021-02-07_seed=10000_sero=true_steps=2000_warmup=1000.csv") ) |> DataFrame
    df1, df2 = make_consistent(smoothed, rt)
    y1 = df2.mq
    p1 = plot()
    p2 = plot()
    for n in [:family, :friends, :strangers, :colleagues]
        y2 = df1[:,n]
        cc = crosscor(y1, y2, lags)
        err = [sum(abs2, y1[1:end-t+1] - y2[t:end])/length(y2[t:end]) for t in 1:101]
        plot!(p1, lags, cc, label=String(n))
        plot!(p2, lags, err, label=String(n))
    end
    plot(p1, p2, layout=(2,1))
end
##
let
    lags = 0:10
    # rt = load( projectdir("reports/2021-04-28/Rt_cases=2020-06-01_chains=10_hospit=true_model=v2_observ=2021-02-07_seed=10000_sero=true_steps=2000_warmup=1000.csv") ) |> DataFrame
    rt = load( projectdir("reports/test_single_observation_AlternativeDelays/Rt_cases=2020-06-01_chains=10_hospit=true_model=cases_observ=2021-02-06_sero=true_steps=2000_warmup=1000.csv") ) |> DataFrame
    rt = rt[rt.date .<= Date("2020-12-16"),:]
    Rt_SSI = National.readsurvey("/home/and/data/covidsurvey/contacts.csv")
    df1, df2 = make_consistent(Rt_SSI, rt)
    y1 = df1.Rt
    y2 = df2.mq
    cc = crosscor(y1, y2, lags)
    err = [sum(abs2, y1[1:end-t+1] - y2[t:end])/length(y2[t:end]) for t in lags.+1]
    p1 = plot(lags, cc)
    p2 = plot(lags, err)
    plot(p1, p2, layout=(2,1))
end

##
plotlyjs()
pgfplotsx()
scalefontsizes(.8)
p = let
    p = let
            plot(legend=:outertopright)
            plot!(t, y, ribbon=(y - lq, uq - y), linewidth=2, title=L"R_t", label=nothing)
            # plot!(Rt_SSI.date, Rt_SSI.Rt, ribbon=(Rt_SSI.Rt - Rt_SSI.uncertainty_lower, Rt_SSI.uncertainty_upper - Rt_SSI.Rt), linewidth=2, label="R_t (SSI)", hover=Rt_SSI.Rt)
    end

    ps = [p]
    for (i,n) in enumerate(["family", "colleagues", "friends", "strangers"])
        p = plot(legend=:outertopright)
        plot!(t, x[:, i], title="$n", label=nothing)
        push!(ps, p)
    end;
    plot(ps..., layout=(5,1), size=(800, 1000), sharex=true)
end
savefig("/home/and/dev/CovidSurvey/figures/zero_contacts.png")
