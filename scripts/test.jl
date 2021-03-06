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
using Base.Threads
using StatsFuns
using ReverseDiff
using Memoization
Turing.setrdcache(true)
setadbackend(:reversediff)
## ============================================================================
nthreads()
name2model = Dict(
    "v3" => National.model_v3,
    "v2"        => National.model_v2,
    #"gp"        => National.model_gp,
    # "contacts" => National.model_contacts,
    "contacts2" => National.model_contacts_v2,
    "contacts3" => National.model_contacts_v3,
    "contacts4" => National.model_contacts_v4,
    "fitted" => National.model_contacts_fitted,
    "const-iar" => National.model_constant_iar,
    "const-iar-c" => National.model_constant_iar_contacts,
    "contacts-shifted" => National.model_contacts_shifted,
    "deaths" => National.model_deaths,
    "cases" => National.model_cases,
    "hospit" => National.model_hospit,
    "parametric-cases" => National.model_parametric_cases
)
#-----------------------------------------------------------------------------
# load data
ps = (
    warmup = 10,
    steps  = 20,
    seed   = 20000,
    observ = "2021-01-13",#"2021-02-06",#"2021-03-25"
    cases  = "2020-06-01",
    model  = "hospit",
    preds = nothing,
    hospit = true,
    sero = true,
)
fname = savename("chains", ps, "")
Random.seed!(ps.seed);
@info ps

data = National.load_data(;
    observations_end  = ps.observ,
    predictors        = ps.preds |> CovidSurvey.parse_predictors,
    rw_step           = 1,
    epidemic_start    = 30,
    num_impute        = 6,
    deathmodel        = National.DeathInit(obs_stop="$( Date(ps.cases) + Day(1))"),
    casemodel         = National.CaseInit(obs_start=ps.cases),
    link              = KLogistic(3.),
    invlink           = KLogit(3.),
    lockdown          = "2020-03-18",
    covariates_kwargs = Dict(
        :fname => projectdir("data","smoothed_contact_rates.csv"),
        :shift => -1,
        :startdate => "2020-11-10",
        :enddate => nothing))
turing_data = data.turing_data;

# num_obs = turing_data.cases|>length
# lockdown = turing_data.lockdown
# covariates_start = turing_data.covariates_start
# num_Rt_steps   = covariates_start-lockdown-1
# num_obs
# lockdown + num_Rt_steps + size(covariates, 1)
# covariates = turing_data.covariates
# size(covariates, 1)
#
# length(turing_data.cases)
#
# fname = normpath( homedir(), "data/covidsurvey/smoothed_contacts.csv" )
# survey = load( fname )|>DataFrame
# National.readcovariates(; covariates_kwargs...
# )

# findfirst(==(Date("2021-01-01")),data.dates) - findfirst(==(Date("2020-05-15")),data.dates) + 1
# findfirst(==(Date("2021-01-01")),data.dates)
# size(survey, 1)
#
#
# data.dates
#
# covariates[140,1]
# size(turing_data.covariates,1) - turing_data.covariates_start
#----------------------------------------------------------------------------
# sample model
model = name2model[ps.model]
m = model(turing_data, false)
Turing.emptyrdcache()
m()
@time chain = sample(m, NUTS(ps.warmup, 0.95), ps.steps + ps.warmup; progress=true)

@time chain = sample(m, NUTS(ps.warmup, 0.95), MCMCThreads(), ps.steps + ps.warmup, 3)
# using MCMCChains
chain = chain[ps.warmup+1:end,:,:]
# chain = chain[:,:,[2,4]]

@info "Saving at: $(projectdir("out", fname))"
safesave(projectdir("out/tmp", fname*".jls"), chain)
##
diagnostics = gelmandiag(chain)
fname_diagnostics = projectdir("out/tmp", fname*"_GELMANDIAG.csv")
safesave(fname_diagnostics, diagnostics)
##
# data = ImperialUSAcases.Data(Dict(),turing_data,Dict("DK"=>dk.date),turing_data.deaths,"DK")
# m = model(turing_data..., false)
# chain = sample(m, Prior(), 2000)
m_pred = model(turing_data, true)
gq = Turing.generated_quantities(m_pred, chain)
generated_posterior = vectup2tupvec( reshape(gq, length(gq)) );
#---------------------------------------------------------------------------
# plot results
plotlyjs()
p = National.plot(data, generated_posterior);
savefig(p, "/home/and/tmp/figures/"*fname*".html")
# savefig(p, projectdir("figures/tmp", fname*".png") )
run(`firefox $("/home/and/tmp/figures/"*fname*".html")`, wait=false)
##
plotlyjs()
n = filter( x->!occursin(r"\[", x), String.(names(chain)))
p = plot(chain[n]);
savefig(p, "/home/and/tmp/figures/"*fname*"_CHAINS.html" )
run(`firefox  $("/home/and/tmp/figures/"*fname*"_CHAINS.html")`, wait=false)
##
using MCMCChains
plotlyjs()
n = filter( x->!occursin(r"\[", x), String.(names(chain)))
p = meanplot(chain[n]);
savefig(p, projectdir("/home/and/tmp/figures/", fname*"_MEANPLOT.html") )
run(`firefox $(projectdir("/home/and/tmp/figures/", fname*"_MEANPLOT.html"))`, wait=false)


##
p = let
    changepoint  = argmin( abs.(mean(Rt)[end-50:end] .- 1) )
    localmin  = argmin(mean(Rt)[end-50:end])
    tickdates = data.dates[1]:Month(2):data.dates[end]
    rt        = hcat(Rt...)
    t1        = Date(2020, 12, 24)
    t1_str    = t1 |> Symbol |> String
    t2        = data.dates[end-50:end][ localmin ]
    t2_str    = t2 |> Symbol |> String
    t3        = data.dates[end-50:end][ changepoint ]
    t3_str    = t3 |> Symbol |> String

    # p1 = plot()
    tickdates = data.dates[1]:Month(3):data.dates[end]
    p1 = plot(title=L"estimated $R_t$")
    rt = hcat(Rt...)
    plot_confidence_timeseries!(p1, data.dates, rt)
    vline!(p1, [t1], ls=:solid, lc=:black, label=t1_str)
    vline!(p1, [t2], ls=:dash, lc=:black, label=t2_str)
    vline!(p1, [t3], ls=:dot, lc=:black, label=t3_str)

    p2 = plot(title="Mobility: change in percent from baseline")
    y = dk.workplaces_percent_change_from_baseline |> rolling
    t = data.dates[4:end-3]
    plot!(p2, t, y, label="work") #data.turing_data.covariates[:,1]
    # plot!(p2, data.dates, -100*data.turing_data.covariates[:,1], label="non-residential") #data.turing_data.covariates[:,1]
    y = dk.residential_percent_change_from_baseline |> rolling
    plot!(p2, t, y, label="residential")

    p = plot(p1, p2, layout=(2,1), sharex=true, xticks=tickdates, legend = :outertopright)
    p
end
savefig(p, "FLA20a/Covid19/figures/reproduction_rate_vs_mobility.png")

rolling(xs) = [maximum(xs[i:i+6]) for i in 1:length(xs)-6]


data_nature = ImperialReport13.load_data(datadir("imperial-report13", "processed.rds"))
data_usa = ImperialUSAcases.load_data(datadir("imperial-usa-cases", "processed.rds"))
dk = begin
    dk = load(datadir("imperial-usa-cases", "DK.rds"))
    i = findfirst(>=(10), skipmissing(dk.deaths))
    startdate = dk.date[i] - Day(30)
    startidx = findfirst(==(startdate), dk.date)
    dk[startidx:end,:]
end
survey = CSV.File(projectdir("data", "survey", "predictors.csv"))|>DataFrame
survey.date = survey.date .|> parsetime


##
figname = savename("chains_seroprev_Nobs=$(length(turing_data.cases))", ps, "")
ImperialDKcases.plot_results(data, expected_daily_cases, expected_daily_deaths, predicted_daily_cases, Rt)
figpath = "/home/and/code/ImperialCollegeReplications/FLA20a/Covid19/figures/$(figname)"
savefig( figpath*".png" )
savefig( figpath*".html" )
run(`firefox /home/and/code/ImperialCollegeReplications/FLA20a/Covid19/figures/fig_Nobs=303_seed=1_steps=500_warmup=500.html`)# $(figpath*".html")`)

#----------------------------------------------------------------
# prior checks
# testing random walk
using StatsFuns
# invlink(x) = 10 / ( 1 + exp( -x ) )
# invlink(x) = exp( -x )


myinvlink(x) = 5 / ( 1 + exp( -x ) )

histogram(rand(Normal(.7, .3), 1000) .|> myinvlink)

@model function testrw(week_index, num_weeks, ::Type{TV} = Vector{Float64}) where {TV}
    # ??         ~ truncated(Normal(0,0.5), 0, Inf)
    # ??         ~ truncated(Normal(3.28, ??), 0, Inf)
    ??         ~ Normal(.7, .3)
    ??1        ~ truncated(Normal(0.8,0.05), 0, 1)
    ??2        ~ truncated(Normal(0.1,0.05), 0, 1) #-??1
    ??w        ~ truncated(Normal(0,1), 0, Inf)
    #WARNING ???x can fail during warm up as we dont guarantee x>0
    ??w_star = ??w * sqrt( max(1-??1^2-??2^2-2??1^2*??2/(1-??2), 1e-5) )#min(1-??1^2-??2^2-2??1^2*??2/(1-??2), 0) ) #sqrt() (1+??2)*(1-??1-??2)*(1+??1-??2)/(1-??2)
    weekly_effect = TV(undef, num_weeks)
    weekly_effect[1] ~ Normal(0,0.01)
    weekly_effect[2] ~ Normal(0,??w_star)
    for w in 3:num_weeks
        weekly_effect[w] ~ Normal(??1*weekly_effect[w-1]+??2*weekly_effect[w-2], ??w_star)
    end
    # invlink.( weekly_effect[week_index] )
    # 3.28 * 2 * logistic.( - weekly_effect[week_index])
    myinvlink.( ?? .+ weekly_effect[week_index])
end

num_weeks  = 52
week_index = vcat( [ones(Int64, 7)*i for i in 1:num_weeks]... )
num_days   = length(week_index)
m = testrw(week_index, num_weeks)
# m = testrw(num_days)
# plot(c)
c = sample(m, Prior(), 1000)
rt = Turing.generated_quantities(m, c)|>vec
y = convert(Matrix{Float64}, hcat(rt...))
p = plot_confidence_timeseries(y)
# plot!(rt[1:3])

## Test GP

using KernelFunctions
using PDMats

invlink(x) = 5 / ( 1 + exp( -x ) )
x = 1:365
k = KernelFunctions.transform(Matern32Kernel(), 1/7)
K = kernelpdmat(k, x[:,1:1], obsdim=1)
y = invlink.(rand(MvNormal(K)))

plot(x, y)
X   = (1:10:365)[:,1:1]
Y   = y[1:10:end] + randn(length(X))*0.01
scatter!(X,Y)

# a ~ LogNormal(-1, .5)
# l ~ LogNormal(5,.5)
# k = KernelFunctions.transform(a*Matern32Kernel(), 1/l)
# K = kernelmatrix(k, day;obsdim=1)
# ?? ~ MvNormal(K)

@model function testgp(X,Y)
    ?? ~ LogNormal(-2, 2)
    l ~ LogNormal(5,.5)
    k = KernelFunctions.transform(Matern32Kernel(), 1/l)
    K = kernelpdmat(k,X;obsdim=1)
    ?? ~ MvNormal(K)
    y = logistic.(??)
    Y ~ MvNormal(y, ??)
    y
end
m = testgp(X,Y)
c = sample(m, Prior(), 1000)
histogram(c["??"])
??s = Turing.generated_quantities(m,c)
plot(x, y)
plot!(X, ??s[2], lc=:black, ??=0.5, lw=.5)

c = sample(m, NUTS(200, .65), 100)
scatter!(X,Y)
plot!(X, ??s[1:100], lc=:black, ??=0.2, lw=.2)
## test auto correlated noise in Rt



histogram(invlink.(-rand(LogNormal(-1, .5), 1000)*1) * 2 )

@model function testgp(X)
    a ~ LogNormal(-1, .5)
    l ~ LogNormal(5,.5)
    k = KernelFunctions.transform(a*Matern32Kernel(), 1/l)
    K = kernelpdmat(k,X;obsdim=1)
    ?? ~ MvNormal(K)
    logistic.(-??)*2
end
m = testgp(x[:,1:1])
c = sample(m, Prior(), 1000)
??s = Turing.generated_quantities(m,c)
plot(x, ??s[1:10], ??=0.5)
## test NegBinomial observation model

@model function testobs()
    ?? ~ truncated(Normal(0, 5), 0, Inf)
    x ~ NegativeBinomial2(10, ??)
end

using StatsPlots
n = 1000
ys = []
for x in 10 .^(1:2)
    ??s = rand(truncated(Normal(0, 5), 0, Inf), n)
    ??s = rand(LogNormal()) y = [rand(NegativeBinomial2(x, 5)) for ?? in ??s]
    push!(ys, y)
end
# plot(ylim=(0.1, 10^7), yaxis=:log)
boxplot(ys; outliers=false, leg=nothing)


histogram(rand(Gamma(2,1), 1000))

##
using StatsPlots
plot(turing_data.??)
function gamma(??, cv)
    ?? = 1/cv^2
    ?? = ??/??
    Gamma(??, ??)
end
d  = gamma(6.5, 0.62)
d1 = gamma(5.1, 0.86)
d2 = gamma(17.8, 0.45)

g  = turing_data.serial_intervals
plot(d)
plot!(g / sum(g))

g  = turing_data.??2
plot(d1)
plot!(g / sum(g))

g = turing_data.??
histogram(rand(d1, 10_000) .+ rand(d2, 10_000), normalize=:pdf)
plot!(g / sum(g))
#--------------------------------------------------------------
# Import Turing and Distributions.
# Import Turing and Distributions.
using Turing, Distributions, Bijectors
using Random

struct RandomWalk{T} <: AbstractMvNormal  where T
    n::Int
    ??1::T#Float64rw
    ??2::T
    ??::T
end

Distributions.length(d::RandomWalk) = d.n

function Base.rand(::AbstractRNG, d::RandomWalk{T} where T)
    #??w_star = d.??w * sqrt( max(1-d.??1^2-d.??2^2-2d.??1^2*d.??2/(1-d.??2), 1e-5) )#min(1-??1^2-??2^2-2??1^2*??2/(1-??2), 0) ) #sqrt() (1+??2)*(1-??1-??2)*(1+??1-??2)/(1-??2)
    rw = zeros(d.n)
    rw[1] = rand(Normal(0, .01))
    rw[2] = rand(Normal(0, d.??))
    for n in 3:d.n
        rw[n] = rand( Normal( d.??1*rw[n-1]+d.??2*rw[n-2], d.?? ) )
    end
    rw
end

function Distributions.logpdf(d::RandomWalk, x::AbstractVector)
    l  = logpdf(Normal(0, .01), x[1])
    l += logpdf(Normal(0, d.??), x[2])
	for i in 3:d.n
        l += logpdf( Normal( d.??1*x[i-1]+d.??2*x[i-2], d.?? ), x[i] )
    end
    l
end

rw = let
    ??1 = .8
    ??2 = .1
    ??w = .2
    ?? = ??w * sqrt( max(1-??1^2-??2^2-2??1^2*??2/(1-??2), 1e-5) )
    RandomWalk(100, ??1, ??2, ??)
end
x = rand(rw) .+ randn(100)*0.01
logpdf(rw, x)

@model function testrw(X)
    ??1        ~ truncated(Normal(0.8,0.05), 0, 1)
    ??2        ~ truncated(Normal(0.1,0.05), 0, 1) #-??1
    ??w        ~ truncated(Normal(0,.2), 0, Inf)
    ?? = ??w * sqrt( max(1-??1^2-??2^2-2??1^2*??2/(1-??2), 1e-5) )
    rw ~ RandomWalk(100, ??1, ??2, ??)
    X  ~ MvNormal(rw, 0.01)
    return rw
end

m = testrw(x)
Turing.setadbackend(:forwarddiff)
@time c =sample(m, NUTS(), 1000)

rws = Turing.generated_quantities(m, c)
let
    p = scatter(x)
    plot!(rws[1:10], lw=.5, lc=:black, ??=.5)
end

## evalueate Rt prior
using StatsFuns

@model function testpred(covariates, num_weeks, week_index, ::Type{TV} = Vector{Float64}) where {TV}
    last_time_step = 300
    num_missing    = findfirst(x->!ismissing(x), covariates[:,1])-1

    ??1        ~ truncated(Normal(0.8,0.05), 0, 1)
    ??2        ~ truncated(Normal(0.1,0.05), 0, 1) #-??1
    ??w        ~ truncated(Normal(0,.2), 0, Inf)
    ?? = ??w * sqrt( max(1-??1^2-??2^2-2??1^2*??2/(1-??2), 1e-5) )
    weekly_effect ~ RandomWalk(num_weeks, ??1, ??2, ??)

    ?? ~ filldist(Normal(0,0.5), 1)
    ?? = 3.28
    Rt                     = TV(undef, last_time_step)
    Rt[1:num_missing]     .= ?? * 2 * logistic.( - weekly_effect[week_index[1:num_missing]]);
    Rt[num_missing+1:end] .= ?? * 2 * logistic.( - weekly_effect[week_index[num_missing+1:last_time_step]]
                                                - covariates[num_missing+1:last_time_step, :] * ??)
    Rt
end

m = testpred(turing_data.covariates, turing_data.num_weeks, turing_data.week_index)
x = [m() for i in 1:100]
plot(x, lw=.5, lc=:black, ??=.5, legend=false)


c = turing_data.covariates
nmis    = findfirst(x->!ismissing(x), c[:,1])-1
c = c[nmis+1:end,:]
n = size(c,1)
x = [2*logistic.(rand(truncated(Normal(0,0.5), 0, Inf), n) .* c) for i in 1:100]
plot(x, lw=.5, lc=:black, ??=.5, legend=false)
plot!(c, c=:red)


#----------------------------------------------------------------------------
dk = load(datadir("imperial-usa-cases", "DK.rds"))
X = data.turing_data.covariates
data_nature = ImperialReport13.load_data(datadir("imperial-report13", "processed.rds"))
t = dk.date

function npi(x)
    x = diff(x)
    pushfirst!(x, 0)
    t = findfirst(==(1), x)
    data_nature.country_to_dates["Denmark"][t]
end

t_npi = let
    ts = []
    for i in 1:6
        t = data_nature.turing_data.covariates[1][:,i] |> npi
        push!(ts, t)
    end
    ts
end
##
# plot(X[:,[3,4]])
# histogram(rand(Beta(5, 35), 10000))
#
# logistic(-.1)
# X[:,3] |>plot
# X[:,2] .-= 1

@model function testrw(num_weeks, week_index, X, ::Type{TV} = Vector{Float64}) where {TV}
    num_missing = findfirst(x->!ismissing(x), X[:,end]) -1

    ??          ~ Beta(35, 15)
    ??2         ~ Beta(10, 25)
    ??3         ~ Beta(10, 25)
    # ??         ~ arraydist([Normal(-1.5, 2), Normal(2,2)])#, truncated(Normal(0,1), 0, Inf)])
    ??1        ~ truncated(Normal(0.8,0.05), 0, 1)
    ??2        ~ truncated(Normal(0.1,0.05), 0, 1) #-??1
    ??w        ~ truncated(Normal(0,.2), 0, Inf)
    ??w_star = ??w * sqrt( max(1-??1^2-??2^2-2??1^2*??2/(1-??2), 1e-5) )
    weekly_effect = TV(undef, num_weeks)
    weekly_effect[1] ~ Normal(0,0.01)
    weekly_effect[2] ~ Normal(0,??w_star)
    for w in 3:num_weeks
        weekly_effect[w] ~ Normal(??1*weekly_effect[w-1]+??2*weekly_effect[w-2], ??w_star)
    end
    lockdown =  1 .- ?? * X[:,1]
    cc = 1 .+ ??2 * X[num_missing+1:end,2]
    cs = 1 .+ ??3 * X[num_missing+1:end,3]
    ?? = @. 2 * logistic( weekly_effect[week_index[1:end]] ) * lockdown
    ??[num_missing+1:end] .*= cs .* cc
    # dailyeffect = weekly_effect[week_index[1:end]] .+ X[1:end,[1,2]]*??[[1,2]]
	# dailyeffect[num_missing+1:end] .+= X[num_missing+1 : end, 3]*??[3]

    # Rt[1:num_missing]     .= 3.28  * 2 * logistic.( X[1:num_missing,1]*??[1] .+ weekly_effect[week_index[1:num_missing]]);
    # Rt[num_missing+1:end] .= 3.28  * 2 * logistic.( X[num_missing+1:end,:]*?? .+ weekly_effect[week_index[num_missing+1:end]])
    3.28 * ??
end

num_weeks = data.turing_data.num_weeks
week_index = data.turing_data.week_index
m = testrw(num_weeks, week_index, X)
# plot(X)
let
    t = data.dates
    x = [m() for i in 1:1000]
    p = plot(t, x[1:10], c=:black, ??=.4, legend=false)
    plot_confidence_timeseries!(p, t, hcat(x...))
end
##
using StatsFuns
lockdown = Date("18/3/2020", dateformat"d/m/y")

2 * logistic.(rand(Normal(-2.2, .2), 1000)) |>histogram

plotlyjs()
plotnpi(x) = plot!(dk.date, x; ticks=:native, label=false)
p1 = plot(data.dates, data.cases, label="cases")
p2 = plot()
plot!(t, dk.workplace_closing, label="workplace")
plot!(t, dk.gatherings_restrictions, label="gatherings_restrictions")
plot!(t, dk.internal_movement_restrictions, label="internal_movement_restrictions")
plot!(t, dk.school_closing, label="school_closing")
plot!(t, dk.transport_closing, label="transport_closing")
plot!(t, dk.cancel_events, label="cancel_events")
plot!(t, dk.stay_home_restrictions, label="stay_home_restrictions")
p4 = vline(t_npi, labels=data_nature.covariate_names)
p3 = plot(t, dk.stringency_index, label="stringency")
plot(p1, p2, p4, p3, layout=(4,1), legend=:outertopright, sharex=true, link=:x )
#------------------------------------------------------------------
# student project
date = data.dates
serial_interval = National.padzeros(National.serialinterval(length(t)))
delay_distr = National.inf2hosp(length(t))
hospit = data.hospit
df = DataFrame((; date, hospit, serial_interval, delay_distr))
save("/home/and/code/StudentProject/covid19model.csv", df)

## ============================================================================
?? = turing_data
predict = false
TV = Vector{Float64}

@unpack num_observations, num_total_days, num_rt_steps, invlink, deathmodel = ??
# If we don't want to predict the future, we only need to compute up-to time-step `num_obs_countries[m]`
num_time_steps   = predict ? num_total_days : num_observations
num_time_steps_2 = predict ? num_total_days : deathmodel.stop
############# 2.) time varying reproduction number

R0         = truncated(Normal(3., 1.), 1., 5.)  |>rand
R1         = truncated(Normal(.8, .1), .5, 1.1)  |>rand
??_rt       = truncated(Normal(0.1, .05), 0, .25)  |>rand
latent_Rt  = CovidSurvey.RandomWalk(num_rt_steps, ??_rt, invlink(R1))  |>rand
# @code_warntype logpdf(CovidSurvey.RandomWalk(num_rt_steps, ??_rt, invlink(R1)), latent_Rt)


Rt = TV(undef, num_time_steps)
CovidSurvey.National.random_walks!(Rt, ??, predict, latent_Rt, R0)
# @code_warntype CovidSurvey.National.random_walks!(Rt, ??, predict, latent_Rt, R0)

############ 3.) infection dynamics
??  = Exponential(1 / 0.03)  |> rand
T  = typeof(??)
y  = truncated(Exponential(??),T(0),T(1000))  |> rand

newly_infected       = TV(undef, num_time_steps)
cumulative_infected  = TV(undef, num_time_steps)
effective_Rt         = TV(undef, num_time_steps)

# infections!(newly_infected, cumulative_infected, ??, ??, y, Rt)
CovidSurvey.National.infections!(newly_infected, cumulative_infected, effective_Rt, ??, ??, y, Rt)
@code_warntype CovidSurvey.National.infections!(newly_infected, cumulative_infected, effective_Rt, ??, ??, y, Rt)

########### 4.) derive observables
??_i2d = truncated(Normal(20., 4.), 10, 30)  |> rand
??_i2c = truncated(Normal(4.5, 1.), 3, 7)  |> rand
iar   = Beta(1,10)  |> rand
ifr   = truncated(Normal(6/1000, 3/1000), 2/1000, 10/1000)  |> rand
# ihr   ~ truncated(Normal(1/100,1/100),.1/100,5/100)

### day of week effect
weekdayeffect_simplex = Dirichlet([7,7,7,7,7,3,1])  |> rand
weekdayeffect = TV(undef, 7)
weekdayeffect!(weekdayeffect, weekdayeffect_simplex)
@code_warntype weekdayeffect!(weekdayeffect, weekdayeffect_simplex)

### holiday effect
holidayeffect = Beta(1,1)  |> rand

########### 3.) observation model
expected_daily_deaths = TV(undef, num_time_steps_2)
expected_daily_cases  = TV(undef, num_time_steps)

??_c = truncated(Normal(50, 10), 30, Inf)  |> rand
??_d = truncated(Normal(30, 10), 5, Inf)  |> rand

cmodel = National.WeekdayHolidayObsModel(
    ??.casemodel, ??_i2d, iar, ??_c, expected_daily_cases, weekdayeffect, holidayeffect
)
dmodel = National.SimpleObsModel(??.deathmodel, ??_i2d, ifr, ??_d, expected_daily_deaths)

National.expected!(expected_daily_deaths, dmodel, newly_infected)
@code_warntype National.expected!(expected_daily_deaths, dmodel, newly_infected)
National.expected!(expected_daily_cases, cmodel, newly_infected)
@code_warntype National.expected!(expected_daily_cases, cmodel, newly_infected)

???  = zero(Float64)
??? += logpdf(cmodel, ??.cases)
??? += logpdf(dmodel, ??.deaths)
@code_warntype logpdf(cmodel, expected_daily_cases)
@code_warntype logpdf(dmodel, expected_daily_deaths)
