using DrWatson
quickactivate(@__DIR__)
@show projectdir()
##
using StatsBase
using Optim
using Distributions
using StatsPlots
#------------------------------------------------------------------------------
# time-varying IAR
using Turing
using StatsFuns
include("/home/and/code/Covid19/src/visualization.jl")

num_tot = 400
num_obs = 365
start_idx = 100
iar_step = 7

function get_iar_idx(num_tot, num_obs, start_idx)
    iar_idx = zeros(Int64, num_tot)
    iar_idx[1:start_idx-1] .= 1
    idx = 2
    for (j,i) in enumerate(start_idx:num_obs)
         iar_idx[i] = idx
         j % iar_step == 0 && (idx += 1)
    end
    if num_obs < num_tot
        iar_idx[num_obs+1:num_tot] .= iar_idx[num_obs]
    end
    iar_idx
end

iar_idx = get_iar_idx(num_tot, num_obs, start_idx)
num_iar_steps = length(unique(iar_idx))

@model function iarmodel(num_iar_steps, iar_idx, last_time_step, ::Type{TV}=Vector{Float64}) where TV
    logit_iar = TV(undef, num_iar_steps)
    σ_iar = 0.1
    iar0 ~ Beta(10,100)
    logit_iar[1] = logit(iar0)
    for i in 2:num_iar_steps
        logit_iar[i] ~ Normal(logit_iar[i-1], σ_iar)
    end
    logistic.( logit_iar[iar_idx[1:last_time_step]] )
end

m = iarmodel(num_iar_steps, iar_idx, num_tot)
d = hcat([m() for _ in 1:100]...)
plot_confidence_timeseries(d)
#---------------------------------------------------------------------------
# seroconversion
using CovidSurvey

pyplot()
nbins = 300
function seropositive(nbins)
    N = 10_000_000
    # For details see Imperial Report34
    rbins = 1.5:1:(nbins + .5)
    inf2symp = rand(GammaMeanCv(5.1, .86), N)
    inf2conv = inf2symp .+ rand(GammaMeanCv(12.6,5.8/12.6), N)
    inf2rev = inf2conv .+ rand(Weibull(3.67, 132.88), N)
    seroconverted = map( ecdf(inf2conv), rbins)
    notseroreverted = 1 .- map( ecdf(inf2rev), rbins)
    serconverted .* notseroreverted
end
inf2conv = rand(GammaMeanCv(12.6,5.8/12.6), N)
seroconverted = map( ecdf(inf2conv), rbins)
inf2symp = rand(GammaMeanCv(5.1, .86), N)
inf2conv = inf2symp .+ rand(Exponential(13.3), N)
plot(1.5:1:(nbins + .5), 1 .- seroconverted)
seroconverted[20]
seropos = seropositive(nbins)
seropos[20]
plot(1.5:1:(nbins + .5), seropositive(nbins))



using QuadGK
bins = 200
λ = 13.3
μ = 143.7
κ = 3.67
f(x, ts, κ, μ, λ) = 1/λ*exp(-x/λ-((ts-x)/μ)^κ)
π = zeros(bins)
for t in 1:200
    integral, err = quadgk(x->f(x, t, κ, μ, λ), 0, t, rtol=1e-8)
    π[t] = integral
end
plot(π)
#---------------------------------------------------------------------------
# estimate parameters
function gamma(μ, cv)
    α = 1/cv^2
    θ = μ/α
    Gamma(α, θ)
end

function loss(θ)
    cv = θ
    d = gamma(2.6, cv)
    x = rand(d, 10_000)
    lq, uq = percentile(x, [2.5, 97.5])
    m = mean(x)
    sum(abs2, [lq-0.4, uq-9] )
end

op = optimize(θ->loss(θ), 0.1, 3)
cv = op.minimizer
percentile(rand(gamma(2.6, .92), 10_000), [2.5, 97.5])
##
function loss(θ, M, LQ, UP)
    m, cv = exp.(θ)
    d = gamma(m, cv)
    # lq, uq = percentile(d, [2.5, 97.5])
    lq, uq = percentile(d, [2.5, 97.5])
    sum(abs2, [m-M, lq-LQ, uq-UQ] )
end

data.seroprev[!,:m] .= 0.
data.seroprev[!,:cv] .= 0.
for v in eachrow(data.seroprev)
    M = v.country
    LQ, UQ = v.CI
    op = optimize(θ->loss(θ, M, LQ, UQ), log.([M, (UQ-M)/1.96]), NelderMead())
    v.m, v.cv = exp.(op.minimizer)
end

let
    t = data.seroprev.date
    CIs = data.seroprev.CI
    Ms  = data.seroprev.country
    le = [m-ci[1] for (m,ci) in zip(Ms, CIs)]
    ue = [ci[2]-m for (m,ci) in zip(Ms, CIs)]
    scatter(t, Ms, yerror=(le,ue))
    cvs = data.seroprev.cv
    ms  = data.seroprev.m
    le = [m-percentile(gamma(m,cv),2.5) for (m,cv) in zip(ms, cvs)]
    ue = [percentile(gamma(m,cv),97.5)-m for (m,cv) in zip(ms, cvs)]
    scatter!(t.+Day(1), ms, yerror=(le,ue))
end
##
function loss(θ, M, LQ)
    m, σ = exp.(θ)
    d = Normal(m, cv)
    # lq, uq = percentile(d, [2.5, 97.5])
    uq = percentile(d, 2.5)
    sum(abs2, [m-M, lq-LQ] )
end

data.seroprev[!,:m] .= 0.
data.seroprev[!,:std] .= 0.
for v in eachrow(data.seroprev)
    M = v.country
    LQ, UQ = v.CI
    op = optimize(θ->loss(θ, M, LQ), log.([M, (M-LQ)/1.96]), NelderMead())
    v.m, v.std = exp.(op.minimizer)
end

let
    t = data.seroprev.date
    CIs = data.seroprev.CI
    Ms  = data.seroprev.country
    le = [m-ci[1] for (m,ci) in zip(Ms, CIs)]
    ue = [ci[2]-m for (m,ci) in zip(Ms, CIs)]
    scatter(t, Ms, yerror=(le,ue))
    stds = data.seroprev.std
    ms  = data.seroprev.m
    le = [m-percentile(Normal(m,std),2.5) for (m,std) in zip(ms, stds)]
    ue = [percentile(Normal(m,std),97.5)-m for (m,std) in zip(ms, stds)]
    scatter!(t.+Day(1), ms, yerror=(le,ue))
    ms  = data.seroprev.m
    le = [m-percentile(Normal(m,(m-ci[1])/1.96),2.5) for (m,ci) in zip(ms, CIs)]
    ue = [percentile(Normal(m,(m-ci[1])/1.96),97.5)-m for (m,ci) in zip(ms, CIs)]
    scatter!(t.+Day(2), ms, yerror=(le,ue))
end

#----------------------------------------------------------------------------
# calculate kernel function
N = 10_000_000
x = rand(gamma(2.6, cv), N)
h = fit(Histogram, x, 1.5:1:30.5, closed=:right)
w = h.weights ./ N
bar(1.5:1:30.5, w)

#----------------------------------------------------------------------------
# calculate kernel function
# infection to death
N = 10_000_000
x = rand(gamma(5.1, .86), N) .+ rand(gamma(17.8, .45), N)
h = fit(Histogram, x, vcat([0], 1.5:1:100.5))

data_nature = ImperialReport13.load_data(datadir("imperial-report13", "processed.rds"))
turing_data = data_nature.turing_data
π = turing_data.π[1][1:100]
w = h.weights / N * 0.0102
# pushfirst!(w, 0.)
plotlyjs()
bar([w, π], fα=.2, lc=:match)
##
data_usa    = ImperialUSAcases.load_data(datadir("imperial-usa-cases", "processed.rds"))
π2 = data_usa.turing_data.π2[51][1:100]
h = fit(Histogram, rand(gamma(5.1, .86), N), vcat([0], 1.5:1:100.5))
w = h.weights ./ N
bar([w, π2], fα=.2, lc=:match)
#-----------------------------------------------------------------------------
# hospitalization rate
using CSV
using DataFrames

pop = begin
    pop = CSV.File("/home/and/data/population_DK.csv", delim=";", header=false) |> DataFrame
    pop = pop[:,end-1:end]
    rename!(pop, Dict(:Column4 => :age, :Column5 => :size))
    parseyear(s) = begin
        y, _ = split(s," ")
        parse(Int64, y)
    end
    pop.age = parseyear.(pop.age)
    pop
end


# Verity et al. Lancet (2020)
# 10 year bands: 0-9, 10-19, ..., ≥ 80
hosp = begin
    percent = [0.00
            0.0408
            1.04
            3.43
            4.25
            8.16
            11.8
            16.6
            18.4
    ]
    age = ["$(10*i)-$(10*(i+1)-1) years" for i in 0:8]
    age[end] = "≥ 80 year"
    DataFrame(Dict(:percent => percent, :age => age))
end

function addpop!(hosp, pop)
    hosp[!,:pop] .= 0
    for i in 0:7
        for j in 1:10
            hosp[i+1, :pop] += pop[i*10+j, :size]
        end
    end
    hosp[end, :pop] = sum(pop[81:end,:size])
    nothing
end
addpop!(hosp, pop)

ihr = hosp.percent'hosp.pop / sum(hosp.pop)
#ihr = 6.09
#----------------------------------------------------------------------------
# infection fatality ration

ifr = 100 / 100_000 # 95%CI (72–211) / 100_000
ifr*(rand(Normal(1,.3),1000) .+ rand(Normal(.2,.1),1000)) |> histogram
ihr = 0.3

rand(truncated(Normal(.25/100,.1/100),.01/100,1/100), 1000) .* 100 |> histogram
rand(Beta(8,20), 1000) |> histogram

rand(truncated(Normal(5, 5), 1e-1, Inf), 1000) |>histogram
rand(Gamma(3,1), 1000) |>histogram

rand(Beta(8,20), 1000).*.6|>histogram
rand(Beta(4,15), 1000).*.6|>histogram
rand(Beta(5,1), 1000) |>histogram

rand(truncated(Normal(0, 5), 1e-1, Inf),1000) |> histogram

function NegativeBinomial3(μ, σ2)
    p = μ/σ2
    r = μ^2 / (σ2 - μ)
    NegativeBinomial(r, p)
end
x = rand(NegativeBinomial3(30, 40),10000)
mean(x)
var(x)
std(x) / mean(x)
x|>histogram

x = rand(NegativeBinomial2(30, ((40-30)/30^3)^(-1)),10000)
mean(x)
var(x)
std(x) / mean(x)

rand(NegativeBinomial3(30, 30+30^2/rand(truncated(Normal(0, 5), 1e-1, Inf)))) for i in 1:1000]|>histogram
rand(NegativeBinomial2(30, ),1000)|>histogram
##
ts = data.dates
ds = data.cases
m  = length(ds)
n  = 1000
ys = zeros(m,n)
for j in 1:n
    for i in 1:m
        d = ds[i] .+ 1e-2
        ϕ = rand(truncated(Normal(0, 2), 0, Inf))
        ys[i,j] = rand(NegativeBinomial3(d, d+d^2/ϕ))
        # ys[i,j] = rand(NegativeBinomial2(d, ϕ))
    end
end
p = plot(ts,ds)
plot_confidence_timeseries!(p, ts, ys)
##

p0 = histogram(rand(Normal(turing_data.seroprev_mean[end], turing_data.seroprev_std[end]), 10000))
σs = rand(InverseGamma2(turing_data.seroprev_std[end], .1), 10000)
p1 = histogram( [rand(Normal(turing_data.seroprev_mean[end], σs[i])) for i in 1:10000] )
μs = rand(Normal(turing_data.seroprev_mean[end], 10000), 10000)
p2 = histogram( [rand(Normal(μs[i], σs[i])) for i in 1:10000] )
plot(p0,p1,p2,layout=(3,1), sharex=true)
# μs|>histogram

##
ts = epidemic_start:num_obs
# μs  = expected_daily_deaths[ts]
# σ2s = @. μs + μs^2 / ϕ
# deaths[ts] ~ arraydist(NegativeBinomial3.(μs, σ2s))
#
#---------------------------------------------------------------------------
n = 365
m = 100
rw = randn(n,m)*.4
rw[1,:] .= 0
rw = cumsum(rw, dims=1)
# plot_confidence_timeseries(rw)
plot(rw[:,1:10])
#--------------------------------------------------------------------------
pyplot()
0.6 * (1 .+ randn(1000)*.2) |> histogram
rand(Beta(15,35),1000)|>histogram
[rand(Exponential( rand(Exponential(1 / 0.03)) )) for i in 1:1000] |> histogram
rand(LogNormal(5, .5), 10000)|>histogram
##
# si = [ 1.832618e-02 6.659231e-02 1.019139e-01 1.177168e-01 1.183856e-01 1.096347e-01 9.612322e-02 8.105485e-02 6.638313e-02 5.314896e-02 4.178950e-02 3.237506e-02 2.477423e-02 1.876128e-02 1.408140e-02 1.048743e-02 7.758059e-03 5.704855e-03 4.172845e-03 3.037799e-03 2.202072e-03 1.590108e-03 1.144187e-03 8.206829e-04 5.869190e-04 4.186079e-04 2.978198e-04 2.113961e-04 1.497302e-04 1.058413e-04 7.467784e-05 5.259832e-05 3.698645e-05 2.596852e-05 1.820643e-05 1.274709e-05 8.913312e-06 6.224997e-06 4.342490e-06 3.025966e-06 2.106388e-06 1.464818e-06 1.017703e-06 7.064292e-07 4.899421e-07 3.395203e-07 2.350969e-07 1.626684e-07 1.124728e-07 7.771288e-08 5.366011e-08 3.702835e-08 2.553596e-08 1.760009e-08 1.212361e-08 8.346653e-09 5.743340e-09 3.949995e-09 2.715288e-09 1.865648e-09 1.281283e-09 8.795663e-10 6.035400e-10 4.139650e-10 2.838226e-10 1.945186e-10 1.332637e-10 9.126477e-11 6.248002e-11 4.275902e-11 2.925304e-11 2.000644e-11 1.367839e-11 9.348966e-12 6.388001e-12 4.363510e-12 2.979728e-12 2.034262e-12 1.388445e-12 9.474643e-13 6.462608e-13 4.407585e-13 3.004264e-13 2.049472e-13 1.395550e-13 9.514611e-14 6.472600e-14 4.418688e-14 3.008704e-14 2.042810e-14 1.387779e-14 9.547918e-15 6.439294e-15 4.329870e-15 2.997602e-15 1.998401e-15 1.443290e-15 8.881784e-16 6.661338e-16 4.440892e-16 ] |> vec
# plot(si)
# plot!(turing_data.serial_intervals)
# xlims!(0,100)

#10.92
##
μs = rand(Normal(10.92, 0.94), 100)
ϕs = rand(Normal(5.4, 0.27), 100)
i2c = NegativeBinomial2.(μs, ϕs)
let
    x = 1:50
    p = plot()
    for i in 1:length(i2c)
        plot!(x, i2c[i], c=:black, α=0.1, label=nothing, ls=:solid)
    end
    plot!(x, NegativeBinomial2.(10.92, 5.4), c=:red, lw=2, ls=:solid)
    plot!(x, turing_data.π2[x], c=:green, lw=2)
end
##
@model function i2c_sampler(::Type{V}  = Int64) where V
    μ_i2o ~ truncated(Normal(1.53, 0.051), 0, Inf)
    σ_i2o ~ truncated(Normal(0.48, 0.0759), 0, Inf)
    i2o   ~ LogNormal(μ_i2o, σ_i2o)

    μ_o2c ~ truncated(Normal(5.82, 0.68), 0, Inf)
    ϕ_o2c ~ truncated(Normal(1.57, 0.054), 0, Inf)
    o2c   ~ NegativeBinomial2(μ_o2c, ϕ_o2c)

    return V(round(i2o+o2c))
end

@model function i2h_sampler(::Type{V}  = Int64) where V
    μ_i2o ~ truncated(Normal(1.53, 0.051), 0, Inf)
    σ_i2o ~ truncated(Normal(0.48, 0.0759), 0, Inf)
    i2o   ~ LogNormal(μ_i2o, σ_i2o)

    μ_o2c ~ truncated(Normal(5.82, 0.68), 0, Inf)
    ϕ_o2c ~ truncated(Normal(1.57, 0.054), 0, Inf)
    o2c   ~ NegativeBinomial2(μ_o2c, ϕ_o2c)

    return V(round(i2o+o2c))
end
## infection 2 symptom onset
plot(0:.1:15, GammaMeanCv(5.1, .86))
plot!(0:.1:15, LogNormal(1.53, 0.48))
## serial interval
# plot(National.serialinterval(15))
plot!(0:.1:15, GammaMeanCv(5.06, 2.11/5.06))
plot!(0:.1:15, GammaMeanCv(6.5, .62))
## infection 2 death
plot(National.inf2death(90))
plot!(1:90, NegativeBinomial2(21.82, 14.26))
##
m = i2c_sampler()
i2c = [m() for _ in 1:10_000]

@model function fitNB(y)
    μ ~ LogNormal(2., .7)
    ϕ ~ LogNormal(0.5, 1.2)
    y .~ NegativeBinomial2(μ, ϕ)
end

c = sample(fitNB(i2c), NUTS(1000, 0.6), 2000, progress=true)
c = c[1001:end,:,:]

plot(c)
c["μ"]|>vec|>mean_and_std
c["ϕ"]|>vec|>mean_and_std
histogram( i2c, label=nothing, normalize=true)
plot!(NegativeBinomial2(c["μ"]|>vec|>mean, c["ϕ"]|>vec|>mean))
## infected 2 death
# from Flaxman et al.
i2d = [rand(GammaMeanCv(5.1,0.86))+rand(GammaMeanCv(17.8,0.45)) for _ in 1:10_000]
#6.49 5 .94-7 .10
@model function fitNB(y)
    μ ~ LogNormal(4.,.5)
    # ϕ ~ LogNormal(1.,1.2)
    cv ~ truncated(Normal(.5, .3), 0, Inf)
    y .~ GammaMeanCv(μ, cv)
    return μ, cv
end
m = fitNB(i2d)
let
    p = histogram( i2d, label=nothing, normalize=true)
    for i in 1:10
        plot!(1:90, GammaMeanCv(m()...))
    end
    p
end
c = sample(m, NUTS(500, 0.6), 2000, progress=true)
c = c[501:end,:,:]

plot(c)
c["μ"]|>vec|>mean_and_std # = 22.8
c["cv"]|>vec|>mean_and_std # = 0.4
histogram( i2d, label=nothing, normalize=true)
plot!(GammaMeanCv(c["μ"]|>vec|>mean, c["cv"]|>vec|>mean))
##
function inf2death(μ, nbins=90)
	r = zeros(nbins+1)
	r[2:end] = 1.5:1:(nbins + .5)
    return diff( cdf.(GammaMeanCv(μ, 0.4), r) )
end

p = let
    p = plot()
    for μ in 20.:2.:30.
        plot!(1:90, GammaMeanCv(μ, 0.4))
        plot!(1:90, inf2death(μ, 90), ls=:dash)
    end
    p
end

#----------------------------------------------------------------------------
# test capacity
function process_ssi(df)
    rename!(df, 1=>:region, 2=>:date, 3=>:counts)
    df.region = replace(df.region, dan2eng...)
    dropmissing!(df, :region)
    df = unstack(df, :date, :region, :counts)
    insert_missing_dates!(df)
    replace_missing!(df)
    DataFrames.transform(df, AsTable(Not(:date)) => sum => :country)
end

df = CSV.File(normpath( homedir(), "data/Statens-Serum-Institut/dashboard/Regionalt_DB/16_pcr_og_antigen_test_pr_region.csv" ) ) |>
    DataFrame #|> National.process_ssi

rename!(df, 1=>:date, 2=>:region, 3=>:tested, 4=>:method)
df.region = replace(df.region, National.dan2eng...)
df.region = replace(df.region, missing=>"unkown")

pcr = df[df.method .== "PCR",:]
pcr = unstack(pcr, :date, :region, :tested)
# National.insert_missing_dates!(pcr)
National.replace_missing!(pcr)
pcr = DataFrames.transform(pcr, AsTable(Not(:date)) => sum => :country)

df = unstack(df, :week, :region, :tested)
DataFrames.transform(df, AsTable(Not(:date)) => sum => :country)
insert_missing_dates!(df)
replace_missing!(df)
DataFrames.transform(df, AsTable(Not(:date)) => sum => :country)
