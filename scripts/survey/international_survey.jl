using DrWatson
quickactivate(@__DIR__)
@show projectdir()
##
using CSV
using DataFrames
using Dates
using Plots
using BSON
using StatsBase
using Underscores
using ColorSchemes
colors = ColorSchemes.tableau_10.colors
# using Makie
# using GLMakie
# GLMakie.activate!()
plotlyjs()

## ============================================================================
# utils

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

datacols(df) = Symbol.(names(df, Not([:date, :region, :batch])))

function rmfirstday(data)
    dates = data.date
    @assert issorted(dates)
    firstday = first(dates)
    i = findfirst(>(firstday), dates)
    data[i:end, :]
end

function maskoutliers(data, p=99.9)
    outliers = zeros(size(data,1))
    for c in datacols(data)
        threshold = p isa Number ? percentile(data[:,c], p) : p[c]
        @show "threshold for $c: $threshold"
        outliers .+= (threshold .< data[!,c])
        outliers .+= (data[!,c] .< 0)
    end
    return outliers .== 0
end

function filteroutliers(data, p=99.9)
    mask = maskoutliers(data, p)
    @show "remove $(sum(mask .== 0)/size(data,1)*100) percent"
    data = data[mask,:]
    data = rmfirstday(data)#NOTE first day seems unusual
    return data
end

function support_index(stepsize, support_position)
    support_position == :start && return 1
    support_position == :end && return stepsize
    iseven(stepsize) && return stepsize ÷ 2
    return stepsize ÷ 2 + 1
end

function sampledown(df; stepsize=1, support_position=:middle)
    @assert issorted(df.date)
    # df  = copy(df)
    s = Day(support_index(stepsize, support_position)-1)
    day1 = first(df.date)
    df[!, :date] =  @. ( ( df.date - day1 ) ÷ stepsize ) * stepsize + s + day1
    return df
end

function continuous_time(df)
    dates = df.date
    data_cols = names(df, Not(:date))
    @assert issorted(dates)
    ts = dates[1] : Day(1) : dates[end]
    long_df = DataFrame("date" => ts, [c => Vector{Union{Float64, Missing}}(undef, length(ts)) for c in data_cols]...)
    next_date = dates[1]
    j = 1
    for (i,t) in enumerate(ts)
        if t == next_date
            long_df[i, data_cols] = df[j, data_cols]
            j += 1
            j > size(df, 1) && return long_df
            next_date = df.date[j]
        end
    end
end

function rollingmean(xs::AbstractVector, window)
    n  = length(xs)-window+1
    ys = Vector{Float64}(undef, n)
    for i in 1:n
        x = xs[i:i+window-1]
        ys[i] = mean(skipmissing(x))
        @assert isfinite(ys[i]) "infinite at i=$i"
    end
    return ys
end

function rollingmean(df, window)
    @assert isodd(window)
    long_df = continuous_time(df)
    Δ = window ÷ 2
    smoothed = DataFrame( :date => long_df.date[1+Δ:end-Δ] )
    for col in names(df, Not(:date))
        @info col
        smoothed[!,col] = rollingmean(long_df[:,col], window)
        @assert all( isfinite.(smoothed[!,col]) ) "$(findall(x-> !isfinite(x), smoothed[!,col]))"
    end
    smoothed
end

function preprocess(df, param, region)
    @_ df |>
        filter(:region => ==(region), __) |>
        filteroutliers(__, param.threshold) |>
        groupby(__, :batch, sort = true) |>
        combine(__, param.ctypes .=> sum, :date => first, nrow, renamecols = false) |>
        select(__, Not(:batch), AsTable(param.ctypes) => ByRow(sum) => :total) |>
        transform(__, [param.ctypes..., :total] .=> ( x -> x ./ __.nrow ), renamecols = false)
end

function plotweeks!(p, dates)
    dates = @_ DataFrame( :date => dates[1] : Day(1) : dates[end] ) |>
        select(__, :date,
            :date =>( x-> week.(x) )=> :week,
            :date =>( x-> year.(x) )=> :year ) |>
        groupby(__, [:year, :week]) |>
        combine(__, :date .=> [first, last])|>
        filter(:week => x->x%2 == 0, __)

    for row in eachrow(dates)
        vspan!(p, [row.date_first, row.date_last], c=:darkblue, α=0.1, label=false)
    end
    p
end

function firefox(p; fname="tmp.html", fdir=normpath(homedir(), ".tmp"))
    # mkpath(dir)
    filepath = normpath(fdir, fname)
    savefig(p, filepath)
    run(`firefox $(filepath)`, wait=false)
    return p
end
## ===========================================================================
# main

param = (
    fname = projectdir("data/contacts/int/international_contacts.csv")
    , ctypes = [:family, :colleagues, :friends, :strangers]
    , threshold = (family=50, colleagues=100, friends=100, strangers=1000)
)

function main(param)
    rawcontacts = readcontacts( param.fname )
    regions  = unique(rawcontacts.region)
    contacts = Dict{String,DataFrame}()
    for region in regions
        regional = preprocess(rawcontacts, param, region)
        contacts[region] = regional
        CSV.write(projectdir("data/contacts/int/mean_contact_rates_region=$(region)_aggregation=batch.csv"), regional)
    end
    bson(projectdir("data/contacts/int/mean_contact_rates_aggregation=batch.bson"), contacts)
    return contacts
end
contacts = main(param)

## plot

let
    for (region, data) in pairs(contacts)
        p = scatter(data.date, data.total, title=region)
        plotweeks!(p, data.date) |> display
    end
end

# plot raw data
let
    rawcontacts = readcontacts( param.fname )
    region = "Denmark"
    rawregional = filter(:region => ==(region), rawcontacts)
    scatter(rawregional.date, rawregional.strangers, mα=0.1, mc=colors[2], msc=:match)
end

for (region, data) in pairs(contacts)
    Δt = unique(diff(data.date))
    @show region, Δt
end












## ===========================================================================
#              Smoothed Contacts
# ============================================================================
using Plots
using AbstractGPs
using Distributions
using ParameterHandling
using ParameterHandling: value, flatten
using Optim
using Zygote

## ===========================================================================
# utils

struct Loss{X,Y,F,X0}
    x::X
    y::Y
    unpack::F
    x0::X0
end

function Loss(X,Y,θ::NamedTuple)
    x0, unflatten = flatten(θ);
    unpack = value ∘ unflatten
    return Loss(X,Y,unpack,x0)
end

function (nlml::Loss)(ps::AbstractVector)
    @unpack x, y, unpack = nlml
    θ = unpack(ps)
    f = build_model(θ)
    return -logpdf(f(x, θ.s), y)
end

function plot_smoothed(x,y,ŷ,sŷ; title = "")
    p = plot(; title)
    scatter!(x,y; label = false)
    plotweeks!(p, x)
    tf = x[1]:Day(1):x[end]
    plot!(tf, ŷ; label=false, ribbon=3sŷ, hover=tf)
end

function smooth(X::AbstractVector, Y::AbstractVector)
    θ = initθ()
    loss = Loss(X,Y,θ)

    results = Optim.optimize(
        loss,
        ps -> gradient(loss, ps)[1],
        loss.x0,
        LBFGS(),
        Optim.Options(
            allow_f_increases=true,
            show_trace=true
        );
        inplace=false
    )
    @show results.ls_success
    ps = loss.unpack(results.minimizer)
    @show ps
    f = build_model(ps)
    fp = AbstractGPs.posterior(f(X, ps.s), Y)
    marg = marginals(fp(0:X[end]))
    (;
        mean = mean.(marg),
        std  = std.(marg)
    )
end

function smooth(df::DataFrame, param::NamedTuple; title = "")
    dates = df.date
    X = [d.value for d in dates .- dates[1]]

    plots = Vector{Plots.Plot}()
    smoothed = Dict{String, AbstractVector}()
    smoothed["date"] = dates[1]:Day(1):dates[end]

    for ctype in names(df, Not([:date, :nrow]))
        @show ctype
        m, s = mean_and_std(df[!,ctype])
        Y = ( df[:,ctype] .- m ) / s
        ŷ, sŷ = smooth(X,Y)

        smoothed[ctype] = ŷ * s .+ m
        smoothed["$(ctype)_std"] = sŷ * s
        p = plot_smoothed(dates, Y, ŷ, sŷ; title=ctype)
        push!(plots, p)
    end

    p = plot(plots..., layout=(5,1), size=(1000, 5*250), sharex=true, link=:x)
    firefox(p, fname="$title.html")
    return DataFrame(smoothed...)
end

function initθ()
    (
        l1 = positive(90.),
        l2 = bounded(30., 5., 50.),
        s1 = positive(1.5),
        s2 = positive(.7),
        α = positive(2.),
        s = bounded(1., 0.1, 2.),
    )
end

function build_model(θ::NamedTuple)
    GP(
        θ.s1 * SqExponentialKernel() ∘ ScaleTransform(1/θ.l1) +
        θ.s2 * SqExponentialKernel() ∘ ScaleTransform(1/14)
        # θ.s2 * RationalQuadraticKernel(α = θ.α) ∘ ScaleTransform(1/θ.l2)
    )
end

## ===========================================================================
function smooth(param::NamedTuple)
    rawcontacts = readcontacts( param.fname )
    regions  = String.( unique(rawcontacts.region) )
    contacts = Dict{String, DataFrame}()
    for region in regions
        @info region
        aggregated_contacts = preprocess(rawcontacts, param, region)[2:end,:]
        smoothed = smooth(aggregated_contacts, param, title=region)
        contacts[region] = smoothed
        CSV.write(projectdir("data/contacts/int/smoothed_contact_rates_region=$(region).csv"), smoothed)
    end
    bson(projectdir("data/contacts/int/smoothed_contact_rates.bson"), contacts)
    return contacts
end

smoothed_contacts = smooth(param)

## ==========================================================================
# replace Denmark with high resolution data
let
    den_lofi = @_ CSV.read(projectdir("data/contacts/int/smoothed_contact_rates_region=Denmark.csv"), DataFrame) |>
        rename(__, Dict(
            col => "$(col)_low_resolution" for col in filter(x->eltype( __[!,x] ) <: Number, names(__))
        ))
    den_hifi = CSV.read(projectdir("data/contacts/dk","smoothed_contact_rates_window=7.csv"), DataFrame)
    den = leftjoin(den_hifi, den_lofi; on=:date, validate=(true, true), )
    CSV.write(projectdir("data/contacts/int/smoothed_contact_rates_region=Denmark.csv"), den)
end
