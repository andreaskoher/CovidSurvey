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
import Downloads
using ColorSchemes
using PrettyTables
colors = ColorSchemes.tableau_10.colors
# using Makie
# using GLMakie
# GLMakie.activate!()
plotlyjs()
##

ps = let
    regions2key =  Dict(
        "Hungary" => "HU",
        "UK" => "GB",
        "France" => "FR",
        "Sweden" => "SE",
        "Denmark" => "DK",
        "Germany" => "DE",
        "Italy" => "IT",
        "USA" => "US")
    key2regions = Dict( c=>k for (k,c) in pairs(regions2key) )
    ks = collect(keys(key2regions))
    regions = collect(values(key2regions))
    (; regions2key, key2regions, keys=ks, regions)
end

function epidata(ps)
    @_ "https://storage.googleapis.com/covid19-open-data/v3/epidemiology.csv" |>
        Downloads.download |>
        CSV.read(__, DataFrame) |>
        filter(:location_key => ∈(ps.keys), __) |>
        select(__, :date, :new_deceased => :deaths, :new_confirmed => :cases,
            :location_key =>ByRow( x->ps.key2regions[x] )=> :region)
end


function popdata(ps)
    @_ "https://storage.googleapis.com/covid19-open-data/v3/demographics.csv" |>
        Downloads.download |>
        CSV.read(__, DataFrame) |>
        filter(:location_key => ∈(ps.keys), __) |>
        select(__, :population,
            :location_key =>ByRow( x->ps.key2regions[x] )=> :region)
end

function vacdata(ps)
    @_ "https://storage.googleapis.com/covid19-open-data/v3/vaccinations.csv" |>
        Downloads.download |>
        CSV.read(__, DataFrame) |>
        filter(:location_key => ∈(ps.keys), __) |>
        select(__, :date, :cumulative_persons_vaccinated => :vaccinated,
            :location_key =>ByRow( x->ps.key2regions[x] )=> :region)
end

function disallownegative!(df)
    for c in [:deaths, :cases]
        idx = findall(<(0), df[!,c])
        if !isempty(idx)
            @info "replace negative counts with zero"
            pretty_table(df[idx,:])
            df[idx,c] .= 0
        end
    end
    return df
end

function rollingmean(xs, window)
    n  = length(xs)-window+1
    ys = Vector{Float64}(undef, n)
    for i in 1:n
        ys[i] = mean(xs[i:i+window-1])
    end
    return ys
end

function smoothing(df, window)
    @assert isodd(window)
    Δ = window ÷ 2
    smoothed = DataFrame(
        :date => df.date[1+Δ:end-Δ],
        :cases => df.cases[1+Δ:end-Δ],
        :deaths => df.deaths[1+Δ:end-Δ], )
    for col in ["cases", "deaths"]
        smoothed[!,"$(col)_smoothed"] = rollingmean(df[:,col], window)
    end
    smoothed
end

function replacemissing!(df)
    cols = filter(x->nonmissingtype( eltype( df[!,x] ) ) <: Number, names(df))
    for col in cols
        for i in findall(ismissing, df[!,col])
            df[i, col] = i == 1 ? 0 : df[i-1, col]
        end
    end
    df
end

function addpadding(df, dates)
    cols = filter(x->nonmissingtype( eltype( df[!,x] ) ) <: Number, names(df))
    padded = DataFrame( :date => dates )
    for col in cols
        df[2:end,col] = diff(df[:,col])
    end
    padded =@_ leftjoin(padded, df; on=:date) |>
        sort(__, :date) |>
        transform(__, cols .=> x->cumsum(coalesce.(x,0)), renamecols=false)
    padded
end

function main(epi, vac, pop)
    epidata = Dict{String,DataFrame}()
    for region in ps.regions
        @info region
        regional = @_ epi |>
            filter(:region => ==(region), __) |>
            sort(__, :date) |>
            filter(:date => <=(Date("2021-09-01")), __) |>
            disallowmissing |>
            disallownegative! |>
            smoothing(__, 7)

        vaccination = @_ vac |>
            filter(:region => ==(region), __) |>
            select(__, Not(:region)) |>
            sort(__, :date) |>
            filter(:date => <(Date("2021-09-01")), __) |>
            replacemissing!(__) |>
            disallowmissing |>
            addpadding(__,regional.date)

        population = @_ pop |>
            filter(:region => ==(region), __) |>
            only(__.population)

        regional[!,:population] .= population
        regional[!,:cases_per_100000] = regional[!,"cases_smoothed"] / population * 100_000
        regional[!,:deaths_per_100000] = regional[!,"deaths_smoothed"] / population * 100_000
        vaccination[!,:fraction_vaccinated] = vaccination[!,"vaccinated"] / population
        regional = leftjoin(regional, vaccination; on=:date)
        disallowmissing!(regional)

        @assert all(regional.date .== regional.date[1]:Day(1):regional.date[end])
        CSV.write(projectdir("data/epidata/", "epidata_region=$region.csv"), regional)
        epidata[region] = regional
    end
    bson(projectdir("data/epidata/", "epidata.bson"), epidata)
    epidata
end

## ========================================================================
data = main(
    epidata(ps),
    vacdata(ps),
    popdata(ps)
    )

## =========================================================================
#          merge data for epidemia
function dict2df(data)
    regions = collect(keys(data))
    r = regions[1]
    epi = @_ data[r] |>
        select(__, All(),
            :date => ( x -> div.(0:length(x)-1, 7) .+ 1 ) => :week)
    epi[!, :region] .= r
    for r in regions[2:end]
        df = @_ data[r] |>
            select(__, All(),
                :date => ( x -> div.(0:length(x)-1, 7) .+ 1 ) => :week)
        df[!, :region] .= r
        epi = vcat(epi, df)
    end
    epi
end
epi = dict2df(data)
CSV.write(projectdir("data/epidata/", "epidata_longformat.csv"), epi)
































## ==============================================================
#     DEPRICATED WHO DATA

who_params = (
    regions = ["Hungary", "The United Kingdom", "France", "Sweden", "Denmark", "Germany", "Italy", "United States of America"]
    , old2new = Dict(
        "Hungary" => "Hungary",
        "The United Kingdom" => "UK",
        "France" => "France",
        "Sweden" => "Sweden",
        "Denmark" => "Denmark",
        "Germany" => "Germany",
        "Italy" => "Italy",
        "United States of America" => "USA"
    )
    , fname = projectdir("data/epinow/WHO-COVID-19-global-data.csv")
)

let
    p1 = bar(rt.date, rt.new_cases, fc=colors[1], lc=colors[1])
    p2 = plot(rt.date, rt.mean, ribbon=(rt.lower_80, rt.upper_80))
    p = plot(p1, p2, layout=(2,1), size=(1000, 2*250), sharex=true, link=:x)
end
who = CSV.read(who_params.fname, DataFrame)



function load_data(regions, params)
    who = CSV.read(params.fname, DataFrame)
    observations = Dict{String, DataFrame}()
    for region in params.regions
        regional = @_ who |>
                filter(:Country => ==(region), __) |>
                select(__, :Date_reported => :date, :New_cases => :cases, :New_deaths => :deaths) #|>
        #__[Date("2020-08-01") <. __.date <. Date("2021-02-01") ]
        observations[params.old2new[region]] = regional
    end
    observations
end

obs = load_data(regions, who_params)
