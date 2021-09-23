using DrWatson
quickactivate(@__DIR__)
@show projectdir()
##
using StatsBase
using Plots
using DataFrames
using Underscores
using CSV
using HTTP, JSON
using Dates
## ============================================================================
#                    apple
#  ============================================================================
old2new_names = Dict(
    "Capital Region of Denmark" => "capital",
    "Central Denmark Region" => "central",
    "North Denmark Region" => "north",
    "Region Zealand" => "zealand",
    "Region of Southern Denmark" => "south"
)
regions = values(old2new_names)

function replacemissing(xs)
    for i in eachindex(xs)
        if ismissing(xs[i])
            xs[i] = @_ vcat( xs[i-2:i-1], xs[i+1:i+2] ) |>
                skipmissing |>
                mean
        end
    end
    convert(Vector{Float64}, xs)
end

function custom_mean(xs)
    n = 0
    s = 0.
    for x in xs
        if !ismissing(x)
            n += 1
            s += x
        end
    end
    if n == 0
        @warn "all missing"
        return missing
    else
        m = s / n
        @assert isfinite(m)
        return m
    end
end

apple = @_ projectdir("data/apple/applemobilitytrends.csv") |>
    CSV.File(__) |>
    DataFrame |>
    filter(:country => x->!ismissing(x) , __) |>
    # filter(:geo_type => ==("country/region"), __) |>
    filter(:country => ==("Denmark") , __) |>
    filter(:region => x-> x ∈ keys(old2new_names) , __) |>
    transform(__, :region => ByRow(x -> old2new_names[x]) => :region) |>
    select(__, Not(["geo_type", "alternative_name", "sub-region", "country"])) |>
    DataFrames.stack(__, Not([:region, :transportation_type]), variable_name="date", value_name="mobility") |>
    unstack(__, :transportation_type, :mobility) |>
    select(__, :date => ByRow(Date) => :date, Not(:date), AsTable([:driving,:walking,:transit]) => ByRow(x->custom_mean(x)) => :apple) |>
    rename(__, [:driving,:walking,:transit] .=> [:apple_driving,:apple_walking,:apple_transit])

## ============================================================================
#                    google
#  ============================================================================
const mobility_names = [
    "retail_and_recreation_percent_change_from_baseline",
    "grocery_and_pharmacy_percent_change_from_baseline",
    "workplaces_percent_change_from_baseline",
    "residential_percent_change_from_baseline"
]

const new_mobility_names = [
    "google_retail",
    "google_grocery",
    "google_workplaces",
    "google_residential"
]

google = @_ "/home/and/data/google-mobility/Global_Mobility_Report.csv" |>
    CSV.File(__) |>
    DataFrame |>
    filter(:country_region => ==("Denmark"), __) |>
    # rename(__, :sub_region_1 => :region) |>
    filter( :sub_region_1 => x->!ismissing(x), __) |>
    transform(__, :sub_region_1 => disallowmissing => :region) |>
    filter(:sub_region_1 => x-> x ∈ keys(old2new_names) , __) |>
    filter(:sub_region_2 => ismissing, __) |>
    select(__, :date, :region, mobility_names .=> new_mobility_names) |>
    select(__, All(), :region => ByRow(x -> old2new_names[x]) => :region) |>
    select(__, All(), AsTable([:google_retail,:google_grocery,:google_workplaces]) => ByRow(mean) => :google)


## ============================================================================
#                    telco
#  ============================================================================
eng2dan = Dict(
        "Brondby" => "Brøndby",
        "Copenhagen" => "København",
        "Lyngby Taarbæk" => "Lyngby Taarbæk",
        "Nordfyn" => "Nordfyns",
        "Aarhus" => "Århus"
    )

kom2reg = @_ "/home/and/data/google-mobility/Global_Mobility_Report.csv" |>
    CSV.File(__) |>
    DataFrame |>
    filter(:country_region => ==("Denmark"), __) |>
    filter( :sub_region_1 => x->!ismissing(x), __) |>
    filter( :sub_region_2 => x->!ismissing(x), __) |>
    select(__, :sub_region_1 => :region, :sub_region_2 => :kommune) |>
    disallowmissing |>
    groupby(__, :kommune) |>
    combine(__, unique) |>
    select(__,
        :region => ByRow(x -> old2new_names[x]) => :region,
        :kommune => ByRow( x->string(first(split(x, " Municipality"))) ) => :kommune,
    ) |>
    transform(__, :kommune => ByRow( x-> x ∈ keys(eng2dan) ? eng2dan[x] : x ) => :kommune ) |>
    transform(__, :kommune => ByRow( x->replace(x, "-"=>" ") ) => :kommune) |>
    append!(__,
        DataFrame( Dict(
        "kommune" => ["Ærø", "Samsø", "Fanø", "Læsø"],
        "region"  => ["south", "central", "south", "north"]
    ))) |>
    Dict([row.kommune => row.region for row in eachrow(__)])

function json2long(raw)
    num_kommunes = length(raw["locations"])
    data = Array(hcat(raw["data"]...)')
    source = string.(repeat(raw["locations"], inner=num_kommunes))
    target = string.(repeat(raw["locations"], outer=num_kommunes))
    dates = Date.(raw["dates"], DT)


    df = DataFrame(
        "date" => Vector{Date}(),
        "source" => Vector{String}(),
        "target" => Vector{String}(),
        "flow" =>  Vector{Float64}()
        )

    for (d, flow) in zip(dates, eachcol(data))
        append!(df, DataFrame( (; date = fill(d, num_kommunes^2), source, target, flow) ) )
    end

    return df
end

const DT = dateformat"yyyy-mm-dd HH:MM:SS"
telco = @_ HTTP.get("https://covid19.compute.dtu.dk/data/telco_data.json") |>
    String(__.body) |>
    JSON.Parser.parse |>
    json2long |>
    transform(__, :source => ByRow(x->kom2reg[x]) => :source) |>
    transform(__, :target => ByRow(x->kom2reg[x]) => :target) |>
    filter(_.source == _.target, __) |>
    select(__, :date, :source => :region, :flow => :telco) |>
    groupby(__, [:date, :region]) |>
    combine(__, :telco => sum => :telco) |>

@_ telco |>
    filter( _.region == "south", __) |>
    plot(__.date, __.telco)
## ============================================================================
#                    combine google, apple and telco
#  ============================================================================
mobil = @_ outerjoin(apple, google; on=[:date,:region]) |>
    outerjoin(__, telco; on=[:date,:region]) |>
    sort(__, :date) |>
    filter( Date("2020-02-15") <= _.date <= Date("2021-06-30"), __)
CSV.write(projectdir("data/mobility/", "mobility_regional.csv"), mobil)

## regional data sets

function rollingmean(xs, window)
    n  = length(xs)-window+1
    ys = Vector{Float64}(undef, n)
    for i in 1:n
        x = xs[i:i+window-1]
        @assert sum( ismissing.(x) ) < 4 "nr of missing: $(sum( ismissing.(x) )) at i=$i"
        ys[i] = mean(skipmissing(x))
        @assert isfinite(ys[i]) "infinite at i=$i"
    end
    return ys
end

numericcols(df::DataFrame) =
    filter(x->nonmissingtype( eltype( df[!,x] ) ) <: Number, names(df))


function postprocessing(df, window)
    @assert isodd(window)
    Δ = window ÷ 2
    smoothed = DataFrame( :date => df.date[1+Δ:end-Δ] )
    cols = filter(x->nonmissingtype( eltype( df[!,x] ) ) <: Number, names(df)) #[eltype(col) <: Number for col in eachcol(mobil)]
    for col in cols
        @info col
        smoothed[!,col] = rollingmean(df[:,col], window)
        smoothed[!,col] ./= abs(smoothed[1,col])
        smoothed[!,col] .-= 1
        smoothed[!,col] ./= StatsBase.std(smoothed[!,col])
        if !all( isfinite.(smoothed[!,col]) )
            @error "is not finite at $col:"
            println("$(findall(x-> !isfinite(x), smoothed[!,col]))")
        end
    end
    smoothed
end

function assertconsistency(df)
    @assert isequal( df.date |> length,
                     df.date[1]:Day(1):df.date[end] |> length)
    @assert all( [!any( @. isnan(col) | ismissing(col) | ! isfinite(col) ) for col in eachcol(df) if eltype(col) <: Number] )
    return df
end

function drop_empy_columns(df)
    drop = []
    for n in names(df)
        col = df[!,n]
        all(ismissing.(col)) && push!(drop, n)
    end
    return select(df, Not(drop))
end

regionalmobil = let
    regionalmobil = Dict()
    for region in regions
        @info region
        regional = @_ mobil |> filter(:region => ==(region), __) |>
            sort(__, :date) |>
            drop_empy_columns |>
            postprocessing(__, 7) |>
            select(__, All(), AsTable([:google,:apple]) => ByRow(mean) => :applegoogle) |>
            disallowmissing |>
            assertconsistency

        CSV.write(projectdir("data/mobility/", "mobility_region=$region.csv"), regional)
        regionalmobil[region] = regional
    end
    regionalmobil
end

## plot
using Plots
plotlyjs()
let
    ps = []
    nplots = length(Regional.regions)
    for region in regions
        mob = regionalmobil[region]
        p = plot(title="$region", xticks=:native)
        plot!(mob.date, mob.apple, lab="apple")
        plot!(mob.date, mob.google, lab="google")
        plot!(mob.date, mob.applegoogle, lab="apple-google")
        plot!(mob.date, mob.telco, lab="telco")
        push!(ps,p)
    end
    plot(ps..., layout=(nplots,1), sharex=true, link=:x, size=(1000, nplots*250))
end
