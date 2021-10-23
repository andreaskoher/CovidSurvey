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
using PrettyTables
colors = ColorSchemes.tableau_10.colors

## ===========================================================================

function consistent!(raw; nonnegative = false, positive = false)
    df = @_ raw |>
        disallowmissing(__) |>
        sort(__, :date) |>
        continuous_time!(__)

    if nonnegative || positive
        dat = @_ df |>
            select(__, Not(:date)) |>
            Array(__)
        nonnegative && @assert all( dat .>= 0 )
        positive && @assert all( dat .> 0 )
    end
    df
end

function Base.filter!(conditions::Vector{Pair{Symbol, Function}}, df::AbstractDataFrame)
    for condition in conditions
        filter!(condition, df)
    end
    df
end
# Base.filter!(condition::Pair{Symbol, Function}, df::AbstractDataFrame) =
#     filter!(condition, df)
Base.filter!(nothing, df::AbstractDataFrame) = df

function read_survey(region; fnames_survey, shift = 0, kwargs...)
    @_ fnames_survey[region] |>
        CSV.read(__, DataFrame) |>
        consistent!(__, positive = true) |>
        transform(__, :date =>( x -> x + Day(shift) )=> :date)
end

function read_mobil(region; fnames_mobil, kwargs...)
    @_ fnames_mobil[region] |>
        CSV.read(__, DataFrame) |>
        consistent!(__)
end

# read_predictors(region; predictors = All(), predictors_kwargs...)
# read_mobil(region; predictors_kwargs...)

function read_predictors(region; predictors = All(), kwargs...)
    @_ outerjoin(
            read_survey(region; kwargs...),
            read_mobil(region; kwargs...);
            on=:date) |>
        select(__, :date, predictors) |>
        sort(__, :date)
end

function read_epidata(region; conditions = nothing, kwargs...)
    @_ projectdir("data/epidata/epidata_region=$(region).csv") |>
        CSV.read(__, DataFrame) |>
        consistent!(__, nonnegative = true) |>
        filter!(conditions, __)
end

function relativechange!(xs::AbstractVector; standardize=false, kwargs...)
    xs[1] == 0. && (@warn "predictor data stream starts with 0, so we assume it is already normalized."; return nothing)
    xs ./= abs(xs[1])
    xs .-= xs[1]
    standardize && (xs ./= std(xs))
    return nothing
end

function relativechange!(df::AbstractDataFrame; kwargs...)
    for n in names(df, Not(:date))
        relativechange!(df[!,n], kwargs...)
    end
    df
end

function continuous_time!(df)
    @assert all( df.date[1]:Day(1):df.date[end] .== df.date )
    df
end

function clean_predictors(pred, epid; conditions, kwargs...)
    df = @_ pred |>
        leftjoin(epid, __; on = :date) |>
        sort(__, :date) |>
        filter!(conditions, __) |>
        select(__, names(pred)) |>
        disallowmissing(__) |>
        relativechange!(__) |>
        continuous_time!(__)
end

function join_predictors!(epid, pred)
    @_ epid |>
        continuous_time!(__) |>
        disallowmissing!(__) |>
        leftjoin(__,
            continuous_time!(pred);
            on=:date) |>
        sort(__, :date) |>
        coalesce.(__, 0.) |>
        disallowmissing!(__)
end

function stepindex(n, j0 = 1; step = 1, kwargs...)
    index = Vector{Int64}(undef, n)
    j = j0
    for i in 1:n
        index[i] = j
        i % step == 0 && ( j += 1 )
    end
    index
end

function earlyphase!(df, startdate)
    j = findlast( <(startdate), df.date)
    !isnothing(j) && (df[1:j,:rw] .= 1)
end

function latephase!(df)
    i = findlast(x->!ismissing(x), df[!, :rw])
    df[i+1:end, :rw] .= df[i, :rw]
end

function last_step(df)
    i = findlast(x->!ismissing(x), df[!, :rw])
    isnothing(i) && return 1
    return df[i, :rw]
end

function step!(df, sdate, edate; kwargs...)
    i0 = last_step(df) + 1
    is = findfirst(==(sdate), df.date)
    ie = findfirst(==(edate), df.date)
    n  = ie - is + 1
    df[is:ie, :rw] = stepindex(n, i0; kwargs...)
end

function wait!(df, sdate, edate)
    i0 = last_step(df)
    is = findfirst(==(sdate), df.date)
    ie = findfirst(==(edate), df.date)
    df[is:ie, :rw] .= i0
end

function randomwalk!(data, pred; semiparam = true, startdate = nothing, enddate = nothing, kwargs...)
    rw = DataFrame(:date => data.date, :rw => Vector{Union{Missing, Int64}}(missing, size(data,1)))
    startdate = isnothing(startdate) ? rw.date[1] : startdate
    enddate   = isnothing(enddate) ? data.date[end] : enddate

    earlyphase!(rw, startdate)
    if semiparam
        s = pred.date[1]
        e = pred.date[end]
        step!(rw, startdate, s; kwargs...)
        wait!(rw, s + Day(1), e)
        step!(rw, e + Day(1), enddate; kwargs...)
    else
        step!(rw, startdate, enddate; kwargs...)
    end
    latephase!(rw)

    @_ leftjoin(data, rw; on=:date) |>
        sort(__, :date)# |>
        # disallowmissing!(__)
end

function load_data(region, epidata_kwargs, predictor_kwargs, rw_kwargs)
    pred = read_predictors(region ; predictor_kwargs... )
    epid = read_epidata(region; epidata_kwargs...)
    pred = clean_predictors(pred, epid; predictor_kwargs... )
    data = @_ join_predictors!(epid, pred) |>
                randomwalk!(__, pred; rw_kwargs... ) |>
                continuous_time!(__) |>
                disallowmissing!(__)
    data[!,:region] .= region
    data
end
## ===========================================================================
regions = ["France", "Hungary", "UK", "Sweden", "Denmark", "Germany", "Italy", "USA"]
params = (
      epidata_kwargs = Dict(
        :conditions => [:date => x -> x >= Date("2020-08-01")]
        )

    , predictors_kwargs = Dict(
        :fnames_survey => Dict(region => projectdir("data/contacts/int/smoothed_contact_rates_region=$(region).csv") for region in regions),
        :fnames_mobil  => Dict(region => projectdir("data/mobility/mobility_country=$(region).csv") for region in regions),
        :shift         => -1,
        :predictors    => [:strangers],#[:strangers, :google, :apple],
        :conditions    => [
            :cases_per_100000    => x -> x >
            20,
            :fraction_vaccinated => x -> x < 5/100
            ]
        )

    , rw_kwargs = Dict(
        :step      => 7,
        :startdate => nothing,
        :enddate   => nothing,
        :semiparam => true
        )
    )


function load_data(regions::AbstractVector, args...)
    data = load_data(regions[1], params...)
    for region in regions[2:end]
        @show region
        data = vcat(data,
            load_data(region, args...)
            )
    end
    data
end

load_data(regions, params...)
