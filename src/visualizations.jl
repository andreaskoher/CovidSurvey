"""
    plot_confidence_timeseries!(p::Plot, data::AbstractMatrix{<:Real}; label="", kwargs...)

Plots confidence intervals for the time-series represented by `data`.
Assumes each row corresponds to the samples for a single time-step.
"""
function plot_confidence_timeseries!(p::Plots.AbstractPlot, data::AbstractMatrix; no_label = false, label="", kwargs...)
    intervals = [0.025, 0.25, 0.5, 0.75, 0.975]

    qs = [quantile(v, intervals) for v in eachrow(data)]
    llq, lq, mq, uq, uuq = (eachrow(hcat(qs...))..., )
    plot!(mq, ribbon=(mq - llq, uuq - mq), linewidth=0, label=(no_label ? "" : "$(label) (95% quantiles)"), kwargs...)
    plot!(mq, ribbon=(mq - lq, uq - mq), linewidth=2, label=(no_label ? "" : "$(label) (50% quantiles)"), kwargs...)

    return p
end

"""
    plot_confidence_timeseries!(p::Plot, time::AbstractVector, data::AbstractMatrix{<:Real}; label="", kwargs...)

Plots confidence intervals for the time-series represented by `data` and a vector of Dates.
Assumes each row corresponds to the samples for a single time-step.
"""
function plot_confidence_timeseries!(p::Plots.AbstractPlot, time::AbstractVector{<:Dates.TimeType}, data::AbstractMatrix; no_label = false, label="", c=:peru, kwargs...)
    intervals = [0.025, 0.25, 0.5, 0.75, 0.975]

    qs = [quantile(v, intervals) for v in eachrow(data)]
    llq, lq, mq, uq, uuq = (eachrow(hcat(qs...))..., )
    strtime = time .|> Symbol .|> String
    #plot!(time, mq, ribbon=(mq - llq, uuq - mq), c=c, α=0.5, linewidth=0, label=(no_label ? "" : "$(label) (95% quantiles)"), hover=strtime, kwargs...)
    plot!(time, mq, ribbon=(mq - lq, uq - mq), c=c, α=0.5, linewidth=2, label=(no_label ? "" : "$(label) (50% quantiles)"), hover=mq, kwargs...)

    return p
end

"""
    plot_confidence_timeseries(data::AbstractMatrix{<:Real}; label="", kwargs...)

See `plot_confidence_timeseries!`.
"""
plot_confidence_timeseries(data::AbstractVector; kwargs...) = plot_confidence_timeseries(hcat(data...) ; kwargs...)
plot_confidence_timeseries(data::AbstractMatrix; kwargs...) = plot_confidence_timeseries!(plot(), data; kwargs...)

"""
    plot_confidence_timeseries(data::AbstractMatrix{<:Real}; label="", kwargs...)

See `plot_confidence_timeseries!`.
"""
plot_confidence_timeseries(
    time::AbstractVector{<:Dates.TimeType},
    data::AbstractVector;
    kwargs...
) = plot_confidence_timeseries!(plot(), time, hcat(data...); kwargs...)
plot_confidence_timeseries!(
    p   ::Plots.AbstractPlot,
    time::AbstractVector{<:Dates.TimeType},
    data::AbstractVector;
    kwargs...
) = plot_confidence_timeseries!(p, time, hcat(data...); kwargs...)
plot_confidence_timeseries(
      time::AbstractVector{<:Dates.TimeType}
    , data::AbstractMatrix; kwargs...
) = plot_confidence_timeseries!(plot(), time, data; kwargs...)
