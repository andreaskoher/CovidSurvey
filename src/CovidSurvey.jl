module CovidSurvey

using DrWatson
using MCMCChains
using Distributions
using Plots
using StatsBase
using Dates
using DistributionsAD
using Random: AbstractRNG
import Bijectors

using DataFrames
using Base:@kwdef

abstract type ObservationInit end
abstract type ObservationsModel end

export
    National,
    Regional,
    LogNormalMeanStd,
    NegativeBinomial2,
    NegativeBinomial3,
    KLogistic,
    KLogit,
    InverseGamma2,
    GammaMeanCv,
    GammaMeanStd,
    vectup2tupvec,
    arrarrarr2arr,
    plot_confidence_timeseries,
    plot_confidence_timeseries!,
    hpdi

include("io.jl")
include("utils.jl")           # <= stuff that might also be included by sub-modules
include("visualizations.jl")   # <= visualization stuff
include("data.jl")

# Simulations on a national level
include("national/National.jl")
include("regional/Regional.jl")

__precompile__(true)
end # module
