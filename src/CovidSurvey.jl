module CovidSurvey

using DrWatson
using MCMCChains
using Distributions
using Plots
using StatsBase



export
    National,
    Regional,
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
    plot_confidence_timeseries!

include("io.jl")
include("utils.jl")           # <= stuff that might also be included by sub-modules
include("visualizations.jl")   # <= visualization stuff

# Simulations on a national level
include("national/National.jl")
include("regional/Regional.jl")

__precompile__(true)
end # module
