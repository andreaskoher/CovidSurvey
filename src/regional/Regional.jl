module Regional

using DrWatson
using Turing
using StatsBase
# using ArgCheck
# using RData
# using DrWatson
using CSV
using DataFrames
using OrderedCollections
using Dates
using StatsBase
using StatsFuns
using Base.Threads# using FillArrays
# using KernelFunctions
using Random
using StatsPlots
using LaTeXStrings

# using QuadGK
# using

using ..CovidSurvey#: NegativeBinomial2

include("models.jl")
include("utils.jl")
include("data.jl")
include("visualization.jl")

# const model = model_v1 # <= defines the "official" model for this sub-module

end
