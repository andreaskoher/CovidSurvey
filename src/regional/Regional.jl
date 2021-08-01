module Regional

using DrWatson
using Turing
using StatsBase
using MCMCChains: Chains, meanplot, gelmandiag
using PrettyTables
using BSON
# using Parameters
# using ArgCheck
# using RData
using CSV
using DataFrames
using OrderedCollections
using Dates
using StatsBase
using StatsFuns
using Base:@kwdef
using Base.Threads# using FillArrays
# using KernelFunctions
using Random
using StatsPlots
using LaTeXStrings

# using QuadGK
# using

abstract type ObservationInit end
abstract type ObservationsModel end

using ..CovidSurvey#: NegativeBinomial2

include("models.jl")
include("utils.jl")
include("data.jl")
include("visualization.jl")

# const model = model_v1 # <= defines the "official" model for this sub-module

end
