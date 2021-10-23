#!/usr/bin/env julia

using DrWatson
quickactivate(@__DIR__)
@show projectdir()
#-----------------------------------------------------------------------------
# load packages
using CovidSurvey
using ArgParse
using Plots
using Dates

# using StatsPlots
using PrettyTables
using DataFrames
using BSON

using Turing
using Memoization
using ReverseDiff
using InlineStrings
using WeakRefStrings
Turing.setrdcache(true)
setadbackend(:reversediff)

plotlyjs()
#-----------------------------------------------------------------------------
# read input
argtable = ArgParseSettings(
    description="This script applies post-processing to Turing.Chains."
)
@add_arg_table! argtable begin
    "--fname", "-f"
        help = "file path to Turing.Chains"
    "--exclude", "-e"
        help = "exclude chains by id. Expected format: 1,2,3 "
        arg_type = String
        default = ""
    "--plot-results", "-p"
        help = "plot and display with firefox"
        arg_type = Bool
        default = false
    "--warmup", "-w"
        help = "set nr. of warmup samples"
        arg_type = Int64
        default = nothing
end

parsed_args = parse_args(ARGS, argtable)
exclude = isempty(parsed_args["exclude"]) ? [] : [parse(Int64,i) for i in split(parsed_args["exclude"], ",")]
#------------------------------------------------------------------------------
Regional.postprocessing(
    parsed_args["fname"];
    plot_results = parsed_args["plot-results"],
    exclude,
    warmup = parsed_args["warmup"]
)

fname = "/home/ankoh/dev/CovidSurvey/reports/regional_effect_estimation/mobility/apple_google_totalcontacts/2021-10-06/CHAIN_chains=10_model=hospit_observ=2021-02-01_preds=CS,MG,MA_regional=true_rwstep=7_semipar=true_std=false_steps=1000_warmup=2000.jls"
isfile(fname)
Regional.postprocessing(
    fname;
    plot_results = false,
    exclude = [],
)
