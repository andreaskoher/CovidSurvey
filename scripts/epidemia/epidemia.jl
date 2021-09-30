using DrWatson
quickactivate(@__DIR__)
@show projectdir()
##
using RCall
using CSV
using BSON
using DataFrames
using Dates
using Plots
using Underscores
using StatsBase
using ColorSchemes
plotlyjs()
##

contacts = BSON.load( projectdir("data/contacts/int/mean_contact_rates_aggregation=14.bson") )
 = contacts["Denmark"].date
diff(rw_step)|>unique
