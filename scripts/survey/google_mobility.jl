using DrWatson
quickactivate(@__DIR__)
@show projectdir()
##
using StatsBase
using StatsPlots
using DataFrames
using CSV
using CovidSurvey
##
const mobility_names = [
    "retail_and_recreation_percent_change_from_baseline",
    "grocery_and_pharmacy_percent_change_from_baseline",
    "workplaces_percent_change_from_baseline",
    "residential_percent_change_from_baseline"
]

const new_mobility_names = [
    "retail",
    "grocery",
    "workplaces",
    "residential"
]
## ============================================================================
#                    national
#  ============================================================================
fname = "/home/and/data/google-mobility/Global_Mobility_Report.csv"
mobil_raw = CSV.File(fname) |> DataFrame

function selectdenmark(mobil_raw)
    mobil_raw = filter(:country_region => ==("Denmark"), mobil_raw)
    mobil_dk = filter(:sub_region_1 => ismissing, mobil_raw)
    mobil_dk[!, "sub_region_1"] = replace(mobil_dk.sub_region_1,
        "Capital Region of Denmark" => "capital"
        ,"Central Denmark Region" => "central"
        ,"North Denmark Region" => "north"
        ,"Region of Southern Denmark" => "south"
        ,"Region Zealand" => "zealand"
    )
    @assert issorted(mobil_dk.date)
    mobil_dk = select(mobil_dk, :date, mobility_names .=> new_mobility_names)
    mobil_dk[!,:nonresidential] = [ mean(r) for r in eachrow( select(mobil_dk, Not([:date, :residential])) ) ]
    disallowmissing!(mobil_dk)
    return mobil_dk
end
mobil_dk = selectdenmark(mobil_raw)
CSV.write(projectdir("data/google/", "mobility_denmark.csv"), mobil_dk)





## ============================================================================
#                    regional
#  ============================================================================
# select all regions
fname = "/home/and/data/google-mobility/Global_Mobility_Report.csv"
mobil_raw = CSV.File(fname) |> DataFrame
function selectregions(mobil_raw)
    mobil_raw = filter(:country_region => ==("Denmark"), mobil_raw)
    mobil_regions = filter(:sub_region_1 => x->!ismissing(x), mobil_raw)
    mobil_regions[!, "sub_region_1"] = replace(mobil_regions.sub_region_1,
        "Capital Region of Denmark" => "capital"
        ,"Central Denmark Region" => "central"
        ,"North Denmark Region" => "north"
        ,"Region of Southern Denmark" => "south"
        ,"Region Zealand" => "zealand"
    )
    return mobil_regions
end
mobil_regions = selectregions(mobil_raw)
##
function selectregion(mobil_regions, region)
    mobil_region = filter(:sub_region_1 => ==(region), mobil_regions)
    mobil_region = filter(:sub_region_2 => ismissing, mobil_region)
    @assert issorted(mobil_region.date)
    mobil_region = select(mobil_region, :date, mobility_names .=> new_mobility_names)
    mobil_region[!,:nonresidential] = [ mean(r) for r in eachrow( select(mobil_region, Not([:date, :residential])) ) ]
    disallowmissing!(mobil_region)
    return mobil_region
end

for region in Regional.regions
    mobil_region = selectregion(mobil_regions, region)
    CSV.write(projectdir("data/google/", "mobility_region=$region.csv"), mobil_region)
end


































##           OLD
# function select_region(df, r)
#     mask = df.sub_region_1 .== r
#     mask = DataFrames.coalesce.(mask, Ref(false))
#     df = df[mask, :]
#     df = df[.! ismissing.(df.sub_region_2), :]
#     df.sub_region_2 = standardize_name.(df.sub_region_2)
#
#     # @assert any(.! ismissing.(Array(mobil[:,mobility_names])))
#     # for n in mobility_names
#     #     df[!,n] = convert(Vector{Float64}, df[:,n])
#     # end
#     return df
# end
#
# function weightedmean(xs, ws)
#     nom = 0.
#     denom = 0.
#     for (x,w) in zip(xs, ws)
#         if !ismissing(x)
#             nom += x*w
#             denom += w
#         end
#     end
#     return nom/denom
# end
#
# function cleanup(mobil)
#     gp = groupby(mobil, "date", sort=true)
#     mobil = combine(gp, [[n,"population"] for n in mobility_names] .=> weightedmean, renamecols=false)
#     mobil[!,"nonresidential"] = mean(eachcol(select(mobil, Not(["residential_percent_change_from_baseline_population", "date"]))))
#     rename!(mobil, [n*"_population" for n in mobility_names] .=> new_mobility_names)
#     @assert issorted(mobil.date)
#     return mobil
# end
#
# function readpopulation()
#     fname = projectdir("data/population/Denmark_2021Q2.csv")
#     pop = CSV.File(fname; header=false, select=[4,5]) |> DataFrame
#     rename!(pop, "Column4"=>"name", "Column5"=>"count")
#     pop.name = standardize_name.(pop.name)
#     pop
# end
#
# const population = readpopulation()
#
# function get_population(s)
#     if s == "Brondby"
#         s = "Br_ndby"
#     end
#     i = findfirst(x->contains(x,s), population.name)
#     isnothing(i) && (@error s; return missing)
#     return population.count[i]
# end
#
# function replace_special_characters(s)
#     s = replace(s, '�'=>'_')
#     s = replace(s, 'ø'=>'_')
#     s = replace(s, 'æ'=>'_')
#     s = replace(s, 'å'=>'_')
#     s = replace(s, '\ue6'=>'_')
#     s = replace(s, '\uf8'=>'_')
#     s = replace(s, '\ue5'=>'_')
#     return s
# end
#
# function standardize_name(s_raw)
#     s = if split(s_raw, ' ')|>last == "Municipality"
#             first(split(s_raw, ' '))
#         else
#             s_raw
#         end
#     replace_special_characters(s)
# end
#
# for region in Regional.regions
#     mobil = select_region(mobil_raw, region)
#     mobil[!,:population] = get_population.(mobil.sub_region_2)
#     @assert any(.! ismissing.(mobil.population))
#
#     mobil = cleanup(mobil)
#     CSV.write(projectdir("data/google/", "mobility_region=$region.csv"), mobil)
# end
