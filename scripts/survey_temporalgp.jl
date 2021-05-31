using DrWatson
quickactivate(@__DIR__)
@show projectdir()
##
using CSV
using DataFrames
# using DataFramesMeta
using Dates
using StatsBase
using Plots
using CovidSurvey

using Stheno, KernelFunctions
using AbstractGPs
using TemporalGPs
using StaticArrays

# Load standard packages from the Julia ecosystem
using Optim # Standard optimisation algorithms.
using ParameterHandling # Helper functionality for dealing with model parameters.
using Zygote # Algorithmic Differentiation

using ParameterHandling: value, flatten
plotlyjs()
#----------------------------------------------------------------------------
# survey data
survey = National.readsurveys()
Rt = load(projectdir("out", "Rt.csv"))|>DataFrame
plot(t, Rt.mq, ribbon=(Rt.mq - Rt.llq, Rt.uuq - Rt.mq), linewidth=0, label="95% quantiles")
plot!(t, Rt.mq, ribbon=(Rt.mq - Rt.lq, Rt.uq - Rt.mq), linewidth=2, label="50% quantiles", hover=mq)
plot!(survey.date, survey.colleagues, ticks=:native, label="colleagues")
plot!(survey.date, survey.family, ticks=:native, label="family")
plot!(survey.date, survey.strangers, ticks=:native, label="strangers")
plot!(survey.date, survey.friends, ticks=:native, label="friends")
##
d = survey.friends
y = zscore(d)
x = RegularSpacing(0.0, 1., length(survey.date))
# x = GPPPInput(:f3, collect(1:size(survey,1)))
# Declare model parameters using `ParameterHandling.jl` types.
params = (
    s1 = positive(1.),
    l1 = positive(50.),
    s2 = positive(1.),
    l2 = positive(10.),
    s3 = positive(1.),
    l3 = positive(2.),
    v  = positive(0.1),
)

# function build_gp(ps)
#     f = @gppp let
#         f1 = ps.s1 * stretch(GP(Matern52Kernel()), 1 / ps.l1)
#         f2 = ps.s2 * stretch(GP(SEKernel()), 1 / ps.l2)
#         f3 = f1 + f2
#     end
# end
function build_kernels(ps)
    k1 = ps.s1 * KernelFunctions.transform(Matern52Kernel(), 1/ps.l1)
    k2 = ps.s2 * KernelFunctions.transform(Matern52Kernel(), 1/ps.l2)
    k3 = ps.s3 * KernelFunctions.transform(Matern32Kernel(), 1/ps.l3)
    return k1, k2, k3
end
function build_gp(ps)
    f_naive = GP( sum( build_kernels(ps) ) )
    return to_sde(f_naive, SArrayStorage(Float64))
end

# Construct mapping between structured and Vector representation of parameters.
flat_initial_params, unflatten = flatten(params)

# Specify an objective function for Optim to minimise in terms of x and y.
# We choose the usual negative log marginal likelihood (NLML).
function objective(flat_params)
    params = value(unflatten(flat_params))
    f = build_gp(params)
    return -logpdf(f(x, params.v), y)
end

# Check that the objective function works:
objective(flat_initial_params)

# Optimise using Optim. This optimiser often works fairly well in practice,
# but it's not going to be the best choice in all situations. Consult
# Optim.jl for more info on available optimisers and their properties.
training_results = Optim.optimize(
    objective,
    θ -> only(Zygote.gradient(objective, θ)),
    flat_initial_params, # Add some noise to make learning non-trivial
    BFGS(
        alphaguess = Optim.LineSearches.InitialStatic(scaled=true),
        linesearch = Optim.LineSearches.BackTracking(),
    ),
    Optim.Options(show_trace = true);
    inplace=false,
)

# Extracting the final values of the parameters.
# Should be close to truth.
flat_final_params = training_results.minimizer
final_params = value(unflatten(flat_final_params))
# f = Stheno.wrap(GP( build_kernel(final_params) ), GPC())
f = build_gp(final_params)

objective(flat_final_params)
fp = posterior(f(x, final_params.v),y)

# plot!(fx(x, final_params.var_noise))
ms = marginals(fp(x));
p = plot(x,y)
plot!(
    x, mean.(ms);
    ribbon=3std.(ms), label="", color=:green, fillalpha=0.2, linewidth=2,
)

function build_marginals(ps)
    build_posterior(f) = posterior(f(x, ps.v),y)
    fs = [GP(k) for k in build_kernels(ps)]
    fps = [build_posterior(f) for f in fs]
    [marginals(fp(x)) for fp in fps]
end

ms = build_marginals(final_params)
p = plot(x,y)
for m in ms
    plot!(p, x, mean.(m))
end
display(p)
