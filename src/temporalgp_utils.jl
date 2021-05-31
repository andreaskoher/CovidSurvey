using AbstractGPs, KernelFunctions
using TemporalGPs
using SpecialFunctions
using StaticArrays
using LinearAlgebra
using Zygote

#----------------------------------------------------------------------------
# periodic kernel
struct ApproxPeriodicKernel{J,T} <: KernelFunctions.SimpleKernel
    l::T
end

# assuming length( k.r ) == 1
ApproxPeriodicKernel(k::PeriodicKernel{T}; degree_of_approx = 3) where {T} = ApproxPeriodicKernel{degree_of_approx,T}( 4*only(k.r) )
ApproxPeriodicKernel(; r=1/4, degree_of_approx = 3 ) = ApproxPeriodicKernel(PeriodicKernel(; r=[r]); degree_of_approx=degree_of_approx)
# k = ApproxPeriodicKernel()

function TemporalGPs.to_sde(k::ApproxPeriodicKernel{J,T}, s::SArrayStorage{T}) where {J,T<:Real}
    D  = 2*(J+1)
    w0 = 2π
    F  = kron( Diagonal(0:J), [0 -w0; w0 0])
    H  = kron( ones(1, J+1),[1 0])
    Fs = SMatrix{D,D,T}(F)
    q  = T(0)
    Hs = SVector{D,T}(H)#SMatrix{1,D,T}(H)
    return Fs, q, Hs
end

function TemporalGPs.stationary_distribution(k::ApproxPeriodicKernel{J,T}, s::SArrayStorage{T}) where {J,T<:Real}
    D = 2*(J+1)
    q = [ j>0 ? 2*besseli(j, 1/k.l^2)*exp(-k.l^2) : besseli(0, 1/k.l^2)*exp(-k.l^2) for j in 0:J]
    P = kron(Diagonal(q), Diagonal([1, 1]))
    m = SVector{D, T}( zeros(T,D) )
    Ps = SMatrix{D,D,T}(P)
    return TemporalGPs.Gaussian(m, Ps)
end

Zygote.@adjoint function to_sde(k::ApproxPeriodicKernel, storage_type)
    return to_sde(k, storage_type), Δ->(nothing, nothing)
end

Zygote.@adjoint function TemporalGPs.stationary_distribution(k::ApproxPeriodicKernel, storage_type)
    return TemporalGPs.stationary_distribution(k, storage_type), Δ->(nothing, nothing)
end

#------------------------------------------------------------------------
# Kernel sum (deep)
KernelFunctions.KernelSum(kernel::Kernel) = kernel

function TemporalGPs.lgssm_components(k::KernelSum, ts::AbstractVector, storage_type::TemporalGPs.StorageType)
    As_l, as_l, Qs_l, emission_proj_l, x0_l = TemporalGPs.lgssm_components(k.kernels[1], ts, storage_type)
    As_r, as_r, Qs_r, emission_proj_r, x0_r = TemporalGPs.lgssm_components( KernelSum( k.kernels[2:end]... ), ts, storage_type)

    As = map(TemporalGPs.blk_diag, As_l, As_r)
    as = map(vcat, as_l, as_r)
    Qs = map(TemporalGPs.blk_diag, Qs_l, Qs_r)
    emission_projections = TemporalGPs._sum_emission_projections(emission_proj_l, emission_proj_r)
    x0 = TemporalGPs.Gaussian(vcat(x0_l.m, x0_r.m), TemporalGPs.blk_diag(x0_l.P, x0_r.P))

    return As, as, Qs, emission_projections, x0
end
