"""
  Tucker decomposition of a N-mode tensor.
"""
immutable Tucker{T<:Number, N} <: TensorDecomposition{T, N}
    factors::NTuple{N, Matrix{T}} # factor matrices
    core::StridedArray{T, N}      # core tensor
    props::Dict{Symbol, Any}      # extra properties

    Tucker(factors::NTuple{N, Matrix{T}}, core::StridedArray{T, N}) =
        new(factors, core, Dict{Symbol, Any}())

    Base.call{T,N}(::Type{Tucker}, factors::NTuple{N, Matrix{T}}, core::StridedArray{T, N}) =
        Tucker{T,N}(factors, core)
end

"""
Returns the core tensor of Tucker decomposition.
"""
core(decomp::Tucker) = decomp.core

"""
Returns the factor matrices of Tucker decomposition.
"""
factors(decomp::Tucker) = decomp.factors

"""
Composes a full tensor from Tucker decomposition.
"""
compose(decomp::Tucker) = tensocontractmatrices(core(decomp), factors(decomp), transpose=true)

compose!{T,N}(dest::Array{T,N}, decomp::Tucker{T,N}) = tensorcontractmatrices!(dest, core(decomp), factors(decomp), transpose=true)
