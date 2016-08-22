"""
  Tucker decomposition of a N-mode tensor.
"""
immutable Tucker{T<:Number, N} <: TensorDecomposition{T, N}
    factors::NTuple{N, Matrix{T}} # factor matrices
    core::StridedArray{T, N}      # core tensor
    props::Dict{Symbol, Any}      # extra properties

    Tucker(factors::NTuple{N, Matrix{T}}, core::StridedArray{T, N}) =
        new(factors, core, Dict{Symbol, Any}())

    (::Type{Tucker}){T, N}(factors::NTuple{N, Matrix{T}}, core::StridedArray{T, N}) =
        Tucker{T, N}(factors, core)
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
compose(decomp::Tucker) = tensorcontractmatrices(core(decomp), factors(decomp), transpose=true)

compose!{T,N}(dest::Array{T,N}, decomp::Tucker{T,N}) = tensorcontractmatrices!(dest, core(decomp), factors(decomp), transpose=true)

"""
Scale the factors and core of the initial decomposition.
Each decompositon component is scaled proportional to the number of its elements.
After scaling, `|decomp|=s`
"""
function rescale!{T,N}(decomp::Tucker{T,N}, s::T)
    total_length = length(decomp.core) + sum(map(length, decomp.factors)) # total elements in the decomposition
    for F in decomp.factors
        f_s = s^(length(F)/total_length)/vecnorm(F)
        map!(x -> x*f_s, F)
    end
    core_s = s^(length(decomp.core)/total_length)/vecnorm(decomp.core)
    map!(x -> x*core_s, decomp.core)
    return decomp
end
