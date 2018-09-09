using SparseArrays

"""
Sparse N-mode array.

`vals` contains all (`L`) non-zero elements.
`pos` is `N`x`L` matrix of non-zero elements indices.
`dims` is `NTuple` of array dimensions.
"""
struct SparseArray{T, N} <: AbstractArray{T, N}
    vals::Vector{T}
    pos::Vector{CartesianIndex{N}}
    dims::NTuple{N,Int}
end

Base.size(a::SparseArray) = a.dims
Base.size(a::SparseArray, i::Int) = a.dims[i]
Base.length(a::SparseArray) = prod(a.dims)
SparseArrays.nnz(a::SparseArray) = length(a.vals)

"""
Convert dense N-dimensional array into N-dimensional a `SparseArray`.
"""
function SparseArray{T,N}(arr::DenseArray{T,N}) where {T,N}
    pos = Vector{CartesianIndex{N}}()
    vals = Vector{T}()

    @inbounds for (i, cart) in enumerate(CartesianIndices(arr))
        (arr[i] == 0) && continue
        push!(vals, arr[i])
        push!(pos, cart)
    end

    return SparseArray{T,N}(vals, pos, size(arr))
end

SparseArray(arr::DenseArray{T, N}) where {T, N} = SparseArray{T,N}(arr)

"""
SS-HOPM (Shifted Symmetric Higher-order Power Method)
for computing tensor eigenpars.
"""
function sshopm(tnsr::AbstractArray{T,N},
                alpha::Real;
                tol::Float64=1e-5,
                maxiter::Int=1000,
                verbose::Bool=false) where {T,N}

    r = size(tnsr, 1)
    all(isequal(r), size(tnsr)) || throw(DimensionMismatch("Input tensor should have the same dimension in all modes, got $(size(tnsr))."))
    x = randn(r)
    x .*= 1/norm(x)
    x_old = similar(x)
    converged = false
    niters = 0
    while !converged && niters < maxiter
        copyto!(x_old, x)
        x .= flipsign.(nmul(tnsr, x) .+ alpha .* x, alpha)
        x .*= 1/norm(x)
        converged = norm(x .- x_old) < tol
        niters += 1
    end

    verbose && _iter_status(converged, niters, maxiter)
    return (dot(x, nmul(tnsr, x)), flipsign.(x, alpha))
end

# (N-1)-way tensor × vector contraction
function nmul(tnsr::StridedArray{T,N}, x::Vector{T}) where {T, N}
    v = copy(tnsr)
    for i in 2:N
        v = tensorcontract(v, collect(i-1:N), x, N)
    end
    return v
end

# (N-1)-way sparse tensor × vector contraction
function nmul(tnsr::SparseArray{T,N}, x::Vector{T}) where {T, N}
    v = zeros(T, size(tnsr, 1))
    @inbounds for i in 1:nnz(tnsr)
        posi = tnsr.pos[i]
        v[posi[1]] += tnsr.vals[i] * prod(ntuple(k -> x[posi[k+1]], N-1))
    end
    return v
end
