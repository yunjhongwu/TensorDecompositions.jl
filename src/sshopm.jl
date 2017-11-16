"""
Sparse N-mode array.

`vals` contains all (`L`) non-zero elements.
`pos` is `N`x`L` matrix of non-zero elements indices.
`dims` is `NTuple` of array dimensions.
"""
struct SparseArray{T, N} <: AbstractArray{T, N}
    vals::Vector{T}
    pos::Matrix{Int}
    dims::NTuple{N,Int}
end

Base.size(a::SparseArray) = a.dims
Base.size(a::SparseArray, i::Int) = a.dims[i]
Base.nnz(a::SparseArray) = length(a.vals)
Base.length(a::SparseArray) = prod(a.dims)

"""
Convert dense N-dimensional array into N-dimensional a `SparseArray`.
"""
function Base.convert{T,N}(::Type{SparseArray}, arr::DenseArray{T,N})
    pos = Int64[]
    vals = Float64[]

    for i in eachindex(arr)
        if arr[i] != 0
            push!(vals, arr[i])
            for j in ind2sub(arr, i)
                push!(pos, j)
            end
        end
    end

    SparseArray{T,N}(vals, reshape(pos, N, length(vals)), size(arr))
end

"""
SS-HOPM (Shifted Symmetric Higher-order Power Method)
for computing tensor eigenpars.
"""
function sshopm{T,N}(tnsr::AbstractArray{T,N},
                alpha::Real;
                tol::Float64=1e-5,
                maxiter::Int=1000,
                verbose::Bool=false)

    r = size(tnsr, 1)
    all(d -> d==r, size(tnsr)) || throw(ArgumentError("Input tensor should have the same dimension in all modes, got $(size(tnsr))."))
    x = randn(r)
    x .*= 1/vecnorm(x)
    x_old = similar(x)
    converged = false
    niters = 0
    while !converged && niters < maxiter
        copy!(x_old, x)
        x = flipsign.(A_mul_B(tnsr, x) + alpha * x, alpha)
        x *= 1/vecnorm(x)
        converged = vecnorm(x - x_old) < tol
        niters += 1
    end

    verbose && _iter_status(converged, niters, maxiter)
    return (dot(x, A_mul_B(tnsr, x)), flipsign.(x, alpha))
end

function A_mul_B{T,N}(tnsr::StridedArray{T,N}, x::Vector{T})
    v = copy(tnsr)
    for i in 2:N
        v = tensorcontract(v, collect(i-1:N), x, N)
    end
    return v
end

function A_mul_B{T,N}(tnsr::SparseArray{T,N}, x::Vector{T})
    v = zeros(T, size(tnsr, 1))
    for i in 1:nnz(tnsr)
        v[tnsr.pos[1, i]] += tnsr.vals[i] * prod(x[tnsr.pos[2:N, i]])
    end
    return v
end
