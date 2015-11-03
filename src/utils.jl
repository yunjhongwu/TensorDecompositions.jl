# utilities

function _random_init(d::Tuple, r::Integer)
    return [randn(i, r)::Matrix{Float64} for i in d]
end

function _KhatriRao(A::Matrix{Float64}, B::Matrix{Float64})
    size(A, 2) == size(B, 2) || error("Input matrices should have the same number of columns.")
    return [[kron(A[:, i], B[:, i])' for i in 1:size(A, 2)]...]'
end

"""
Unfolds the tensor into matrix, such that the specified
group of modes becomes matrix rows and the other one becomes columns.

  * `row_modes` vector of modes to be unfolded as rows
  * `col_modes` vector of modes to be unfolded as columns
"""
function _unfold{T,N}(tnsr::StridedArray{T,N}, row_modes::Vector{Int}, col_modes::Vector{Int})
    length(row_modes) + length(col_modes) == N ||
        throw(ArgumentError("column and row modes should be disjoint subsets of 1:N"))

    dims = size(tnsr)
    return reshape(permutedims(tnsr, [row_modes; col_modes]),
                   prod(dims[row_modes]), prod(dims[col_modes]))
end

"""
Unfolds the tensor into matrix such that the specified mode becomes matrix row.
"""
_row_unfold{T,N}(tnsr::StridedArray{T,N}, mode::Integer) = _unfold(tnsr, [mode], [1:mode-1; mode+1:N])

"""
Unfolds the tensor into matrix such that the specified mode becomes matrix column.
"""
_col_unfold{T,N}(tnsr::StridedArray{T,N}, mode::Integer) = _unfold(tnsr, [1:mode-1; mode+1:N], [mode])

function _iter_status(converged::Bool, niters::Integer, maxiter::Integer)
    println(converged ? string("Algorithm converged after ", string(niters)::ASCIIString, " iterations.") :
                        string("Warning: Maximum number (", string(maxiter)::ASCIIString, ") of iterations exceeded."))
end


function _check_sign(v::StridedVector)
    return sign(v[findmax(abs(v))[2]]) * v
end

function _check_tensor(T::StridedArray, r::Integer)
    num_modes = ndims(T)
    num_modes > 2 || error("This method does not support scalars, vectors, or matrices input.")
    r <= minimum(size(T)) && r > 0 || error("r should satisfy 1 <= r <= minimum(size(T)).")
    isreal(T) || error("This package currently only supports real-number-valued tensors.")
    return num_modes
end
