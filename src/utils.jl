# utilities

function tensorcontractmatrix(src::StridedArray{T,N}, mtx::StridedMatrix{T}, n::Int;
                              transpose::Bool=false, method::Symbol=:BLAS) where {T,N}
    #@info "TTM: src=$(size(src)) mtx=$(size(mtx)) n=$n transpose=$transpose method=$method"
    tensorcontract(src, 1:N,
                   mtx, [transpose ? N+1 : n, transpose ? n : N+1],
                   [1:(n-1); N+1; (n+1):N], method=method)
end

function tensorcontractmatrix!(dest::StridedArray{T,N}, src::StridedArray{T,N},
                               mtx::StridedMatrix{T}, n::Int;
                               transpose::Bool=false, method::Symbol=:BLAS) where {T,N}
    #@info "TTM: dest=$(size(dest)) src=$(size(src)) mtx=$(size(mtx)) n=$n transpose=$transpose method=$method"
    tensorcontract!(1, src, 1:N, 'N',
                    mtx, [transpose ? N+1 : n, transpose ? n : N+1], 'N',
                    0, dest, [1:(n-1); N+1; (n+1):N], method=method)
end

"""
Contract N-mode tensor and M matrices.

  * `dest` array to hold the result
  * `src`  source tensor to contract
  * `matrices` matrices to contract
  * `modes` corresponding modes of matrices to contract
  * `transpose` if true, matrices are contracted along their columns
"""
function tensorcontractmatrices!(dest::Array{T,N}, src::Array{T,N}, matrices::Any,
                                 modes::Any = 1:length(matrices);
                                 transpose::Bool=false, method::Symbol=:BLAS) where {T,N}
    length(matrices) == length(modes) ||
        throw(ArgumentError("The number of matrices doesn't match the length of mode sequence"))
    for i in 1:length(matrices)-1
        src = tensorcontractmatrix(src, matrices[i], modes[i],
                                   transpose=transpose, method=method)
    end
    tensorcontractmatrix!(dest, src, matrices[end], modes[end],
                          transpose=transpose, method=method)
end

"""
Contract N-mode tensor and M matrices.

  * `tensor` tensor to contract
  * `matrices` matrices to contract
  * `modes` corresponding modes of matrices to contract
  * `transpose` if true, matrices are contracted along their columns
"""
function tensorcontractmatrices(tensor::StridedArray, matrices::Any,
                                modes::Any = 1:length(matrices);
                                transpose::Bool=false, method::Symbol=:BLAS)
    length(matrices) == length(modes) ||
        throw(ArgumentError("The number of matrices doesn't match the length of mode sequence"))
    res = tensor
    for i in eachindex(matrices)
        res = tensorcontractmatrix(res, matrices[i], modes[i],
                                   transpose=transpose, method=method)
    end
    return res
end

"""
Generates random factor matrices for Tucker/CANDECOMP etc decompositions.

  * `orig_dims` original tensor dimensions
  * `core_dims` core tensor dimensions

Returns:
  * a vector of `N` (orig[n], core[n])-sized matrices
"""
_random_factors(orig_dims::NTuple{N, Int}, core_dims::NTuple{N, Int}) where {N} =
    Matrix{Float64}[randn(o_dim, c_dim) for (o_dim, c_dim) in zip(orig_dims, core_dims)]

"""
Generates random factor matrices for Tucker/CANDECOMP decompositions if core tensor is `r^N` hypercube.

Returns:
  * a vector of `N` (orig[n], r)-sized matrices
"""
_random_factors(dims::NTuple{N, Int}, r::Integer) where {N} =
    _random_factors(dims, ntuple(_ -> r, N))

"""
    khatrirao!(dest::AbstractMatrix{T}, mtxs::NTuple{N, <:AbstractMatrix{T}})

In-place Khatri-Rao matrices product (column-wise Kronecker product) calculation.
"""
@generated function khatrirao!(dest::AbstractMatrix{T},
                               mtxs::NTuple{N, <:AbstractMatrix{T}}) where {N, T}
    (N === 1) && return quote
        size(dest) == size(mtxs[1]) ||
            throw(DimensionMismatch("Output and single input matrix have different sizes ($(size(dest)) and $(size(mtxs[1])))"))
        return copyto!(dest, mtxs[1])
    end
    # generate the code for looping over the matrices 2:N
    _innerloop = Base.Cartesian.lreplace(:(desti[offsj_k + 1] = destij_k), :k, N)
    for k in N:-1:2
        _innerloop = Base.Cartesian.lreplace(quote
            for j_k in axes(mtxs[k], 1)
                destij_k = destij_{k-1}*coli_k[j_k]
                offsj_k = offsj_{k-1}*size(mtxs[k], 1) + j_k - 1
                $_innerloop
            end
        end, :k, k)
    end
    # main code
    quote
    # dimensions check
    ncols = size(dest, 2)
    for i in 1:length(mtxs)
        (size(mtxs[i], 2) == ncols) ||
            throw(DimensionMismatch("Output matrix and input matrix #$i have different number of columns ($ncols and $(size(mtxs[i], 2)))"))
    end
    nrows = prod(@ntuple($N, i -> size(mtxs[i], 1)))
    size(dest, 1) == nrows ||
        throw(DimensionMismatch("Output matrix rows and the expected number of rows do not match ($(size(dest, 1)) and $nrows)"))
    # multiplication
    @inbounds for i in axes(dest, 2)
        @nexprs($N, k -> (coli_k = view(mtxs[k], :, i)))
        desti = view(dest, :, i)
        for j_1 in axes(mtxs[1], 1)
            destij_1 = coli_1[j_1]
            offsj_1 = j_1 - 1
            $_innerloop
        end
    end
    return dest
    end
end

khatrirao!(dest::AbstractMatrix, mtxs::AbstractVector) =
    khatrirao!(dest, tuple(mtxs...))
khatrirao!(dest::AbstractMatrix, mtxs...) = khatrirao!(dest, tuple(mtxs...))

"""
    khatrirao(mtxs::NTuple{N, <:AbstractMatrix{T}})

Calculates Khatri-Rao product of a sequence of matrices (column-wise Kronecker product).
"""
function khatrirao(mtxs::NTuple{N, <:AbstractMatrix{T}}) where {N, T}
    (N === 1) && return copy(first(mtxs))
    ncols = size(first(mtxs), 2)
    for i in 2:length(mtxs)
        (size(mtxs[i], 2) == ncols) ||
            throw(DimensionMismatch("Input matrices have different number of columns ($ncols and $(size(mtxs[i], 2)))"))
    end
    return khatrirao!(Matrix{T}(undef, prod(ntuple(i -> size(mtxs[i], 1), N)), ncols), mtxs)
end

khatrirao(mtxs::AbstractVector) = khatrirao(tuple(mtxs...))
khatrirao(mtxs...) = khatrirao(tuple(mtxs...))

"""
Unfolds the tensor into matrix, such that the specified
group of modes becomes matrix rows and the other one becomes columns.

  * `row_modes` vector of modes to be unfolded as rows
  * `col_modes` vector of modes to be unfolded as columns
"""
function _unfold(tnsr::StridedArray, row_modes::Vector{Int}, col_modes::Vector{Int})
    length(row_modes) + length(col_modes) == ndims(tnsr) ||
        throw(ArgumentError("column and row modes should be disjoint subsets of 1:$(ndims(tnsr))"))

    dims = size(tnsr)
    return reshape(permutedims(tnsr, [row_modes; col_modes]),
                   prod(dims[row_modes]), prod(dims[col_modes]))
end

"""
Unfolds the tensor into matrix such that the specified mode becomes matrix row.
"""
_row_unfold(tnsr::StridedArray, mode::Integer) =
    _unfold(tnsr, [mode], [1:mode-1; mode+1:ndims(tnsr)])

"""
Unfolds the tensor into matrix such that the specified mode becomes matrix column.
"""
_col_unfold(tnsr::StridedArray, mode::Integer) =
    _unfold(tnsr, [1:mode-1; mode+1:ndims(tnsr)], [mode])

function _iter_status(converged::Bool, niters::Integer, maxiter::Integer)
    converged ? @info("Algorithm converged after $(niters) iterations.") :
                @warn("Maximum number $(maxiter) of iterations exceeded.")
end

_check_sign(v::StridedVector) = sign(v[findmax(abs.(v))[2]]) * v

"""
Checks the validity of the core tensor dimensions.
"""
function _check_tensor(tnsr::StridedArray{T, N}, core_dims::NTuple{N, Int}) where {T<:Real,N}
    ndims(tnsr) > 2 || throw(ArgumentError("This method does not support scalars, vectors, or matrices input."))
    for i in 1:N
        0 < core_dims[i] <= size(tnsr, i) ||
            throw(ArgumentError("core_dims[$i]=$(core_dims[i]) given, 1 <= core_dims[$i] <= size(tensor, $i) = $(size(tnsr, i)) expected."))
    end
    #isreal(T) || throw(ArgumentError("This package currently only supports real-number-valued tensors."))
    return N
end

"""
Checks the validity of the core tensor dimensions, where core tensor is `r^N` hypercube.
"""
_check_tensor(tensor::StridedArray{<:Number, N}, r::Integer) where N =
    _check_tensor(tensor, ntuple(_ -> r, N))
