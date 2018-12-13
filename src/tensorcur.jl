"""
CUR (columns-U-rows) 3-tensor decomposition.
"""
struct CUR{T<:Number, N} <: TensorDecomposition{T, 3}
    Cindex::Vector{Int64}
    Cweight::Vector{Int64}
    Rindex::Vector{CartesianIndex{2}}
    Rweight::Vector{Int64}
    U::Matrix{T}
    error::Vector{T}

    function CUR(tnsr::StridedArray{T, N},
                 slab_axis::Integer,
                 fiber_axes::NTuple{2, Int64},
                 fiber_size::NTuple{2, Int64},
                 Cindex::Vector{Int64},
                 Cweight::Vector{Int64},
                 Rindex::Vector{Int64},
                 Rweight::Vector{Int64},
                 U::Matrix{T},
                 compute_u::Bool) where {T,N}

        if compute_u
            output_index = collect(1:3)
            output_index[slab_axis] = 4
            S = tensorcontract(copy(selectdim(tnsr, slab_axis, Cindex)), [1, 2, 3], U, [4, slab_axis], output_index)
            W = dropdims(permutedims(mapslices(slab -> slab[Rindex], tnsr, dims=fiber_axes),
                                     [slab_axis, fiber_axes[1], fiber_axes[2]]), dims=3)
            S = tensorcontract(S, output_index, W, [slab_axis, 4], [1, 2, 3])
            S .-= tnsr
            err = dropdims(mapslices(norm, S, dims=fiber_axes) ./ mapslices(norm, tnsr, dims=fiber_axes), dims=fiber_axes)
        else
            err = Vector{T}()
        end
        new{T,N}(Cindex, Cweight,
            CartesianIndices(fiber_size)[Rindex], Rweight,
            U, err)
    end
end

"""
Calculates CUR decomposition for 3-mode tensors.
"""
function tensorcur3(tnsr::StridedArray,
                    c::Integer, r::Integer,
                    slab_axis::Integer=3;
                    compute_u::Bool=true)

    ndims(tnsr) == 3 || throw(ArgumentError("CUR3 method currently supports only 3-mode tensors."))
    1 <= slab_axis <= 3 || throw(ArgumentError("Invalid slab_axis; slab_axis should be 1, 2, or 3"))
    fiber_axes = tuple(circshift([1, 2, 3], 1 - slab_axis)[2:3]...)
    fiber_size = ntuple(i -> size(tnsr, fiber_axes[i]), 2)
    tnsr_sqnorm = sum(abs2, tnsr)
    p = sum(abs2, tnsr, dims=fiber_axes) ./ tnsr_sqnorm
    q = sum(abs2, tnsr, dims=slab_axis) ./ tnsr_sqnorm

    Cindex = rand(Categorical(vec(p)), c)
    Rindex = rand(Categorical(vec(q)), r)
    Cweight = fit(Histogram, Cindex, 0:size(tnsr, slab_axis), closed=:right).weights
    Rweight = fit(Histogram, Rindex, 0:prod(fiber_size), closed=:right).weights
    Cindex = sort(unique(Cindex))
    Rindex = sort(unique(Rindex))
    Cweight = Cweight[Cindex]
    Rweight = Rweight[Rindex]

    U = zeros(0, 0)
    if compute_u
        P = (Cweight*Rweight') ./ (p[Cindex]*q[Rindex]')
        W = dropdims(permutedims(mapslices(slab -> slab[Rindex],
                                           selectdim(tnsr, slab_axis, Cindex),
                                           dims=fiber_axes),
                                [slab_axis, fiber_axes[1], fiber_axes[2]]), dims=3)
        U = pinv(W .* P) .* P'
    else
        U = zeros(eltype(tnsr), 0, 0)
    end

    return CUR(tnsr, slab_axis, fiber_axes, fiber_size,
               Cindex, Cweight, Rindex, Rweight, U, compute_u)
end
