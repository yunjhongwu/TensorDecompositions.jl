"""
CUR (columns-U-rows) 3-tensor decomposition.
"""
struct CUR <: TensorDecomposition{Float64, 3}
    Cindex::Vector{Int64}
    Cweight::Vector{Int64}
    Rindex::Vector{NTuple{2, Int64}}
    Rweight::Vector{Int64}
    U::Matrix{Float64}
    error::Vector{Float64}

    function CUR(tnsr::StridedArray{Float64},
                 slab_axis::Integer,
                 fiber_axes::NTuple{2, Int64},
                 fiber_size::NTuple{2, Int64},
                 Cindex::Vector{Int64},
                 Cweight::Vector{Int64},
                 Rindex::Vector{Int64},
                 Rweight::Vector{Int64},
                 U::Matrix{Float64},
                 compute_u::Bool)

        res = zeros(0)
        if compute_u
            output_index = collect(1:3)
            output_index[slab_axis] = 4
            S = tensorcontract(slicedim(tnsr, slab_axis, Cindex), [1, 2, 3], U, [4, slab_axis], output_index)
            W = squeeze(permutedims(mapslices(slab -> slab[Rindex], tnsr, fiber_axes),
                                    [slab_axis, fiber_axes[1], fiber_axes[2]]), 3)
            S = tensorcontract(S, output_index, W, [slab_axis, 4], [1, 2, 3])
            res = mapslices(vecnorm, S - tnsr, fiber_axes)[:] ./ mapslices(vecnorm, tnsr, fiber_axes)[:]
        end
        new(Cindex, Cweight, [zip(ind2sub(fiber_size, Rindex)...)...], Rweight, U, res)
    end
end

"""
Calculates CUR decomposition for 3-mode tensors.
"""
function tensorcur3(tnsr::StridedArray,
                    c::Integer, r::Integer,
                    slab_axis::Integer=3;
                    compute_u::Bool=true)

    ndims(tnsr) == 3 || error("This method currently only supports 3-mode tensors.")
    slab_axis > 0 && slab_axis < 4 || error("Invalid slab_axis; slab_axis should be 1, 2, or 3")
    fiber_axes = tuple(circshift([1, 2, 3], 1 - slab_axis)[2:3]...)
    fiber_size = (size(tnsr, fiber_axes[1]), size(tnsr, fiber_axes[2]))
    T2 = tnsr .^ 2
    T2_sum = sum(T2)
    p = sum(T2, fiber_axes)[:] / T2_sum
    q = sum(T2, slab_axis)[:] / T2_sum

    Cindex = rand(Categorical(p), c)
    Rindex = rand(Categorical(q), r)
    Cweight = fit(Histogram, Cindex, 0:size(tnsr, slab_axis), closed=:right).weights
    Rweight = fit(Histogram, Rindex, 0:prod(fiber_size), closed=:right).weights
    Cindex = sort(unique(Cindex))
    Rindex = sort(unique(Rindex))
    Cweight = Cweight[Cindex]
    Rweight = Rweight[Rindex]

    U = compute_u ? Array{Float64}(r, c) : zeros(0, 0)
    if compute_u
        P = (Cweight*Rweight') ./ (p[Cindex]*q[Rindex]')
        W = squeeze(permutedims(mapslices(slab -> slab[Rindex],
                                slicedim(tnsr, slab_axis, Cindex), fiber_axes),
                                [slab_axis, fiber_axes[1], fiber_axes[2]]), 3)
        U = pinv(W .* P) .* P'
    end

    return CUR(tnsr, slab_axis, fiber_axes, fiber_size, Cindex, Cweight, Rindex, Rweight, U, compute_u)
end
