"""
High-order singular value decomposition (HO-SVD).
"""
function hosvd{T,N}(tensor::StridedArray{T,N}, core_dims::NTuple{N, Int};
                    pad_zeros::Bool=false, compute_error::Bool=false)
    pad_zeros || _check_tensor(tensor, core_dims)

    factors = map(1:N) do i
        X = _col_unfold(tensor, i)
        f = eigs(X'X, nev=core_dims[i])[2]
        if pad_zeros && size(f, 2) < core_dims[i] # fill missing factors with zeros
            f = hcat(f, zeros(T, size(tensor, i), core_dims[i]-size(f, 2)))
        end
        mapslices(_check_sign, f, 1)
    end

    res = Tucker((factors...), tensorcontractmatrices(tensor, factors))
    if compute_error
        _set_rel_residue(res, tensor)
    end
    return res
end

hosvd{T,N}(tensor::StridedArray{T,N}, r::Int; compute_error::Bool=false) =
    hosvd(tensor, (fill(r, N)...); compute_error=compute_error)
