function hosvd(T::StridedArray, 
               r::Integer; 
               compute_core::Bool=true)

    num_modes = _check_tensor(T, r)

    factors = map(1:N) do i
        X = _col_unfold(T, i)
        f = eigs(X'X, nev=core_dims[i])[2]
        mapslices(_check_sign, f, 1)
    end
    return Tucker(T, factors, compute_res=compute_core)
end
