function hosvd(T::StridedArray, rank::Integer; compute_core::Bool=true)
    num_modes = _check_tensor(T, rank)

    factors = Array(Array{Float64, 2}, num_modes)

    for i = 1:ndims(T)
        X = _unfold(T, i)
        factors[i] = eigs(X * X', nev=rank)[2]
    end

    factors = map(factor -> mapslices(_check_sign, factor, 1), factors)
    return Tucker(T, factors, compute_res=compute_core)
end
