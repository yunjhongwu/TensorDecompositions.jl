function hosvd(T::StridedArray, rank::Integer; compute_core::Bool=true)
    num_modes = _check_tensor(T, rank)

    factors = Array(Array{Float64, 2}, num_modes)

    for i = 1:ndims(T)
        X = _unfold(T, i)
        X = Symmetric(convert(Array{Float64, 2}, X * X'))
        factors[i] = eigvecs(X)[:, end:-1:end-rank+1]
    end

    factors = map(factor -> mapslices(_check_sign, factor, 1), factors)
    return Factors(T, factors, compute_res=compute_core)
end
