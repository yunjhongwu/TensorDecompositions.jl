function hosvd(T::StridedArray, 
               r::Integer; 
               compute_core::Bool=true)

    num_modes = _check_tensor(T, r)

    factors = Array(Matrix{Float64}, num_modes)

    for i in 1:ndims(T)
        X = _unfold(T, i)
        factors[i] = eigs(X * X', nev=r)[2]
    end

    factors = map(F -> mapslices(_check_sign, F, 1), factors)
    return Tucker(T, factors, compute_res=compute_core)
end
