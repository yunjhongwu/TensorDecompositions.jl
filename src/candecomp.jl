function candecomp(T::StridedArray,
                   r::Integer;
                   tol::Float64=1e-5,
                   maxiter::Integer=100,
                   compute_res::Bool=true,
                   hosvd_init::Bool=false,
                   verbose::Bool=true)

    num_modes = _check_tensor(T, r)

    factors = (hosvd_init) ? hosvd(T, r, compute_core=false).factors : _random_init(size(T), r)
    gram = [factor' * factor for factor = factors]
    T_norm = vecnorm(T) 
    T_flat = _unfold(T, num_modes)'
    niters = 0
    converged = false
    res = T_norm 
    lbds = Array(Float64, r)
    while !converged && niters < maxiter
        V = []
        for i = 1:num_modes
            idx = [num_modes:-1:i+1, i-1:-1:1]
            V = reduce(_KhatriRao, factors[idx])
            factors[i] = _unfold(T, i) * V / reduce(.*, gram[idx])
            lbds = sum(abs(factors[i]), 1)
            factors[i] ./= lbds
            gram[i] = factors[i]' * factors[i]
        end
        res_old = res 
        res = vecnorm(V * (factors[num_modes] .* lbds)'- T_flat) 
        converged = abs(res - res_old) < tol * res_old 
        niters += 1
    end

    verbose && println(converged ? string("The algorithm converged after ", niters, " iterations.") :
                                   string("Warning: Maximum number (", maxiter, ") of iterations exceeded."))

    return Tucker(T, factors, lbds, compute_res=compute_res)
end

