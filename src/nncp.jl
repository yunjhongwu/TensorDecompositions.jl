function nncp(T::StridedArray,
              r::Integer;
              tol::Float64=1e-4,
              maxiter::Integer=100,
              verbose::Bool=true)

    minimum(T) >= 0 || error("Input tensor must be nonnegative.")
    num_modes = _check_tensor(T, r)
    T_norm = vecnorm(T)

    factors = [abs(F) * (T_norm ^ (1/num_modes) / vecnorm(F)) for F::Matrix{Float64} in _random_factors(size(T), r)]
    factors_old = deepcopy(factors)
    factors_exp = deepcopy(factors)
    gram = Matrix{Float64}[F'F for F in factors]

    niters = 0
    converged = false
    obj_old = 0.0
    t = 1.0
    t_old = 1.0

    LB = ones(num_modes)
    LB_old = ones(num_modes)


    U = Array(Float64, num_modes, num_modes)
    M = Array(Float64, r, r)
    while !converged && niters < maxiter
        LB_old = copy(LB)
        for i in 1:num_modes
            idx = [num_modes:-1:i+1; i-1:-1:1]
            U = reduce(.*, gram[idx])
            LB[i] = vecnorm(U)
            M = _row_unfold(T, i) * reduce(_KhatriRao, factors[idx])
            factors[i] = max(0, factors_exp[i] - (factors_exp[i] * U - M) * (1 / LB[i]))
            At_mul_B!(gram[i], factors[i], factors[i])
        end

        obj = sum(gram[num_modes] .* U) - 4 * sum(factors[num_modes] .* M)

        if obj > obj_old
            factors_exp = factors_old
        else
            t = 1 + sqrt(1 + 4 * t_old^2)
            weights = min((t_old - 2) / t, sqrt( LB_old ./ LB ))
            factors_exp = map((F, F_old, w) -> F + w * (F - F_old), factors, factors_old, weights)
            factors_old = deepcopy(factors)
            converged = obj_old - obj < tol * abs(obj_old)
            t_old = t
            obj_old = obj
        end
        niters += 1

    end

    verbose && _iter_status(converged, niters, maxiter)

    return Tucker(T, factors, ones(1, r)) 
end
