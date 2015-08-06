function nncp(T::StridedArray, 
              r::Integer;
              tol::Float64=1e-4,
              maxiter::Integer=100,
              verbose::Bool=true)

    minimum(T) >= 0 || error("Input tensor must be nonnegative.")
    num_modes = _check_tensor(T, r)
    T_norm = vecnorm(T)
    
    factors = [abs(factor) * (T_norm ^ (1/num_modes) / vecnorm(factor)) for factor = _random_init(size(T), r)]
    factors_old = deepcopy(factors)
    factors_exp = deepcopy(factors)
    gram = [factor' * factor for factor = factors]

    niters = 0
    converged = false
    obj_old = 0 
    t = 1
    t_old = 1

    LB = ones(num_modes)
    LB_old = ones(num_modes)

    while !converged && niters < maxiter
        U = Array(Float64, num_modes, num_modes) 
        M = []
        for i = 1:num_modes
            idx = [num_modes:-1:i+1, i-1:-1:1]
            U = reduce(.*, gram[idx])
            LB_old[i] = LB[i]
            LB[i] = vecnorm(U)
            M = _unfold(T, i) * reduce(_KhatriRao, factors[idx])
            factors[i] = max(0, factors_exp[i] - (factors_exp[i] * U - M) * (1 / LB[i]))
            gram[i] = factors[i]' * factors[i]
        end
        
        obj = sum(gram[num_modes] .* U) - 4 * sum(factors[num_modes] .* M)  
        converged = abs(obj - obj_old) < tol * abs(obj_old)

        if obj > obj_old
            factors_exp = factors_old
        else
            t = (1 + sqrt(1 + 4 * t_old ^2)) / 2
            weights = min((t_old - 1) / t, sqrt( LB_old ./ LB ))
            factors_exp = map((factor, factor_old, wi) -> factor + wi * (factor - factor_old), factors, factors_old, weights) 
            factors_old = deepcopy(factors)
            t_old = t
            obj_old = obj
        end
        niters += 1

    end

    verbose && println(converged ? string("Algorithm converged after ", niters, " iterations.") :
                                   string("Warning: Maximum number (", maxiter, ") of iterations exceeded."))

    return Tucker(T, factors, ones(1, r)) 
end


