"""
Non-negative CANDECOMP tensor decomposition.
"""
function nncp(tnsr::StridedArray,
              r::Integer;
              tol::Float64=1e-4,
              maxiter::Integer=100,
              compute_error::Bool=false,
              verbose::Bool=true)

    minimum(tnsr) >= 0 || error("Input tensor must be nonnegative.")
    num_modes = _check_tensor(tnsr, r)
    T_norm = vecnorm(tnsr)

    factors = [abs.(F) * (T_norm ^ (1/num_modes) / vecnorm(F)) for F::Matrix{Float64} in _random_factors(size(tnsr), r)]
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


    U = Array{Float64}(num_modes, num_modes)
    M = Array{Float64}(r, r)
    while !converged && niters < maxiter
        LB_old = copy(LB)
        for i in 1:num_modes
            idx = [num_modes:-1:i+1; i-1:-1:1]
            U = reduce(((x, y) -> broadcast(*, x, y)), gram[idx])
            LB[i] = vecnorm(U)
            M = _row_unfold(tnsr, i) * reduce(khatrirao, factors[idx])
            factors[i] = max.(0, factors_exp[i] - (factors_exp[i] * U - M) * (1 / LB[i]))
            At_mul_B!(gram[i], factors[i], factors[i])
        end

        obj = sum(broadcast(*, gram[num_modes], U)) - 4 * sum(broadcast(*, factors[num_modes], M))

        if obj > obj_old
            factors_exp = factors_old
        else
            t = 1 + sqrt.(1 + 4 * t_old^2)
            weights = min.((t_old - 2) / t, sqrt.( LB_old ./ LB ))
            factors_exp = map((F, F_old, w) -> F + w * (F - F_old), factors, factors_old, weights)
            factors_old = deepcopy(factors)
            converged = obj_old - obj < tol * abs.(obj_old)
            t_old = t
            obj_old = obj
        end
        niters += 1

    end

    verbose && _iter_status(converged, niters, maxiter)

    res = CANDECOMP((factors...), ones(r))
    if compute_error
      _set_rel_residue(res, tnsr)
    end
    return res
end
