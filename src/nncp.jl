"""
Non-negative CANDECOMP tensor decomposition.
"""
function nncp(tnsr::StridedArray{T,N},
              r::Integer;
              tol::Float64=1e-4,
              maxiter::Integer=100,
              compute_error::Bool=false,
              verbose::Bool=true) where {T,N}

    all(x -> x >= 0, tnsr) || throw(ArgumentError("Input tensor must be nonnegative."))
    num_modes = _check_tensor(tnsr, r)
    T_norm = norm(tnsr)

    factors = [abs.(F) .* (T_norm ^ (1/num_modes) / norm(F)) for F::Matrix{T} in _random_factors(size(tnsr), r)]
    factors_old = deepcopy(factors)
    factors_exp = deepcopy(factors)
    gram = [F'F for F in factors]

    niters = 0
    converged = false
    obj_old = 0.0
    t = 1.0
    t_old = 1.0

    LB = fill!(similar(tnsr, num_modes), 1.0)
    LB_old = fill!(similar(tnsr, num_modes), 1.0)

    U = similar(tnsr, r, r)

    pb = Progress(maxiter, "NNCP iterations ")
    while !converged && niters < maxiter
        update!(pb, niters)
        LB_old .= LB
        local M::typeof(U)
        for i in 1:num_modes
            idx = [num_modes:-1:i+1; i-1:-1:1] # num_modes except i
            U .= reduce((x, y) -> x .* y, gram[idx]) # element-wise product of gram matrices
            LB[i] = norm(U)
            M = _row_unfold(tnsr, i) * khatrirao(factors[idx])
            factors[i] = max.(0, factors_exp[i] .- (factors_exp[i] * U .- M) .* (1 / LB[i]))
            mul!(gram[i], factors[i]', factors[i])
        end

        obj = dot(gram[num_modes], U) - 4 * dot(factors[num_modes], M)
        if obj > obj_old
            for i in eachindex(factors_old)
                copyto!(factors_exp[i], factors_old[i])
            end
        else
            t = 1 + sqrt(1 + 4 * t_old^2)
            min_w = (t_old - 2) / t
            for i in eachindex(factors)
                w = min(min_w, sqrt(LB_old[i] / LB[i]))
                factors_exp[i] .= factors[i] .+ w .* (factors[i] .- factors_old[i])
                copyto!(factors_old[i], factors[i])
            end
            converged = abs(obj_old - obj) < tol * abs(obj_old)
            t_old = t
            obj_old = obj
        end
        niters += 1
    end
    finish!(pb)
    verbose && _iter_status(converged, niters, maxiter)

    res = CANDECOMP(tuple(factors...), fill!(similar(tnsr, r), 1.0))
    compute_error && _set_rel_residue(res, tnsr)
    return res
end
