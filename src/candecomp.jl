function candecomp(T::StridedArray,
                   r::Integer;
                   algo::String="als",
                   tol::Float64=1e-5,
                   maxiter::Integer=100,
                   compute_res::Bool=true,
                   hosvd_init::Bool=false,
                   verbose::Bool=true)
    methods = { "als" => _candecomp_als, "sgsd" => _candecomp_sgsd } 
    haskey(methods, algo) || error("Algorithm does not exist.")

    num_modes = _check_tensor(T, r)
    factors = (hosvd_init) ? hosvd(T, r, compute_core=false).factors : _random_init(size(T), r)
    
    return methods[algo](T, r, num_modes, factors, tol, maxiter, compute_res, verbose)
end

function _candecomp_als(T::StridedArray,
                        r::Integer,
                        num_modes::Integer,
                        factors::Array{Array{Float64, 2}, 1},
                        tol::Float64,
                        maxiter::Integer,
                        compute_res::Bool,
                        verbose::Bool)
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

    verbose && println(converged ? string("Algorithm converged after ", niters, " iterations.") :
                                   string("Warning: Maximum number (", maxiter, ") of iterations exceeded."))

    return Tucker(T, factors, lbds, compute_res=compute_res)
end

function _candecomp_sgsd(T::StridedArray,
                         r::Integer,
                         num_modes::Integer,
                         factors::Array{Array{Float64, 2}, 1},
                         tol::Float64,
                         maxiter::Integer,
                         compute_res::Bool,
                         verbose::Bool)
 
    num_modes == 3 || error("This algorithm only applies to 3-mode tensors.")

    (n1, n2, n3) = size(T)
    Q = qr(factors[1], thin=false)[1]
    Z = qr(fliplr(factors[2]), thin=false)[1]
    q = Array(Float64, n1, n1)
    z = Array(Float64, n2, n2)
    R = tensorcontract(tensorcontract(T, [1, 2, 3], Q, [1, 4], [4, 2, 3]), [1, 2, 3], Z, [2, 4], [1, 4, 3])

    res = vecnorm(T)
    converged = false
    niters = 0
    while !converged && niters < maxiter
        q = eye(n1)
        z = eye(n2)

        for i = 1:min(n1 - 1, r)
            q[:, i:end] *= svd(q[:, i:end]' * squeeze(R[:, n2 - r + i, :], 2))[1]
        end
        R = tensorcontract(R, [1, 2, 3], q, [1, 4], [4, 2, 3])
        for i = r:-1:(n2 == r ? 2 : 1)
            z[:, 1:n2 - r + i] *= fliplr(svd(squeeze(R[i, :, :], 1)' * z[:, 1:n2 - r + i])[3])
        end
        Q *= q
        Z *= z
        R = tensorcontract(tensorcontract(T, [1, 2, 3], Q, [1, 4], [4, 2, 3]), [1, 2, 3], Z, [2, 4], [1, 4, 3])

        res_old = res
        res = vecnorm(tril(squeeze(sum(R .^ 2, 3), 3), n2 - r - 1))
        converged = abs(res - res_old) < tol 
        niters += 1
    end

    verbose && println(converged ? string("Algorithm converged after ", niters, " iterations.") :
                                   string("Warning: Maximum number (", maxiter, ") of iterations exceeded."))

    R = R[1:r, end-r+1:end, :]
    M = cat(3, eye(r, r), eye(r, r))
    for i = r - 1:-1:1, j = i + 1:r
        d = i + 1:j - 1
        M[i, j, :] = [R[j, j, :][:] R[i, i, :][:]] \ (R[i, j, :][:] - mapslices(R3 -> sum(M[i, d, 1] * (diag(R3)[d] .* M[d, j, 2])), R, [1, 2])[:])
    end
    
    factors[1] = Q[:, 1:r] * M[:, :, 1] 
    factors[2] = Z[:, end - r + 1:end] * M[:, :, 2]'
    factors[3] = _unfold(T, 3) * _KhatriRao(factors[2], factors[1]) / ((factors[2]' * factors[2]) .* (factors[1]' * factors[1]))
    
    lbds = ones(1, r)
    for i = 1:num_modes
        lbd = mapslices(vecnorm, factors[i], 1) 
        factors[i] ./= lbd
        lbds .*= lbd
    end

    return Tucker(T, factors, lbds, compute_res=compute_res)
end
