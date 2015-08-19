function sshopm(T::StridedArray,
                alpha::Real;
                tol::Float64=1e-5,
                maxiter::Integer=100,
                verbose::Bool=false)

    num_modes = _check_tensor(T, 1)
    all(size(T)) > 0 || error("Input tensor should have the same dimension in all modes.")

    x = randn(size(T, 1))
    x *= 1 / vecnorm(x)
    converged = false
    niters = 0
    while !converged && niters < maxiter
        x_old = copy(x)
        x = flipsign(_left_mult(T, x, num_modes) + alpha * x, alpha)
        x *= 1 / vecnorm(x) 
        converged = vecnorm(x - x_old) < tol
        niters += 1
    end

    verbose && _iter_status(converged, niters, maxiter)
    return (dot(x, _left_mult(T, x, num_modes)), flipsign(x, alpha))
end

function _left_mult(T::StridedArray, x::Vector{Float64}, num_modes::Int64)
    v = copy(T)
    for i = 2:num_modes
        v = tensorcontract(v, [i-1:num_modes], x, num_modes)
    end
    return v
end
