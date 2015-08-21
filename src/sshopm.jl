function sshopm(T::Union(StridedArray, (Matrix{Int64}, StridedVector, Integer)),
                alpha::Real;
                tol::Float64=1e-5,
                maxiter::Integer=100,
                verbose::Bool=false)

    if typeof(T)<:StridedArray
        num_modes = _check_tensor(T, 1)
        all(size(T)) > 0 || error("Input tensor should have the same dimension in all modes.")
        _left_mult = _left_mult_dense(T, num_modes)
        x = randn(size(T, 1))
    else
        num_modes = size(T[1], 1)
        _left_mult = _left_mult_sparse(T, num_modes)
        x = randn(T[3])
    end


    x *= 1 / vecnorm(x)
    converged = false
    niters = 0
    while !converged && niters < maxiter
        x_old = copy(x)
        x = flipsign(_left_mult(x) + alpha * x, alpha)
        x *= 1 / vecnorm(x) 
        converged = vecnorm(x - x_old) < tol
        niters += 1
    end

    verbose && _iter_status(converged, niters, maxiter)
    return (dot(x, _left_mult(x)), flipsign(x, alpha))
end

function _left_mult_dense(T::StridedArray, num_modes::Int64)
    return function(x::Vector{Float64})
               v = copy(T)
               for i in 2:num_modes
                   v = tensorcontract(v, [i-1:num_modes], x, num_modes)
               end
               return v
           end
end

function _left_mult_sparse(T::(Matrix{Int64}, Vector{Float64}, Integer), num_modes::Int64)
    return function(x::Vector{Float64})
               v = zeros(T[3])
               for i in 1:length(T[2])
                   for w in @task _rep_permute(T[1][:, i])
                       v[w[1]] += prod(x[w[2:num_modes]]) * T[2][i]
                   end
               end
               return v
           end
end

 
function _rep_permute(v::Vector{Int64}, w::Vector{Int64}=Int64[])
    if length(v) > 0
        _rep_permute(v[2:end], [w, v[1]])
        for i in 2:length(v)
            if v[i] != v[i - 1]
                _rep_permute(vcat(v[1:i-1], v[i+1:end]), [w, v[i]])
            end
        end
    else
        produce(w)
    end
end
