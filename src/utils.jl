# utilities

function _random_init(d::Tuple, r::Integer)
    return [randn(i, r)::Matrix{Float64} for i in d]
end

function _KhatriRao(A::Matrix{Float64}, B::Matrix{Float64})
    size(A, 2) == size(B, 2) || error("Input matrices should have the same number of columns.")
    return [[kron(A[:, i], B[:, i])' for i in 1:size(A, 2)]...]'
end

function _unfold(T::StridedArray, mode::Integer)
    num_modes = ndims(T)
    mode <= num_modes && mode > 0 || error("Unable to unfold the tensor: invalid mode index")

    idx = [mode, 1:mode-1, mode+1:num_modes] 
    return reshape(permutedims(T, idx), 
                   size(T, mode), 
                   prod(size(T)[idx[2:num_modes]]))
end

function _iter_status(converged::Bool, niters::Integer, maxiter::Integer)
    println(converged ? string("Algorithm converged after ", string(niters)::ASCIIString, " iterations.") :
                        string("Warning: Maximum number (", string(maxiter)::ASCIIString, ") of iterations exceeded."))
end


function _check_sign(v::StridedVector)
    return sign(v[findmax(abs(v))[2]]) * v
end

function _check_tensor(T::StridedArray, r::Integer)
    num_modes = ndims(T)
    num_modes > 2 || error("This method does not support scalars, vectors, or matrices input.")
    r <= minimum(size(T)) && r > 0 || error("r should satisfy 1 <= r <= minimum(size(T)).")
    isreal(T) || error("This package currently only supports real-number-valued tensors.")
    return num_modes
end
