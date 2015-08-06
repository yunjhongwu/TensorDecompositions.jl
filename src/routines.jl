type Tucker 
    factors::Array{Array{Float64, 2}, 1}
    core::Array{Float64}
    error::Float64

    function Tucker(T::StridedArray,
                    factors::Array{Array{Float64, 2}, 1}, 
                    S::Array{Float64}=zeros(0);
                    compute_res::Bool=true)

        num_modes = ndims(T)
        res = NaN 
        if compute_res
            if length(S) > 0
                res = vecnorm(_unfold(T, num_modes) - (factors[end] .* S) * reduce(_KhatriRao, factors[end-1:-1:1])')
            else
                d = num_modes + 1
                S = copy(T)
                for i = 1:num_modes
                    S = tensorcontract(S, [1:num_modes], factors[i], [i, d], [1:i-1, d, i+1:num_modes])
                end
    
                L = copy(S)
                for i = 1:num_modes
                    L = tensorcontract(L, [1:num_modes], factors[i], [d, i], [1:i-1, d, i+1:num_modes])
                end
                res = vecnorm(L - T)
            end
        end
        new(factors, S, res / vecnorm(T)) 
    end
end


function _random_init(d::Tuple, n_components::Integer)
    return [randn(d[i], n_components) for i = 1:length(d)]
end

function _KhatriRao(A::StridedMatrix, B::StridedMatrix)
    size(A, 2) == size(B, 2) || error("Input matrices should have the same number of columns.")
    return [[kron(A[:, i], B[:, i])' for i = 1:size(A, 2)]...]'
end

function _unfold(T::StridedArray, mode::Integer)
    mode <= ndims(T) && mode > 0 || error("Unable to unfold the tensor: invalid mode index")

    idx = [mode, 1:mode-1, mode+1:ndims(T)] 
    return reshape(permutedims(T, idx), 
                   size(T, mode), 
                   prod(size(T)[idx[2:end]]))
end

function _check_sign(v::StridedVector)
    return sign(v[findmax(abs(v))[2]]) * v
end

function _check_tensor(T::StridedArray, r::Integer)
    num_modes = ndims(T)
    num_modes > 2 || error("This method does not support scalars, vectors, or matrices input.")
    r <= minimum(size(T)) && r > 0 || error("r should satisfy 0 < r <= minimum(size(T)).")
    isreal(T) || error("This package currently only supports real-number-valued tensors.")
    return num_modes
end


