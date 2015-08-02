type Factors
    factors::Array{Array{Float64, 2}, 1}
    core::Array{Float64}
    residual::Float64

    function Factors(factors::Array{Array{Float64, 2}, 1}, 
                     T::StridedArray, 
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
        new(factors, S, res) 
    end
end


function _random_init(d::Tuple, n_components::Integer)
    return [randn(d[i], n_components) for i = 1:length(d)]
end

function _KhatriRao(A::StridedMatrix, B::StridedMatrix)
    @assert size(A, 2) == size(B, 2)
    return [[kron(A[:, i], B[:, i])' for i = 1:size(A, 2)]...]'
end

function _unfold(T::StridedArray, mode::Integer)
    @assert mode <= ndims(T) && mode > 0

    idx = [mode, 1:mode-1, mode+1:ndims(T)] 
    return reshape(permutedims(T, idx), 
                   size(T, mode), 
                   prod(size(T)[idx[2:end]]))
end

function _check_sign(v::StridedVector)
    return sign(v[findmax(abs(v))[2]]) * v
end

function _check_tensor(T::StridedArray, rank::Integer)
    num_modes = ndims(T)
    @assert num_modes > 2
    @assert rank <= minimum(size(T)) && rank > 0
    return num_modes
end


