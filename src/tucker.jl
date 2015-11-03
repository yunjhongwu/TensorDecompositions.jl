"""
  Tucker decomposition of a tensor
"""
immutable Tucker 
    factors::Vector{Matrix{Float64}}
    core::StridedArray
    error::Float64

    function Tucker(T::StridedArray,
                    factors::Vector{Matrix{Float64}}, 
                    S::Array{Float64}=zeros(0);
                    compute_res::Bool=true)

        num_modes = ndims(T)
        res = NaN 
        if compute_res
            if length(S) > 0
                res = vecnorm(_row_unfold(T, num_modes) - (factors[num_modes] .* S) * reduce(_KhatriRao, factors[num_modes-1:-1:1])')
            else
                d = num_modes + 1
                S = copy(T)
                for i in 1:num_modes
                    S = tensorcontract(S, [1:num_modes], factors[i], [i, d], [1:i-1, d, i+1:num_modes])
                end
    
                L = copy(S)
                for i in 1:num_modes
                    L = tensorcontract(L, [1:num_modes], factors[i], [d, i], [1:i-1, d, i+1:num_modes])
                end
                res = vecnorm(L - T)
            end
        end
        new(factors, S, res / vecnorm(T)) 
    end
end

