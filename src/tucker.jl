"""
  Tucker decomposition of a N-mode tensor.
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
                res = vecnorm(T - tensorcontractmatrices(S, factors, collect(1:num_modes), transpose=true))
            else
                d = num_modes + 1
                S = tensorcontractmatrices(T, factors, collect(1:num_modes))
    
                L = tensorcontractmatrices(S, factors, collect(1:num_modes), transpose=true)
                res = vecnorm(L - T)
            end
        end
        new(factors, S, res / vecnorm(T)) 
    end
end

