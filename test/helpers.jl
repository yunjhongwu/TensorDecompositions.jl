randn_tensor(dims, nonneg::Bool) = nonneg ? abs.(randn(dims)) : randn(dims)

rand_candecomp(r::Int64, dims::NTuple{N, Int};
               lambdas_nonneg::Bool=false, factors_nonneg::Bool=false) where N =
    CANDECOMP(ntuple(i -> randn_tensor((dims[i], r), factors_nonneg), N),
              randn_tensor(r, lambdas_nonneg))

rand_kruskal3(r::Int64, dims::NTuple{N, Int}, nonnegative::Bool) where N =
    compose(rand_candecomp(r, dims, lambdas_nonneg=nonnegative, factors_nonneg=nonnegative))

rand_tucker(core_dims::NTuple{N, Int}, dims::NTuple{N, Int};
            core_nonneg::Bool=false, factors_nonneg::Bool=false) where N =
    Tucker(ntuple(i -> randn_tensor((dims[i], core_dims[i]), factors_nonneg), N),
           randn_tensor(core_dims, core_nonneg))

function add_noise(tnsr::Array{T,N}, sn_ratio = 0.6, nonnegative::Bool = false) where {T,N}
    tnsr_noise = randn(size(tnsr)...)
    if nonnegative
        tnsr_noise .= max.(tnsr_noise, 0.0)
    end
    noise_scale = 10^(-sn_ratio/0.2)*norm(tnsr)/norm(tnsr_noise)
    return tnsr .+ noise_scale .* tnsr_noise
end
