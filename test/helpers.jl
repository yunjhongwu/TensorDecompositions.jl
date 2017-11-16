function rand_candecomp(r::Int64, dims::NTuple{N, Int};
                        lambdas_nonneg::Bool=false, factors_nonneg::Bool=false) where N
    rnd_factor = factors_nonneg ? x -> abs.(randn(x...)) : randn
    rnd_lambda = lambdas_nonneg ? x -> abs.(randn(x...)) : randn
    return CANDECOMP((Matrix{Float64}[rnd_factor((s, r)) for s in dims]...),
                     rnd_lambda(r))
end

rand_kruskal3(r::Int64, dims::NTuple{N, Int}, nonnegative::Bool) where {N} =
    compose(rand_candecomp(r, dims, lambdas_nonneg=nonnegative, factors_nonneg=nonnegative))

function rand_tucker(core_dims::NTuple{N, Int}, dims::NTuple{N, Int};
                     core_nonneg::Bool=false, factors_nonneg::Bool=false) where {N}
    rnd_factor = factors_nonneg ? x -> abs.(randn(x...)) : randn
    rnd_core = core_nonneg ? x -> abs.(randn(x...)) : randn
    return Tucker((Matrix{Float64}[rnd_factor((dims[i], core_dims[i])) for i in 1:N]...),
                  rnd_core(core_dims))
end

function add_noise(tnsr::Array{T,N}, sn_ratio = 0.6, nonnegative::Bool = false) where {T,N}
    tnsr_noise = randn(size(tnsr)...)
    if nonnegative
        map!(x -> max(0.0, x), tnsr_noise, tnsr_noise)
    end
    tnsr + 10^(-sn_ratio/0.2)*vecnorm(tnsr)/vecnorm(tnsr)*tnsr_noise
end
