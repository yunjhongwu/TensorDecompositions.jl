"""
Canonical polyadic N-mode tensor decomposition (CANDECOMP/PARAFAC).
"""
struct CANDECOMP{T<:Number, N} <: TensorDecomposition{T, N}
  factors::NTuple{N, Matrix{T}} # rank X i-th tensor mode factor matrices
  lambdas::Vector{T}
  props::Dict{Symbol, Any}      # extra properties

  CANDECOMP{T, N}(factors::NTuple{N, Matrix{T}}, lambdas::Vector{T}) where {T<:Number, N} =
    new(factors, lambdas, Dict{Symbol, Any}())
end


CANDECOMP(factors::NTuple{N, Matrix{T}}, lambdas::Vector{T}) where {T<:Number, N} =
    CANDECOMP{T, N}(factors, lambdas)

"""
Returns CANDECOMP rank.
"""
LinearAlgebra.rank(decomp::CANDECOMP) = length(decomp.lambdas)

"""
Re-composes the tensor from CANDECOMP decomposition.
"""
@generated function compose!(dest::Array{T,N}, factors::NTuple{N, Matrix{T}}, lambdas::Vector{T}) where {T,N}
  quote
    csize = @ntuple($N, k -> size(factors[k], 2))
    @nall($N, k -> csize[k] == length(lambdas)) ||
        throw(DimensionMismatch("length of lambdas ($(length(lambdas))) doesn't match the factors columns ($csize)"))
    rsize = @ntuple($N, k -> size(factors[k], 1))
    size(dest) == rsize ||
        throw(DimensionMismatch("dest dimensions $(size(dest)) do not match the factors rows ($rsize)"))
    @nloops $N i dest begin
        elm = zero(T)
        # FIXME traversal of factors is not very efficient, would be better to have them transposed
        @inbounds for j in eachindex(lambdas)
            elm += lambdas[j] * (*(@ntuple($N, k -> factors[k][i_k, j])...))
        end
        @inbounds @nref($N, dest, i) = elm
    end
    return dest
  end
end

compose(factors::NTuple{N, Matrix{T}}, lambdas::Vector{T}) where {T,N} =
    compose!(Array{T, N}(undef, ntuple(i -> size(factors[i], 1), N)), factors, lambdas)

compose(decomp::CANDECOMP) = compose(decomp.factors, decomp.lambdas)

compose!(dest::Array{T,N}, decomp::CANDECOMP{T,N}) where {T,N} =
    compose!(dest, decomp.factors, decomp.lambdas)

"""
Calculates canonical polyadic tensor decomposition (CANDECOMP/PARAFAC).

Returns:
  `CANDECOMP` object
"""
function candecomp(tnsr::StridedArray{T,N},
                   r::Integer,
                   initial_guess::NTuple{N, Matrix{T}};
                   method::Symbol=:ALS,
                   tol::Float64=1e-5,
                   maxiter::Integer=100,
                   compute_error::Bool=false,
                   verbose::Bool=true) where {T,N}

    _check_tensor(tnsr, r)
    verbose && @info("initializing factor matrices...")
    all([(size(tnsr, i), r) == size(initial_guess[i]) for i in 1:N]) || throw(ArgumentError("dimension of initial guess does not match input tensor."))
    factors = collect(Matrix{T}, initial_guess)
    verbose && @info("applying candecomp $method method...")
    res = _candecomp(Val{method}, tnsr, r, factors, tol, maxiter, verbose)
    compute_error && _set_rel_residue(res, tnsr)
    return res
end

"Stub for non-implemented CANDECOMP algorithms."
_candecomp(
  method::Val{S},
  tnsr::StridedArray{T,N},
  r::Integer,
  factors::Vector{Matrix{Float64}},
  tol::Float64,
  maxiter::Integer,
  verbose::Bool) where {T,N,S<:Symbol} =
    throw(ArgumentError("Unknown CANDECOMP method: $method"))

"""
Computes CANDECOMP by ALS (Alternating Least Squares) method.
"""
function _candecomp(
    method::Type{Val{:ALS}},
    tnsr::StridedArray{T,N},
    r::Integer,
    factors::Vector{Matrix{Float64}},
    tol::Float64,
    maxiter::Integer,
    verbose::Bool) where {T,N}

    gram = [F'F for F in factors]
    tnsr_norm = norm(tnsr)
    tnsr_flat = _col_unfold(tnsr, N)
    tnsr_size = size(tnsr)
    niters = 0
    converged = false
    resid = tnsr_norm
    lbds = Matrix{Float64}(undef, 1, r)
    V = Matrix{Float64}(undef, length(tnsr) ÷ minimum(tnsr_size), r)
    while !converged && niters < maxiter
        nVi = 0
        @inbounds for i in 1:N
            idx = [N:-1:i + 1; i - 1:-1:1]
            nVi = length(tnsr) ÷ tnsr_size[i]
            Vi = view(V, 1:nVi, :)
            khatrirao!(Vi, factors[idx])
            factors[i] .= _row_unfold(tnsr, i) * Vi / reduce((x, y) -> x .* y, gram[idx])
            sum!(abs, lbds, factors[i])
            factors[i] ./= lbds
            mul!(gram[i], factors[i]', factors[i])
        end
        resid_old = resid
        resid = norm(view(V, 1:nVi, :) * (factors[N] .* lbds)' .- tnsr_flat)
        converged = abs(resid - resid_old) < tol * resid_old
        niters += 1
    end

    verbose && _iter_status(converged, niters, maxiter)

    return CANDECOMP(tuple(factors...), dropdims(lbds, dims=1))
end

"""
Computes CANDECOMP by SGSD (Simultaneous Generalized Schur Decomposition) method.
"""
function _candecomp(
    method::Type{Val{:SGSD}},
    tnsr::StridedArray{<:Number},
    r::Int,
    factors::Vector{<:StridedMatrix},
    tol::Float64,
    maxiter::Integer,
    verbose::Bool)

    ndims(tnsr) == 3 || throw(ArgumentError("This algorithm only applies to 3-mode tensors."))
    length(factors) == 3 || throw(ArgumentError("3 factor matrices expected."))
    (n1, n2, n3) = size(tnsr)
    IB1 = min(n1 - 1, r)
    IB2 = (n2 == r) ? 2 : 1
    Q = qr(factors[1]).Q * Matrix(I, n1, n1)
    Z = qr(reverse(factors[2], dims=2)).Q * Matrix(I, n2, n2)
    q = Matrix{Float64}(undef, n1, n1)
    z = Matrix{Float64}(undef, n2, n2)

    R = similar(tnsr)
    @tensor R[4,5,3] = tnsr[1,2,3] * Q[1,4] * Z[2,5]

    res = norm(tnsr)
    converged = false
    niters = 0
    @inbounds while !converged && niters < maxiter
        # reset to diagonal
        @inbounds for i in 1:n1, j in 1:n1
            q[i, j] = ifelse(i == j, 1.0, 0.0)
        end
        @inbounds for i in 1:n2, j in 1:n2
            z[i, j] = ifelse(i == j, 1.0, 0.0)
        end

        for i in 1:IB1
            q[:, i:n1] *= svd(view(q, :, i:n1)' * view(R, :, n2 - r + i, :)).U
        end

        @tensor R[4,2,3] = R[1,2,3] * q[1,4]

        for i in r:-1:IB2
            z[:, 1:n2 - r + i] *= reverse(svd(view(R, i, :, :)' * view(z, :, 1:n2 - r + i)).V, dims=2)
        end
        Q *= q
        Z *= z

        @tensor R[4,5,3] = tnsr[1,2,3] * Q[1,4] * Z[2,5]

        res_old = res
        res = norm(tril(dropdims(sum(abs2, R, dims=3), dims=3), n2 - r - 1))
        converged = abs(res - res_old) < tol * res_old
        niters += 1
    end

    verbose && _iter_status(converged, niters, maxiter)

    R = R[1:r, n2-r+1:n2, :]
    M = Array{Float64,3}(undef, r, r, 2)
    @inbounds for k in 1:2, j in 1:r, i in 1:r
        M[i, j, k] = i == j ? 1.0 : 0.0
    end
    @inbounds for i in (r - 1):-1:1, j = (i + 1):r
        if i + 1 < j
            d = (i + 1):(j - 1)
            M[i, j, :] = hcat(view(R, j, j, :), view(R, i, i, :)) \
                (view(R, i, j, :) .- mapslices(R3 -> sum(view(M, i, d, 1) .* diag(R3)[d] .* view(M, d, j, 2)), R, dims=(1, 2)))
        end
    end

    factors[1] = view(Q, :, 1:r) * view(M, :, :, 1)
    factors[2] = view(Z, :, n2 - r + 1:n2) * view(M, :, :, 2)'
    factors[3] = _row_unfold(tnsr, 3) * khatrirao(factors[2], factors[1]) /
        ((factors[2]'factors[2]) .* (factors[1]'factors[1]))

    # normalize
    lbds = ones(1, r)
    for i in 1:3
        lbd = mapslices(norm, factors[i], dims=1)
        factors[i] ./= lbd
        lbds .*= lbd
    end

    return CANDECOMP(tuple(factors...), dropdims(lbds, dims=1))
end

"""
Codes of implemented CANDECOMP methods.
"""
const CANDECOMP_methods = Symbol[t.parameters[1] for t in filter(t -> isa(t, DataType),
                                                                 [Base.unwrap_unionall(m.sig).parameters[2].parameters[1] for m in methods(_candecomp)])] |> Set{Symbol}
