"""
Canonical polyadic N-mode tensor decomposition (CANDECOMP/PARAFAC).
"""
immutable CANDECOMP{T<:Number, N} <: TensorDecomposition{T, N}
  factors::NTuple{N, Matrix{T}} # rank X i-th tensor mode factor matrices
  lambdas::Vector{T}
  props::Dict{Symbol, Any}      # extra properties

  CANDECOMP(factors::NTuple{N, Matrix{T}}, lambdas::Vector{T}) =
    new(factors, lambdas, Dict{Symbol, Any}())

  Base.call{T,N}(::Type{CANDECOMP}, factors::NTuple{N, Matrix{T}}, lambdas::Vector{T}) =
    CANDECOMP{T,N}(factors, lambdas)
end

"""
Returns CANDECOMP rank.
"""
Base.rank(decomp::CANDECOMP) = length(decomp.lambdas)

"""
Re-composes the tensor from CANDECOMP decomposition.
"""
@generated function compose!{T,N}(dest::Array{T,N}, factors::NTuple{N, Matrix{T}}, lambdas::Vector{T})
  quote
    @nloops $N i dest begin
        elm = zero(T)
        # FIXME traversal of factors is not very efficient, would be better to have them transposed
        for j in 1:length(lambdas)
            elm += lambdas[j] * (*(@ntuple($N, k -> factors[k][i_k, j])...))
        end
        @nref($N, dest, i) = elm
    end
    dest
  end
end

@generated function compose{T,N}(factors::NTuple{N, Matrix{T}}, lambdas::Vector{T})
  quote
    compose!(Array{T}(@ntuple $N i -> size(factors[i], 1)), factors, lambdas)
  end
end

compose(decomp::CANDECOMP) = compose(decomp.factors, decomp.lambdas)

compose!{T,N}(dest::Array{T,N}, decomp::CANDECOMP{T,N}) = compose!(dest, decomp.factors, decomp.lambdas)

"""
Calculates canonical polyadic tensor decomposition (CANDECOMP/PARAFAC).

Returns:
  `CANDECOMP` object
"""
function candecomp{T,N}(tnsr::StridedArray{T,N},
                   r::Integer;
                   method::Symbol=:ALS,
                   tol::Float64=1e-5,
                   maxiter::Integer=100,
                   hosvd_init::Bool=false,
                   compute_error::Bool=false,
                   verbose::Bool=true)

    _check_tensor(tnsr, r)
    # don't use HO-SVD init if later the call would fail because of the wrong method
    verbose && info("Initializing factor matrices...")
    factors = in(method, CANDECOMP_methods) && hosvd_init ?
              hosvd(tnsr, r, compute_error=false).factors :
              _random_factors(size(tnsr), r)

    verbose && info("Applying CANDECOMP $method method...")
    res = _candecomp(Val{method}, tnsr, r, factors, tol, maxiter, verbose)
    if compute_error
      _set_rel_residue(res, tnsr)
    end
    return res
end

"Stub for non-implemented CANDECOMP algorithms."
_candecomp{T,N,S<:Symbol}(
  method::Val{S},
  tnsr::StridedArray{T,N},
  r::Integer,
  factors::Vector{Matrix{Float64}},
  tol::Float64,
  maxiter::Integer,
  verbose::Bool) = throw(ArgumentError("Unknown CANDECOMP method: $method"))

"""
Computes CANDECOMP by ALS (Alternating Least Squares) method.
"""
function _candecomp{T,N}(
    method::Type{Val{:ALS}},
    tnsr::StridedArray{T,N},
    r::Integer,
    factors::Vector{Matrix{Float64}},
    tol::Float64,
    maxiter::Integer,
    verbose::Bool)

    gram = [F'F for F in factors]
    tnsr_norm = vecnorm(tnsr)
    tnsr_flat = _col_unfold(tnsr, N)
    tnsr_size = size(tnsr)
    niters = 0
    converged = false
    resid = tnsr_norm
    lbds = Matrix{Float64}(1, r)
    V = Matrix{Float64}(div(length(tnsr), minimum(tnsr_size)), r)
    while !converged && niters < maxiter
        VB = 0
        @inbounds for i in 1:N
            idx = [N:-1:i + 1; i - 1:-1:1]
            VB = prod(i -> tnsr_size[i], idx)
            V[1:VB, :] = reduce(khatrirao, factors[idx])
            factors[i] = _row_unfold(tnsr, i) * V[1:VB, :] / reduce(.*, gram[idx])
            sum!(lbds, abs(factors[i]))
            factors[i] ./= lbds
            At_mul_B!(gram[i], factors[i], factors[i])
        end
        resid_old = resid
        resid = vecnorm(V[1:VB, :] * (factors[N] .* lbds)' - tnsr_flat)
        converged = abs(resid - resid_old) < tol * resid_old
        niters += 1
    end

    verbose && _iter_status(converged, niters, maxiter)

    return CANDECOMP((factors...), squeeze(lbds, 1))
end

"""
Computes CANDECOMP by SGSD (Simultaneous Generalized Schur Decomposition) method.
"""
function _candecomp{T,N}(
  method::Type{Val{:SGSD}},
  tnsr::StridedArray{T,N},
  r::Int,
  factors::Vector{Matrix{Float64}},
  tol::Float64,
  maxiter::Integer,
  verbose::Bool)

    N==3 || throw(ArgumentError("This algorithm only applies to 3-mode tensors."))
    (n1, n2, n3) = size(tnsr)
    IB = [min(n1 - 1, r), (n2 == r) ? 2 : 1]
    Q = qr(factors[1], thin=false)[1]
    Z = qr(flipdim(factors[2], 2), thin=false)[1]
    q = Array(Float64, n1, n1)
    z = Array(Float64, n2, n2)
    R = tensorcontract(tensorcontract(tnsr, [1, 2, 3], Q, [1, 4], [4, 2, 3]), [1, 2, 3], Z, [2, 4], [1, 4, 3])

    res = vecnorm(tnsr)
    converged = false
    niters = 0
    @inbounds while !converged && niters < maxiter
        q = eye(n1)
        z = eye(n2)

        for i in 1:IB[1]
            q[:, i:n1] *= svd(q[:, i:n1]' * slice(R, :, n2 - r + i, :))[1]
        end
        R = tensorcontract(R, [1, 2, 3], q, [1, 4], [4, 2, 3])
        for i in r:-1:IB[2]
            z[:, 1:n2 - r + i] *= flipdim(svd(slice(R, i, :, :)' * z[:, 1:n2 - r + i])[3], 2)
        end
        Q *= q
        Z *= z
        R = tensorcontract(tensorcontract(tnsr, [1, 2, 3], Q, [1, 4], [4, 2, 3]), [1, 2, 3], Z, [2, 4], [1, 4, 3])

        res_old = res
        res = vecnorm(tril(squeeze(sum(R .^ 2, 3), 3), n2 - r - 1))
        converged = abs(res - res_old) < tol
        niters += 1
    end

    verbose && _iter_status(converged, niters, maxiter)

    R = R[1:r, n2-r+1:n2, :]
    M = cat(3, eye(r, r), eye(r, r))
    @inbounds for i in r - 1:-1:1, j = i + 1:r
        d = i + 1:j - 1
        M[i, j, :] = hcat(R[j, j, :][:], R[i, i, :][:]) \ (R[i, j, :][:] - mapslices(R3 -> sum(M[i, d, 1] * (diag(R3)[d] .* M[d, j, 2])), R, [1, 2])[:])
    end

    factors[1] = Q[:, 1:r] * M[:, :, 1]
    factors[2] = Z[:, n2 - r + 1:n2] * M[:, :, 2]'
    factors[3] = _row_unfold(tnsr, 3) * khatrirao(factors[2], factors[1]) / ((factors[2]'factors[2]) .* (factors[1]'factors[1]))

    lbds = ones(1, r)
    for i in 1:3
        lbd = mapslices(vecnorm, factors[i], 1)
        factors[i] ./= lbd
        lbds .*= lbd
    end

    return CANDECOMP((factors...), squeeze(lbds, 1))
end

"""
Codes of implemented CANDECOMP methods.
"""
const CANDECOMP_methods = Symbol[t.parameters[1] for t in filter(t -> isa(t, DataType),
                                                                 [m.sig.parameters[1].parameters[1] for m in methods(_candecomp)])] |> Set{Symbol}
