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
Base.rank(decomp::CANDECOMP) = length(decomp.lambdas)

"""
Re-composes the tensor from CANDECOMP decomposition.
"""
@generated function compose!(dest::Array{T,N}, factors::NTuple{N, Matrix{T}}, lambdas::Vector{T}) where {T,N}
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

@generated function compose(factors::NTuple{N, Matrix{T}}, lambdas::Vector{T}) where {T,N}
  quote
    compose!(Array{T}(@ntuple $N i -> size(factors[i], 1)), factors, lambdas)
  end
end

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
    verbose && info("initializing factor matrices...")
    all([(size(tnsr, i), r) == size(initial_guess[i]) for i in 1:N]) || throw(ArgumentError("dimension of initial guess does not match input tensor."))
    factors = collect(Matrix{T}, initial_guess)
    verbose && info("applying candecomp $method method...")
    res = _candecomp(Val{method}, tnsr, r, factors, tol, maxiter, verbose)
    if compute_error
      _set_rel_residue(res, tnsr)
    end
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
            VB = prod(k -> tnsr_size[k], idx)
            V[1:VB, :] = reduce(khatrirao, factors[idx])
            factors[i] = _row_unfold(tnsr, i) * V[1:VB, :] / reduce(((x, y) -> broadcast(*, x, y)), gram[idx])
            sum!(lbds, abs.(factors[i]))
            factors[i] ./= lbds
            At_mul_B!(gram[i], factors[i], factors[i])
        end
        resid_old = resid
        resid = vecnorm(V[1:VB, :] * broadcast(*, factors[N], lbds)' - tnsr_flat)
        converged = abs.(resid - resid_old) < tol * resid_old
        niters += 1
    end

    verbose && _iter_status(converged, niters, maxiter)

    return CANDECOMP(tuple(factors...), squeeze(lbds, 1))
end

"""
Computes CANDECOMP by SGSD (Simultaneous Generalized Schur Decomposition) method.
"""
function _candecomp(
    method::Type{Val{:SGSD}},
    tnsr::StridedArray{T,N},
    r::Int,
    factors::Vector{Matrix{Float64}},
    tol::Float64,
    maxiter::Integer,
    verbose::Bool) where {T,N}

    ndims(tnsr) == 3 || throw(ArgumentError("This algorithm only applies to 3-mode tensors."))
    (n1, n2, n3) = size(tnsr)
    IB = [min(n1 - 1, r), (n2 == r) ? 2 : 1]
    Q = qr(factors[1], thin=false)[1]
    Z = qr(flipdim(factors[2], 2), thin=false)[1]
    q = Array{Float64}(n1, n1)
    z = Array{Float64}(n2, n2)

    R = zeros(size(tnsr))
    @tensor R[4,5,3] = tnsr[1,2,3] * Q[1,4] * Z[2,5]

    res = vecnorm(tnsr)
    converged = false
    niters = 0
    @inbounds while !converged && niters < maxiter
        q = eye(n1)
        z = eye(n2)

        for i in 1:IB[1]
            q[:, i:n1] *= svd(q[:, i:n1]' * view(R, :, n2 - r + i, :))[1]
        end

        @tensor R[4,2,3] = R[1,2,3] * q[1,4]

        for i in r:-1:IB[2]
            z[:, 1:n2 - r + i] *= flipdim(svd(view(R, i, :, :)' * z[:, 1:n2 - r + i])[3], 2)
        end
        Q *= q
        Z *= z

        @tensor R[4,5,3] = tnsr[1,2,3] * Q[1,4] * Z[2,5]

        res_old = res
        res = vecnorm(tril(squeeze(sum(R .^ 2, 3), 3), n2 - r - 1))
        converged = abs.(res - res_old) < tol
        niters += 1
    end

    verbose && _iter_status(converged, niters, maxiter)

    R = R[1:r, n2-r+1:n2, :]
    M = cat(3, eye(r, r), eye(r, r))
    @inbounds for i in (r - 1):-1:1, j = (i + 1):r
        if i + 1 < j
            d = (i + 1):(j - 1)
            println(d)

            M[i, j, :] = hcat(r[j, j, :][:], r[i, i, :][:]) \ (r[i, j, :][:] - mapslices(R3 -> sum(M[i, d, 1] * broadcast(*, diag(R3)[d], M[d, j, 2])), R, [1, 2])[:])
        end
    end

    factors[1] = Q[:, 1:r] * M[:, :, 1]
    factors[2] = Z[:, n2 - r + 1:n2] * M[:, :, 2]'
    factors[3] = _row_unfold(tnsr, 3) * khatrirao(factors[2], factors[1]) / broadcast(*, factors[2]'factors[2], factors[1]'factors[1])

    lbds = ones(1, r)
    for i in 1:3
        lbd = mapslices(vecnorm, factors[i], 1)
        factors[i] ./= lbd
        lbds = broadcast(*, lbds, lbd)
    end

    return CANDECOMP(tuple(factors...), squeeze(lbds, 1))
end

"""
Codes of implemented CANDECOMP methods.
"""
const CANDECOMP_methods = Symbol[t.parameters[1] for t in filter(t -> isa(t, DataType),
                                                                 [Base.unwrap_unionall(m.sig).parameters[2].parameters[1] for m in methods(_candecomp)])] |> Set{Symbol}
