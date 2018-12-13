"""
Parallel Factor Analysis (PARAFAC) decomposition
"""
struct PARAFAC2{T<:Number, N} <: TensorDecomposition{T, N}
    factors::NTuple{N, Matrix{T}}
    B::Matrix{T}
    A::Matrix{T}
    props::Dict{Symbol, Any}

    function PARAFAC2(
                   X::Vector{Matrix{T}},
                   F::Matrix{T},
                   B::Matrix{T},
                   A::Matrix{T}) where T
        N = length(X)
        @assert size(B, 2) == N
        factors = ntuple(N) do i
            u = svd((F .* view(B, :, i)) * (X[i] * A)')
            return u.V * (u.U'F)
        end
        return new{T, N}(factors, B, A, Dict{Symbol,Any}())
    end

end

"""
PARAFAC2 model
"""
function parafac2(X::AbstractVector{<:StridedMatrix{T}},
                  r::Integer;
                  tol::Float64=1e-5,
                  maxiter::Integer=100,
                  compute_error::Bool=false,
                  verbose::Bool=true) where T
    m = length(X)
    n = size(X[1], 2)
    all(Xi -> size(Xi, 2) == n, X) ||
        throw(DimensionMismatch("All input matrices should have the same number of rows."))

    F = Matrix{T}(I, r, r)
    B = fill(T(1.0), m, r)
    A = eigen(Symmetric(sum(Xi -> Xi'Xi, X)), (n-r+1):n).vectors
    G1 = Matrix{T}(I, r, r); G2 = copy(G1); G3 = fill(T(m), r, r)
    H = Matrix{T}[(size(X[i], 1) > size(X[i], 2) ? qr(X[i]).R : X[i]) for i in 1:m]
    P = [Matrix{T}(undef, r, size(H[i], 1)) for i in 1:m]
    PH = Array{T, 3}(undef, r, n, m)

    niters = 0
    converged = false
    resid = resid0 = norm(X)
    pb = Progress(maxiter, "PARAFAC2 iterations ")
    while !converged && niters < maxiter
        update!(pb, niters)
        for i in eachindex(H)
            u = svd((F .* view(B, i, :)) * (H[i] * A)')
            mul!(P[i], u.U, u.Vt)
            mul!(view(PH, :, :, i), P[i], H[i])
        end
        mul!(F, _row_unfold(PH, 1), khatrirao(B, A) / (G3 .* G2))
        #F = _row_unfold(PH, 1) * khatrirao(B, A) / (G3 .* G2)
        mul!(G1, F', F)
        mul!(A, _row_unfold(PH, 2), khatrirao(B, F) / (G3 .* G1))
        #A = _row_unfold(PH, 2) * khatrirao(B, F) / (G3 .* G1)
        mul!(G2, A', A)
        mul!(B, _row_unfold(PH, 3), khatrirao(A, F) / (G2 .* G1))
        #B = _row_unfold(PH, 3) * khatrirao(A, F) / (G2 .* G1)
        mul!(G3, B', B)

        resid_old = resid
        resid = sum(k -> sum(abs2, H[k] .- P[k]' * ((F .* view(B, k, :)) * A')), 1:m)
        converged = abs(resid - resid_old) < tol * resid_old

        niters += 1
    end
    finish!(pb)
    verbose && _iter_status(converged, niters, maxiter)
    res = PARAFAC2(X, F, Matrix(B'), A)
    compute_error && _set_rel_residue(res, sqrt(resid) / resid0)
    return res
end
