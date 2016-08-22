"""
Parallel Factor Analysis (PARAFAC) decomposition
"""
immutable PARAFAC2{T<:Number, N} <: TensorDecomposition{T, N}
    factors::NTuple{N, Matrix{T}}
    D::Vector{Matrix{T}}
    A::Matrix{T}
    props::Dict{Symbol, Any}

    function (::Type{PARAFAC2}){T}(
                   X::Vector{Matrix{T}},
                   F::Matrix{Float64},
                   D::Vector{Matrix{Float64}},
                   A::Matrix{Float64})
        factors = (map(function (Xi, Di)
            U = svd(A_mul_Bt(F .* Di, Xi * A))
            U[3] * (U[1]'F)
        end, X, D)...)
        return new{T, length(factors)}(factors, D, A, Dict{Symbol,Any}())
    end

end

"""
PARAFAC2 model
"""
function parafac2{S<:Matrix}(X::Vector{S},
                             r::Integer;
                             tol::Float64=1e-5,
                             maxiter::Integer=100,
                             compute_error::Bool=false,
                             verbose::Bool=true)
    m = length(X)
    n = size(X[1], 2)
    for i in 2:m
        size(X[i], 2) == n || error("All input matrices should have the same number of rows.")
    end

    F = eye(r)
    D = Matrix{Float64}[ones(1, r) for _ in 1:m]
    A = eigs(sum(map(Xi -> Xi'Xi, X)), nev=r)[2]
    G = Matrix{Float64}[eye(r), eye(r), ones(r, r) * m]
    H = Matrix{Float64}[(size(X[i], 1) > size(X[i], 2) ? qr(X[i])[2]: X[i]) for i in 1:m]
    P = Array(Matrix{Float64}, m)

    niters = 0
    converged = false
    resid = vecnorm(vcat(X...))
    while !converged && niters < maxiter
        map!(function (Hi, Di)
            U = svd((F .* Di) * (Hi * A)')
            U[3]*U[1]'
        end, P, H, D)
        T = cat(3, [P[i]'H[i] for i in 1:m]...)

        B = vcat(D...)
        F = _row_unfold(T, 1) * khatrirao(B, A) / (G[3] .* G[2])
        At_mul_B!(G[1], F, F)
        A = _row_unfold(T, 2) * khatrirao(B, F) / (G[3] .* G[1])
        At_mul_B!(G[2], A, A)
        B = _row_unfold(T, 3) * khatrirao(A, F) / (G[2] .* G[1])
        At_mul_B!(G[3], B, B)

        D = Matrix{Float64}[B[i, :] for i in 1:m]

        resid_old = resid
        resid = sum(k -> sumabs2(H[k] - P[k] * A_mul_Bt(F .* D[k], A)), 1:m)
        converged = abs(resid - resid_old) < tol * resid_old

        niters += 1
    end

    verbose && _iter_status(converged, niters, maxiter)
    res = PARAFAC2(X, F, D, A)
    if compute_error
        _set_rel_residue(res, sqrt(resid) / vecnorm(vcat(X...)))
    end
    return res
end
