type PARAFAC2 
    factors::Array{Array{Float64, 2}, 1}
    D::Array{Array{Float64, 2}, 1}
    A::Array{Float64, 2}
    error::Float64

    function PARAFAC2{S<:Matrix}(X::Array{S, 1},
                                 F::Array{Float64, 2},
                                 D::Array{Array{Float64, 2}, 1},
                                 A::Array{Float64, 2},
                                 res::Float64)

        factors = map(U -> U[3] * U[1]' * F, map((Xi, Di) -> svd(F .* Di * A' * Xi'), X, D))
        return new(factors, D, A, sqrt(res) / vecnorm(vcat(X...)))
    end

end

function parafac2{S<:Matrix}(X::Array{S, 1}, 
                             r::Integer;
                             tol::Float64=1e-5,
                             maxiter::Integer=100,
                             verbose::Bool=true)
    m = length(X)
    n = size(X[1], 2)
    for i = 2:m
        size(X[i], 2) == n || error("All input matrices should have the same number of rows.")
    end
    
    F = eye(r)
    D = Array{Float64, 2}[ones(1, r) for i = 1:m]
    A = eigs(sum(map(Xi -> Xi' * Xi, X)), nev=r)[2]
    G = Array{Float64, 2}[eye(r), eye(r), ones(r, r) * m]
    H = Array{Float64, 2}[(size(X[i], 1) > size(X[i], 2) ? qr(X[i])[2]: X[i]) for i = 1:m]
    P = Array(Array{Float64, 2}, m) 
   
    niters = 0
    converged = false
    res = vecnorm(vcat(X...))
    while !converged && niters < maxiter
        P = map(U -> U[3] * U[1]', map((Hi, Di) -> svd(F .* Di * A' * Hi'), H, D))
        T = cat(3, [P[i]' * H[i] for i = 1:m]...) 

        B = vcat(D...)
        F = _unfold(T, 1) * _KhatriRao(B, A) / (G[3] .* G[2])
        G[1] = F' * F
        A = _unfold(T, 2) * _KhatriRao(B, F) / (G[3] .* G[1])
        G[2] = A' * A
        B = _unfold(T, 3) * _KhatriRao(A, F) / (G[2] .* G[1])
        G[3] = B' * B

        D = Array{Float64, 2}[B[i, :] for i = 1:m]

        res_old = res 
        res = sum(map((Hi, Pi, Di) -> sum((Hi - Pi * F .* Di * A') .^ 2), H, P, D))
        converged = abs(res - res_old) < tol * res_old 

        niters += 1
    end
    P = map(U -> U[3] * U[1]', map((Xi, Di) -> svd(F .* Di * A' * Xi'), X, D))
 
    verbose && println(converged ? string("Algorithm converged after ", niters, " iterations.") :
                                   string("Warning: Maximum number (", maxiter, ") of iterations exceeded."))

    return PARAFAC2(X, F, D, A, res)
end

