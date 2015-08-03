type PARAFAC2 
    factors::Array{Array{Float64, 2}, 1}
    A::Array{Float64, 2}
    error::Float64

    function PARAFAC2(factors::Array{Array{Float64, 2}, 1},
                      A::Array{Float64, 2}, err::Float64)

    end
end

function parafac2{S<:Matrix}(X::Array{S, 1}, 
                             r::Integer;
                             tol::Float64=1e-5,
                             max_iters::Integer=100,
                             verbose::Bool=true)
    println("start")

    m = length(X)
    n = size(X[1], 2)
    for i = 2:m
        @assert size(X[i], 2) == n
    end
    
    D = [ones(1, r) for i = 1:m]
    F = eye(r)
    C = [X[i]' * X[i] for i = 1:m]
    A = eigs(sum(C), nev=r)[2]
    G = Array{Float64, 2}[eye(r), eye(r), ones(r, r) * m]
   
    niters = 0
    conv = false
    err = vecnorm(A)
    while !conv && niters < max_iters
        println(niters)
        P = map(U -> U[3] * U[1]', map((Xi, Di) -> svd(F .* Di * A' * Xi'), X, D))
        T = cat(3, [P[i]' * X[i] for i = 1:m]...)
#=
        F = _unfold(T, 1) * _KhatriRao(D, A) * pinv(G[3] .* G[2])
        G[1] = F' * F
        F = _unfold(T, 2) * _KhatriRao(D, F) * pinv(G[3] .* G[1])
        G[2] = A' * A
        F = _unfold(T, 3) * _KhatriRao(A, F) * pinv(G[2] .* G[1])
        G[3] = D' * D

    =#
        D = [D[i, :] for i = 1:m]
        err_old = err
        err = sum(map((Xi, Pi, Di) -> vecnorm(Xi - Pi * F .* Di * A'), X, P, D))
        println(niters, " ", err / sum(map(vecnorm, X)), " ", abs(err - err_old) / err_old)
        println(vecnorm(A), " ", vecnorm(F))

        niters += 1
    end

    if !conv && verbose
        println("Warning: Iterations did not converge.")
    end

end
