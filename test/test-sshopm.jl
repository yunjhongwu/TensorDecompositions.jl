println("SS-HOPM")
println(" - Case 1: Dense represenetation")
eigvec = randn(20)
eigval = vecnorm(eigvec)
eigvec /= eigval

T = eigval * eigvec .* eigvec' .* reshape(eigvec, 1, 1, 20)

@time (lbd, x) = sshopm(T, 1)
@test vecnorm(eigvec - x) < 1e-5
@test vecnorm(eigval - lbd) < 1e-5

println(" - Case 2: Sparse represenetation")
U = Int64[]
V = Float64[]

for i = 1:20, j=i:20, k=j:20
    append!(U, [i, j, k])
    push!(V, T[i, j, k])
end

@time (lbd, x) = sshopm((reshape(U, 3, div(length(U), 3)), V, 20), 1)
@test vecnorm(eigvec - x) < 1e-5
@test vecnorm(eigval - lbd) < 1e-5

