facts("SS-HOPM") do

T = eigval * eigvec .* eigvec' .* reshape(eigvec, 1, 1, 20)

context("Dense represenetation") do
    eigvec = randn(20)
    eigval = vecnorm(eigvec)
    eigvec /= eigval

    @time (lbd, x) = sshopm(T, 1)
    @fact vecnorm(eigvec - x) --> less_than(1e-5)
    @fact vecnorm(eigval - lbd) --> less_than(1e-5)
end

context("Sparse represenetation") do
    U = Int64[]
    V = Float64[]

    for i = 1:20, j=i:20, k=j:20
        append!(U, [i, j, k])
        push!(V, T[i, j, k])
    end

    @time (lbd, x) = sshopm((reshape(U, 3, div(length(U), 3)), V, 20), 1)
    @fact vecnorm(eigvec - x) --> less_than(1e-5)
    @fact vecnorm(eigval - lbd) --> less_than(1e-5)
end

end
