println("CANDECOMP")
r = 2
srand(1)
T = zeros(10, 20, 30)
for i = 1:r
    T += tensorcontract(tensorcontract(randn(size(T, 1), 1, 1), [1, 2, 3],
                                       randn(size(T, 2), 1), [4, 2], [1, 4, 3]), [1, 4, 3],
                                       randn(size(T, 3), 1), [5, 3], [1, 4, 5])
end                   
println(" - Case 1: Alternating least square")
@time factors = candecomp(T, r, algo="als")
@test length(factors.factors) == ndims(T)
for i = 1:ndims(T)
    @test size(factors.factors[i]) == (size(T, i), r)
end
@test length(factors.core) == r
@test factors.error < 1e-5

println(" - Case 2: Simultaneous generalized Schur decomposition")
@time factors = candecomp(T, r, algo="sgsd")
@test length(factors.factors) == ndims(T)
for i = 1:ndims(T)
    @test size(factors.factors[i]) == (size(T, i), r)
end
@test length(factors.core) == r
@test factors.error < 1e-5


