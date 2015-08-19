println("CANDECOMP")
r = 2
T = _kruskal3_generator(r, (10, 20, 30), 1, false)

println(" - case 1: alternating least square")
@time factors = candecomp(T, r, algo="als")
@test length(factors.factors) == ndims(T)
for i in 1:ndims(T)
    @test size(factors.factors[i]) == (size(T, i), r)
end
@test length(factors.core) == r
@test factors.error < 1e-5

println(" - Case 2: Simultaneous generalized Schur decomposition")
@time factors = candecomp(T, r, algo="sgsd")
@test length(factors.factors) == ndims(T)
for i in 1:ndims(T)
    @test size(factors.factors[i]) == (size(T, i), r)
end
@test length(factors.core) == r
@test factors.error < 1e-5

