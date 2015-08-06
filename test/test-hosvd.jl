println("HOSVD")
r = 2
T = _kruskal3_generator(r, (10, 20, 30), 1, false)

println(" - Case 1: HOSVD algorithm")
@time factors = hosvd(T, r, compute_core=false)
@test length(factors.factors) == ndims(T)
for i in 1:ndims(T)
    @test size(factors.factors[i]) == (size(T, i), r)
end
@test size(factors.core) == (0,)
@test isnan(factors.error) 

println(" - Case 2: Core reconstruction and residuals")
@time factors = hosvd(T, r)
@test size(factors.core) == (r, r, r)
@test factors.error < 1e-5


