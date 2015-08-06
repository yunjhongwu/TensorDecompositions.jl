println("Non-negative CANDECOMP")
r = 2
T = _kruskal3_generator(r, (10, 20, 30), 1, true)

@time factors = nncp(T, r)
@test length(factors.factors) == ndims(T)
for i in 1:ndims(T)
    @test size(factors.factors[i]) == (size(T, i), r)
end
@test length(factors.core) == r
@test factors.error < 0.05 
