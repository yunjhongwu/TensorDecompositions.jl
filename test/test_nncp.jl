@testset "Non-negative CANDECOMP" begin
Random.seed!(12345)

r = 2
T = rand_kruskal3(r, (10, 20, 30), true)

@time factors = nncp(T, r, compute_error=true)
@test length(factors.factors) == ndims(T)
@test size.(factors.factors) == ntuple(i -> (size(T, i), r), 3)
@test rank(factors) == r
@test rel_residue(factors) < 0.05

end
