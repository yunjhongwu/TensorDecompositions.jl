@testset "PARAFAC2" begin

Random.seed!(1)
r = 2
A = randn(r, 10)
T = [(randn(4 * (i + 1), r) * A) for i in 1:5]
@time factors = parafac2(T, 2, compute_error=true)
@test size.(factors.factors) == ntuple(i -> (size(T[i], 1), r), length(T))
@test size(factors.B) == (r, length(T))
@test size(factors.A) == (size(T[1], 2), r)
@test rel_residue(factors) < 0.05

end
