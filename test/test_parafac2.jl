facts("PARAFAC2") do

srand(1)
r = 2
A = randn(r, 10)
T = Array{Float64, r}[(randn(4 * (i + 1), r) * A) for i in 1:5]
@time factors = parafac2(T, 2, compute_error=true)
@fact map(size, factors.factors) --> ([(size(t, 1), r) for t in T]...)
@fact length(factors.D) --> length(T)
@fact map(size, factors.D) --> fill((1, r), length(T))
@fact size(factors.A) --> (size(T[1], 2), r)
@fact rel_residue(factors) --> less_than(0.05)

end
