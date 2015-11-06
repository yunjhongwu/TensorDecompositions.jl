facts("PARAFAC2") do

srand(1)
r = 2
A = randn(r, 10)
X = Array{Float64, r}[(randn(4 * (i + 1), r) * A) for i in 1:5]
@time factors = parafac2(X, 2)
@fact map(size, factors.factors) --> ([(size(t, 1), r) for t in T]...)
@fact length(factors.D) --> length(X)
@fact map(size, factors.D) --> fill((1, r), length(X))
@fact size(factors.A) --> (size(X[1], 2), r)
@fact factors.error --> less_than(0.05)

end
