println("PARAFAC2")
srand(1)
r = 2
A = randn(r, 10)
X = Array{Float64, r}[(randn(4 * (i + 1), r) * A) for i = 1:5]

@time factors = parafac2(X, 2)
for i = 1:length(X)
    @assert size(factors.factors[i], 1) == size(X[i], 1)
    @assert size(factors.factors[i], 2) == r
end
@assert size(factors.D) == (length(X), r)
@assert size(factors.A, 1) == size(X[1], 2)
@assert size(factors.A, 2) == r
@assert factors.error < 0.5
