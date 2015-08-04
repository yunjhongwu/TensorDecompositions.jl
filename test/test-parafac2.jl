println("PARAFAC2")
srand(1)
r = 2
A = randn(r, 10)
X = Array{Float64, r}[(randn(4 * (i + 1), r) * A) for i = 1:5]

@time factors = parafac2(X, 2)
for i = 1:length(X)
    @assert size(factors.factors[i]) == (size(X[i], 1), r)
    @assert size(factors.D[i]) == (1, r)
    
end

@assert length(factors.D) == length(X)
@assert size(factors.A) == (size(X[1], 2), r)
@assert factors.error < 0.05
