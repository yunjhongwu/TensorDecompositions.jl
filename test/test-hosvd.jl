facts("HO-SVD") do
r = 2
T = rand_kruskal3(r, (10, 20, 30), false)

context("no residuals calculation") do
    @time factors = hosvd(T, r)
    @fact length(factors.factors) --> ndims(T)
    @fact map(size, factors.factors) --> (collect(zip(size(T), (r, r, r)))...)
    @fact size(factors.core) --> (r, r, r)
    @fact rel_residue(factors) --> isnan
end

context("core reconstruction and residuals") do
    @time factors = hosvd(T, r, compute_error=true)
    @fact size(factors.core) --> (r, r, r)
    @fact rel_residue(factors) --> less_than(1e-5)
end

context("core dimension equal to the original dimension") do
    @time factors = hosvd(randn(10, 20, 30), (10, 20, 5), compute_error=true)
    @fact size(factors.core) --> (10, 20, 5)
    @fact rel_residue(factors) --> less_than(1e-5)
end

end
