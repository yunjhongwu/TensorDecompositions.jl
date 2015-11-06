facts("CANDECOMP") do
r = 2
T = _kruskal3_generator(r, (10, 20, 30), 1, false)

context("Incorrect method") do
  @fact_throws candecomp(T, r, compute_error=true, method=:ALdS) ArgumentError
end

context("ALS (Alternating least squares)") do
    @time factors = candecomp(T, r, compute_error=true, method=:ALS)
    @fact length(factors.factors) --> ndims(T)
    @fact map(size, factors.factors) --> (collect(zip(size(T), (r, r, r)))...)
    @fact rank(factors) --> r
    @fact rel_residue(factors) --> less_than(1e-5)
end

context("SGSD (Simultaneous generalized Schur decomposition)") do
    @time factors = candecomp(T, r, compute_error=true, method=:SGSD)
    @fact length(factors.factors) --> ndims(T)
    @fact map(size, factors.factors) --> (collect(zip(size(T), (r, r, r)))...)
    @fact rank(factors) --> r
    @fact length(factors.lambdas) --> r
    @fact rel_residue(factors) --> less_than(1e-5)
end

end
