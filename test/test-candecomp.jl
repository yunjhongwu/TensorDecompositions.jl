facts("CANDECOMP") do
r = 2
T = _kruskal3_generator(r, (10, 20, 30), 1, false)

context("ALS (Alternating least squares)") do
    @time factors = candecomp(T, r, algo="als")
    @fact length(factors.factors) --> ndims(T)
    @fact map(size, factors.factors) --> (collect(zip(size(T), (r, r, r)))...)
    @fact length(factors.lmbds) --> r
    @fact factors.error --> less_than(1e-5)
end

context("SGSD (Simultaneous generalized Schur decomposition)") do
    @time factors = candecomp(T, r, algo="sgsd")
    @fact length(factors.factors) --> ndims(T)
    @fact map(size, factors.factors) --> (collect(zip(size(T), (r, r, r)))...)
    @fact length(factors.lmbds) --> r
    @fact length(factors.core) --> r
    @fact factors.error --> less_than(1e-5)
end

end
