facts("Non-negative CANDECOMP") do

r = 2
T = _kruskal3_generator(r, (10, 20, 30), 1, true)

@time factors = nncp(T, r)
@fact length(factors.factors) --> ndims(T)
@fact map(size, factors.factors) --> (collect(zip(size(T), (r, r, r)))...)
@fact length(factors.core) --> r
@fact factors.error --> less_than(0.05)

end
