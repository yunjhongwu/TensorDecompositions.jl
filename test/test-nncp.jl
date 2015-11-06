facts("Non-negative CANDECOMP") do

r = 2
T = rand_kruskal3(r, (10, 20, 30), true)

@time factors = nncp(T, r, compute_error=true)
@fact length(factors.factors) --> ndims(T)
@fact map(size, factors.factors) --> (collect(zip(size(T), (r, r, r)))...)
@fact rank(factors) --> r
@fact rel_residue(factors) --> less_than(0.05)

end
