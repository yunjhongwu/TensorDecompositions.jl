facts("CANDECOMP") do
    r = 2
    T = rand_kruskal3(r, (10, 20, 30), false)
    initial_guess = tuple([randn(dim, r) for dim in size(T)]...)
    
    context("ALS (Alternating least squares)") do
        @time factors = candecomp(T, r, initial_guess, tol=1e-6, compute_error=true, method=:ALS)
        @fact length(factors.factors) --> ndims(T)
        @fact map(size, factors.factors) --> (collect(zip(size(T), (r, r, r)))...)
        @fact rank(factors) --> r
        @fact rel_residue(factors) --> less_than(1e-5)
    end
    
    context("SGSD (Simultaneous generalized Schur decomposition)") do
        @time factors = candecomp(T, r, initial_guess, compute_error=true, method=:SGSD)
        @fact length(factors.factors) --> ndims(T)
        @fact map(size, factors.factors) --> (collect(zip(size(T), (r, r, r)))...)
        @fact rank(factors) --> r
        @fact length(factors.lambdas) --> r
        @fact rel_residue(factors) --> less_than(1e-5)
    end
    
end
