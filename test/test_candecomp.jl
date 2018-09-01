@testset "CANDECOMP" begin
    Random.seed!(12345)

    r = 2 # core tensor dimensions
    T = rand_kruskal3(r, (10, 20, 30), false)
    initial_guess = ntuple(k -> randn(size(T, k), r), ndims(T))

    @testset "ALS (Alternating least squares)" begin
        @time factors = candecomp(T, r, initial_guess, tol=1e-6, compute_error=true, method=:ALS)
        @test length(factors.factors) == ndims(T)
        @test size.(factors.factors) == ntuple(i -> (size(T, i), r), ndims(T))
        @test rank(factors) == r
        @test rel_residue(factors) < 1e-5
    end

    @testset "SGSD (Simultaneous generalized Schur decomposition)" begin
        @time factors = candecomp(T, r, initial_guess, compute_error=true, method=:SGSD)
        @test length(factors.factors) == ndims(T)
        @test size.(factors.factors) == ntuple(i -> (size(T, i), r), ndims(T))
        @test rank(factors) == r
        @test length(factors.lambdas) == r
        @test_broken rel_residue(factors) < 1e-5
    end

end
