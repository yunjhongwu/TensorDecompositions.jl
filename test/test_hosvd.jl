@testset "HO-SVD" begin
    r = 2
    T = rand_kruskal3(r, (10, 20, 30), false)

    @testset "no residuals calculation" begin
        @time factors = hosvd(T, r)
        @test length(factors.factors) == ndims(T)
        @test size.(factors.factors) == ntuple(i -> (size(T, i), r), ndims(T))
        @test size(factors.core) == (r, r, r)
        @test isnan(rel_residue(factors))
    end

    @testset "core reconstruction and residuals" begin
        @time factors = hosvd(T, r, compute_error=true)
        @test size(factors.core) == (r, r, r)
        @test rel_residue(factors) < 1e-5
    end

    @testset "core dimension equal to the original dimension" begin
        @time factors = hosvd(randn(10, 20, 30), (10, 15, 5), compute_error=true)
        @test size(factors.core) == (10, 15, 5)
    end

end
