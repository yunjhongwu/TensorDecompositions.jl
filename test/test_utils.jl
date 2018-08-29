@testset "Utilities" begin

    T = rand_kruskal3(2, (10, 20, 30), true)
    @test size(T) == (10, 20, 30)

    @testset "_row_unfold()" begin
        res = TensorDecompositions._row_unfold(T, 1)
        @test size(res) == (10, 600)

        res = TensorDecompositions._row_unfold(T, 2)
        @test size(res) == (20, 300)

        res = TensorDecompositions._row_unfold(T, 3)
        @test size(res) == (30, 200)
    end

    @testset "_col_unfold()" begin
        res = TensorDecompositions._col_unfold(T, 1)
        @test size(res) == (600, 10)

        res = TensorDecompositions._col_unfold(T, 2)
        @test size(res) == (300, 20)

        res = TensorDecompositions._col_unfold(T, 3)
        @test size(res) == (200, 30)
    end

    @testset "tensorcontractmatrices()" begin
        factors = TensorDecompositions._random_factors(size(T), (5, 2, 6))
        res = tensorcontractmatrices(T, factors)
        @test size(res) == (5, 2, 6)
    end

end
