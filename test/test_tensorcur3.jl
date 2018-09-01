@testset "Tensor-CUR" begin
Random.seed!(12345)

@testset "Small case" begin
    T = rand_kruskal3(2, (20, 40, 60), false)
    @testset "slab axis: $i" for i in 1:3
        @time cur3 = tensorcur3(T, 3, 15, i)
        @test sum(cur3.Cweight) == 3
        @test sum(cur3.Rweight) == 15
        @test all(err -> err < 1e-5, cur3.error)
    end
end

@testset "Large case without reconstruction" begin
    T = rand_kruskal3(2, (100, 120, 60), false)
    @time factors = tensorcur3(T, 10, 200, compute_u=false)
    @test sum(factors.Cweight) == 10
    @test sum(factors.Rweight) == 200
    @test size(factors.U) == (0, 0)
    @test size(factors.error) == (0,)
end

end
