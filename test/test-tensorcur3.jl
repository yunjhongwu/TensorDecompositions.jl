println("Tensor-CUR")
T = _kruskal3_generator(2, (20, 40, 60), 1, false)

println(" - Case 1: Small case")
for i in 1:3
    println(string("    - Slab axis: ", i))
    @time factors = tensorcur3(T, 3, 15, i)
    @test sum(factors.Cweight) == 3
    @test sum(factors.Rweight) == 15 
    @test all(factors.error .< 1e-5)
end

T = _kruskal3_generator(2, (100, 120, 60), 1, false)

println(" - Case 2: Large case without reconstruction")
@time factors = tensorcur3(T, 10, 200, compute_u=false)
@test sum(factors.Cweight) == 10 
@test sum(factors.Rweight) == 200 
@test size(factors.U) == (0, 0) 
@test size(factors.error) == (0,)

