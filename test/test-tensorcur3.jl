facts("Tensor-CUR") do

context("Small case") do
    T = _kruskal3_generator(2, (20, 40, 60), 1, false)

    for i in 1:3
        context("slab axis: $i") do
            @time factors = tensorcur3(T, 3, 15, i)
            @fact sum(factors.Cweight) --> 3
            @fact sum(factors.Rweight) --> 15
            @fact all(factors.error .< 1e-5) --> true
        end
    end
end

context("Large case without reconstruction") do
    T = _kruskal3_generator(2, (100, 120, 60), 1, false)

    @time factors = tensorcur3(T, 10, 200, compute_u=false)
    @fact sum(factors.Cweight) --> 10
    @fact sum(factors.Rweight) --> 200
    @fact size(factors.U) --> (0, 0)
    @fact size(factors.error) --> (0,)
end

end
