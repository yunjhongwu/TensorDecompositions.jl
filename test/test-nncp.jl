println("Non-negative CANDECOMP")
r = 2
srand(1)
T = zeros(10, 20, 30)
for i = 1:r
    T += tensorcontract(tensorcontract(rand(size(T, 1), 1, 1), [1, 2, 3],
                                       rand(size(T, 2), 1), [4, 2], [1, 4, 3]), [1, 4, 3],
                                       rand(size(T, 3), 1), [5, 3], [1, 4, 5])
end                   

@time factors = nncp(T, r)
@test length(factors.factors) == ndims(T)
for i = 1:ndims(T)
    @test size(factors.factors[i]) == (size(T, i), r)
end
@test length(factors.core) == r
@test factors.error < 0.05 
