println("Tensor-CUR")
r = 2
srand(1)
T = zeros(20, 40, 60)
for i = 1:r
    T += tensorcontract(tensorcontract(rand(size(T, 1), 1, 1), [1, 2, 3],
                                       rand(size(T, 2), 1), [4, 2], [1, 4, 3]), [1, 4, 3],
                                       rand(size(T, 3), 1), [5, 3], [1, 4, 5])
end                   

@time factors = tensorcur3(T, 3, 20)
@test sum(factors.Cweight) == 3
@test sum(factors.Rweight) == 20 
@test all(factors.error .< 1e-5)

@time factors = tensorcur3(T, 10, 200, compute_u=false)
@test sum(factors.Cweight) == 10 
@test sum(factors.Rweight) == 200 
@test size(factors.U) == (0, 0) 
@test size(factors.error) == (0,)

