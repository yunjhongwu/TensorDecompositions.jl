using TensorDecompositions
using Base.Test

srand(1)
T = reshape([1:60], 3,4,5) 
S = abs(randn(10, 15, 20))
X = randn(50, 60, 70)

println("HOSVD, Case 1")
@test @time hosvd(T, 2).error < 1e-7  
println("CANDECOMP, Case 1")
@test @time candecomp(T, 2).error < 0.01
println("Non-negative CP, Case 1")
@test @time ntfcp(T, 2).error < 0.1
println("HOSVD, Case 2")
@test @time hosvd(S, 5).error < 0.6 
println("CANDECOMP, Case 2")
@test @time candecomp(S, 5).error < 0.6
println("Non-negative CP, Case 2")
@test @time ntfcp(S, 5).error < 0.6
println("Tensor-CUR")
@test @time mean(tensorcur3(X, 25, 2100).error .< 0.01) > 0.3

println("All tests passed.")

