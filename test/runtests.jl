using TensorDecompositions
using Base.Test

srand(1)
T = reshape([1:60], 3,4,5) 
T_norm = vecnorm(T)
S = abs(randn(10, 15, 20))
S_norm = vecnorm(S)
#X = randn(50, 60, 70)
#X_norm = vecnorm(X)

println("HOSVD, Case 1")
@test @time hosvd(T, 2).residual < 1e-7 * T_norm 
println("CANDECOMP, Case 1")
@test @time candecomp(T, 2).residual < 0.005 * T_norm
println("Non-negative CP, Case 1")
@test @time ntfcp(T, 2).residual < 0.06 * T_norm
println("HOSVD, Case 2")
@test @time hosvd(S, 2).residual < 0.6 * S_norm 
println("CANDECOMP, Case 2")
@test @time candecomp(S, 2).residual < 0.6 * S_norm
println("Non-negative CP, Case 2")
@test @time ntfcp(S, 2).residual < 0.6 * S_norm
#println("Tensor-CUR")
#println(tensorcur(X, 10, 30).residual / X_norm)


println("All tests passed.")

