# Tensor Decompositions 

A Julia implementation of tensor decomposition algorithms 

-------

### Available functions
All functions return a `Factors`, which contains `factors::Array{Array{Float64, 2}, 1}`, `core::Array{Float64}` (1-dimensional array if the core is a diagonal tensor), and a residual value `residual::Float64`.

- `hosvd(T::StridedArray, rank::Integer; core::Bool=false)`, (truncated) High-order SVD (HOSVD); `hosvd` returns the residual only when `core=true`
- `candecomp(T::StridedArray, rank::Integer; tol::Float64=1e-5, max_iter::Integer=100, hosvd\_init::Bool=false])`, canonical polyadic decomposition (CANDECOMP/PARAFAC) 
- `ntfcp(T::StridedArray, rank::Integer; tol::Float64=1e-5, max_iter::Integer=100)`, non-negative CANDECOMP/PARAFAC by block-coordinate update [2]

### Future plan

- Binary tensor decompositions
- CUR tensor decomposition
- Tensor completion algorithms

### Reference

 - [1] Kolda, T. G., & Bader, B. W. (2009). Tensor decompositions and applications. SIAM review, 51(3), 455-500.

 - [2] Xu, Y., & Yin, W. (2013). A block coordinate descent method for regularized multiconvex optimization with applications to nonnegative tensor factorization and completion. SIAM Journal on imaging sciences, 6(3), 1758-1789.

