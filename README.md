# TensorDecompositions.jl

A Julia implementation of tensor decomposition algorithms 

[![Build Status](https://travis-ci.org/yunjhongwu/TensorDecompositions.jl.svg?branch=master)](https://travis-ci.org/yunjhongwu/TensorDecompositions.jl) [![Coverage Status](https://coveralls.io/repos/yunjhongwu/TensorDecompositions.jl/badge.svg?branch=master&service=github)](https://coveralls.io/github/yunjhongwu/TensorDecompositions.jl?branch=master)

------- 

### Available functions 

1. The following functions for Tucker decompositions return a `Tucker`, which contains `factors::Array{Array{Float64, 2}, 1}`, `core::Array{Float64}` (1-dimensional array if the core is a diagonal tensor), and the relative reconstruction error.

  - **High-order SVD (HOSVD)** `hosvd(T::StridedArray, r::Integer; compute_core::Bool=false)`; `hosvd` returns the residual only when `core=true` 
  - **Canonical polyadic decomposition (CANDECOMP/PARAFAC)** `candecomp(T::StridedArray, r::Integer, algo::String="als"; tol::Float64=1e-5, maxiter::Integer=100, hosvd_init::Bool=false, compute_res::Bool=true, verbose=true)`; this function provides two algorithms, set by `algo` argument, for fitting the CANDECOMP model:
    - *als*: Alternating least square method [3] 
    - *sgsd*: Simultaneous generalized Schur decomposition [1]
  - **Non-negative CANDECOMP/PARAFAC** by the block-coordinate update method [5] `nncp(T::StridedArray, r::Integer; tol::Float64=1e-5, maxiter::Integer=100, compute_res::Bool=true, verbose::Bool=true)`

2. **Tensor-CUR** for 3-mode tensors [4] is a randomized algorithm and returns a `CUR`, which includes indexes of *c* slabs (along axis *slab_index*) and *r* fibers, matrix *U*, and the relative reconstruction error of slabs. Note that this function samples with replacement, the numbers of repeated samples are stored in `Cweight` and `Rweight`.

  - `tensorcur3(T::StridedArray, c::Integer, r::Integer, slab_index::Integer=3, compute_u::Bool=true)`

3. **PARAFAC2** for 3-mode tensors by the alternating least square method [2] takes an array of matrices, which may not be equally sized, and factorizes the matrices under a constraint on row space. The below function returns a `PARAFAC2`, which contains factors and diagonal effects `D` of each matrix, a common loading matrix `A`, and the relative error.

  - `parafac2{S<:Matrix}(X::Array{S, 1}, r::Integer; tol::Float64=1e-5, maxiter::Integer=100, verbose::Bool=true)`


+ Common parameters:
  - `T::StridedArray`: Data tensor
  - `r::Integer`: Number of components/factors
  - `tol::Float64`: Tolerance to achieve 
  - `maxiter::Integer`: Maximum number of iterations
  - `verbose::Bool`: Print status when iterative algorithms terminate

### Requirements
  - Julia 0.3
  - TensorOperations
  - Distributions

### Future plan

- More algorithms for fitting CANDECOMP/PARAFAC and non-negative tensor decompositions
- Probabilistic tensor decompositions
- Tensor completion algorithms

### Reference

 - [1] De Lathauwer, L., De Moor, B., & Vandewalle, J. (2004). Computation of the canonical decomposition by means of a simultaneous generalized Schur decomposition. SIAM Journal on Matrix Analysis and Applications, 26(2), 295-327.

 - [2] Kiers, H. A., Ten Berge, J. M., & Bro, R. (1999). PARAFAC2-Part I. A direct fitting algorithm for the PARAFAC2 model. Journal of Chemometrics, 13(3-4), 275-294.

 - [3] Kolda, T. G., & Bader, B. W. (2009). Tensor decompositions and applications. SIAM review, 51(3), 455-500.

 - [4] Mahoney, M. W., Maggioni, M., & Drineas, P. (2008). Tensor-CUR decompositions for tensor-based data. SIAM Journal on Matrix Analysis and Applications, 30(3), 957-987.

 - [5] Xu, Y., & Yin, W. (2013). A block coordinate descent method for regularized multiconvex optimization with applications to nonnegative tensor factorization and completion. SIAM Journal on imaging sciences, 6(3), 1758-1789.
