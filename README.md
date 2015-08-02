# Tensor Decompositions 

A Julia implementation of tensor decomposition algorithms 

[![Build Status](https://travis-ci.org/yunjhongwu/TensorDecompositions.svg?branch=master)](https://travis-ci.org/yunjhongwu/TensorDecompositions)


-------

### Available functions
All the following functions (except `tensorcur`) return a `Factors`, which contains `factors::Array{Array{Float64, 2}, 1}`, `core::Array{Float64}` (1-dimensional array if the core is a diagonal tensor), and the relative reconstruction error.

- High-order SVD (HOSVD) `hosvd(T::StridedArray, rank::Integer; core::Bool=false)`; `hosvd` returns the residual only when `core=true`
- Canonical polyadic decomposition (CANDECOMP/PARAFAC) `candecomp(T::StridedArray, rank::Integer; tol::Float64=1e-5, max_iter::Integer=100, hosvd_init::Bool=false])`
- Non-negative CANDECOMP/PARAFAC by block-coordinate update [2] `ntfcp(T::StridedArray, rank::Integer; tol::Float64=1e-5, max_iter::Integer=100)`

Tensor-CUR for 3-mode tensor returns indexes of *c* slabs (along axis *slab_index*) and *r* fibers, matrix *U*, and the relative reconstruction error of slabs. Note that this function samples with replacement, the numbers of repeated samples are stored in `Cweight` and `Rweight`.

- Tensor-CUR decomposition [3] for 3-mode tensors `tensorcur3(T::StridedArray, c::Integer, r::Integer, slab_index::Integer=3)`

### Future plan

- Binary tensor decompositions
- Tensor completion algorithms

### Reference

 - [1] Kolda, T. G., & Bader, B. W. (2009). Tensor decompositions and applications. SIAM review, 51(3), 455-500.

 - [2] Xu, Y., & Yin, W. (2013). A block coordinate descent method for regularized multiconvex optimization with applications to nonnegative tensor factorization and completion. SIAM Journal on imaging sciences, 6(3), 1758-1789.

 - [3] Mahoney, M. W., Maggioni, M., & Drineas, P. (2008). Tensor-CUR decompositions for tensor-based data. SIAM Journal on Matrix Analysis and Applications, 30(3), 957-987.
