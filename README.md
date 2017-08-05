# TensorDecompositions.jl

A Julia implementation of tensor decomposition algorithms

[![Build Status](https://travis-ci.org/yunjhongwu/TensorDecompositions.jl.svg?branch=master)](https://travis-ci.org/yunjhongwu/TensorDecompositions.jl) [![Coverage Status](https://coveralls.io/repos/yunjhongwu/TensorDecompositions.jl/badge.svg?branch=master&service=github)](https://coveralls.io/github/yunjhongwu/TensorDecompositions.jl?branch=master)
[![TensorDecompositions](http://pkg.julialang.org/badges/TensorDecompositions_0.6.svg)](http://pkg.julialang.org/?pkg=TensorDecompositions&ver=release)

------- 

### Available functions 

1. The following functions for **Tucker decompositions**, except for `sshopm`, return a `Tucker`, which contains `factors::Vector{Matrix{Float64}}`, `core::Array{Float64}` (1-dimensional array for **Kruskal decompositions**), and the relative reconstruction error `error::Float64`.

  - **High-order SVD (HOSVD)** [3] `hosvd{T,N}(tnsr::StridedArray{T,N}, core_dims::NTuple{N, Int}; pad_zeros::Bool=false, compute_error::Bool=false)` 

  - **Canonical polyadic decomposition (CANDECOMP/PARAFAC)** `candecomp{T,N}(tnsr::StridedArray{T,N}, r::Integer, initial_guess::NTuple{N, Matrix{T}}; method::Symbol=:ALS, tol::Float64=1e-5, maxiter::Integer=100, compute_error::Bool=false, verbose::Bool=true)`. This function provides two algorithms, set by `method` argument, for fitting the CANDECOMP model:
    - *ALS* (default): Alternating least square method [3] 
    - *SGSD*: Simultaneous generalized Schur decomposition [1]

  - **Non-negative CANDECOMP/PARAFAC** by the block-coordinate update method [7] `nncp(tnsr::StridedArray, r::Integer; tol::Float64=1e-4, maxiter::Integer=100, compute_error::Bool=false, verbose::Bool=true)`

  > Remark. Choose a smaller `r` if the above functions throw `ERROR: SingularException`.

  - **Symmetric rank-1 approximation for symmetric tensors** by shifted symmetric higher-order power method (SS-HOPM) [4] `sshopm{T,N}(tnsr::AbstractArray{T,N}, alpha::Real; tol::Float64=1e-5, maxiter::Int=100, verbose::Bool=false)`. This function requires a shifting parameter `alpha` and returns an eigenpair of *tnsr*, represented by a tuple `(Float64, Vector{Float64})`. Note that this functions does NOT check symmetry of input tensors. This implementation takes both dense representation `StridedArray` and sparse representation `(Matrix{Int64}, StridedVector, Integer)`, which contains indexes, as column vectors of the matrix in the first component of the tuple, corresponding values, and dimension of a mode.

  - **Sparse (semi-)nonnegative Tucker decomposition** by the alternating proximal gradient method [6] `spnntucker{T,N}(tnsr::StridedArray{T, N}, core_dims::NTuple{N, Int}; core_nonneg::Bool=true, tol::Float64=1e-4, hosvd_init::Bool=false, max_iter::Int=500, max_time::Float64=0.0, lambdas::Vector{Float64} = fill(0.0, N+1), Lmin::Float64 = 1.0, rw::Float64=0.9999, bounds::Vector{Float64} = fill(Inf, N+1), ini_decomp = nothing, verbose::Bool=false)`. 
  

2. **Tensor-CUR** for 3-mode tensors [5] is a randomized algorithm and returns a `CUR`, which includes indexes of *c* slabs (along axis *slab_index*) and *r* fibers, matrix *U*, and the relative reconstruction error of slabs. Note that this function samples with replacement, and the numbers of repeated samples are stored in `Cweight` and `Rweight`.

  - `tensorcur3(tnsr::StridedArray, c::Integer, r::Integer, slab_axis::Integer=3; compute_u::Bool=true)`

3. **PARAFAC2** for 3-mode tensors by the alternating least square method [2] takes an array of matrices, which may not be equally sized, and factorizes the matrices under a constraint on row space. The below function returns a `PARAFAC2`, which contains factors and diagonal effects *D* of each matrix, a common loading matrix *A*, and the relative error.

  - `parafac2{S<:Matrix}(X::Vector{S}, r::Integer; tol::Float64=1e-5, maxiter::Integer=100, verbose::Bool=true)`


+ Common parameters:
  - `tnsr::StridedArray`: Data tensor
  - `r::Integer`: Number of components/factors
  - `tol::Float64`: Tolerance to achieve 
  - `maxiter::Integer`: Maximum number of iterations
  - `verbose::Bool`: Print status when iterative algorithms terminate

### Quick example
Here is a quick example of code that fits the CANDECOMP model:
```julia
julia> using TensorDecompositions

julia> u = randn(10); v = randn(20); w = randn(30)

# Generate a noisy rank-1 tensor
julia> T = cat(3, map(x -> x * u * v', w)...) + 0.2 * randn(10, 20, 30)

julia> size(T)
(10, 20, 30)

julia> F = candecomp(T, 1, (randn(10, 1), randn(20, 1), randn(30, 1)), compute_error=true, method=:ALS);
NFO: Initializing factor matrices...
INFO: Applying CANDECOMP ALS method...
INFO: Algorithm converged after 4 iterations.

julia> [size(F.factors[i]) for i = 1:3]
3-element Array{Any,1}:
 (10,1)
 (20,1)
 (30,1)

julia> F.props
Dict{Symbol,Any} with 1 entry:
  :rel_residue => 0.18735979193091348

julia> F.factors[1]
10x1 Array{Float64,2}:
  0.0676683 
 -0.0985366 
 -0.239748  
  0.0821674 
 -0.0547672 
 -0.00892641
  0.0220593 
 -0.058075  
 -0.135493  
  0.23256 

```

### Requirements
  - Julia 0.6
  - TensorOperations
  - Distributions
  - ProgressMeter
  - FactCheck

### Performance issues
  - Inefficiency of unfolding (matricizing), which has a significant impact on the performance of `candecomp` (`method:=ALS`) and `parafac2`

### Future plan
  - Improving performance 
  - More algorithms for fitting CANDECOMP/PARAFAC and non-negative tensor decompositions
  - Probabilistic tensor decompositions
  - Algorithms for factorizing sparse tensors
  - Tensor completion algorithms

### Reference
 - [1] De Lathauwer, L., De Moor, B., & Vandewalle, J. (2004). Computation of the canonical decomposition by means of a simultaneous generalized Schur decomposition. *SIAM Journal on Matrix Analysis and Applications*, 26(2), 295-327.

 - [2] Kiers, H. A., Ten Berge, J. M., & Bro, R. (1999). PARAFAC2-Part I. A direct fitting algorithm for the PARAFAC2 model. *Journal of Chemometrics*, 13(3-4), 275-294.

 - [3] Kolda, T. G., & Bader, B. W. (2009). Tensor decompositions and applications. *SIAM Review*, 51(3), 455-500.

 - [4] Kolda, T. G., & Mayo, J. R. (2011). Shifted power method for computing tensor eigenpairs. *SIAM Journal on Matrix Analysis and Applications*, 32(4), 1095-1124.

 - [5] Mahoney, M. W., Maggioni, M., & Drineas, P. (2008). Tensor-CUR decompositions for tensor-based data. *SIAM Journal on Matrix Analysis and Applications*, 30(3), 957-987.

 - [6] Xu, Y. (2015). Alternating proximal gradient method for sparse nonnegative Tucker decomposition. *Mathematical Programming Computation*, 7(1), 39-70.

 - [7] Xu, Y., & Yin, W. (2013). A block coordinate descent method for regularized multiconvex optimization with applications to nonnegative tensor factorization and completion. *SIAM Journal on Imaging Sciences*, 6(3), 1758-1789.
