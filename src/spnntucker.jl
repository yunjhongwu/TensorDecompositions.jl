"""
State of sparse (semi-)nonnegative Tucker decomposition
"""
immutable SPNNTuckerState
	sqr_residue::Float64          # residue, i.e. 0.5 vecnorm(tnsr - recomposed)^2
	rel_residue::Float64          # residue relative to the ||tnsr||
	rel_residue_delta::Float64    # residue delta relative to the current residue

	function SPNNTuckerState(sqr_residue::Float64, prev_sqr_residue::Float64, tnsr_nrm::Float64)
		sqr_residue < -1E-10*tnsr_nrm^2 && warn("Negative residue: $sqr_residue")
		sqr_residue = max(0.0, sqr_residue)
		new(sqr_residue, sqrt(2*sqr_residue)/tnsr_nrm, abs(sqr_residue-prev_sqr_residue)/(prev_sqr_residue+1E-5))
	end

end

"""
Helper object for spnntucker().
"""
immutable SPNNTuckerHelper{T<:Number, N}
	tnsr::Array{T, N}
	tnsr_nrm::Float64
	core_dims::NTuple{N,Int}
	tnsrXfactors_low::Vector{Array{T, N}}
	lambdas::Vector{T}
	bounds::Vector{T}
	Lmin::Float64
	#tmp_core_unfold::Vector{Matrix{T}}
	L::Vector{Float64}   # previous Lipschitz constants
	L0::Vector{Float64}
	arr_pool::ArrayPool{T}

	function (::Type{SPNNTuckerHelper}){T, N}(tnsr::Array{T,N}, core_dims::NTuple{N,Int},
											  lambdas::Vector{Float64}, bounds::Vector{T},
											  Lmin::Float64; verbose::Bool=false)
		verbose && info("Precomputing input tensor unfoldings...")
		tnsr_dims = size(tnsr)
		new{T,N}(tnsr, vecnorm(tnsr), core_dims,
				 Array{T,N}[Array{T}(core_dims[1:n]..., tnsr_dims[(n+1):N]...) for n in 1:N],
				 lambdas, bounds,
				 Lmin, fill(1.0, N+1), fill(1.0, N+1), ArrayPool{T}()
		)
	end
end

acquire!(helper::SPNNTuckerHelper, size) = acquire!(helper.arr_pool, size)
release!(helper::SPNNTuckerHelper, size) = release!(helper.arr_pool, size)

function _spnntucker_update_tensorXfactors_low!{T,N}(helper::SPNNTuckerHelper{T,N}, decomp::Tucker{T,N})
	tensorcontractmatrix!(helper.tnsrXfactors_low[1], helper.tnsr,
						  decomp.factors[1], 1)
	for n in 2:N
		tensorcontractmatrix!(helper.tnsrXfactors_low[n],
							  helper.tnsrXfactors_low[n-1], decomp.factors[n], n)
	end
	return helper
end

function _spnntucker_factor_grad_components!{T,N}(helper::SPNNTuckerHelper{T,N}, decomp::Tucker{T,N}, n::Int)
	all_but_n = [1:(n-1); (n+1):N]
	cXa_size = (size(helper.tnsr)[1:n-1]..., helper.core_dims[n], size(helper.tnsr)[(n+1):N]...)
	coreXantifactor = tensorcontractmatrices!(acquire!(helper, cXa_size),
											  decomp.core,
											  decomp.factors[all_but_n], all_but_n, transpose=true)
	cXa2 = tensorcontract!(1, coreXantifactor, 1:N, 'N',
					coreXantifactor, [1:(n-1); N+1; (n+1):N], 'N',
					0, acquire!(helper, (helper.core_dims[n], helper.core_dims[n])), [n, N+1], method=:BLAS)
	tXcXa = tensorcontract!(1, helper.tnsr, 1:N, 'N',
					coreXantifactor, [1:(n-1); N+1; (n+1):N], 'N',
					0, acquire!(helper, size(decomp.factors[n])), [n, N+1], method=:BLAS)
	release!(helper, coreXantifactor)
	return cXa2, tXcXa
end

function _spnntucker_reg_penalty{T,N}(decomp::Tucker{T,N}, lambdas::Vector{T})
	res = 0.0
	for i in 1:N
		res += lambdas[i] > 0.0 ? (lambdas[i] * sum(decomp.factors[i])) : 0.0
	end
	return res + (lambdas[N+1] > 0.0 ? (lambdas[N+1] * sum(abs, decomp.core)) : 0.0)
end

_spnntucker_project{PRJ}(::Type{Val{PRJ}}, x, lambda, bound) = throw(ArgumentError("Unknown project type: $PRJ"))

_spnntucker_project(::Type{Val{:Nonneg}}, x, lambda, bound) = max(x, 0.0)
_spnntucker_project(::Type{Val{:NonnegReg}}, x, lambda, bound) = max(x - lambda, 0.0)
_spnntucker_project(::Type{Val{:NonnegBounded}}, x, lambda, bound) = clamp(x, 0.0, bound)

_spnntucker_project(::Type{Val{:Unbounded}}, x, lambda, bound) = x
_spnntucker_project(::Type{Val{:SignedReg}}, x, lambda, bound) = x > lambda ? x - lambda : (x < -lambda ? x + lambda : 0.0)
_spnntucker_project(::Type{Val{:SignedBounded}}, x, lambda, bound) = x > bound ? bound : (x < -bound ? -bound : x)

# update core tensor of dest
function _spnntucker_update_core!{T,N,PRJ}(prj::Type{Val{PRJ}},
	helper::SPNNTuckerHelper{T,N}, dest::Tucker{T,N}, src::Tucker{T,N},
	src_factor2s::Vector{Matrix{T}}, n::Int)

	tensorXfactors_all = _as_vector(n < N ?
		tensorcontractmatrices!(acquire!(helper, helper.core_dims),
								helper.tnsrXfactors_low[n], dest.factors[(n+1):N], (n+1):N) :
		helper.tnsrXfactors_low[N])::Vector{T}
	s = (1.0/helper.L[N+1])::Float64
	core_grad = _as_vector(tensorcontractmatrices!(acquire!(helper, helper.core_dims), src.core, src_factor2s))::Vector{T}
	s_lambda = (helper.lambdas[N+1]/helper.L[N+1])::Float64
	bound = helper.bounds[N+1]::T
	src_core = _as_vector(src.core)::Vector{T}
	dest_core = _as_vector(dest.core)::Vector{T}
	@simd for i in eachindex(dest_core)
		@inbounds dest_core[i] = _spnntucker_project(prj, src_core[i] - s*(core_grad[i] - tensorXfactors_all[i]),
													 s_lambda, bound)
	end
	release!(helper, tensorXfactors_all)
	release!(helper, core_grad)
	dest
end

# update n-th factor matrix of dest
# return new residual
function _spnntucker_update_factor!{T,N}(
	helper::SPNNTuckerHelper{T,N}, dest::Tucker{T,N}, src::Tucker{T,N},
	dest_factor2s::Vector{Matrix{T}}, n::Int
)
	coreXantifactor2, tnsrXcoreXantifactor = _spnntucker_factor_grad_components!(helper, dest, n)::Tuple{Matrix{T}, Matrix{T}}
	factorXcoreXantifactor2 = _as_vector(A_mul_B!(acquire!(helper, size(src.factors[n])), src.factors[n], coreXantifactor2))

	# update Lipschitz constant
	helper.L0[n] = helper.L[n]
	helper.L[n] = max(helper.Lmin, vecnorm(coreXantifactor2))
	s = (1.0/helper.L[n])
	# update n-th factor matrix
	src_factor = _as_vector(src.factors[n])
	dest_factor = _as_vector(dest.factors[n])
	lambda = helper.lambdas[n]
	bound = helper.bounds[n]
	for i in eachindex(src_factor)
		@inbounds f = src_factor[i] - s*(factorXcoreXantifactor2[i] - tnsrXcoreXantifactor[i]+lambda)
		if lambda == 0.0 && isfinite(bound) && f > bound
			f = bound
		elseif f < 0.0
			f = 0.0
		end
		@inbounds dest_factor[i] = f
	end
	dest_factor2 = _as_vector(At_mul_B!(dest_factor2s[n], dest.factors[n], dest.factors[n]))
	coreXantifactor2_v = _as_vector(coreXantifactor2)
	factor2XcoreXantifactor2 = 0.0
	@simd for i in eachindex(dest_factor2)
		@inbounds factor2XcoreXantifactor2 += dest_factor2[i] * coreXantifactor2_v[i]
	end
	factorXtnsrXcoreXantifactor = 0.0
	tnsrXcoreXantifactor_v = _as_vector(tnsrXcoreXantifactor)
	@simd for i in eachindex(dest_factor)
		@inbounds factorXtnsrXcoreXantifactor += dest_factor[i] * tnsrXcoreXantifactor_v[i]
	end
	release!(helper, coreXantifactor2)
	release!(helper, tnsrXcoreXantifactor)
	release!(helper, factorXcoreXantifactor2)

	return 0.5*(factor2XcoreXantifactor2-2*factorXtnsrXcoreXantifactor+helper.tnsr_nrm^2) +
			_spnntucker_reg_penalty(dest, helper.lambdas)
end

function _spnntucker_update_proxy_factor!{T,N}(
	proxy::Tucker{T,N}, cur::Tucker{T,N}, prev::Tucker{T,N},
	n::Int, w::Float64
)
	factor_p = proxy.factors[n]
	factor0 = prev.factors[n]
	factor1 = cur.factors[n]
	@simd for i in eachindex(factor_p)
		@inbounds factor_p[i] = factor1[i] + w*(factor1[i]-factor0[i])
	end
	factor_p
end

function _spnntucker_update_proxy_core!{T,N}(
	proxy::Tucker{T,N}, cur::Tucker{T,N}, prev::Tucker{T,N}, w::Float64
)
	core_p = reinterpret(T, proxy.core, (length(proxy.core),))::Vector{T}
	core0 = reinterpret(T, prev.core, (length(proxy.core),))::Vector{T}
	core = reinterpret(T, cur.core, (length(proxy.core),))::Vector{T}
	@simd for i in eachindex(core_p)
		@inbounds core_p[i] = core[i] + w*(core[i]-core0[i])
	end
	proxy
end

"""
Sparse (semi-)nonnegative Tucker decomposition

Decomposes nonnegative tensor `tnsr` into optionally nonnegative `core` tensor
and sparse nonnegative factor matrices `factors`.

 * `tnsr` nonnegative `N`-mode tensor to decompose
 * `core_dims` size of a core densor
 * `core_nonneg` if true, the output core tensor is nonnegative
 * `tol` the target error of decomposition relative to the Frobenius norm of `tnsr`
 * `max_iter` maximum number of iterations if error stays above `tol`
 * `max_time` max running time
 * `lambdas` `N+1` vector of non-negative sparsity regularizer coefficients for the factor matrices and the core tensor
 * `Lmin` lower bound for Lipschitz constant for the gradients of the residual error eqn{l(Z,U) = fnorm(tnsr - ttl(Z, U))` by `Z` and each `U`
 * `rw` controls the extrapolation weight
 * `bounds` `N+1` vector of the maximal absolute values allows for the elements of core tensor and factor matrices (effective only if the regularization is disabled)
 * `ini_decomp` initial decomposition, if equals to `:hosvd`, `hosvd()` is used to generate the starting decomposition, if `nothing`, a random decomposition is used
 * `verbose` more output algorithm progress

Returns:
  * `Tucker` decomposition object with additional properties:
	* `:converged` method convergence indicator
	* `:rel_residue` the Frobenius norm of the residual error `l(Z,U)` plus regularization penalty (if any)
	* `:niter` number of iterations
	* `:nredo` number of times `core` and `factor` were recalculated to avoid the increase in objective function
	* `:iter_diag` convergence info for each iteration, see `SPNNTuckerState`

The function uses the alternating proximal gradient method to solve the following optimization problem:
 deqn{min 0.5 |tnsr - Z times_1 U_1 ldots times_K U_K |_{F^2} +
 sum_{n=1}^{K} lambda_n |U_n|_1 + lambda_{K+1} |Z|_1, ;text{where; Z geq 0, U_i geq 0.}
 If `core_nonneg` is `FALSE`, core tensor `Z` is allowed to have negative
 elements and eqn{z_{i,j}=max(0,z_{i,j}-lambda_{K+1}/L_{K+1}}) rule is replaced by eqn{z_{i,j}=sign(z_{i,j})max(0,|z_{i,j}|-lambda_{K+1}/L_{K+1})}.
 The method stops if either the relative improvement of the error is below the tolerance `tol` for 3 consequitive iterations or
 both the relative error improvement and relative error (wrt the `tnsr` norm) are below the tolerance.
 Otherwise it stops if the maximal number of iterations or the time limit were reached.

The implementation is based on ntds_fapg() MATLAB code by Yangyang Xu and Wotao Yin.

See Y. Xu, "Alternating proximal gradient method for sparse nonnegative Tucker decomposition", Math. Prog. Comp., 7, 39-70, 2015.
See http://www.caam.rice.edu/~optimization/bcu/`
"""
function spnntucker{T,N}(tnsr::StridedArray{T, N}, core_dims::NTuple{N, Int};
						 eigmethod=trues(N),
						 core_nonneg::Bool=true, tol::Float64=1e-4, hosvd_init::Bool=false,
						 max_iter::Int=500, max_time::Float64=0.0,
						 lambdas::Vector{Float64} = fill(0.0, N+1),
						 Lmin::Float64 = 1.0, rw::Float64=0.9999,
						 bounds::Vector{Float64} = fill(Inf, N+1), ini_decomp = nothing,
						 verbose::Bool=false,
						 progressbar::Bool=true)
	start_time = time()

	# define "kernel" functions for "fixing" the core tensor after iteration
	core_bound = bounds[N+1]
	core_lambda = lambdas[N+1]
	if core_nonneg
		if core_lambda > 0.0
			# regularization
			projection_type = Val{:NonnegReg}
		elseif isfinite(core_bound)
			projection_type = Val{:NonnegBounded}
		else
			projection_type = Val{:Nonneg}
		end
	else
		if core_lambda > 0.0
			# regularization
			projection_type = Val{:SignedReg}
		elseif isfinite(core_bound)
			projection_type = Val{:SignedBounded}
		else
			projection_type = Val{:Unbounded}
		end
	end

	if ini_decomp === nothing
		verbose && info("Generating random initial factor matrices and core tensor estimates...")
		ini_decomp = Tucker((Matrix{T}[randn(size(tnsr, i), core_dims[i]) for i in 1:N]...), randn(core_dims...))
		rescale_ini = true
	elseif ini_decomp == :hosvd
		verbose && info("Using High-Order SVD to get initial decomposition...")
		# "solve" Z = tnsr x_1 U_1' ... x_K U_K'
		ini_decomp = hosvd(tnsr, core_dims, eigmethod; pad_zeros=true, compute_error=true)
		rescale_ini = false
	elseif isa(ini_decomp, Tucker{T,N})
		rescale_ini = false
	else
		throw(ArgumentError("Incorrect ini_decomp value $(typeof(ini_decomp))"))
	end

	#verbose && info("Initializing helper object...")
	helper = SPNNTuckerHelper(tnsr, core_dims, lambdas, bounds, Lmin, verbose = verbose)
	verbose && info("|tensor|=$(helper.tnsr_nrm)")
	verbose && info("Initial Frobenius norm: $(vecnorm(tnsr - compose(ini_decomp)))")

	verbose && info("Rescaling initial decomposition...")
	decomp0 = deepcopy(ini_decomp)
	if rescale_ini
		rescale!(decomp0, helper.tnsr_nrm)
	end
	decomp = deepcopy(decomp0)     # current decomposition
	decomp_p = deepcopy(decomp0)   # proxy decomposition

	#verbose && info("Calculating factors squares...")
	factor2s0 = Matrix{T}[f'f for f in decomp0.factors]
	factor2s = deepcopy(factor2s0)
	factor2_nrms = map(vecnorm, factor2s)::Vector{Float64}

	#verbose && info("Calculating initial residue...")
	resid = resid0 = 0.5*vecnorm(tnsr - compose(decomp0))^2 + _spnntucker_reg_penalty(decomp0, lambdas)
	resid = resid0 # current residual error
	verbose && info("Initial residue=$resid0")

	# Iterations of block-coordinate update
	# iteratively updated variables:
	# GradU: gradients with respect to each component matrix of U
	# GradZ: gradient with respect to Z
	t0 = fill(1.0, N+1)
	t = deepcopy(t0)

	iter_diag = Vector{SPNNTuckerState}()
	nstall = 0
	nredo = 0
	converged = false

	#verbose && info("Starting iterations...")
	pb = Progress(max_iter, "Alternating proximal gradient iterations")
	niter = 1
	while !converged
		progressbar && update!(pb, niter) # progress bar slows parallel executions; also becomes unreadable due to overlaps

		residn0 = resid
		_spnntucker_update_tensorXfactors_low!(helper, decomp0)

		for n in N:-1:1
			# -- update the core tensor Z --
			helper.L0[N+1] = helper.L[N+1]
			helper.L[N+1] = max(helper.Lmin, prod(factor2_nrms))

			# try to make a step using extrapolated decompositon (Zm,Um)
			_spnntucker_update_core!(projection_type, helper, decomp, decomp_p, factor2s, n)
			residn = _spnntucker_update_factor!(helper, decomp, decomp_p, factor2s, n)
			if residn > residn0
				# extrapolated Zm,Um decomposition lead to residual norm increase,
				# revert extrapolation and make a step using Z0,U0 to ensure
				# objective function is decreased
				nredo += 1
				# re-update to make objective nonincreasing
				copy!(factor2s[n], factor2s0[n]) # restore factor square, core update needs it
				_spnntucker_update_core!(projection_type, helper, decomp, decomp0, factor2s, n)
				residn = _spnntucker_update_factor!(helper, decomp, decomp0, factor2s, n)
				verbose && (residn > residn0) && warn("Iteration $niter - Redo $nredo: residue increase at redo step ($residn > $residn0)")
				verbose && (residn <= residn0) && warn("Iteration $niter - Redo $nredo: residue ok at redo step ($residn <= $residn0)")
			end
			# --- correction and extrapolation ---
			t[n] = (1.0+sqrt(1.0+4.0*t0[n]^2))/2.0
			#verbose && info("Updating proxy factors $n...")
			_spnntucker_update_proxy_factor!(decomp_p, decomp, decomp0, n, min((t0[n]-1)/t[n], rw*sqrt(helper.L0[n]/helper.L[n])))
			t[N+1] = (1.0+sqrt(1.0+4.0*t0[N+1]^2))/2.0
			#verbose && info("Updating proxy core $n...")
			_spnntucker_update_proxy_core!(decomp_p, decomp, decomp0, min((t0[N+1]-1)/t[N+1], rw*sqrt(helper.L0[N+1]/helper.L[N+1])))

			#verbose && info("Storing updated core and factors...")
			copy!(decomp0.core, decomp.core)
			copy!(decomp0.factors[n], decomp.factors[n])
			copy!(factor2s0[n], factor2s[n])
			factor2_nrms[n] = vecnorm(factor2s[n])
			t0[n] = t[n]
			t0[N+1] = t[N+1]
			residn0 = residn
		end

		# --- diagnostics, reporting, stopping checks ---
		resid0 = resid
		resid = residn0

		#verbose && info("Storing statistics...")
		cur_state = SPNNTuckerState(resid, resid0, helper.tnsr_nrm)
		push!(iter_diag, cur_state)

		# check stopping criterion
		niter += 1
		nstall = cur_state.rel_residue_delta < tol ? nstall + 1 : 0
		if nstall >= 3 || cur_state.rel_residue < tol
			verbose && (cur_state.rel_residue == 0.0) && info("Residue is zero. Exact decomposition was found")
			verbose && (nstall >= 3) && info("Relative error below $tol $nstall times in a row")
			verbose && (cur_state.rel_residue < tol) && info("Relative error is $(cur_state.rel_residue) times below input tensor norm")
			verbose && info("spnntucker() converged in $niter iteration(s), $nredo redo steps")
			converged = true
			finish!(pb)
			break
		elseif (max_time > 0) && ((time() - start_time) > max_time)
			cancel(pb, "Maximal time exceeded ($(time() - start_time) > $(max_time)), might be not an optimal solution")
			verbose && info("Final relative error $(cur_state.rel_residue)")
			verbose && info("Frobenius norm: $(vecnorm(tnsr - compose(decomp0)))")
			break
		elseif niter >= max_iter
			cancel(pb, "Maximal number of iterations reached ($(niter) >= $(max_iter)), might be not an optimal solution")
			verbose && info("Final relative error $(cur_state.rel_residue)")
			verbose && info("Frobenius norm: $(vecnorm(tnsr - compose(decomp0)))")
			break
		end
	end # iterations
	finish!(pb)

	res = decomp0
	res.props[:niter] = niter
	res.props[:nredo] = nredo
	res.props[:converged] = converged
	res.props[:rel_residue] = 2*sqrt(resid-_spnntucker_reg_penalty(decomp, lambdas))/helper.tnsr_nrm
	res.props[:iter_diag] = iter_diag
	return res
end
