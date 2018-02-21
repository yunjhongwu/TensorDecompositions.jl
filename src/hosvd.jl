import TensorToolbox

"""
High-order singular value decomposition (HO-SVD).
"""
function hosvd{T,N}(tensor::StridedArray{T,N}, core_dims::NTuple{N, Int}, eigmethod=trues(N); pad_zeros::Bool=false, compute_error::Bool=false)
	pad_zeros || _check_tensor(tensor, core_dims)

	factors = map(1:N) do i
		X = _col_unfold(tensor, i)
		if eigmethod[i]
			f = eigs(X'X, nev=core_dims[i])[2]
		else
			f = eig(X'X)[2]
		end
		if pad_zeros && size(f, 2) < core_dims[i] # fill missing factors with zeros
			warn("Zero slices ($(core_dims[i]-size(f, 2))) added in dimension $i ")
			f = hcat(f, zeros(T, size(tensor, i), core_dims[i]-size(f, 2)))
		end
		mapslices(_check_sign, f, 1)
	end

	res = Tucker((factors...), tensorcontractmatrices(tensor, factors))
	if compute_error
		_set_rel_residue(res, tensor)
		info("Error: $(res.props[:rel_residue])")
	end
	info("HOSVD core rank: $(TensorToolbox.mrank(res.core))")
	return res
end

hosvd{T,N}(tensor::StridedArray{T,N}, r::Int; compute_error::Bool=false, pad_zeros::Bool=false) =
	hosvd(tensor, (fill(r, N)...); compute_error=compute_error, pad_zeros=pad_zeros)
