"""
Abstract decomposition of N-mode tensor.
"""
abstract TensorDecomposition{T<:Number, N}

"""
Returns relative error between the re-composed and the original tensors.
"""
rel_residue{T,N}(recomposed::StridedArray{T, N}, tensor::StridedArray{T, N}) = vecnorm(recomposed - tensor) / vecnorm(tensor)

"""
Returns relative error between re-composed and the original tensor.
"""
rel_residue{T,N}(decomp::TensorDecomposition{T, N}, tensor::StridedArray{T, N}) = rel_residue(compose(decomp), tensor)

"""
Returns relative tensor decomposition error, NaN if not available.
"""
rel_residue(decomp::TensorDecomposition) = get(decomp.props, :rel_residue, NaN)

_set_rel_residue(decomp::TensorDecomposition, error::Float64) = setindex!(decomp.props, error, :rel_residue)

_set_rel_residue{T,N}(decomp::TensorDecomposition{T,N}, tensor::StridedArray{T,N}) =
    _set_rel_residue(decomp, rel_residue(decomp, tensor))
