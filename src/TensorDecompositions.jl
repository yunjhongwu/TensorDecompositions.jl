module TensorDecompositions
    using TensorOperations
    using Distributions
    using ProgressMeter
    using Base.Cartesian
    using StatsBase
    using LinearAlgebra

    export

    # types
    SparseArray,
    TensorDecompositions, PARAFAC2, CANDECOMP, CUR, Tucker,

    # TensorDecomposition methods
    rel_residue,
    compose, compose!,

    # additional tensor algebra methods
    tensorcontractmatrix, tensorcontractmatrices!, tensorcontractmatrices,

    # tensor decomposition algorithms
    hosvd,                   # High-order SVD
    candecomp,               # Canonical polyadic decomposition (CANDECOMP/PARAFAC)
    sshopm,                  # Shifted symmetric higher-order power method
    nncp,                    # Non-negative CANDECOMP
    spnntucker,              # sparse (semi) non-negative Tucker
    tensorcur3,              # Tensor-CUR for 3-mode tensors
    parafac2                 # PARAFAC2 model

    include("array_pool.jl")
    include("utils.jl")
    include("abstract_decomposition.jl")
    include("tucker.jl")
    include("hosvd.jl")
    include("candecomp.jl")
    include("sshopm.jl")
    include("nncp.jl")
    include("tensorcur.jl")
    include("parafac2.jl")
    include("spnntucker.jl")

end
