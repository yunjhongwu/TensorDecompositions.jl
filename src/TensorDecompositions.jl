# require("TensorOperations")

module TensorDecompositions
    using TensorOperations
    using Distributions
    using Base.Cartesian

    export

    # additional tensor algebra methods
    tensorcontractmatrix, tensorcontractmatrices!, tensorcontractmatrices,

    # tensor decomposition algorithms
    hosvd,                   # High-order SVD
    candecomp,               # Canonical polyadic decomposition (CANDECOMP/PARAFAC)
    sshopm,                  # Shifted symmetric higher-order power method
    nncp,                    # Non-negative CANDECOMP
    tensorcur3,              # Tensor-CUR for 3-mode tensors
    parafac2,                # PARAFAC2 model

    include("utils.jl")      # Sub-routines
    include("tucker.jl")
    include("hosvd.jl")
    include("candecomp.jl")
    include("sshopm.jl")
    include("nncp.jl")
    include("tensorcur.jl")
    include("parafac2.jl")

end
