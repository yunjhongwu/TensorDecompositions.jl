# require("TensorOperations")

module TensorDecompositions
    using TensorOperations
    using Distributions
    using Compat

    export 

    hosvd,                   # High-order SVD
    candecomp,               # Canonical polyadic decomposition (CANDECOMP/PARAFAC)
    nncp,                    # Non-negative CANDECOMP 
    tensorcur3,              # Tensor-CUR for 3-mode tensors
    parafac2                 # PARAFAC2 model

    include("utils.jl")      # Sub-routines
    include("hosvd.jl")
    include("candecomp.jl")
    include("nncp.jl")
    include("tensorcur.jl")
    include("parafac2.jl")

end
