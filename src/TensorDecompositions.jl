# require("TensorOperations")

module TensorDecompositions
    using TensorOperations
    using Distributions

    import TensorOperations: tensorcontract
    import Distributions: Categorical, rand

    export 

    hosvd,                   # High-order SVD
    candecomp,               # Canonical polyadic decomposition (CANDECOMP/PARAFAC)
    ntfcp,                   # Non-negative CANDECOMP 
    tensorcur3,              # Tensor-CUR for 3-mode tensors
    parafac2                 # PARAFAC2 model

    include("routines.jl")   #sub-routines
    include("hosvd.jl")
    include("candecomp.jl")
    include("ntfcp.jl")
    include("tensorcur.jl")
    include("parafac2.jl")

end
