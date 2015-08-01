# require("TensorOperations")

module TensorDecompositions
    using TensorOperations
    import TensorOperations: tensorcontract

    export 

    hosvd,                   # High-order SVD
    candecomp,               # Canonical polyadic decomposition
    ntfcp                    # Non-negative CANDECOMP 

    include("routines.jl")   #sub-routines
    include("hosvd.jl")
    include("candecomp.jl")
    include("ntfcp.jl")

end
