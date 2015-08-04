using TensorDecompositions
using TensorOperations
using Base.Test

println("Running tests:")

include("test-hosvd.jl")
include("test-candecomp.jl")
include("test-nncp.jl")
include("test-tensorcur3.jl")
include("test-parafac2.jl")

