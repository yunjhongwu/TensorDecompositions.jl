using TensorDecompositions
using TensorOperations
using FactCheck

include("helpers.jl")

include("test_utils.jl")
include("test-hosvd.jl")
include("test-candecomp.jl")
include("test-sshopm.jl")
include("test-nncp.jl")
include("test-tensorcur3.jl")
include("test-parafac2.jl")
include("test_spnntucker.jl")

#=
using Lint
using TypeCheck
lintpkg("TensorDecompositions")
checkreturntypes(TensorDecompositions)
checklooptypes(TensorDecompositions)
checkmethodcalls(TensorDecompositions)
=#
