using TensorDecompositions
using TensorOperations
using FactCheck

import TensorDecompositions._KhatriRao
function _kruskal3_generator(r::Int64, s::Tuple, 
                             seed::Int64, nonnegative::Bool)
    srand(seed)
    rnd = nonnegative ? rand : randn
    return reshape(reduce(_KhatriRao, [rnd(s[i], r) for i in length(s)-1:-1:1]) * rnd(s[end], r)', s...)
end

include("test_utils.jl")
include("test-hosvd.jl")
include("test-candecomp.jl")
include("test-sshopm.jl")
include("test-nncp.jl")
include("test-tensorcur3.jl")
include("test-parafac2.jl")

#=
using Lint
using TypeCheck
lintpkg("TensorDecompositions")
checkreturntypes(TensorDecompositions)
checklooptypes(TensorDecompositions)
checkmethodcalls(TensorDecompositions)
=#
