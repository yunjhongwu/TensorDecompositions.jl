using TensorDecompositions
using TensorOperations
using Base.Test

import TensorDecompositions._KhatriRao
function _kruskal3_generator(r::Int64, s::Tuple, 
                             seed::Int64, nonnegative::Bool)
    srand(seed)
    rnd = nonnegative ? rand : randn
    return reshape(reduce(_KhatriRao, [rnd(s[i], r) for i in length(s)-1:-1:1]) * rnd(s[end], r)', s...)
end


println("Running tests:")

#include("test-hosvd.jl")
#include("test-candecomp.jl")
#include("test-nncp.jl")
include("test-tensorcur3.jl")
#include("test-parafac2.jl")
