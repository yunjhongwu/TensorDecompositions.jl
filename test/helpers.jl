# helper functions for the TensorDecompositions tests

import TensorDecompositions._KhatriRao
function _kruskal3_generator(r::Int64, s::Tuple, 
                             seed::Int64, nonnegative::Bool)
    srand(seed)
    rnd = nonnegative ? rand : randn
    return reshape(reduce(_KhatriRao, [rnd(s[i], r) for i in length(s)-1:-1:1]) * rnd(s[end], r)', s...)
end
