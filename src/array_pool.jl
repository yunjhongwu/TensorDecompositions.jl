"""
Helps to maintain the pool of reusable arrays of different sizes
and reduce the burden on garbage collection.
"""
immutable ArrayPool{T<:Number}
    length_pools::Dict{Int, Vector{Vector{T}}}

    ArrayPool() = new(Dict{Int, Vector{Vector{T}}}())
end

"""
Gets an array of specific size from the pool.
The returned array should be returned back to the pool using `release!()`.
"""
function acquire!{T}(pool::ArrayPool{T}, size)
    len = prod(size)
    len_pool = haskey(pool.length_pools, len) ?
               pool.length_pools[len] :
               get!(pool.length_pools, len, Vector{Vector{T}}())
    isempty(len_pool) ? Array{T}(size) : reinterpret(T, pop!(len_pool), size)
end

"""
Releases an array returned by `acquire!()` back into the pool.
"""
function release!{T}(pool::ArrayPool{T}, arr::Array{T})
    len = length(arr)
    len_pool = haskey(pool.length_pools, len) ?
               pool.length_pools[len] :
               get!(pool.length_pools, len, Vector{Vector{T}}())
    push!(len_pool, reinterpret(T, arr, (len,)))
    pool
end
