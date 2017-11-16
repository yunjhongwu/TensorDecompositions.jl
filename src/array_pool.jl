"""
Helps to maintain the pool of reusable arrays of different sizes
and reduce the burden on garbage collection.
"""
struct ArrayPool{T<:Number}
    length_pools::Dict{Int, Vector{Vector{T}}}

    ArrayPool{T}() where {T<:Number} = new(Dict{Int, Vector{Vector{T}}}())
end

"""
Gets an array of specific size from the pool.
The returned array should be returned back to the pool using `release!()`.
"""
function acquire!(pool::ArrayPool{T}, size) where T
    len = prod(size)
    len_pool = haskey(pool.length_pools, len) ?
               pool.length_pools[len] :
               get!(pool.length_pools, len, Vector{Vector{T}}())
    isempty(len_pool) ? Array{T}(size) : reinterpret(T, pop!(len_pool), size)
end

"""
Releases an array returned by `acquire!()` back into the pool.
"""
function release!(pool::ArrayPool{T}, arr::Array{T}) where T
    len = length(arr)
    len_pool = haskey(pool.length_pools, len) ?
               pool.length_pools[len] :
               get!(pool.length_pools, len, Vector{Vector{T}}())
    push!(len_pool, reinterpret(T, arr, (len,)))
    pool
end
