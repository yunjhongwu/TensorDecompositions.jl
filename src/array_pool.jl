"""
Helps to maintain the pool of reusable arrays of different sizes
and reduce the burden on garbage collection.
"""
struct ArrayPool{T<:Number}
    length_pools::Dict{Int, Vector{Vector{T}}}

    ArrayPool{T}() where {T<:Number} = new{T}(Dict{Int, Vector{Vector{T}}}())
end

"""
Gets an array of specific size from the pool.
The returned array should be returned back to the pool using `release!()`.
"""
function acquire!(pool::ArrayPool{T}, size) where T
    len = prod(size)
    len_pool = haskey(pool.length_pools, len) ?
               pool.length_pools[len] :
               get!(() -> Vector{Vector{T}}(), pool.length_pools, len)
    return isempty(len_pool) ?
            Array{T}(undef, size) :
            reshape(pop!(len_pool), size)
end

"""
Releases an array returned by `acquire!()` back into the pool.
"""
function release!(pool::ArrayPool{T}, arr::Array{T}) where T
    len = length(arr)
    len_pool = get(pool.length_pools, len, nothing)
    if len_pool !== nothing
        #@info "release($(size(arr))) ($(length(len_pool)))"
        (length(len_pool) <= 100) || error("Overflow of $len-sized vectors pool")
        push!(len_pool, vec(arr))
    else
        throw(DimensionMismatch("No $len-element arrays were acquired before"))
    end
    return pool
end
