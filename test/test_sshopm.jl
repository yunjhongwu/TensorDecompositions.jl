@testset "SS-HOPM" begin

Random.seed!(1)
eigvec = randn(20)
eigval = norm(eigvec)
eigvec ./= eigval
T = eigval * eigvec .* eigvec' .* reshape(eigvec, 1, 1, 20)

@testset "Dense representation" begin
    @time (lbd, x) = sshopm(T, 1, verbose=true)
    @test norm(eigvec .- x) < 1e-5
    @test norm(eigval .- lbd) < 1e-5
end

@testset "Sparse representation" begin
    spT = SparseArray(T)

    @time (lbd, x) = sshopm(spT, 1, verbose=true)
    @test norm(eigvec .- x) < 1e-5
    @test norm(eigval .- lbd) < 1e-5
end

end
