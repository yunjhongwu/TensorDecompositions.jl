@testset "Sparse (semi-)nonnegative Tucker decomposition" begin
Random.seed!(12345)

@testset "nonnegative decomposition" begin
    # An example of nonnegative and semi-nonnegative Tucker decomposition
    tucker_orig = rand_tucker((4, 5, 6), (40, 50, 60), factors_nonneg=true, core_nonneg=true)
    tnsr_orig = compose(tucker_orig)

    tnsr_max = maximum(tnsr_orig)
    tucker_orig.core ./= tnsr_max
    tnsr_orig ./= tnsr_max

    tnsr = add_noise(tnsr_orig, 0.6, true)

    # Solve the problem
    @time tucker_spnn = spnntucker(tnsr, size(tucker_orig.core), tol=1E-5, ini_decomp=:hosvd,
                            core_nonneg=true,
                            max_iter=1000, verbose=true, lambdas=fill(0.1, 4))

    # Reporting
    @test rel_residue(tucker_spnn) < 0.05
    @info "Relative error of decomposition : $(rel_residue(tucker_spnn))"
end

@testset "semi-nonnegative decomposition" begin
    tucker_orig = rand_tucker((4, 5, 6), (40, 50, 60), factors_nonneg=true, core_nonneg=false)
    tnsr_orig = compose(tucker_orig)

    tnsr_max = maximum(tnsr_orig)
    tucker_orig.core ./=  tnsr_max
    tnsr_orig ./= tnsr_max

    tnsr = add_noise(tnsr_orig, 0.6, false)

    # Solve the problem
    @time tucker_spnn = spnntucker(tnsr, size(tucker_orig.core), tol=1E-5, ini_decomp=:hosvd,
                             core_nonneg=false,
                             max_iter=1000, verbose=true, lambdas=fill(0.1, 4))

    # Reporting
    @test rel_residue(tucker_spnn) < 0.05
    @info("Relative error of decomposition : $(rel_residue(tucker_spnn))")
end

end
