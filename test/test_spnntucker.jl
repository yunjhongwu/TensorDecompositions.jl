facts("Sparse (semi-)nonnegative Tucker decomposition") do

context("nonnegative decomposition") do
    # An example of nonnegative and semi-nonnegative Tucker decomposition
    tucker_orig = rand_tucker((4, 5, 6), (40, 50, 60), factors_nonneg=true, core_nonneg=true)
    tnsr_orig = compose(tucker_orig)

    tnsr_max = maximum(tnsr_orig)
    map!(x -> x / tnsr_max, tucker_orig.core)
    map!(x -> x / tnsr_max, tnsr_orig)

    tnsr = add_noise(tnsr_orig, 0.6, true)

    # Solve the problem
    @time tucker_spnn = spnntucker(tnsr, size(tucker_orig.core), tol=1E-4, ini_decomp=:hosvd,
                             core_nonneg=true,
                             max_iter=1000, verbose=true, lambdas=fill(0.1, 4))

    # Reporting
    @fact rel_residue(tucker_spnn) --> less_than(0.05)
    info("Relative error of decomposition : $(rel_residue(tucker_spnn))")
end

context("semi-nonnegative decomposition") do
    tucker_orig = rand_tucker((4, 5, 6), (40, 50, 60), factors_nonneg=true, core_nonneg=false)
    tnsr_orig = compose(tucker_orig)

    tnsr_max = maximum(tnsr_orig)
    map!(x -> x / tnsr_max, tucker_orig.core)
    map!(x -> x / tnsr_max, tnsr_orig)

    tnsr = add_noise(tnsr_orig, 0.6, false)

    # Solve the problem
    @time tucker_spnn = spnntucker(tnsr, size(tucker_orig.core), tol=1E-4, ini_decomp=:hosvd,
                             core_nonneg=false,
                             max_iter=1000, verbose=true, lambdas=fill(0.1, 4))

    # Reporting
    @fact rel_residue(tucker_spnn) --> less_than(0.05)
    info("Relative error of decomposition : $(rel_residue(tucker_spnn))")
end

end
