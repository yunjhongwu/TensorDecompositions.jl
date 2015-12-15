facts("SS-HOPM") do

srand(1)
eigvec = randn(20)
eigval = vecnorm(eigvec)
eigvec /= eigval
T = eigval * eigvec .* eigvec' .* reshape(eigvec, 1, 1, 20)

context("Dense representation") do
    @time (lbd, x) = sshopm(T, 1, verbose=true)
    @fact vecnorm(eigvec - x) --> less_than(1e-5)
    @fact vecnorm(eigval - lbd) --> less_than(1e-5)
end

context("Sparse representation") do
    spT = SparseArray(T)

    @time (lbd, x) = sshopm(spT, 1, verbose=true)
    @fact vecnorm(eigvec - x) --> less_than(1e-5)
    @fact vecnorm(eigval - lbd) --> less_than(1e-5)
end

end
