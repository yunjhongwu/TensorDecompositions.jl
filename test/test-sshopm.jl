println("SS-HOPM")
eigvec = randn(20)
eigval = vecnorm(eigvec)
eigvec /= eigval

T = eigval * eigvec .* eigvec' .* reshape(eigvec, 1, 1, 20)

@time (lbd, x) = sshopm(T, 1)
@test vecnorm(eigvec - x) < 1e-5
@test vecnorm(eigval - lbd) < 1e-5
