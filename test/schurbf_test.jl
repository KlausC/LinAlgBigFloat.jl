
srand(1)




T = BigFloat
setprecision(BigFloat, 512) do

  @testset "transform_Hess $n" for n in (3, 4, 5)
    AB = BigFloat.(randn(n, n)) + BigFloat.(randn(n,n))*im
    A = copy(AB)
    Q = Complex{BigFloat}.(eye(n,n))
    ev = eig(Complex128.(AB))[1]
    s = ev[n:n]
    LinAlgBigFloat.transform_Hess!(A, 1, n, Q, s, n, n)
    @test checkfactors(AB, Q, A, rtol = eps(T) * n * n)
  end

  @testset "schurfact $n" for n in (1, 2, 3, 4, 5, 9, 20)
    AB = BigFloat.(randn(n, n))
    HF = hessfact(AB)
    @test checkfactors(AB, HF[:Q], HF[:H]) rtol = eps(T) * n * 2
    SF = schurfact(AB)
    @test checkfactors(AB, SF[:vectors], SF[:T]) rtol = eps(T) * n * 10

    AB = BigFloat.(randn(n, n)) + BigFloat.(randn(n,n)) * im
    HF = hessfact(AB)
    @test checkfactors(AB, HF[:Q], HF[:H]) rtol = eps(T) * n * 2
    SF = schurfact(AB)
    @test checkfactors(AB, SF[:vectors], SF[:T]) rtol = eps(T) * n * 20
  end
end



