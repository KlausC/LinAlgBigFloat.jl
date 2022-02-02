
using LinAlgBigFloat.VectorSpaces
using Random

Random.seed!(1)

l, n, m = 2, 6, 4
A = randn(n, m)
B = randn(l, n)

vz = ZeroSpace(n)
@test rank(vz) == 0 && dim(vz) == n

vf = VectorSpace(n)
@test rank(vf) == n && dim(vf) == n

va = VectorSpace(A)
@test rank(va) == min(m, n) && dim(va) == n
vb = VectorSpace(B)
@test rank(vb) == min(l, n) && dim(vb) == l

@test va == image(A)
@test vb == image(B)

@test vz ⊆ va ⊆ vf

@test kernel(A) == ZeroSpace(m)
@test image(B) == VectorSpace(l)

@test kernel(A) == preimage(A)


