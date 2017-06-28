
import Base: A_mul_Bc!, A_mul_B!, (*)
import Base: getindex, setindex!

type Dummy end
AbstractM  = Union{AbstractMatrix, LinAlg.AbstractRotation, Dummy}

"""
Add missing multiplication to LinAlg.Rotation
"""
function A_mul_Bc!(R::LinAlg.Rotation, G::LinAlg.Givens)
  insert!(R.rotations, 1, G')
  R
end

function A_mul_B!{T<:Number,S<:Number}(A::AbstractMatrix{T}, R::LinAlg.Rotation{S})
  n = length(R.rotations)
  @inbounds for i = 1:n
    A_mul_Bc!(A, R.rotations[n+1-i]')
  end
  A
end

function A_mul_B!(R::LinAlg.Rotation, A::AbstractVecOrMat)
    @inbounds for i = 1:length(R.rotations)
        A_mul_B!(R.rotations[i], A)
    end 
    return A
end

(*){T<:Number,S<:Number}(A::AbstractMatrix{T}, R::LinAlg.Rotation{S}) = A_mul_B!(copy(A), R)

A_mul_Bc!(Q::Dummy, ::Any) = Q
A_mul_B!(Q::Dummy, ::Any) = Q
A_mul_B!(A::Any, Q::Dummy) = Q
(*)(Q::Dummy, A::Any) = Q
(*)(A::Any, Q::Dummy) = Q

getindex(Q::Dummy, ::Any, ::Any) = Q
setindex!(Q::Dummy, ::Any, ::Any, ::Any) = Q

"""
Discriminant of 2 x 2 submatrix
"""
@inline discriminant(A, i, j) = ((A[i,i] - A[j,j]) / 2 ) ^2 + A[i,j] * A[j,i]

"""
Extract eigenvalues from quasi triangular matrix
"""
function seigendiag{T<:Real}(A::AbstractMatrix{T}, ilo::Integer, ihi::Integer)
  # ev = eig(A[ilo:ihi,ilo:ihi])[1]  # TODO replace by specialized eig
  # filter( x -> imag(x) >= 0, ev)
  ev = Complex{eltype(A)}[]
  k = ilo
  while k <= ihi
    if k >= ihi || A[k+1,k] == 0
      push!(ev, A[k,k])
    else
      r = ( A[k,k] + A[k+1,k+1] ) / 2
      disc = discriminant(A, k, k + 1)
      if disc < 0
        dd = sqrt(-disc) * im
        push!(ev, r + dd)
        push!(ev, r - dd)
      else
        dd = sqrt(disc)
        push!(ev, r + dd)
        push!(ev, r - dd)
      end
      k += 1
    end
    k += 1
  end
  isreal(ev) ? real(ev) : ev
end
function seigendiag{T<:Complex}(A::AbstractMatrix{T}, ilo::Integer, ihi::Integer)
  ev = diag(A[ilo:ihi,ilo:ihi])
  isreal(ev) ? real(ev) : ev
end

"""
Calculate vector of eigenvalues of a real 1x1 or 2x2 matrix A.
If eigenvalue is complex, return one complex number with positive imaginary part.
"""
function eig2(A::AbstractMatrix)
  n = size(A, 1)
  if n == 1
    [ A[1,1] ]
  else
    disc = discriminant(A, 1, 2)
    re = (A[1,1] + A[2,2]) / 2
    sq = sqrt(abs(disc))
    if disc < 0
      [ Complex(re, sq) ]
    else
      [ re - sq; re + sq ]
    end
  end
end

"""

  `is_hessenberg(::AbstractMatrix) -> bool`

  Test if matrix is exact Hessenberg form.
  No square matrix required.
  (Below subdiagonal exact zero except optional spike column)
"""
function is_hessenberg(A::AbstractMatrix, ispike::Integer = 0)
  n, m = size(A)
  for k = 1:m
    if k != ispike
      for i = k+2:n
        if A[i,k] != 0
          return false
        end
      end
    end
  end
  true
end

"""
Finish up to standardized 2x2 diagonal blocks
"""
function finish!(A::AbstractMatrix, Q::AbstractM)
  n = size(A, 1)
  for k = 1:n-1
    if A[k+1,k] != 0
      r = k:k+1
      AA, G = givens1(A, k, k+1)
      A_mul_B!(G, view(A, :, k+1:n))
      A_mul_Bc!(view(A, 1:k-1, :), G)
      A_mul_Bc!(Q, G)
      A[r,r] = AA
      # println("finish($k:$(k+1))")
      # display(@view A[k:k+1,k:k+1])
      # display(AA)
    end
  end
end

"""
Set elements of matrix to zero, which should be, but are not due to numerical errors.
"""
@inline function reschur!(A::AbstractMatrix, k::Integer)
  A[k+1:end,1:k] = 0
end

"""
Produce Givens Rotation, which transforms 2x2 matrix to another 2x2 matrix with
either: subdiagonal zero if 2 real eigenvalues.
The eigenvalue of lowest absolute value is placed in lower right position.
or: equal diagonal elements if 2 non-real complex eigenvalues.
The lower left element is <= upper right element absolutely.
"""
function givens1{T<:Number}(A::AbstractMatrix{T}, i1::Integer, i2::Integer)
  a, b, x, d = A[i1,i1], A[i1,i2], A[i2,i1], A[i2,i2]
  btx = b * x
  apd = ( a + d ) / 2
  bpx = ( b + x ) / 2
  da  = ( d - a ) / 2
  disc = da ^ 2 + btx
  if disc >= 0
    root = sqrt( disc )
    root = copysign(root, apd)
    e1 = apd + root
    e2 = ( a * d - btx ) / e1
    ea = a - e2
    ed = d - e2
    G, r = abs(b) + abs(ed) > abs(b) + abs(ea) ? givens(b, ed, i1, i2) : givens(ea, x, i1, i2)
    v = 2( da * G.c - bpx * G.s) * G.s + b
    [e1 v; 0 e2], G
  else
    bx = ( b - x ) / 2
    root = hypot(da, bpx)
    root = copysign(root, bx)
    w = bx + root
    v = disc / w
    G, r = abs(b) >= abs(x) ? givens(w + x, da, i1, i2) : givens(da, w - b, i1, i2)
    [ apd w; v apd], G
  end
end
