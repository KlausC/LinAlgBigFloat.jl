
import Base: A_mul_Bc!, A_mul_B!, (*)
import Base: getindex, setindex!

type Dummy end
typealias AbstractM Union{AbstractMatrix, LinAlg.AbstractRotation, Dummy}

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

