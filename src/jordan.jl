
# functions to support transformation of general real or complex matrices to
# Jordan normal form.
#

"""
Transform upper (quasi) triangular input matrix to block-diagonal form.
  The eigenvalues of the matrix must be ordered in a way, that multiple
  eigenvalues are stored in contiguous parts of the diagonal.
  The results stored in place of the input matrix.
"""
function blockdiagonal!(A::StridedMatrix{T}, ilo::Int, ihi::Int) where T <: Number

  while ihi > ilo
    # find lower right part of diagonal with equal eigenvalues
    isep = findlastequal(A, ilo, ihi)
    if isep > ilo
      rA = ilo:isep-1
      rB = isep:ihi
      A[rA, rB] .= sylvester(view(A, rA, rA), -view(A, rB, rB), A[rA, rB])
    end
    ihi = isep - 1
  end
  A
end

"""
Find diagonal position with first eigenvalue equal to last eigenvalue
"""
function findlastequal(A::AbstractMatrix{T}, ilo::Int, ihi::Int) where T <: Complex
  k = ihi
  d = A[k,k]
  while k > ilo && A[k-1,k-1] == d
    k -= 1
  end
  k
end


function findlastequal(A::AbstractMatrix{T}, ilo::Int, ihi::Int) where T <: Real
  k = ihi
  if ihi - ilo > 1
    if A[k,k-1] == 0
      d = A[k,k]
      while k > ilo && A[k-1,k-1] == d
        k -= 1
      end
    else
      k -= 1
      d, di = ev2(A, k)
      di > 0 || error("not expecting real eigenvalue in 2x2-block")
      while k > ilo + 1
        d2, di2 = ev2(A, k-2)
        if di2 > 0 && di2 == di && d2 == d
          k -= 2
        else
          break
        end
      end
    end
  end
  k
end

function ev2(A::AbstractMatrix{T}, k::Int) where T <: Real
  dis = discriminant(A, k, k+1)
  di = sqrt(max(-dis, zero(dis)))
  d = (A[k,k] + A[k+1,k+1]) / 2
  d, di
end
