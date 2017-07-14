

import Base.LinAlg.rank

function rank(A::AbstractMatrix{T}) where T <: Number
  rank(qrfact(A, Val{true}))
end

function rank(QR::LinAlg.QRPivoted{T, S} ) where S <: AbstractMatrix{T} where T <: Number
  R = QR[:R]
  sv = sort(abs.(diag(R)), rev = true)
  tol = eps(sv[1]) * maximum(size(A))
  n = length(sv)
  while n > 0 && sv[n] <= tol
    n -= 1
  end
  n
end

"""
Produce orthogonal Matrix, whose vectors are basis of the range of Matrix A.
If the matrix is surjective, the identity matrix is returned.
"""
function span(A::AbstractMatrix{T}) where T <: Number

  QR = qrfact(A, Val{true})
  n, m = size(A)
  r = rank(QR)
  if r < n
    QR[:Q][:,1:r]
  else
    eye(T, n, n)
  end
end

"""
Produce othogonal Matrix, whose vectors are basis of the kernel of Matrix A.
If the matrix is injective, the identity matrix is returned.
"""
function kern(A::AbstractMatrix{T}) where T <: Number

  QR = qrfact(A, Val{true})
  r = rank(QR)
  n, m = size(A)
  K = zeros(T, m, m - r)
  for k = r+1:m
    K[k,k-r] = one(T)
  end
  if r < m
    r1r = 1:r
    rrm = r+1:m
    R = QR[:R]
    RV = view(R, r1r, r1r)
    K[r1r,:] = -RV \ R[r1r, rrm]
    p = sortperm(QR[:p]) # inverse of permutation
    K, R = qr(K[p,:])
  end
  K
end


