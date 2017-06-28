
"""
Refine Float64 to BigFloat precision.
"""
function iterate!{T<:BigFloat}(A::AbstractMatrix{T}, ilo::Integer, ihi::Integer, Q::AbstractM)

  # approximation of submatrix eigenvalues in Float64 precision
  r = ilo:ihi
  ev = seigendiag(smalleig(Float64.(view(A, r, r)))[1], 1, ihi-ilo+1)
  ihi = refineprecision!(A, ilo, ihi, Q, ev)
  ilo, ihi
end

"""
Refine all eigenvalues of a list close to estimation value
"""
function refineprecision!{T<:Union{Real,Complex}}(A::AbstractMatrix{T}, ilo::Integer, ihi::Integer, Q::AbstractM, ev::AbstractVector)
  
  realcase = T <: Real
  evr = reverse(ev)
  k = length(ev)
  while k >= 1
    evk = ev[k]
    evk2 = ! realcase || isreal(evk) ? T(evk) : Complex{T}(evk)

    ihi0 = ihi
    ihi = refineprecision!(A, ilo, ihi, Q, evk)
    if ihi0 - ihi >= 2 && k > 1 && imag(evk) == -imag(ev[k-1])
      k -= 1
    end
    k -= 1
  end
  ihi
end

"""
Refine one eigenvalue close to estimation value
"""
function refineprecision!{T<:Union{Real,Complex}}(A::AbstractMatrix{T}, ilo::Integer, ihi::Integer, Q::AbstractM, ev::Number)
 
  ih2 = ! (T <: Real) || isreal(ev) ? ihi : ihi - 1
  r = ih2:ihi
  while !converged(A, ilo, ihi)
    transform_Hess!(A, ilo, ihi, Q, [ev], ihi, 0)
    ev = eig2(view(A, r, r))[end]
  end
  ihi = ih2 - 1
  if ihi >= ilo
    A[ihi+1,ihi] = 0
  end
  ihi
end

"""
Detect convergence of last eigenvalue block of hessenberg matrix.
"""
function converged{T<:Union{Real,Complex}}(A::AbstractMatrix{T}, ilo::Integer, ihi::Integer)
  if ilo >= ihi
    return true
  end
  if deflation_criterion(A[ihi,ihi-1], A[ihi-1,ihi-1], A[ihi,ihi])
    return true
  end
  if ! (T <: Real) || discriminant(A, ihi-1, ihi) > 0
    return false
  end
  if ilo >= ihi - 1
    return true
  end
  deflation_criterion(A[ihi-1,ihi-2], A[ihi-2,ihi-2], A[ihi-1,ihi-1]) # TODO
end

