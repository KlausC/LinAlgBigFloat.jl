
using Base.LinAlg

import Base.LinAlg: sylvester, lyap

function sylvester(A::StridedMatrix{T},B::StridedMatrix{T},C::StridedMatrix{T}) where T<:BigFloatOrComplex

  _sylvester_lyapunov(A, Nullable(B), C, sylvester)
end


function _sylvester_lyapunov(A::StridedMatrix{T}, B::Nullable{<:StridedMatrix{T}}, C::StridedMatrix{T}, syly::Function) where T <: BigFloatOrComplex

  # make standard precision copy of all input matrices calculate first approximation
  P = Float64
  S = T <: Real ? P : Complex{P}
  AS = S.(A)
  if isnull(B)
    B = A'
    BS = nothing
  else
    B = get(B)
    BS = S.(B)
  end
  CS = S.(C)
  XS = syly(AS, BS, CS)
  X = T.(XS)
  rtol = eps(real(T))
  ftol = cbrt(eps(P)) # 0.9 
  mres = rtol * norm(CS, 1)
  mdx  = rtol * norm(XS, 1)
  pres = nres = pdx = ndx = big"Inf"
  
  while nres >= mres && ndx >= mdx && nres <= pres && ndx <= pdx
    pres = nres * ftol
    pdx = ndx * ftol
    R = A * X + X * B + C
    nres = norm(R, 1)
    CS = S.(R)
    dX = syly(AS, BS, CS)
    ndx = norm(dX, 1)
    if nres <= pres && ndx <= pdx
      X .+= dX
    end
    println("nres = $(Float64(nres)/Float64(mres)), ndx = $(Float64(ndx)/Float64(mdx))")
  end
  X
end

function lyap(A::StridedMatrix{T}, C::StridedMatrix{T}) where T<:BigFloatOrComplex

  lyapunov(A, B, C) = lyap(A, C)
  _sylvester_lyapunov(A, Nullable{typeof(C)}(), C, lyapunov)
end

