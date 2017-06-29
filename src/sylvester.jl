
include("util.jl")

using Base.LinAlg

import Base.LinAlg.sylvester

function sylvester(A::StridedMatrix{T},B::StridedMatrix{T},C::StridedMatrix{T}) where T<:BigFloatOrComplex

  # make standard precision copy of all input matrices calculate first approximation
  P = Float64
  S = T <: Real ? P : Complex{P}
  AS = S.(A)
  BS = S.(B)
  CS = S.(C)
  XS = sylvester(AS, BS, CS)
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
    RS = S.(R)
    dX = sylvester(AS, BS, RS)
    ndx = norm(dX, 1)
    X .+= dX
    # println("nres = $(Float64(nres)/Float64(mres)), ndx = $(Float64(ndx)/Float64(mdx))")
  end
  X
end
  




