
"""
Verify, that |A * Q - Q * T| ≈ 0
A, Q and T must be Matrices with appropriate sizes.
"""
function checkfactors(A::AbstractMatrix{T1}, Q::AbstractMatrix{T2}, T::AbstractMatrix{T3}; atol::Real = zero(real(eltype(A))), rtol::Real = eps(real(eltype(A)))) where {T1<:Number, T2<:Number, T3<:Number}

  res = norm(A * Q - Q * T, 1)
  tol = rtol * norm(A, 1) + atol
  # println("checkfactors() = $(Float64(res)) < ! $(Float64(tol)) $(Float64(res/tol))")
  res <= rtol * norm(A, 1) + atol
end

