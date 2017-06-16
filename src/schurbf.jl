
import Base.LinAlg.schurfact!
import Base.getindex

# Schur decomposition
# redefine Base.LinAlg structure in order to support BigFloatOrComplex
immutable Schur{Ty<:BigFloatOrComplex, S<:AbstractMatrix} <: Factorization{Ty}
    T::S
    Z::S
    values::Vector
    Schur(T::AbstractMatrix{Ty}, Z::AbstractMatrix{Ty}, values::Vector) = new(T, Z, values)
end

function Schur{Ty}(T::AbstractMatrix{Ty}, Z::AbstractMatrix{Ty}, values::Vector)
  Schur{Ty, typeof(T)}(T, Z, values)
end

function getindex(F::Schur, sym::Symbol)

  if sym == :T || sym == :Schur
    F.T
  elseif sym == :Z || sym == :vectors
    F.Z
  elseif sym == :values
    F.values
  else
    error("invalid key $(sym) for Schur")
  end
end

function schurfact!{T<:BigFloatOrComplex}(A::StridedMatrix{T})
  chkstride1(A)
  n = checksquare(A)
  # transform to Hessenberg form
  HF = hessfact!(A)
  Q = full(HF[:Q])
  A = HF[:H]
  separate!(A, 1, n, Q, processPart!)
  eigv = seigendiag(A, 1, size(A, 1))
  Schur(A, Q, eigv)
end

function processPart!{T<:BigFloatOrComplex}(A::StridedMatrix{T}, ilo::Integer, ihi::Integer, Q::AbstractMatrix)
  S = ifelse(T <: Real, Float64, Complex128)
  # 1. step: estimate eigenvalues in lower precision
  F = schurfact!(S.(view(A, ilo:ihi, ilo:ihi)))
  ev = F[:values]
  # 2. step: use low precision values ot iterate to high precision
  refineprecision!(A, ilo, ihi, Q, ev)
end

