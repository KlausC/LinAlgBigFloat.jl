
import LinearAlgebra.schurfact!
import Base.getindex

# Schur decomposition
# redefine LinearAlgebra structure in order to support BigFloatOrComplex
struct SchurBig{Ty<:BigFloatOrComplex, S<:AbstractMatrix} <: Factorization{Ty}
    T::S
    Z::S
    values::Vector
    SchurBig{Ty,S}(T::AbstractMatrix{Ty}, Z::AbstractMatrix{Ty}, values::Vector) where {Ty,S} = new(T, Z, values)
end

function SchurBig(T::AbstractMatrix{Ty}, Z::AbstractMatrix{Ty}, values::Vector) where {Ty}
  SchurBig{Ty, typeof(T)}(T, Z, values)
end

function getindex(F::SchurBig, sym::Symbol)

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

function schurfact!(A::StridedMatrix{T}) where {T<:BigFloatOrComplex}
  chkstride1(A)
  n = checksquare(A)
  # transform to Hessenberg form
  HF = hessfact!(A)
  Q = full(HF[:Q])
  A = HF[:H]
  separate!(A, 1, n, Q, processPart!)
  eigv = seigendiag(A, 1, size(A, 1))
  finish!(A, Q)
  SchurBig(A, Q, eigv)
end

function nonhessenberg(A::AbstractMatrix)
  n, m = size(A)
  sum = 0.0
  for i = 1:n
    for j = 1:min(i-2,m-2)
      sum += Float64(abs2(A[i,j]))
    end
  end
  for i = 2:n
    x = Float64(abs2(A[i,i-1]))
    if eltype(A) <: Complex || x < 1e-10 
      # bigger elements in subdiagonal are considered block
      sum += x
    end
  end
  sqrt(sum)
end

function processPart!(A::StridedMatrix{T}, ilo::Integer, ihi::Integer, Q::AbstractMatrix) where {T<:BigFloatOrComplex}
  S = ifelse(T <: Real, Float64, Complex128)
  # 1. step: estimate eigenvalues in lower precision
  F = schurfact!(S.(view(A, ilo:ihi, ilo:ihi)))
  ev = F[:values]
  # 2. step: use low precision values ot iterate to high precision
  refineprecision!(A, ilo, ihi, Q, ev)
end

