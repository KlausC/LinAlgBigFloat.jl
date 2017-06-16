
# reimplements methods: hessfact, hessfact!
# and minor functions from LinAlg: full, A_mul_B! ..., reflectorApply! 

import Base.LinAlg
import Base: A_mul_B!, Ac_mul_B!, A_mul_Bc!, copymutable, full
import Base.LinAlg: hessfact, hessfact!, Hessenberg, HessenbergQ
import Base.LinAlg: chkstride1, checksquare

import Base.LinAlg: reflector!, reflectorApply!
import Base.LinAlg.LAPACK: gehrd!

include("util.jl")
include("separate.jl")
include("refineprecision.jl")
include("deflationcrit.jl")
include("transformhess.jl")

typealias BigFloatOrComplex Union{Complex{BigFloat}, BigFloat}

function Hessenberg{T<:BigFloatOrComplex}(A::StridedMatrix{T})
  Hessenberg(gehrd!(A)...)
end

hessfact!{T<:BigFloatOrComplex}(A::StridedMatrix{T}) = Hessenberg(A)
hessfact{T<:BigFloatOrComplex}(A::StridedMatrix{T}) = hessfact!(copy(A))

# Perfom Householder/Hessenberg
# The Householder matrices are unitary but not hermitian in the complex case
# to provide real subdiagonal of the Hessenberg matrix.
# mimic the corresponding LAPACK.gehrd! function.
function gehrd!{T<:BigFloatOrComplex}(ilo::Integer, ihi::Integer, A::StridedMatrix{T})
  chkstride1(A)
  n = checksquare(A)
  τ = zeros(T, max(0, n - 1))
  
  ilo = max(ilo, 1)
  ihi = min(ihi, n)

  for k = ilo:ihi-1
    x = view(A, k+1:ihi, k)
    τk = reflector!(x)
    τ[k] = τk
    reflectorApply!(x, τk, view(A, k+1:n, k+1:n))
    reflectorApply!(view(A, :, k+1:n), x, τk)
  end
  A, τ
end

"""
  
  `reflectorApply!(x::AbstractVector, τ::Number, A::StridedMatrix)`
  `reflectorApply!(A::StridedMatrix, x::AbstractVector, τ::Number)`

Apply reflector `x, τ` from left or right to matrix `A`, see `linalg/generic.jl`.
Matrix is modified by this operation.
"""
function reflectorApply!(A::StridedMatrix, x::AbstractVector, τ::Number)
  n, m = size(A)
  if length(x) != m
    throw(DimensionMismatch("reflector has length $(length(x)), which must match the second d    imension of matrix A, $m"))
  end
  @inbounds begin
  for j = 1:n
    # dot
    vAj = A[j,1]
    for i = 2:m
      vAj += x[i]*A[j,i]
    end

    vAj = τ*vAj

    # ger
    A[j, 1] -= vAj
    for i = 2:m
      A[j, i] -= x[i]'*vAj
    end
   end
  end
  return A
end

# various multiplications with HessenbergQ
# note the ctranspose(τ[k]) - supports the case of complex τ!
function A_mul_B!{T<:BigFloatOrComplex}(HQ::HessenbergQ{T}, A::StridedVecOrMat{T})
  n = size(A, 1)
  τ = HQ.τ
  for k = length(τ):-1:1
    τk = τ[k]'
    if τk != 0
      x = view(HQ.factors, k+1:n, k)
      reflectorApply!(x, τk, view(A, k+1:n, :))
    end
  end
  A
end

function A_mul_B!{T<:BigFloatOrComplex}(A::StridedMatrix{T}, HQ::HessenbergQ{T})
  n = size(A, 2)
  τ = HQ.τ
  for k = 1:length(τ)
    τk = τ[k]
    if τk != 0
      x = view(HQ.factors, k+1:n, k)
      reflectorApply!(view(A, :, k+1:n), x, τk)
    end
  end
  A
end

function Ac_mul_B!{T<:BigFloatOrComplex}(HQ::HessenbergQ{T}, A::StridedVecOrMat{T})
  n = size(A, 1)
  τ = HQ.τ
  for k = 1:length(τ)
    τk = τ[k]
    if τk != 0
      x = view(HQ.factors, k+1:n, k)
      reflectorApply!(x, τk, view(A, k+1:n, :))
    end
  end
  A
end

function A_mul_Bc!{T<:BigFloatOrComplex}(A::StridedMatrix{T}, HQ::HessenbergQ{T})
  n = size(A, 2)
  τ = HQ.τ
  for k = length(τ):-1:1
    τk = τ[k]'
    if τk != 0
      x = view(HQ.factors, k+1:n, k)
      reflectorApply!(view(A, :, k+1:n), x, τk)
    end
  end
  A
end

function copymutable(HQ::HessenbergQ)
  Q = eye(HQ.factors)
  A_mul_B!(HQ, Q)
  Q
end

full{T<:BigFloatOrComplex,S<:AbstractMatrix}(HQ::HessenbergQ{T,S}) = copymutable(HQ)
