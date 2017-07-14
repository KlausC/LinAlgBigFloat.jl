
module SpecialMatrices

import Base:  getindex, size, *

struct JordanBlock{T<:Number} <: AbstractMatrix{T}
  blocks::Vector{Int}
  diag::T
end

size(A::JordanBlock) = Tuple([1; 1] * sum(A.blocks))

function getindex(A::JordanBlock{T}, i::Int, j::Int) where T <: Number
  if i == j
    A.diag
  elseif i > j || j > i + 1
    zero(T)
  else
    if i in cumsum(A.blocks)
      zero(T)
    else
      one(T)
    end
  end
end

function *(J::JordanBlock, A::AbstractMatrix)
  n = size(J, 2)
  na, ma = size(A)
  n == na || error("dimension mismatch")
  M = A * J.diag
  for i = 1:n-1
    if J[i,i+1] != 0
      M[i,:] += A[i+1,:]
    end
  end
  M
end

end #module


