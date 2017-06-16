module RandomMatrix

export unitvector, orthogonal

using Base.LinAlg
using Base.Random

"""
return random unit vector, uniformly distributed on unit sphere of n dimensions.
"""
function unitvector{T<:AbstractFloat}(n::Integer, ::Type{T} = Float64, rng::AbstractRNG = Random.GLOBAL_RNG)

  r = zero(T)
  v = zeros(T, n)
  sqnh = sqrt(n) / 2
  while r < sqnh
    randn!(rng, v)
    r = norm(v)
  end
  for k = 1:n
    v[k] /= r
  end
  v
end

function unitvector{T<:AbstractFloat}(n::Integer, ::Type{Complex{T}}, rng::AbstractRNG = Random.GLOBAL_RNG)

  r = zero(T)
  v = zeros(Complex{T}, n)
  sqnh = sqrt(n) / 2
  while r < sqnh
    for k = 1:n
      v[k] = complex(randn(rng, T), randn(rng, T))
    end
    r = norm(v)
  end
  for k = 1:n
    v[k] /= r
  end
  v
end

"""
Generate random orthogonal real matrix. Determinant may be 1 or -1.
"""

function orthogonal{T<:AbstractFloat}(n::Integer, m::Integer, S::Union{Type{T},Type{Complex{T}}} = Float64, rng ::AbstractRNG = Random.GLOBAL_RNG)

  Q = zeros(S,0)
  for k = 1:m
    append!(Q, unitvector(n, S, rng))
  end
  Q = reshape(Q, n, m)
  qr(Q)[1]
end

function orthogonal2{T<:AbstractFloat}(n::Integer, m::Integer, ::Type{T} = Float64, rng::AbstractRNG = Random.GLOBAL_RNG)

  n >= m || error("orthogonal random matrix requires n >= m")
  Q = eye(T, n, m)
  if m > 0 && rand(rng, Bool)
    Q[m,m] = -Q[m,m]
  end
  # R2 = zeros(T, n, m)

  for k = m:-1:1
    v = unitvector(n - k + 1, T, rng)
    # R2[k:end,k] = v
    R = UnitRotation(T)
    r = v[n-k+1]
    for i = n-k:-1:1
      g, r = givens(v[i], r, i + k - 1, i + k)
      A_mul_B!(g, R)
    end
    A_mul_B!(R', Q)
  end
  Q #, R2
end

UnitRotation{T<:AbstractFloat}(::Type{T}) = LinAlg.Rotation(LinAlg.Givens{T}[])

import Base.randn, Base.rand

function rand(rng::AbstractRNG, ::Type{BigFloat})
  bigFloat_randfill(rng, rand(rng, Float64))
end

function randn(rng::AbstractRNG, ::Type{BigFloat})
  bigFloat_randfill(rng, randn(rng, Float64))
end

"""
Append random bits in BigFloat representation of input small float number.
"""
function bigFloat_randfill{T<:Union{Float64,Float32,Float16}}(rng::AbstractRNG, a::T)
  missing = precision(BigFloat) - precision(a)
  bigi = BigInt(1) << missing
  add = ( rand(rng, BigInt(0):bigi-1) / bigi )
  add *= big"2.0"^(exponent(a) - precision(a))
  add = copysign(add, a)
  a + add
end

"""
Create quasi diagonal real matrix with given diagonal (real or complex, 
without negative imaginary parts).
"""
function diagonal{T<:AbstractFloat}(dia::Union{AbstractVector{T}, AbstractVector{Complex{T}}})

  m = length(dia)
  mr = m
  j = 1
  while j <= m
    a = dia[j]
    if !isreal(a)
      if j >= m || imag(a) != -imag(dia[j+1])
        return diagm(dia)
      else
        j += 1
      end
    end
    j += 1
  end

  A = zeros(T, mr, mr)

  k = 1
  while k <= m
    dr = real(dia[k])
    di = imag(dia[k])
    if di == 0
      A[k,k] = dr
      k += 1
    else
      A[k,k] = A[k+1,k+1] = dr
      A[k+1,k] = di
      A[k,k+1] = -di
      k += 2
    end
  end
  A
end

function normal{T<:AbstractFloat}(dia::Union{AbstractVector{T}, AbstractVector{Complex{T}}}, rng::AbstractRNG = Random.GLOBAL_RNG)

  D = diagonal(dia)
  m = size(D, 1)
  Q = orthogonal(m, m, eltype(D), rng)
  Q * D * Q'
end

end


