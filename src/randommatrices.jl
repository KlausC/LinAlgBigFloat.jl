module RandomMatrices

export unitvector, orthogonal, normal, general, symmetric, hermitian

using Base.Random: GLOBAL_RNG
using Base.LinAlg:  reflector!, QR

import Base.randn, Base.rand

"""
unitvector(m, T::Type [, rng::AbstractRNG]) -> Vector{T}

  create random unit vector, uniformly distributed on unit sphere of `m` dimensions.
  `T` may be any real or complex float type.
"""
function unitvector(m::Integer, S::Union{Type{T},Type{Complex{T}}}=Float64, rng::AbstractRNG = GLOBAL_RNG) where T<:AbstractFloat

  unitvector!(zeros(S, m), rng)
end

"""
  unitvector!(v::AbstractVector{T} [, rng::AbstractRNG]) -> v
  
  Same as unitvector, populating the given vector or 1-dim view `v`.
"""
function unitvector!(v::Union{AbstractVector{T},AbstractVector{Complex{T}}}, rng::AbstractRNG = GLOBAL_RNG) where T<:AbstractFloat

  n = length(v)
  r = zero(T)
  sqnh = sqrt(n) / 2
  while r < sqnh
    randn!(rng, v)
    r = norm(v)
  end
  normalize!(v)
end

"""
  orthogonal(m, n, ::Type{T}=Float64 [, rng::AbstractRNG]) -> LinAlg.QRCompactWYQ

Generate random orthogonal real or unitary complex matrix.
"""
function orthogonal(m::Integer, n::Integer, S::Type=Float64, rng::AbstractRNG=GLOBAL_RNG) 
  S = S == Complex ? Complex{Float64} : S 
  S <: Union{T,Complex{T}} where T <: AbstractFloat || error("illegal type $S")
  m >= n || error("more independent vectors($n) than dimension($m) requested")
  A = zeros(S, m, n)
  τ = zeros(S, min(m,n))

  n2 = n == m && (S <: Real) && rand(Bool) ? n - 1 : n
  for k = 1:n2
    x = view(A, k:m, k)
    unitvector!(x)
    τk = reflector!(x)
    τ[k] = τk
  end
  QR(A, τ)[:Q]
end

# modify rand and randn to produce BigFloat and complex values
function rand(rng::AbstractRNG, ::Type{BigFloat})
  bigFloat_randfill(rng, rand(rng, Float64))
end
function randn(rng::AbstractRNG, ::Type{BigFloat})
  bigFloat_randfill(rng, randn(rng, Float64))
end
function rand(rng::AbstractRNG, ::Type{Complex{T}}) where T<:AbstractFloat
  a, b = 2rand(T, 2) - 1
  while hypot(a,b) > 1
    a, b = 2rand(T, 2) - 1
  end
  complex(a, b)
end
function randn(rng::AbstractRNG, ::Type{Complex{T}}) where T<:AbstractFloat
  complex(randn(rng, T), randn(rng, T))
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
  diagonal(dia::AbstractVector{T}) -> Array{T,2}

Create quasi diagonal real matrix with diagonal blocks given by `dia`.
`dia` must contain only real and pairs of conjugate complex numbers.
If complex output matrix is required, use `diagm(dia)`.
"""
function diagonal(dia::Union{AbstractVector{T}, AbstractVector{Complex{T}}}) where T<:AbstractFloat

  m = length(dia)
  mr = m
  j = 1
  while j <= m
    a = dia[j]
    if !isreal(a)
      if j >= m || imag(a) != -imag(dia[j+1])
        throw(InexactError())
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

"""
  normal(dia::AbstractVector, [,rng]) -> Array{T,2}

Create real or complex normal matrix with eigenvalues given by dia.
"""
function normal(dia::Union{AbstractVector{T}, AbstractVector{Complex{T}}}, S::Union{Type{T},Type{Complex{T}}}=Float64, rng::AbstractRNG = GLOBAL_RNG) where T<:AbstractFloat

  D = S<:Real ? diagonal(dia) : diagm(dia)
  m = size(D, 1)
  Q = orthogonal(m, m, S, rng)
  Q * D * Q'
end

"""
  general(dia::Vector{<:AbstractFloat}, m, n, S::Type=Float64, [,rng]) -> Array{S,2}

Create general real or complex (m,n)-matrix with singular values given.

"""
function general(singular_values::AbstractVector{T}, m::Int, n::Int, S::Union{Type{T},Type{Complex{T}}}=Float64, rng::AbstractRNG=GLOBAL_RNG) where T<:AbstractFloat
 
  j = length(singular_values)
  j <= min(m, n) || error("too many singular values")
  all(singular_values .> 0) || error("singular values must be positive")

  D = zeros(S, m, n)
  for k = 1:j
    D[k,k] = singular_values[k]
  end
  orthogonal(m, m, S, rng) * D * orthogonal(n, n, S, rng)'
end

end # module
