
export AbstractTransformation, PlaneTrans, Rotation, Reflection, Transvection, Transformation
export transform!, A_mul_Binv!
import Base: convert, (*), A_mul_B!, A_mul_Bc!, inv, ctranspose, transpose, full

"""
Generalized Givens rotation.
"""
abstract AbstractTransformation{T}

"""
2-dimensional linear transformation acting on 2 indices of multi-dimensional space
of real numbers
"""
abstract PlaneTrans{T<:Number} <: AbstractTransformation{T}

"""
Sequence of plane transformations
"""
immutable Transformation{T<:Number} <: AbstractTransformation{T}
  trans::Tuple{Vararg{PlaneTrans{T}}}
end

Transformation{T}(t::PlaneTrans{T}...) = Transformation(tuple(t...))
"""
Two-dimensional rotation mapping - aka Givens rotation
"""
immutable Rotation{T<:Number} <: PlaneTrans{T}
  i1::Int
  i2::Int
  c::T
  s::T
  function Rotation(i1::Int, i2::Int, c::T, s::T)
    i1 > 0 && i2 > 0 && i1 != i2 || error("indices must be positive and different")
    new(i1, i2, c, s)
  end
end

"""
Two-dimensinal reflection mapping
"""
immutable Reflection{T<:Number} <: PlaneTrans{T}
  i1::Int
  i2::Int
  c::T
  s::T
  function Reflection(i1::Int, i2::Int, c::T, s::T)
    i1 > 0 && i2 > 0 && i1 != i2 || error("indices must be positive and different")
    new(i1, i2, c, s)
  end
end

"""
Transvection or aka shear mapping (German: Scherung)
"""
immutable Transvection{T<:Number} <: PlaneTrans{T}
  i1::Int
  i2::Int
  s::T
  function Transvection(i1::Int, i2::Int, s::T)
    i1 > 0 && i2 > 0 && i1 != i2 || error("indices must be positive and different")
    new(i1, i2, s)
  end
end

Transvection{T<:Number}(i1::Int, i2::Int, s::T) = Transvection{T}(i1, i2, s)
Rotation(i1::Int, i2::Int, phi::Real) = Rotation(i1, i2, cos(phi), sin(phi))
Rotation{T<:Number}(i1::Int, i2::Int, c::T, s::T) = Rotation{T}(i1, i2, c, s)
Reflection(i1::Int, i2::Int, phi::Real) = Reflection(i1, i2, cos(phi), sin(phi))
Reflection{T<:Number}(i1::Int, i2::Int, c::T, s::T) = Reflection{T}(i1, i2, c, s)

function *{T,S}(p1::PlaneTrans{T}, p2::PlaneTrans{S})
  TS = promote_type(T, S)
  Transformation(p2, p1)
end

function *{T,S}(G::PlaneTrans{S}, tr::Transformation{T})
  TS = promote_type(T, S)
  trans = T == TS ? tr.trans : [convert(PlaneTrans{TS}, g) for g in tr.trans]
  Transformation(trans..., PlaneTrans{TS}(G))
end

function *{T,S}(tr::Transformation{T}, G::PlaneTrans{S})
  TS = promote_type(T, S)
  trans = T == TS ? tr.trans : [convert(PlaneTrans{TS}, g) for g in tr.trans]
  Transformation(PlaneTrans{TS}(G), trans...)
end

function *{T,S}(tr1::Transformation{S}, tr2::Transformation{T})
  TS = promote_type(T, S)
  trans1 = T == TS ? tr1.trans : [convert(PlaneTrans{TS}, g) for g in tr1.trans]
  trans2 = T == TS ? tr2.trans : [convert(PlaneTrans{TS}, g) for g in tr2.trans]
  Transformation(trans2..., trans1...)
end

function *{T,S}(R::AbstractTransformation{T}, A::AbstractVecOrMat{S})
  TS = promote_type(T, S)
  A_mul_B!(convert(AbstractTransformation{TS}, R), TS == S ? copy(A) : convert(AbstractArray{TS}, A))    
end

function *{T,S}(A::AbstractMatrix{S}, R::Transformation{T})
  TS = promote_type(T, S)
  AA = TS == S ? copy(A) : convert(AbstractMatrix{TS}, A)
  RR = convert(AbstractTransformation{TS}, R)    
  A_mul_Bc!(A, ctranspose(RR))
end

function A_mul_B!(tr::Transformation, A::AbstractVecOrMat)
  @inbounds for i = 1:length(tr.trans)
    A_mul_B!(tr.trans[i], A)
  end
  A
end

function A_mul_Bc!(A::AbstractMatrix, tr::Transformation)
  @inbounds for i = 1:length(tr.trans)
    A_mul_Bc!(A, tr.trans[i])
  end
  A
end

function A_mul_Binv!(A::AbstractMatrix, tr::Transformation)
  @inbounds for i = 1:length(tr.trans)
    A_mul_Binv!(A, tr.trans[i])
  end
  A
end

function A_mul_B!(G::Rotation, A::AbstractVecOrMat)
    m, n = size(A, 1), size(A, 2)
    if G.i2 > m
        throw(DimensionMismatch("column indices for rotation are outside the matrix"))
    end
    @inbounds @simd for i = 1:n
        a1, a2 = A[G.i1,i], A[G.i2,i]
        A[G.i1,i] = G.c * a1 - G.s * a2
        A[G.i2,i] = G.s * a1 + G.c * a2
    end
    return A
end

function A_mul_Binv!(A::AbstractMatrix, G::Rotation)
    m, n = size(A, 1), size(A, 2)
    if G.i2 > n 
        throw(DimensionMismatch("column indices for rotation are outside the matrix"))
    end 
    @inbounds @simd for i = 1:m 
        a1, a2 = A[i,G.i1], A[i,G.i2]
        A[i,G.i1] = a1 * G.c - a2 * G.s
        A[i,G.i2] = a1 * G.s + a2 * G.c
    end 
    return A
end

A_mul_Bc!(A::AbstractMatrix, G::Rotation) = A_mul_Binv!(A, G)

function A_mul_B!(G::Reflection, A::AbstractVecOrMat)
    m, n = size(A, 1), size(A, 2)
    if G.i2 > m
        throw(DimensionMismatch("column indices for reflection are outside the matrix"))
    end
    @inbounds @simd for i = 1:n
        a1, a2 = A[G.i1,i], A[G.i2,i]
        A[G.i1,i] = G.c * a1 + G.s * a2
        A[G.i2,i] = G.s * a1 - G.c * a2
    end
    return A
end

function A_mul_Binv!(A::AbstractMatrix, G::Reflection)
    m, n = size(A, 1), size(A, 2)
    if G.i2 > n 
        throw(DimensionMismatch("column indices for reflection are outside the matrix"))
    end 
    @inbounds @simd for i = 1:m 
        a1, a2 = A[i,G.i1], A[i,G.i2]
        A[i,G.i1] =  a1 * G.c + a2 * G.s
        A[i,G.i2] =  a1 * G.s - a2 * G.c
    end 
    return A
end

A_mul_Bc!(A::AbstractMatrix, G::Reflection) = A_mul_Binv!(A, G)

function A_mul_B!(G::Transvection, A::AbstractVecOrMat)
    m, n = size(A, 1), size(A, 2)
    if G.i2 > m
        throw(DimensionMismatch("column indices for transvection are outside the matrix"))
    end
    @inbounds @simd for i = 1:n
        A[G.i1,i] += G.s * A[G.i2,i]
    end
    return A
end

function A_mul_Binv!(A::AbstractMatrix, G::Transvection)
    m, n = size(A, 1), size(A, 2)
    if G.i2 > n 
        throw(DimensionMismatch("column indices for transvection are outside the matrix"))
    end 
    @inbounds @simd for i = 1:m 
        A[i,G.i2] -=  A[i,G.i1] * G.s
    end 
    return A
end

function transform!(A::AbstractMatrix, r::AbstractTransformation)
  A_mul_B!(r, A)
  A_mul_Binv!(A, r)
  A
end

ctranspose(G::Rotation) = Rotation(G.i1, G.i2, G.c, -G.s)
ctranspose(G::Reflection) = Reflection(G.i1, G.i2, -G.c, G.s)
ctranspose(G::Transvection) = Transvection(G.i2, G.i1, G.s)
ctranspose(G::Transformation) = Transformation(reverse!([ctranspose(g) for g in G.trans]))

inv(G::Rotation) = ctranspose(G)
inv(G::Reflection) = G
inv(G::Transvection) = Transvection(G.i1, G.i2, -G.s)
inv(G::Transformation) = Transformation(reverse!([inv(g) for g in G.trans]...))

function full(t::Transformation, n::Int = 0)
  m = max(n, maximum([maxind(g) for g in t.trans]))
  A = eye(m,m)
  for g in t.trans
    A_mul_B!(g, A)
  end
  A
end

function full(G::PlaneTrans, n::Int = 0)
  full(Transformation(G), n)
end

maxind(G::Rotation) = max(G.i1, G.i2)
maxind(G::Reflection) = max(G.i1, G.i2)
maxind(G::Transvection) = max(G.i1, G.i2)

convert{T<:Number}(::Type{Matrix}, r::Rotation{T}) = [r.c r.s; -r.s r.c]
convert{T<:Number}(::Type{Matrix}, r::Reflection{T}) = [r.c r.s; r.s -r.c]
convert{T<:Number}(::Type{Matrix}, r::Transvection{T}) = [one(T) r.s; zero(T) one(T)]
convert{T<:Number}(::Type{Matrix}, r::Transformation{T}) = full(r)

convert{T}(::Type{Rotation{T}}, G::Rotation{T}) = G
convert{T}(::Type{Rotation{T}}, G::Rotation) = Rotation(G.i1, G.i2, convert(T,G.c), convert(T, G.s))
convert{T}(::Type{Reflection{T}}, G::Reflection{T}) = G
convert{T}(::Type{Reflection{T}}, G::Reflection) = Reflection(G.i1, G.i2, convert(T,G.c), convert(T, G.s))
convert{T}(::Type{Transvection{T}}, G::Transvection{T}) = G
convert{T}(::Type{Transvection{T}}, G::Transvection) = Transvection(G.i1, G.i2, convert(T, G.s))

convert{T}(::Type{PlaneTrans{T}}, G::Rotation) = convert(Rotation{T}, G)
convert{T}(::Type{PlaneTrans{T}}, G::Reflection) = convert(Reflection{T}, G)
convert{T}(::Type{PlaneTrans{T}}, G::Transvection) = convert(Transvection{T}, G)

convert{T}(::Type{Transformation{T}}, G::PlaneTrans{T}) = Transformation{T}(G)
convert{T}(::Type{Transformation{T}}, G::Transformation{T}) = G
convert{T}(::Type{Transformation{T}}, G::Transformation) = Transformation{T}([convert(PlansTrans{T}, g) for g in G.trans])

convert{T}(::Type{AbstractTransformation{T}}, G::PlaneTrans) = convert(PlaneTrans{T}, G)
convert{T}(::Type{AbstractTransformation{T}}, G::Transformation) = convert(Transformation{T}, G)

