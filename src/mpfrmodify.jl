"""
BigFloat operations modifying a buffer. The first argument is overwritten by the result.
All operations of MPRF returning a BigFloat are simulated.
According to the MPFR C-functions, the result variable may be identical to one of the arguments.
"""

module MPFR_MODIFY

export
    dot!, rotate!, neg!,
    mul!, add!, sub!, dv!, pow!, ceil!, copysign!, div!,
    exp!, exp2!, exponent!, factorial!, floor!, fma!, fms!, hypot!,
    ldexp!, log!, log2!, log10!, max!, min!, mod!, modf!,
    nextfloat!, prevfloat!, rem!, rem2pi!, round!,
    sqrt!, trunc!, exp10!, expm1!,
    gamma!, lgamma!, log1p!,
    sin!, cos!, tan!, sec!, csc!, cot!, acos!, asin!, atan!,
    cosh!, sinh!, tanh!, sech!, csch!, coth!, acosh!, asinh!, atanh!, atan2!,
    cbrt!, significand, frexp!

import Base: broadcast!, copy!, sum!, A_mul_B!
import Base.GMP: ClongMax, CulongMax, CdoubleMax, Limb
import Base.MPFR.ROUNDING_MODE

# Basic arithmetic without promotion
for (fJ, fC) in ((:add!,:add), (:mul!,:mul))
    @eval begin
        # BigFloat
        function ($fJ)(z::BigFloat, x::BigFloat, y::BigFloat)
            ccall(($(string(:mpfr_,fC)),:libmpfr), Int32, (Ptr{BigFloat}, Ptr{BigFloat}, Ptr{BigFloat}, Int32), &z, &x, &y, ROUNDING_MODE[])
            return z
        end

        # Unsigned Integer
        function ($fJ)(z::BigFloat, x::BigFloat, c::CulongMax)
            ccall(($(string(:mpfr_,fC,:_ui)), :libmpfr), Int32, (Ptr{BigFloat}, Ptr{BigFloat}, Culong, Int32), &z, &x, c, ROUNDING_MODE[])
            return z
        end
        ($fJ)(z::BigFloat, c::CulongMax, x::BigFloat) = ($fJ)(z,x,c)

        # Signed Integer
        function ($fJ)(z::BigFloat, x::BigFloat, c::ClongMax)
            ccall(($(string(:mpfr_,fC,:_si)), :libmpfr), Int32, (Ptr{BigFloat}, Ptr{BigFloat}, Clong, Int32), &z, &x, c, ROUNDING_MODE[])
            return z
        end
        ($fJ)(z::BigFloat, c::ClongMax, x::BigFloat) = ($fJ)(z,x,c)

        # Float32/Float64
        function ($fJ)(z::BigFloat, x::BigFloat, c::CdoubleMax)
            ccall(($(string(:mpfr_,fC,:_d)), :libmpfr), Int32, (Ptr{BigFloat}, Ptr{BigFloat}, Cdouble, Int32), &z, &x, c, ROUNDING_MODE[])
            return z
        end
        ($fJ)(z::BigFloat, c::CdoubleMax, x::BigFloat) = ($fJ)(z,x,c)

        # BigInt
        function ($fJ)(z::BigFloat, x::BigFloat, c::BigInt)
            ccall(($(string(:mpfr_,fC,:_z)), :libmpfr), Int32, (Ptr{BigFloat}, Ptr{BigFloat}, Ptr{BigInt}, Int32), &z, &x, &c, ROUNDING_MODE[])
            return z
        end
        ($fJ)(z::BigFloat, c::BigInt, x::BigFloat) = ($fJ)(z,x,c)
    end
end

for (fJ, fC) in ((:sub!,:sub), (:dv!,:div))
    @eval begin
        # BigFloat
        function ($fJ)(z::BigFloat, x::BigFloat, y::BigFloat)
            ccall(($(string(:mpfr_,fC)),:libmpfr), Int32, (Ptr{BigFloat}, Ptr{BigFloat}, Ptr{BigFloat}, Int32), &z, &x, &y, ROUNDING_MODE[])
            return z
        end

        # Unsigned Int
        function ($fJ)(z::BigFloat, x::BigFloat, c::CulongMax)
            ccall(($(string(:mpfr_,fC,:_ui)), :libmpfr), Int32, (Ptr{BigFloat}, Ptr{BigFloat}, Culong, Int32), &z, &x, c, ROUNDING_MODE[])
            return z
        end
        function ($fJ)(z::BigFloat, c::CulongMax, x::BigFloat)
            ccall(($(string(:mpfr_,:ui_,fC)), :libmpfr), Int32, (Ptr{BigFloat}, Culong, Ptr{BigFloat}, Int32), &z, c, &x, ROUNDING_MODE[])
            return z
        end

        # Signed Integer
        function ($fJ)(z::BigFloat, x::BigFloat, c::ClongMax)
            ccall(($(string(:mpfr_,fC,:_si)), :libmpfr), Int32, (Ptr{BigFloat}, Ptr{BigFloat}, Clong, Int32), &z, &x, c, ROUNDING_MODE[])
            return z
        end
        function ($fJ)(z::BigFloat, c::ClongMax, x::BigFloat)
            ccall(($(string(:mpfr_,:si_,fC)), :libmpfr), Int32, (Ptr{BigFloat}, Clong, Ptr{BigFloat}, Int32), &z, c, &x, ROUNDING_MODE[])
            return z
        end

        # Float32/Float64
        function ($fJ)(z::BigFloat, x::BigFloat, c::CdoubleMax)
            ccall(($(string(:mpfr_,fC,:_d)), :libmpfr), Int32, (Ptr{BigFloat}, Ptr{BigFloat}, Cdouble, Int32), &z, &x, c, ROUNDING_MODE[])
            return z
        end
        function ($fJ)(z::BigFloat, c::CdoubleMax, x::BigFloat)
            ccall(($(string(:mpfr_,:d_,fC)), :libmpfr), Int32, (Ptr{BigFloat}, Cdouble, Ptr{BigFloat}, Int32), &z, c, &x, ROUNDING_MODE[])
            return z
        end

        # BigInt
        function ($fJ)(z::BigFloat, x::BigFloat, c::BigInt)
            ccall(($(string(:mpfr_,fC,:_z)), :libmpfr), Int32, (Ptr{BigFloat}, Ptr{BigFloat}, Ptr{BigInt}, Int32), &z, &x, &c, ROUNDING_MODE[])
            return z
        end
        # no :mpfr_z_div function
    end
end

function sub!(z::BigFloat, c::BigInt, x::BigFloat)
    ccall((:mpfr_z_sub, :libmpfr), Int32, (Ptr{BigFloat}, Ptr{BigInt}, Ptr{BigFloat}, Int32), &z, &c, &x, ROUNDING_MODE[])
    return z
end

function fma!(r::BigFloat, x::BigFloat, y::BigFloat, z::BigFloat)
    ccall(("mpfr_fma",:libmpfr), Int32, (Ptr{BigFloat}, Ptr{BigFloat}, Ptr{BigFloat}, Ptr{BigFloat}, Int32), &r, &x, &y, &z, ROUNDING_MODE[])
    return r
end

function fms!(r::BigFloat, x::BigFloat, y::BigFloat, z::BigFloat)
    ccall(("mpfr_fms",:libmpfr), Int32, (Ptr{BigFloat}, Ptr{BigFloat}, Ptr{BigFloat}, Ptr{BigFloat}, Int32), &r, &x, &y, &z, ROUNDING_MODE[])
    return r
end

# div
# BigFloat
function div!(z::BigFloat, x::BigFloat, y::BigFloat)
    ccall((:mpfr_div,:libmpfr), Int32, (Ptr{BigFloat}, Ptr{BigFloat}, Ptr{BigFloat}, Int32), &z, &x, &y, to_mpfr(RoundToZero))
    ccall((:mpfr_trunc, :libmpfr), Int32, (Ptr{BigFloat}, Ptr{BigFloat}), &z, &z)
    return z
end

# Unsigned Int
function div!(z::BigFloat, x::BigFloat, c::CulongMax)
    ccall((:mpfr_div_ui, :libmpfr), Int32, (Ptr{BigFloat}, Ptr{BigFloat}, Culong, Int32), &z, &x, c, to_mpfr(RoundToZero))
    ccall((:mpfr_trunc, :libmpfr), Int32, (Ptr{BigFloat}, Ptr{BigFloat}), &z, &z)
    return z
end
function div!(z::BigFloat, c::CulongMax, x::BigFloat)
    ccall((:mpfr_ui_div, :libmpfr), Int32, (Ptr{BigFloat}, Culong, Ptr{BigFloat}, Int32), &z, c, &x, to_mpfr(RoundToZero))
    ccall((:mpfr_trunc, :libmpfr), Int32, (Ptr{BigFloat}, Ptr{BigFloat}), &z, &z)
    return z
end

# Signed Integer
function div!(z::BigFloat, x::BigFloat, c::ClongMax)
    ccall((:mpfr_div_si, :libmpfr), Int32, (Ptr{BigFloat}, Ptr{BigFloat}, Clong, Int32), &z, &x, c, to_mpfr(RoundToZero))
    ccall((:mpfr_trunc, :libmpfr), Int32, (Ptr{BigFloat}, Ptr{BigFloat}), &z, &z)
    return z
end
function div!(z::BigFloat, c::ClongMax, x::BigFloat)
    ccall((:mpfr_si_div, :libmpfr), Int32, (Ptr{BigFloat}, Clong, Ptr{BigFloat}, Int32), &z, c, &x, to_mpfr(RoundToZero))
    ccall((:mpfr_trunc, :libmpfr), Int32, (Ptr{BigFloat}, Ptr{BigFloat}), &z, &z)
    return z
end

# Float32/Float64
function div!(z::BigFloat, x::BigFloat, c::CdoubleMax)
    ccall((:mpfr_div_d, :libmpfr), Int32, (Ptr{BigFloat}, Ptr{BigFloat}, Cdouble, Int32), &z, &x, c, to_mpfr(RoundToZero))
    ccall((:mpfr_trunc, :libmpfr), Int32, (Ptr{BigFloat}, Ptr{BigFloat}), &z, &z)
    return z
end
function div!(z::BigFloat, c::CdoubleMax, x::BigFloat)
    ccall((:mpfr_d_div, :libmpfr), Int32, (Ptr{BigFloat}, Cdouble, Ptr{BigFloat}, Int32), &z, c, &x, to_mpfr(RoundToZero))
    ccall((:mpfr_trunc, :libmpfr), Int32, (Ptr{BigFloat}, Ptr{BigFloat}), &z, &z)
    return z
end

# BigInt
function div!(z::BigFloat, x::BigFloat, c::BigInt)
    ccall((:mpfr_div_z, :libmpfr), Int32, (Ptr{BigFloat}, Ptr{BigFloat}, Ptr{BigInt}, Int32), &z, &x, &c, to_mpfr(RoundToZero))
    ccall((:mpfr_trunc, :libmpfr), Int32, (Ptr{BigFloat}, Ptr{BigFloat}), &z, &z)
    return z
end


# More efficient commutative operations
for (fJ, fC, fI) in ((:add!, :add, 0), (:mul!, :mul, 1))
    @eval begin
        function ($fJ)(z::BigFloat, a::BigFloat, b::BigFloat, c::BigFloat)
            ccall(($(string(:mpfr_,fC)), :libmpfr), Int32, (Ptr{BigFloat}, Ptr{BigFloat}, Ptr{BigFloat}, Int32), &z, &a, &b, ROUNDING_MODE[])
            ccall(($(string(:mpfr_,fC)), :libmpfr), Int32, (Ptr{BigFloat}, Ptr{BigFloat}, Ptr{BigFloat}, Int32), &z, &z, &c, ROUNDING_MODE[])
            return z
        end
        function ($fJ)(z::BigFloat, a::BigFloat, b::BigFloat, c::BigFloat, d::BigFloat)
            ccall(($(string(:mpfr_,fC)), :libmpfr), Int32, (Ptr{BigFloat}, Ptr{BigFloat}, Ptr{BigFloat}, Int32), &z, &a, &b, ROUNDING_MODE[])
            ccall(($(string(:mpfr_,fC)), :libmpfr), Int32, (Ptr{BigFloat}, Ptr{BigFloat}, Ptr{BigFloat}, Int32), &z, &z, &c, ROUNDING_MODE[])
            ccall(($(string(:mpfr_,fC)), :libmpfr), Int32, (Ptr{BigFloat}, Ptr{BigFloat}, Ptr{BigFloat}, Int32), &z, &z, &d, ROUNDING_MODE[])
            return z
        end
        function ($fJ)(z::BigFloat, a::BigFloat, b::BigFloat, c::BigFloat, d::BigFloat, e::BigFloat)
            ccall(($(string(:mpfr_,fC)), :libmpfr), Int32, (Ptr{BigFloat}, Ptr{BigFloat}, Ptr{BigFloat}, Int32), &z, &a, &b, ROUNDING_MODE[])
            ccall(($(string(:mpfr_,fC)), :libmpfr), Int32, (Ptr{BigFloat}, Ptr{BigFloat}, Ptr{BigFloat}, Int32), &z, &z, &c, ROUNDING_MODE[])
            ccall(($(string(:mpfr_,fC)), :libmpfr), Int32, (Ptr{BigFloat}, Ptr{BigFloat}, Ptr{BigFloat}, Int32), &z, &z, &d, ROUNDING_MODE[])
            ccall(($(string(:mpfr_,fC)), :libmpfr), Int32, (Ptr{BigFloat}, Ptr{BigFloat}, Ptr{BigFloat}, Int32), &z, &z, &e, ROUNDING_MODE[])
            return z
        end
    end
end

function neg!(z::BigFloat)
    ccall((:mpfr_neg, :libmpfr), Int32, (Ptr{BigFloat}, Ptr{BigFloat}, Int32), &z, &z, ROUNDING_MODE[])
    return z
end

function neg!(z::BigFloat, x::BigFloat)
    ccall((:mpfr_neg, :libmpfr), Int32, (Ptr{BigFloat}, Ptr{BigFloat}, Int32), &z, &x, ROUNDING_MODE[])
    return z
end

function sqrt!(z::BigFloat, x::BigFloat)
    isnan(x) && return x
    ccall((:mpfr_sqrt, :libmpfr), Int32, (Ptr{BigFloat}, Ptr{BigFloat}, Int32), &z, &x, ROUNDING_MODE[])
    if isnan(z)
        throw(DomainError())
    end
    return z
end

sqrt!(z::BigFloat, x::BigInt) = sqrt!(z, BigFloat(x))

function pow!(z::BigFloat, x::BigFloat, y::BigFloat)
    ccall((:mpfr_pow, :libmpfr), Int32, (Ptr{BigFloat}, Ptr{BigFloat}, Ptr{BigFloat}, Int32), &z, &x, &y, ROUNDING_MODE[])
    return z
end

function pow!(z::BigFloat, x::BigFloat, y::CulongMax)
    ccall((:mpfr_pow_ui, :libmpfr), Int32, (Ptr{BigFloat}, Ptr{BigFloat}, Culong, Int32), &z, &x, y, ROUNDING_MODE[])
    return z
end

function pow!(z::BigFloat, x::BigFloat, y::ClongMax)
    ccall((:mpfr_pow_si, :libmpfr), Int32, (Ptr{BigFloat}, Ptr{BigFloat}, Clong, Int32), &z, &x, y, ROUNDING_MODE[])
    return z
end

function pow!(z::BigFloat, x::BigFloat, y::BigInt)
    ccall((:mpfr_pow_z, :libmpfr), Int32, (Ptr{BigFloat}, Ptr{BigFloat}, Ptr{BigInt}, Int32), &z, &x, &y, ROUNDING_MODE[])
    return z
end

pow!(z::BigFloat, x::BigFloat, y::Integer)  = typemin(Clong)  <= y <= typemax(Clong)  ? pow!(z, x, Clong(y))  : pow!(z, x, BigInt(y))
                                                                                    
pow!(z::BigFloat, x::BigFloat, y::Unsigned) = typemin(Culong) <= y <= typemax(Culong) ? pow!(z, x, Culong(y)) : pow!(z, x, BigInt(y))

for f in (:exp, :exp2, :exp10, :expm1, :cosh, :sinh, :tanh, :sech, :csch, :coth, :cbrt)
@eval function $(Symbol(f, "!"))(z::BigFloat, x::BigFloat)
        ccall(($(string(:mpfr_,f)), :libmpfr), Int32, (Ptr{BigFloat}, Ptr{BigFloat}, Int32), &z, &x, ROUNDING_MODE[])
        return z
    end
end

# return log(2)
function big_ln2!(c::BigFloat)
    ccall((:mpfr_const_log2, :libmpfr), Cint, (Ptr{BigFloat}, Int32),
          &c, MPFR.ROUNDING_MODE[])
    return c
end

function ldexp!(z::BigFloat, x::BigFloat, n::Clong)
    ccall((:mpfr_mul_2si, :libmpfr), Int32, (Ptr{BigFloat}, Ptr{BigFloat}, Clong, Int32), &z, &x, n, ROUNDING_MODE[])
    return z
end
function ldexp!(z::BigFloat, x::BigFloat, n::Culong)
    ccall((:mpfr_mul_2ui, :libmpfr), Int32, (Ptr{BigFloat}, Ptr{BigFloat}, Culong, Int32), &z, &x, n, ROUNDING_MODE[])
    return z
end
ldexp!(z::BigFloat, x::BigFloat, n::ClongMax) = ldexp!(z, x, convert(Clong, n))
ldexp!(z::BigFloat, x::BigFloat, n::CulongMax) = ldexp!(z, x, convert(Culong, n))
ldexp!(z::BigFloat, x::BigFloat, n::Integer) = mul!(z, x, exp2!(BigFloat(n)))

function factorial!(z::BigFloat, x::BigFloat)
    if x < 0 || !isinteger(x)
        throw(DomainError())
    end
    ui = convert(Culong, x)
    ccall((:mpfr_fac_ui, :libmpfr), Int32, (Ptr{BigFloat}, Culong, Int32), &z, ui, ROUNDING_MODE[])
    return z
end

function hypot!(z::BigFloat, x::BigFloat, y::BigFloat)
    ccall((:mpfr_hypot, :libmpfr), Int32, (Ptr{BigFloat}, Ptr{BigFloat}, Ptr{BigFloat}, Int32), &z, &x, &y, ROUNDING_MODE[])
    return z
end

for f in (:log, :log2, :log10)
  @eval function $(Symbol(f, "!"))(z::BigFloat, x::BigFloat)
        if x < 0
            throw(DomainError())
        end
        ccall(($(string(:mpfr_,f)), :libmpfr), Int32, (Ptr{BigFloat}, Ptr{BigFloat}, Int32), &z, &x, ROUNDING_MODE[])
        return z
    end
end

function log1p!(z::BigFloat, x::BigFloat)
    if x < -1
        throw(DomainError())
    end
    ccall((:mpfr_log1p, :libmpfr), Int32, (Ptr{BigFloat}, Ptr{BigFloat}, Int32), &z, &x, ROUNDING_MODE[])
    return z
end

function max!(z::BigFloat, x::BigFloat, y::BigFloat)
    isnan(x) && return x
    isnan(y) && return y
    ccall((:mpfr_max, :libmpfr), Int32, (Ptr{BigFloat}, Ptr{BigFloat}, Ptr{BigFloat}, Int32), &z, &x, &y, ROUNDING_MODE[])
    return z
end

function min!(z::BigFloat, x::BigFloat, y::BigFloat)
    isnan(x) && return x
    isnan(y) && return y
    ccall((:mpfr_min, :libmpfr), Int32, (Ptr{BigFloat}, Ptr{BigFloat}, Ptr{BigFloat}, Int32), &z, &x, &y, ROUNDING_MODE[])
    return z
end

function modf!(zint::BigFloat, zfloat::BigFloat, x::BigFloat)
  if isinf(x)
    copy!(zint, x)
    copy!(zfloat, BigFloat(NaN))
  end
  ccall((:mpfr_modf, :libmpfr), Int32, (Ptr{BigFloat}, Ptr{BigFloat}, Ptr{BigFloat}, Int32), &zint, &zfloat, &x, ROUNDING_MODE[])
end

function rem!(z::BigFloat, x::BigFloat, y::BigFloat)
    ccall((:mpfr_fmod, :libmpfr), Int32, (Ptr{BigFloat}, Ptr{BigFloat}, Ptr{BigFloat}, Int32), &z, &x, &y, ROUNDING_MODE[])
    return z
end

function rem!(z::BigFloat, x::BigFloat, y::BigFloat, ::RoundingMode{:Nearest})
    ccall((:mpfr_remainder, :libmpfr), Int32, (Ptr{BigFloat}, Ptr{BigFloat}, Ptr{BigFloat}, Int32), &z, &x, &y, ROUNDING_MODE[])
    return z
end

# TODO: use a higher-precision BigFloat(pi) here?
function rem2pi!(z::BigFloat, x::BigFloat, r::RoundingMode)
  mul!(z, BigFloat(pi), 2)
  rem!(z, x, z, r)
end

function sum!(z::BigFloat, arr::AbstractArray{BigFloat})
copy!(z, BigFloat(0))
    for i in arr
        ccall((:mpfr_add, :libmpfr), Int32,
            (Ptr{BigFloat}, Ptr{BigFloat}, Ptr{BigFloat}, Cint),
            &z, &z, &i, 0)
    end
    return z
end

# Functions for which NaN results are converted to DomainError, following Base
for f in (:sin,:cos,:tan,:sec,:csc,
          :acos,:asin,:atan,:acosh,:asinh,:atanh, :gamma)
  @eval begin
    function $(Symbol(f, "!"))(z::BigFloat, x::BigFloat)
            if isnan(x)
                return x
            end
            ccall(($(string(:mpfr_,f)), :libmpfr), Int32, (Ptr{BigFloat}, Ptr{BigFloat}, Int32), &z, &x, ROUNDING_MODE[])
            if isnan(z)
                throw(DomainError())
            end
            return z
        end
    end
end

# log of absolute value of gamma function
const lgamma_signp2 = Ref{Cint}()
function lgamma!(z::BigFloat, x::BigFloat)
    ccall((:mpfr_lgamma,:libmpfr), Cint, (Ptr{BigFloat}, Ptr{Cint}, Ptr{BigFloat}, Int32), &z, lgamma_signp2, &x, ROUNDING_MODE[])
    return z
end

lgamma_r!(z::BigFloat, x::BigFloat) = (lgamma!(z, x), lgamma_signp2[])

function atan2!(z::BigFloat, y::BigFloat, x::BigFloat)
    ccall((:mpfr_atan2, :libmpfr), Int32, (Ptr{BigFloat}, Ptr{BigFloat}, Ptr{BigFloat}, Int32), &z, &y, &x, ROUNDING_MODE[])
    return z
end

function copysign!(z::BigFloat, x::BigFloat, y::BigFloat)
    ccall((:mpfr_copysign, :libmpfr), Int32, (Ptr{BigFloat}, Ptr{BigFloat}, Ptr{BigFloat}, Int32), &z, &x, &y, ROUNDING_MODE[])
    return z
end

function frexp!(z::BigFloat, x::BigFloat)
    c = Ref{Clong}()
    ccall((:mpfr_frexp, :libmpfr), Int32, (Ptr{Clong}, Ptr{BigFloat}, Ptr{BigFloat}, Cint), c, &z, &x, ROUNDING_MODE[])
    return (z, c[])
end

function significand!(z::BigFloat, x::BigFloat)
    c = Ref{Clong}()
    ccall((:mpfr_frexp, :libmpfr), Int32, (Ptr{Clong}, Ptr{BigFloat}, Ptr{BigFloat}, Cint), c, &z, &x, ROUNDING_MODE[])
    # Double the significand to make it work as Base.significand
    ccall((:mpfr_mul_si, :libmpfr), Int32, (Ptr{BigFloat}, Ptr{BigFloat}, Clong, Int32), &z, &z, 2, ROUNDING_MODE[])
    return z
end

function round!(z::BigFloat, x::BigFloat)
    ccall((:mpfr_rint, :libmpfr), Int32, (Ptr{BigFloat}, Ptr{BigFloat}, Cint), &z, &x, ROUNDING_MODE[])
    return z
end
function round!(z::BigFloat, x::BigFloat,::RoundingMode{:NearestTiesAway})
    ccall((:mpfr_round, :libmpfr), Int32, (Ptr{BigFloat}, Ptr{BigFloat}), &z, &x)
    return z
end

function copy!(z::BigFloat, x::BigFloat)
    ccall((:mpfr_set, :libmpfr), Int32, (Ptr{BigFloat}, Ptr{BigFloat}, Int32),
          &z, &x, ROUNDING_MODE[])
    return z
end

function copy!(z::BigFloat, x::ClongMax)
    ccall((:mpfr_set_si, :libmpfr), Int32, (Ptr{BigFloat}, Clong, Int32),
          &z, x, ROUNDING_MODE[])
    return z
end

function copy!(z::BigFloat, x::CulongMax)
    ccall((:mpfr_set_ui, :libmpfr), Int32, (Ptr{BigFloat}, Culong, Int32),
          &z, x, ROUNDING_MODE[])
    return z
end

function copy!(z::BigFloat, x::CdoubleMax)
    ccall((:mpfr_set_d, :libmpfr), Int32, (Ptr{BigFloat}, Cdouble, Int32),
          &z, x, ROUNDING_MODE[])
    return z
end

function copy!(z::BigFloat, x::BigInt)
    ccall((:mpfr_set_z, :libmpfr), Int32, (Ptr{BigFloat}, Ptr{BigInt}, Int32),
          &z, &x, ROUNDING_MODE[])
    return z
end

copy!(z::BigFloat, x::Real) = copy!(z, BigFloat(x))

function nextfloat!(z::BigFloat, x::BigFloat)
  copy!(z, x)
  ccall((:mpfr_nextabove, :libmpfr), Int32, (Ptr{BigFloat},), &z) != 0
  return z
end

function prevfloat!(z::BigFloat, x::BigFloat)
  copy!(z, x)
  ccall((:mpfr_nextbelow, :libmpfr), Int32, (Ptr{BigFloat},), &z) != 0
  return z
end

# broadcast!(::typeof(identity), z::BigFloat, x::Real) = copy!(z, x)

function checklength(n::Int, m::Int)
  n != m &&
    throw(DimensionMismatch("first array has length $(n) which does not match the length of the second, $(m)."))
  n
end

checklength(a, b) = checklength(length(a), length(b))
checkmult(a::AbstractArray{T,2}, b) where T = checklength(size(a, 2), length(b))
checkmult(a, b::AbstractArray{T,2}) where T = checklenght(length(a), size(b, 2))

function dot!(z::BigFloat, a::AbstractArray{T,1}, b::AbstractArray{U,1}) where T <: Real where U <: Real

  n = checklength(a, b)
  _dot!(z, n, a, b)
end

function _dot!(z::BigFloat, n::Int, a::AbstractArray{T,1}, b::AbstractArray{T,1}) where T <: BigFloat
  copy!(z, 0)
  for i in 1:n
    fma!(z, a[i], b[i], z)
  end
  z
end

function _dot!(z::BigFloat, n::Int, a::AbstractArray{T,1}, b::AbstractArray{U,1}) where T <: BigFloat where U <: Real

  copy!(z, 0)
  c = BigFloat()
  for i in 1:n
    copy!(c, b[i])
    fma!(z, a[i], c, z)
  end
  z
end

function _dot!(z::BigFloat, n::Int, a::AbstractArray{T,1}, b::AbstractArray{U,1}) where T <: Real where U <: BigFloat
  _dot!(z, n, b, a)
end

function _dot!(z::BigFloat, n::Int, a::AbstractArray{T,1}, b::AbstractArray{U,1}) where T <: Real where U <: Real

  copy!(z, 0)
  c = BigFloat()
  for i in 1:n
    copy!(c, a[i])
    mul!(c, c, b[i])
    add!(z, z, c)
  end
  z
end

function mul!(z::AbstractVector{BigFloat}, a::AbstractArray{BigFloat,2}, b::AbstractVector{BigFloat})
  
  checkmult(a, b)
  n = checkmult(z, a)
  for i = 1:n
    isassigned(z, i) || ( z[i] = BigFloat() )
    dot!(z[i], view(a, i,:), b)
  end
  z
end

function mul!(z::AbstractVector{BigFloat}, a::AbstractVector{BigFloat}, b::Real)
  n = checklength(z, a)
  for i = 1:n
    isassigned(z, i) || ( z[i] = BigFloat() )
    mul!(z[i], a[i], b)
  end
  z
end

function add!(z::AbstractVector{BigFloat}, a::AbstractVector{BigFloat}, b::AbstractVector{<:Real})
  n = checklength(z, a)
  for i = 1:n
    isassigned(z, i) || ( z[i] = BigFloat() )
    add!(z[i], a[i], b[i])
  end
  z
end

function sub!(z::AbstractVector{BigFloat}, a::AbstractVector{BigFloat}, b::AbstractVector{<:Real})
  n = checklength(z, a)
  for i = 1:n
    isassigned(z, i) || ( z[i] = BigFloat() )
    sub!(z[i], a[i], b[i])
  end
  z
end

"""
Apply givens rotation, defined by `(c, s)` to `[a1, a2]`.
Use temporary work space `t[1:2]`. No aliasing of any used varables allowed.
"""
function rotate!(a1::BigFloat, a2::BigFloat, c::BigFloat, s::BigFloat, t::AbstractVector{BigFloat})

  mul!(t[1], a1, c)
  mul!(t[2], a1, s)
  fma!(a1, a2, s, t[1])
  fms!(a2, a2, c, t[2])
  nothing
end


end # module
