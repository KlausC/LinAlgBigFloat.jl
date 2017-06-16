module Diagonalize


include("rotgivens.jl")

using Optim

"""
Given 2 vectors u[1:3] and x[1:3] find a rotation matrix R[1:3,1:3] with the properties:
R * u = [*; a; 0] and R * x = [*; b; 0] and a²+b² is minimal under these conditions.
If not u[3] == 0 == x[3] the solution is unique. 
"""
function rot3matrix{T<:Real}(x::Vector{T}, u::Vector{T})
  S = promote_type(Float32, typeof(one(T)/one(T)))
  Z = zero(S)
  E = one(S)
  sina = sinb = sinc = Z
  cosa = cosb = cosc = E

  norm3(a, b, c) = hypot(hypot(a, b), c)

  nx, nu = norm3(x...), norm3(u...)
  if nx < nu
    u, x = x, u
    nu, nx = nx, nu
  end

  if x[3] != Z || u[3] != Z
    a = cross(x, u)
    na = norm3(a...)
    if na == Z
      a = [Z; -x[3]; x[2]]
      na = norm3(a...)
    end
    a /= na
    if a[3] < Z
      a = -a
    end

    cosb = a[3]
    sinb = sqrt(one(T) - cosb^2)
    na = hypot(a[1], a[2])
    sina, cosa = a[1] / na , -a[2] / na
    if cosa < Z
      cosa = -cosa
      sina = -sina
      sinb = -sinb
    end
  end

  sf(y) = -cosa * y[1] - sina * y[2]
  tf(y) = -sina * cosb * y[1] + cosa * cosb * y[2] + sinb * y[3]

  sx = sf(x) * nu
  tx = tf(x) * nu
  su = sf(u) * nx
  tu = tf(u) * nx

  # sin2c = -(su * tu + sx * tx)
  # cos2c = (su^2 + sx^2 - tu^2 - tx^2) / 2

  sin2c = -(tx * tu - sx * su)
  cos2c =   tx * su + tu * sx

  if cos2c != Z
    na = hypot(sin2c, cos2c)
    sin2c, cos2c = sin2c / na, cos2c / na
    cosc = sqrt((cos2c + E) / 2)
    sinc = sin2c / cosc / 2
  elseif sin2c != Z
    cosc = Z
    sinc = E
  end

  @inline nonegz(x) = x == Z ? Z : x

  nonegz.([ cosa*cosc - sina*cosb*sinc   sina*cosc + cosa*cosb*sinc   sinb*sinc
           -cosa*sinc - sina*cosb*cosc  -sina*sinc + cosa*cosb*cosc   sinb*cosc
            sina*sinb                   -cosa*sinb                    cosb    ])
end

"""
Transform real matrix to matrix B th some zeros by orthogonal tranformation Q.
A = Q * B * Q'
"""
function orthtoband!{T<:Real}(A::AbstractMatrix{T})
  n = size(A,1)
  n == size(A,2) || error("need square matrix")

  for i = 1:n÷2-1
    for j = n:-1:2i+2
      x = A[j-2:j,i]
      u = vec(A[i,j-2:j])
      R = rot3matrix(x, u)
      A[j-2:j,:] = R * A[j-2:j,:]
      A[:,j-2:j] = A[:,j-2:j] * R'
      display(A)
    end
  end
  A
end

""" Similarity transformation to tridiagonal.
The transformation is not unitary
Use pivoting (row -and column swapping) in order to use the least condition number
for the non-unitary tranformation.
Then perform a unitary transformation, which makes a[4:end,1] and a[1,4:end] zero
and leaves a[2:3,1] and a[1,2:3]. If dot(a[2:3,1], a[1,2:3]) != 0,
a linear 2x2-transformation can be applied. The condition number of this transformation
is (1 + sin(ϕ)) / cos(ϕ), where cos(ϕ) is the angle between 2-vectors.
The case of one of the vectors being zero can be handled using a 2x2 unitry matrix.
"""
function tridiag!{T<:Real}(A::AbstractMatrix{T})
  n = size(A, 1)
  n == size( A, 2) || error("need square matrix")
  Z = zero(T)
  Q = eye(A)
  p = [i for i = 1:n]

  pivstart = 1
  # pivot_tri!(A, Q, p, pivstart) # re-ordering A, Q and p

  for i = 1:n-2
    rp0 = i:n
    rp1 = i+1:n
    uv = [A[rp1, i] A[i, rp1]]
    QR = qrfact(uv, Val{true}) # QR for nx2 matrix
    R = QR[:R]
    ta = R[2,2] / R[1,2] # actually tan(ϕ)
    println("cos(ϕ) = $(sqrt(1/(ta^2+1)))")

    # Apply uniform transformation on A and Q
    A[rp1,:] = QR[:Q]' * A[rp1,:]
    Q[rp1,:] = QR[:Q]' * Q[rp1,:]
    A[:,rp1] = A[:,rp1] * QR[:Q]

    # Apply non-uniform transformation with [1 ta; 0 1] or [1 0; ta 1]
    if QR[:p][1] == 1
      A[rp0,i+2] -= A[rp0,i+1] * ta
      A[i+1,rp0] += A[i+2,rp0] * ta
      Q[i+1,:]   += Q[i+2,:]   * ta
    else
      A[rp0,i+1] += A[rp0,i+2] * ta
      A[i+2,rp0] -= A[i+1,rp0] * ta
      Q[i+2,:]   -= Q[i+1,:]   * ta
    end
    # replace rounding errors by correct zeros
    A[i+2:n,i] = Z
    A[i,i+2:n] = Z
    pivstart = i + 2
  end
  A, Q, p
end

"""
Re-order square Matrix A by the abs(cos(ϕ)).
Use only part A[start:end,start:end] for permutations.
Swap rows and columns of A.
Swap rows of Q and p.
"""
function pivot_tri!{T<:Real}(A::AbstractMatrix{T}, Q::AbstractMatrix{T}, p::Vector{Int}, start::Int=1)
  n = size(A, 1)

  u = zeros(T, n)
  v = zeros(T, n)
  uv = zeros(T, n)
  rsn = start:n
  for i = rsn
    for j = rsn
      if i != j
       aij = A[i,j]
       aji = A[j,i]
       u[i] += aji^2
       v[i] += aij^2
       uv[i] += aij * aji
     end
    end
  end
  c = [abs(uv[i])/sqrt(u[i])/sqrt(v[i]) for i = rsn]
  # list of all cos values
  p1 = sortperm(c, lt = (>)) + (start - 1)
  println("cos: $c")
  println("p1: $p1")

  if any( i-> p1[i-start+1] != i, rsn)
    A[rsn,:] = A[p1,:]
    A[:,rsn] = A[:,p1]
    Q[rsn,:] = Q[p1,:]
    p[rsn] = p[p1]
  end
  nothing
end

"""
Transform real matrix A in a way, the vectors A[2:n,1] and A[1,2:n] are co-linear
Use orthogonal Givens rotations and reflections and low-condition Transvections as
transformation matrices.
"""
function precondition!(A::AbstractMatrix)
  n = size(A, 1)
  n == size(A, 2) || error("need square matrix")
  tr = Transformation{eltype(A)}(())
  for i = 1:n-1
    for j = i+1:n
      rot = optgivens(A, i, i, j)
      transform!(A, rot)
      tr = rot * tr
      println("$i $j cosi after rot: $(cosi(A, i))")
    end
  end
end

"""
extract rows i and j and columns i and j of matrix A.
swap indices i <-> 1 and j <-> 2
"""
function extract(A::AbstractMatrix, s::Int, i::Int, j::Int)
  n = size(A, 1)
  n == size(A, 2) || error("need square matrix")
  1 <= s <= i < j <= n || error("1 <= s <= i < j <= n required")
  
  ii = i - s + 1
  jj = j - s + 1

  @inline function swap(a)
    a[1], a[ii] = a[ii], a[1]
    a[2], a[jj] = a[jj], a[2]
  end

  u = vec(A[s:n,i])
  v = vec(A[s:n,j])
  w = vec(A[i,s:n])
  x = vec(A[j,s:n])
  swap(u)
  swap(v)
  swap(w)
  swap(x)
  u, v, w ,x
end

function cosi(A, i)
  vr = A[i,i+1:end]
  vc = A[i+1:end,i]
  dot(vc, vr) / norm(vc) / norm(vr)
end

"""
Find a givens rotation  or reflection which transforms 4 vectors of Matrix
to obtain maximal co-linearity (cos(ϕ))
"""
function optgivens{T<:Real}(A::Matrix{T}, s::Int, i::Int, j::Int)
  n = size(A, 1)
  u, v, w, x = extract(A, s, i, j)
  c1, s1, v1 = optcos(u, v, w, x, false)
  println("cos-rot $v1")
  Rotation(i, j, c1, s1)
end

"""
find optimum of abs(cos(ϕ))
Evaluate at m equidistant angles i.
Look for maximum of integral of interpolation square polynome over i-1 to i+1
return maximum spot of polynome (which is in i-1, i+1) in terms of cos and sin.
"""
function optcos{T<:Real}(u::Vector{T}, v::Vector{T}, w::Vector{T}, x::Vector{T}, reflection::Bool = false)
  const m = 32
  y0 = zero(T)
  c0, s0 = one(T), zero(T)
  i0 = 0
  f = zeros(T, m+2)
  for i = 0:m-1
    cs = exp(im * pi * i / m)
    c, s = real(cs), imag(cs)
    f[i+2] = calccos(u, v, w, x, c, s, reflection)
  end
  f[1], f[m+2] = f[m+1], f[2]

  for i = 0:m-1
    y = abs(f[i+2])
    if y > y0
      y0 = y; i0 = i;
    end
  end
  sig = -sign(f[i0+2])
  fcos, dcos = cosfunctions([u[1] v[1]; u[2] v[2]], u[3:end], v[3:end], w[3:end], x[3:end])
  r = optimize(x->sig*fcos(x), (i0-1) * pi / m, (i0+1)*pi/m)
  x0 = Optim.minimizer(r)
  cs = exp(im * x0)
  c0, s0 = real(cs), imag(cs)
  y1 = fcos(x0)
  println("opt: x0 = $x0 y0 = $y0 y1 = $y1")
  c0, s0, y1
end


function cosfunctions{T<:Number}(a::Matrix{T}, u, v, w, x)
  
  uu = norm(u)^2
  vv = norm(v)^2
  ww = norm(w)^2
  xx = norm(x)^2
  uv = dot(u, v)
  uw = dot(u, w)
  ux = dot(u, x)
  vw = dot(v, w)
  vx = dot(v, x)
  wx = dot(w, x)

  a21p12 = (a[2,1] + a[1,2]) / 2
  a21m12 = (a[2,1] - a[1,2]) / 2
  a11m22 = (a[1,1] - a[2,2]) / 2
  uwmvx = (uw - vx) / 2
  uwpvx = (uw + vx) / 2
  uxpvw = (ux + vw) / 2
  uumvv = (uu - vv) / 2
  uupvv = (uu + vv) / 2
  wwmxx = (ww - xx) / 2
  wwpxx = (ww + xx) / 2

  c2s2(c::T, s::T) =  c^2 - s^2, 2c*s

  f1a(c2, s2) = (c2 * a21p12 + s2 * a11m22)
  fuwnew(f1a, c2, s2) = f1a ^ 2 - a21m12 ^ 2 + c2 * uwmvx - s2 * uxpvw + uwpvx
  fuunew(f1a, c2, s2) = ( f1a + a21m12 ) ^ 2 + c2 * uumvv - s2 * uv + uupvv
  fwwnew(f1a, c2, s2) = ( f1a - a21m12 ) ^ 2 + c2 * wwmxx - s2 * wx + wwpxx

  function fcosc2s2(c2, s2)
    z = f1a(c2, s2)
    fuwnew(z, c2, s2) / sqrt(fuunew(z, c2, s2) * fwwnew(z, c2, s2))
  end

  fcoscs(c, s) = fcosc2s2(c2s2(c, s)...)
  fcosphi(phi) = fcoscs(cos(phi), sin(phi))

  function dfcosphi(phi)
    c, s = cos(phi), sin(phi)
    dot(dfcoscs(c, s), [-s; c])
  end
  
  function dfcoscs(c, s)
    z = 2 * dfcosc2s2(c2s2(c, s)...)
    [dot(z, [c; s]); dot(z, [-s c])]
  end

  function dfcosc2s2(c2, s2)
    z = f1a(c2, s2)
    dfuw = dfuwnew( z, c2, s2)
    dfuu = dfuunew( z, c2, s2)
    dfww = dfwwnew( z, c2, s2)
    zuw = fuwnew(z, c2, s2)
    zuu = fuunew(z, c2, s2)
    zww = fwwnew(z, c2, s2)
    sq = sqrt(zuu * zww)
    dfuw / sq - zuw / sq / zuu / 2 * dfuu - zuw / sq / zww / 2 * dfww
  end

  dfuwnew(z, c2, s2) = [2z * a21p12 + uwmvx; 2z * a11m22 - uxpvw]
  dfuunew(z, c2, s2) = [2(z + a21m12) * a21p12 + uumvv; 2(z + a21m12) * a11m22 - uv]
  dfwwnew(z, c2, s2) = [2(z - a21m12) * a21p12 + wwmxx; 2(z - a21m12) * a11m22 - wx]

  fcosphi, dfcosphi
end

"""
Perform rotation or reflection on column vectors u, v and row vectors w, x.
Return cos of the rotated vectors u and w.
"""
function calccos{T<:Real}(u::Vector{T}, v::Vector{T}, w::Vector{T}, x::Vector{T}, c::T, s::T, reflection::Bool = false)

  cc, ss = reflection ? (-c, s) : (c, -s)
  # matrix [c ss; s cc] represents rotation or reflection

  ww = w * c + ss * x
  uu = u * c + ss * v
  uu[2] = uu[1] * s + uu[2] * cc
  ww[2] = ww[1] * s + ww[2] * cc

  k = 2
  nu = norm(u[k:end])
  nw = norm(w[k:end])
  nuu = norm(uu[k:end])
  nww = norm(ww[k:end])
  uw = dot(uu[k:end], ww[k:end])
  res = uw / nuu / nww # , uw / nu / nw

  # println("\n[$c $ss; $s $cc]")
  # println("u  = $u\nw  = $w")
  # println("uu = $uu\nww = $ww")
  # println("cos = $res")
  res
end

end # module



