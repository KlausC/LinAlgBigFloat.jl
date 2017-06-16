
"""
One step of Francis transformation Transform hessenberg matrix
H is in upper Hessenberg form.
shifts is a vector of shift values.
With each shift value, its conjugate complex value is implicitly used as shift
All shift values should be distinct.
"""
function transform_Hess!{T<:Union{Real,Complex}, S<:Union{Real,Complex}}(A::AbstractMatrix{T}, ilo::Integer, ihi::Integer, Q::AbstractM, s::Vector{S}, maxchase::Integer, iwindow::Integer)

  !isa(Q, AbstractMatrix) || size(A, 2) == size(Q, 2) || error("A and Q have incompatible sizes")

  n = size(A, 1)
  z = zero(T)
  m = counteigs(s)
  realcase = T <: Real
  minn = ifelse(realcase, 2, 1)
  #println("def_sub ilo = $ilo ihi = $ihi")
  
  # supress further calculations if A is already quasi-diagonal
  isize = ihi - ilo + 1
  if isize <= 1 || isize == minn && discriminant(A, ilo, ihi) < z
    m = 0
    maxchase = 0
  end
  
  # Any shifts provided
  if m > 0

    # First step: shift bulges blocking the undisturbed insertion
    if isize > 4
      i0, m0 = lastbulge(A, ilo, ihi, m + 1)
      if i0 > 0
        chase!(A, ilo, ihi, Q, m + 2 - i0, iwindow)
      end
    end

    # Second step: compute  prod(A-s(k))e1
    pe = prode1(A, s, ilo)

    # Third step: set in new upper left bulge
    set_in!(A, ilo, Q, pe)
  end

  # Forth step: chase all remaining bulks, oldest first
  if maxchase > 0
    chase!(A, ilo, ihi, Q, maxchase, iwindow)
  end
  ilo, ihi, Q
end

function transform_Hess!{T<:Real, S<:Union{Real,Complex}}(A::AbstractMatrix{T}, Q::AbstractM, s::Vector{S}, maxchase::Integer, iwindow::Integer = 0)

  transform_Hess!(A, 1, size(A, 1), Q, s, maxchase, iwindow)
end

function transform_Hess!{T<:Real, S<:Union{Real,Complex}}(A::AbstractMatrix{T}, s::Vector{S}, maxchase::Integer, iwindow::Integer = 0)

  #Q = LinAlg.Rotation(LinAlg.Givens{T}[])
  Q = eye(T, size(A, 2))
  transform_Hess!(A, Q, s, maxchase, iwindow)
end

# calculate product over k of (A - s[k]*I) * e1
function prode1{T<:Real}(A::AbstractMatrix{T}, s::Vector, ilo::Integer)
  n = size(A, 1)
  pe = zeros(T, n)
  pe[ilo] = one(T)
  k = j = 1
  while k <= length(s)
    sr = real(s[k])
    si = imag(s[k])
    if si == 0
      pe = A * pe - sr * pe
      j += 1
    else
      pe = A * ( A * pe - 2sr * pe) + ( hypot(sr, si) ^ 2 ) * pe
      if k < length(s) && imag(s[k+1]) == -si
        k += 1
      end
      j += 2
    end
    k += 1
  end
  pe
end

function prode1{T<:Complex}(A::AbstractMatrix{T}, s::Vector, ilo::Integer)
  n = size(A, 1)
  pe = zeros(T, n)
  pe[ilo] = one(T)
  k = j = 1
  for sk in s
    pe = A * pe - sk * pe
  end
  pe
end

function set_in!(A::AbstractMatrix, ilo::Integer, Q::AbstractM, pe::Vector)
  m2 = findlast(x -> x != 0, pe)
  for k = m2:-1:ilo+1
    G, r = givens(pe, k-1, k)
    pe[k-1] = r
    pe[k] = 0
    A_mul_B!(G, A)
    A_mul_Bc!(A, G)
    A_mul_Bc!(Q, G)
  end
end

"""
Count number of eigenvalues.
Each non-real eigenvalue is counted twice, if not paired with its conjugate.
"""
counteigs{T<:Real}(s::Vector{T}) = length(s)
function counteigs{T}(s::Vector{Complex{T}})
  z = zero(T)
  m = length(s)
  count = 0
  k = 1
  while k <= m
    sk = s[k]
    si = imag(sk)
    if si == z
      count += 1
    else
      if k < m && imag(s[k+1]) == -si
        k += 1
      end
      count += 2
    end
    k += 1
  end
  count
end

"""
Chase all bulges towards right lower corner
"""
function chase!(A, ilo::Integer, ihi::Integer, Q, maxchase::Integer, iwindow::Integer)
  n1, n2 = size(A)
  n2 = min(ihi, n2)
  n1 = min(ihi, n1)
  m = i0 = n2 - 1 # column to start with

  while i0 >= ilo && m > 0
    i0, m = lastbulge(A, ilo, ihi, i0 - 1)

    if m > 0
      i1 = i0 + m > ihi -iwindow ? n2 - 2 : min(i0 + maxchase - 1, n2 - 2)
      for  i = i0:min(i0 + maxchase - 1, n2 - 2)
        for k = min(i+m+1,n1):-1:i+2
          if A[k,i] != 0
            G, r = givens(A, k-1, k, i)
            A_mul_B!(G, A)
            A_mul_Bc!(A, G)
            A_mul_Bc!(Q, G)
            A[k-1,i] = r
            A[k,i] = 0
          end
        end
      end
    end
  end
  A, Q
end

"""
Find last bulge start column and size. Start search in column i.
"""
function lastbulge{T<:Number}(A::AbstractMatrix{T}, ilo::Integer, ihi::Integer, i::Integer)
  n, m = size(A)
  n == m || error("need square matrix")
  #i <= n - 1 || error("column number must be less than n")

  maxpos = 2
  while i >= ilo
    m = findlast(A[i+2:ihi,i])
    if m == 0 
      if maxpos > 2
        break
      end
    else
      maxpos = max(maxpos, m + i + 1)
    end
    i -= 1
  end
  m = maxpos - i - 2
  m == 0 ? 0 : i + 1, m
end

