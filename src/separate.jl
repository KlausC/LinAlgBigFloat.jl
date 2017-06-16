
"""
Separately process parts, if Hessenberg matrix A decays.
"""
function separate!(A::AbstractMatrix, jlo::Integer, jhi::Integer, Q::AbstractM, it!::Function)
  ihi = jhi
  while ihi >= jlo
    ilo = separation_point(A, jlo, ihi)
    it!(A, ilo, ihi, Q)
    ihi = ilo - 1
  end
end

"""
Find last zero in subdiagonal
"""
function separation_point(A, jlo, ihi)
  ilo = ihi
  while ilo > jlo && A[ilo,ilo-1] != 0
    ilo -= 1
  end
  ilo
end

