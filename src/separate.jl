
"""
  `separate(A::AbstractMatrix, jlo::Integer, jhi::Integer, Q::AbstractM, it!::Function)`

  Process parts separately, if Hessenberg matrix A has zero in subdiagonal.
  ilo and ihi are the index bounds of the matrix (typically (1, size(A, 1) ).
  The iterator function is called like `it!(A, ilo, ihi, Q)` where `ilo, ihi` are set 
  reflecting the bounds of the detected submatrices. If ilo > 1, A[ilo-1,ilo] == 0.
  Processing order is from end to begin of A.
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

