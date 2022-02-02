
"""
deflation criterion.
"""
function deflation_criterion(sub::T, da::T, db::T) where {T<:Number}
  deflation_criterion1(abs(sub), (abs(da) + abs(db) ))
end

@inline deflation_criterion1(sub::T, da::T) where {T<:Number} = sub <= eps(da) / 4

