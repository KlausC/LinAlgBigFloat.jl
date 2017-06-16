
"""
deflation criterion.
"""
function deflation_criterion{T<:Number}(sub::T, da::T, db::T)
  deflation_criterion1(abs(sub), (abs(da) + abs(db) ))
end

@inline deflation_criterion1{T<:Number}(sub::T, da::T) = sub <= eps(da) / 4

