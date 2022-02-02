"""
Module providing extensions of linalg to BigFloat and Complex{BigFloat}
"""
module LinAlgBigFloat

using LinearAlgebra

include("util.jl")
#include("mpfrmodify.jl")
include("deflationcrit.jl")
include("refineprecision.jl")
include("separate.jl")
include("transformhess.jl")
#include("hessenbergbf.jl")
include("schurbf.jl")
include("sylvester.jl")
include("spaces.jl")
include("jordan.jl")
include("matrices.jl")
include("randommatrices.jl")

export is_hessenberg

end # module
