using LinAlgBigFloat
using Test

@time @testset "Utils" begin include("utils_test.jl") end
# @time @testset "Schur BigFloat" begin include("schurbf_test.jl") end
@time @testset "VectorSpaces" begin include("spaces_test.jl") end

0
