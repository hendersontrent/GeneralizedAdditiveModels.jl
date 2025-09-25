include("../src/GAM.jl")
using .GAM

using Test
using RDatasets, Plots

#-------------------- Set up data -----------------

df = dataset("datasets", "trees");

#-------------------- Run tests -----------------

@testset "Plotting" begin
    mod = gam("Volume ~ s(Girth, k=10, degree=3) + s(Height, k=10, degree=3)", df)

    p = plot(mod, var=1)
    @test p isa Plots.Plot

    p2 = plot(mod)
    @test p2 isa Plots.Plot

    p = plotgam(mod)
    @test p isa Plots.Plot

    p = partialdependenceplot(mod, 1)
    @test p isa Plots.Plot
end

@testset "Gamma" begin
    # Gamma version

    mod2 = gam("Volume ~ s(Girth, k=10, degree=3) + s(Height, k=10, degree=3)", df; Family = "Gamma", Link = "Log")

    p1 = plotgam(mod2)
    @test p1 isa Plots.Plot
end