module GeneralizedAdditiveModels

using Distributions, GLM, Optim, BSplines, LinearAlgebra, DataFrames, Plots, Optim
using StatsModels
using StatsModels: FormulaTerm, @formula, Term, ConstantTerm, InterceptTerm, AbstractTerm

include("Links-Dists.jl")
include("GAMData.jl")
include("BuildBasis.jl")
include("diffm.jl")
include("DifferenceMatrix.jl")
include("dcat.jl")
include("ModelDiagnostics.jl")
include("FitOLS.jl")
include("FitWPS.jl")
include("alpha.jl")
include("PIRLS.jl")
include("Predictions.jl")
include("Plots.jl")
include("SmoothTerm.jl")
include("GAMFormula.jl")
include("FitGAM.jl")

export Links
export Dists
export Dist_Map
export Link_Map
export GAMData
export PartialDependencePlot
export plotGAM
export gam
export @formula, s, SmoothTerm, ParseFormula

end
