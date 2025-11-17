module GAM

using Distributions, GLM, Optim, BSplines, LinearAlgebra, DataFrames, Plots, Optim

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
include("FitGAM.jl")
include("GAMFormula.jl")

export Links
export Dists
export Dist_Map
export Link_Map
export GAMData
export PartialDependencePlot
export plotGAM
export gam
export summary
export residuals
export fitted
end
