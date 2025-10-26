"""
    BuildPredictionMatrix(x, Basis, ColMeans)
Build prediction matrix.

Usage:
```julia-repl
BuildPredictionMatrix(x, Basis, ColMeans)
```
Arguments:
- `x` : `Vector` of data for a variable.
- `Basis` : `AbstractMatrix` containing basis matrix.
- `ColMeans` : `Matrix` containing column means.
"""
function BuildPredictionMatrix(x::AbstractArray, Basis::BSplineBasis, ColMeans::AbstractArray)
    basisMatrix = DropCol(BuildBasisMatrix(Basis, x), length(Basis.breakpoints))
    return CenterBasisMatrix(basisMatrix, ColMeans)
end

"""
    PredictPartial(mod, ix)
Predict partial values.

Usage:
```julia-repl
PredictPartial(mod, ix)
```
Arguments:
- `mod` : `GAMData` containing the model.
- `ix` : `Int` denoting the variable to plot.
"""
function PredictPartial(mod, ix)
    bi = mod.Basis[ix]
    if bi === :linear
        μ  = mod.ColMeans[ix][1]         # stored as 1×1
        Xi = reshape(mod.x[ix] .- μ, :, 1)
        β  = mod.Coef[mod.CoefIndex[ix]] # scalar
        return Xi * β
    else
        predMatrix = BuildPredictionMatrix(mod.x[ix], bi, mod.ColMeans[ix])
        predBeta = mod.Coef[mod.CoefIndex[ix]]
        return predMatrix * predBeta
    end
end
