"""
    GAMData(x, y, Basis, Family, Link, Coef, ColMeans, CoefIndex, Fitted, Diagnostics)
Holds information relevant to the final fitted GAM model.

Usage:
```julia-repl
GAMData(x, y, Basis, Family, Link, Coef, ColMeans, CoefIndex, Fitted, Diagnostics)
```
Arguments:
- `x` : `AbstractArray` of input data.
- `y` : `AbstractArray` for the response variable.
- `Basis` : `AbstractArray` of the basis Matrix.
- `Family` : `Dict` of the likelihood distribution.
- `Link` : `Dict` of link function.
- `Coef` : `AbstractArray` of coefficients.
- `ColMeans` : `AbstractArray` of column means.
- `CoefIndex` : `AbstractArray` of the coefficient index.
- `Fitted` : `AbstractArray` of fitted values.
- `Diagnostics` : `Dict` of model diagnostic information.
"""
mutable struct GAMData
    y::AbstractArray
    x::AbstractArray
    Basis::AbstractArray
    Family::Dict
    Link::Dict
    Coef::AbstractArray
    ColMeans::AbstractArray
    CoefIndex::AbstractArray
    Fitted::AbstractArray
    Diagnostics::Dict

    function GAMData(
        y::AbstractArray,
        x::AbstractArray,
        Basis::AbstractArray,
        Family::Dict,
        Link::Dict,
        Coef::AbstractArray,
        ColMeans::AbstractArray,
        CoefIndex::AbstractArray,
        Fitted::AbstractArray,
        Diagnostics::Dict
    )
        new(y, x, Basis, Family, Link, Coef, ColMeans, CoefIndex, Fitted, Diagnostics)
    end
end
