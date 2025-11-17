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
    Basis::AbstractArray{BSplineBasis}
    Family::Dict
    Link::Dict
    Coef::AbstractArray
    ColMeans::AbstractArray
    CoefIndex::AbstractArray
    Fitted::AbstractArray
    Diagnostics::Dict
    #TODO Why is separate constructor needed? spk
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

"Calculate residuals from fitted data"
residuals(data)=abs.(data.y - data.Fitted)

"Return fitted data to match R function"
fitted(data)=data.Fitted

"Print summary information of GAMData"
function summary(data::GAMData, out::IO=stdout)
    println("Summary:")
    println("Coef $(data.Coef)")
    println(out, "EDF: $(data.Diagnostics[:EDF])")
    println(out, "GCV: $(data.Diagnostics[:GCV])")
    println(out, "RSS: $(data.Diagnostics[:RSS])")
end