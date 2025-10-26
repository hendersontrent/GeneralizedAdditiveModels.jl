"""
    BuildUniformBasis(x, n_knots, order)
Builds uniform basis matrix.

Usage:
```julia-repl
BuildUniformBasis(x, n_knots, order)
```
Arguments:
- `x` : `Vector` of input data.
- `n_knots` : `Int64` specifying the number of knots for the B-spline.
- `order` : `Int64` specifying the polynomial order of the spline.
"""
function BuildUniformBasis(x::Vector, n_knots::Int64, order::Int64)
    KnotsList = range(minimum(x), maximum(x), length = n_knots);
    Basis = BSplineBasis(order, KnotsList);
    return Basis
end

"""
    BuildQuantileBasis(x, n_knots, order)
Builds quantile basis matrix.

Usage:
```julia-repl
BuildQuantileBasis(x, n_knots, order)
```
Arguments:
- `x` : `Vector` of input data.
- `n_knots` : `Int64` specifying the number of knots for the B-spline.
- `order` : `Int64` specifying the polynomial order of the spline.
"""
function BuildQuantileBasis(x::Vector, n_knots::Int64, order::Int64)
    KnotsList = quantile(x, range(0, 1; length=n_knots));
    Basis = BSplineBasis(order, KnotsList);
    return Basis
end

"""
    BuildBasisMatrix(Basis, x)
Builds full basis matrix.

Usage:
```julia-repl
BuildBasisMatrix(Basis, x)
```
Arguments:
- `Basis` : `BSplineBasis` containing the basis function.
- `x` : `AbstractVector` of input data.
"""
function BuildBasisMatrix(Basis::BSplineBasis, x::AbstractVector)
    splines = vec(
        mapslices(
            x -> Spline(Basis,x), 
            diagm(ones(length(Basis))),
            dims=1
        )
    );
    X = hcat([s.(x) for s in splines]...)
    return X 
end

"""
    BuildBasisMatrixColMeans(BasisMatrix)
Builds column means for a full basis matrix.

Usage:
```julia-repl
BuildBasisMatrixColMeans(BasisMatrix)
```
Arguments:
- `Basis` : `AbstractArray` containing the basis matrix.
"""
function BuildBasisMatrixColMeans(BasisMatrix::AbstractArray)
    # Calculate means of each column
    return mean(BasisMatrix, dims=1)
end

"""
    CenterBasisMatrix(BasisMatrix, BasisMatrixColMeans)
Centers columns of a basis matrix using column means.

Usage:
```julia-repl
CenterBasisMatrix(BasisMatrix, BasisMatrixColMeans)
```
Arguments:
- `Basis` : `AbstractArray` containing the basis matrix.
- `BasisMatrixColMeans` : `AbstractArray` containing the column means for the basis matrix.
"""
function CenterBasisMatrix(BasisMatrix::AbstractMatrix, BasisMatrixColMeans::AbstractArray)
    return BasisMatrix .- BasisMatrixColMeans
    #return map((c, m) -> (c .- m), gamDataBasisMatrix, gamDataBasisMatrixColMeans)
end

"""
    DropCol(X, ix)
Drop a column.

Usage:
```julia-repl
DropCol(X, ix)
```
Arguments:
- `Basis` : `AbstractMatrix` containing the matrix.
- `ix` : `Int` denoting column to drop.
"""
function DropCol(X::AbstractMatrix, ix::Int)
    cols = [1:size(X,2);]
    deleteat!(cols, ix)
    return X[:, cols]
end

"""
    BuildCoefIndex(BasisMatrix)
Builds coefficient index from a basis matrix.

Usage:
```julia-repl
BuildCoefIndex(BasisMatrix)
```
Arguments:
- `Basis` : `AbstractArray` containing the basis matrix.
"""
function BuildCoefIndex(BasisMatrix::AbstractArray)
    ix = [1:1]
    ix_end = cumsum(vcat([1], size.(BasisMatrix,2)))
    append!(ix, [ix_end[i-1]+1:ix_end[i] for i in 2:length(ix_end)])
    return ix[2:end]
end

"""
    BuildPenaltyMatrix(y, x, sp, Basis)
Builds coefficient index from a basis matrix.

Usage:
```julia-repl
BuildPenaltyMatrix(y, x, sp, Basis)
```
Arguments:
- `y` : `Vector` containing the response variable.
- `x` : `Vector` of input data.
- `sp` : `xxx` xx.
- `Basis` : `AbstractArray` containing the basis matrix.
"""
function BuildPenaltyMatrix(y, x, sp, Basis)
    n = length(y)

    # Prepare per-term blocks
    Xblocks = Vector{AbstractMatrix}(undef, length(x))
    Dblocks = Vector{AbstractMatrix}(undef, length(x))
    ColMeans = Vector{AbstractArray}(undef, length(x))

    for i in eachindex(x)
        bi = Basis[i]
        xi = x[i]
        if bi === :linear
            # centered 1-column linear block; no penalty
            μ = mean(xi)
            Xi = reshape(xi .- μ, :, 1)
            Di = zeros(1, 1)
            Xblocks[i] = Xi
            Dblocks[i] = Di
            ColMeans[i] = reshape([μ], 1, :)
        else
            # smooth block
            nk  = length(bi.breakpoints)
            Xi0 = BuildBasisMatrix(bi, xi)
            Di0 = BuildDifferenceMatrix(bi)
            Xi  = DropCol(Xi0, nk)
            Di  = DropCol(Di0, nk)
            # drop for identifiability
            cm  = BuildBasisMatrixColMeans(Xi)
            Xi  = CenterBasisMatrix(Xi, cm)
            Xblocks[i] = Xi
            Dblocks[i] = Di
            ColMeans[i] = cm
        end
    end

    # Coefficient index per-term by block width
    CoefIndex = BuildCoefIndex(Xblocks)

    # Assembled design with intercept
    X = hcat(repeat([1], n), hcat(Xblocks...))

    # Block-diagonal penalty (sqrt(sp) scaling on smooth blocks; linear blocks are zero already)
    Dscaled = map((p, d) -> sqrt(p) .* d, sp, Dblocks)
    D = dcat(Dscaled)
    D = hcat(repeat([0], size(D, 1)), D) # intercept column

    return X, y, D, ColMeans, CoefIndex
end


"""
    HatMatrix(X, D, W)
Builds a hat matrix.

Usage:
```julia-repl
HatMatrix(X, D, W)
```
Arguments:
- `X` : `Matrix` containing data.
- `D` : `Matrix` containing data.
- `W` : `Vector` of weights.
"""
function HatMatrix(X, D, W)
    return W * (X * inv(X' * W * X + D' * D) ) * X'
end
