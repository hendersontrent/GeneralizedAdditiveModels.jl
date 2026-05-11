"""
    gam(ModelFormula, Data; Family, Link, Optimizer, maxIter, tol)
Fit generalised additive model.

Usage:
```julia-repl
# Using string formula (original syntax)
gam("Y ~ s(MPG, k=5, degree=3) + WHT", Data)

# Using @formula macro (new StatsModels.jl syntax)
using StatsModels
gam(@formula(Y ~ s(MPG, 5, 3) + WHT), Data)
```
Arguments:
- `ModelFormula` : Either a `String` or `FormulaTerm` (@formula macro) containing the expression of the model. Continuous covariates are wrapped in s() like `mgcv` in R. For strings, use `s(var, k=N, degree=D)` syntax. For @formula macro, use `s(var, N, D)` with positional arguments. An example string expression is `"Y ~ s(MPG, k=5, degree=3) + WHT"` and an example @formula is `@formula(Y ~ s(MPG, 5, 3) + WHT)`
- `Data` : `DataFrame` containing the covariates and response variable to use.
- `Family` : `String` specifying Likelihood distribution. Should be one of "Normal", "Poisson", "Gamma", or "Bernoulli". Defaults to "Normal"
- `Link` : `String` specifying link function distribution. Should be one of "Identity", "Log", or "Logit". Defaults to "Identity"
- `Optimizer` : Algorithm to use for optimisation. Defaults to `NelderMead()`.
- `maxIter` : Maximum number of iterations for algorithm. Defaults to 25.
- `tol` : Tolerance for solver. Defaults to 1e-6.
"""
function gam(ModelFormula::String, Data::DataFrame; Family="Normal", Link="Identity", Optimizer = NelderMead(), maxIter = 25, tol = 1e-6)

    family_name = Dist_Map[Family]
    family_name = Dists[family_name]
    link_name = Link_Map[Link]
    link_name = Links[link_name]

    # Parse formula and generate design matrix and response variable vector
    GAMForm = ParseFormula(ModelFormula)
    y = Data[!, GAMForm.y]
    
    # Validate response for Bernoulli family
    if Family == "Bernoulli"
        @assert all(y .∈ Ref([0, 1])) "Response must be binary (0 or 1) for Bernoulli family"
    end
    
    # Collect covariate columns and term meta
    xdf = Data[!, GAMForm.covariates.variable]
    x = [xdf[!, col] for col in names(xdf)]
    BasisArgs  = [(GAMForm.covariates.k[i], GAMForm.covariates.degree[i]) for i in 1:nrow(GAMForm.covariates)]
    smoothmask = collect(GAMForm.covariates.smooth)

    # Per-term basis: smooth -> BSplineBasis, linear -> :linear
    Basis = Vector{Any}(undef, length(x))
    for i in eachindex(x)
        if smoothmask[i]
            k, degree = BasisArgs[i]
            Basis[i] = BuildUniformBasis(x[i], k, degree)
        else
            Basis[i] = :linear
        end
    end

    # Fit PIRLS procedure
    gam = OptimPIRLS(y, x, Basis, family_name, link_name; Optimizer, maxIter, tol)
    return gam
end

"""
    gam(ModelFormula::FormulaTerm, Data; Family, Link, Optimizer, maxIter, tol)
Fit generalised additive model using StatsModels.jl @formula macro.

Usage:
```julia-repl
using StatsModels
f = @formula(Y ~ s(MPG, 5, 3) + WHT)
gam(f, Data; Family="Normal", Link="Identity")
```
Arguments:
- `ModelFormula` : `FormulaTerm` from StatsModels @formula macro
- `Data` : `DataFrame` containing the covariates and response variable to use.
- `Family` : `String` specifying Likelihood distribution. Should be one of "Normal", "Poisson", "Gamma", or "Bernoulli". Defaults to "Normal"
- `Link` : `String` specifying link function distribution. Should be one of "Identity", "Log", or "Logit". Defaults to "Identity"
- `Optimizer` : Algorithm to use for optimisation. Defaults to `NelderMead()`.
- `maxIter` : Maximum number of iterations for algorithm. Defaults to 25.
- `tol` : Tolerance for solver. Defaults to 1e-6.
"""
function gam(ModelFormula::FormulaTerm, Data::DataFrame; Family="Normal", Link="Identity", Optimizer = NelderMead(), maxIter = 25, tol = 1e-6)
    # Delegate to the String version by parsing the FormulaTerm first
    # This allows us to reuse all the existing logic
    GAMForm = ParseFormula(ModelFormula)

    family_name = Dist_Map[Family]
    family_name = Dists[family_name]
    link_name = Link_Map[Link]
    link_name = Links[link_name]

    # Extract response and covariates
    y = Data[!, GAMForm.y]

    # Validate response for Bernoulli family
    if Family == "Bernoulli"
        @assert all(y .∈ Ref([0, 1])) "Response must be binary (0 or 1) for Bernoulli family"
    end

    x = Data[!, GAMForm.covariates.variable]
    BasisArgs = [(GAMForm.covariates.k[i], GAMForm.covariates.degree[i]) for i in 1:nrow(GAMForm.covariates)]
    x = [x[!, col] for col in names(x)]

    # Build basis
    Basis = map((xi, argi) -> BuildUniformBasis(xi, argi[1], argi[2]), x, BasisArgs)

    # Fit PIRLS procedure
    gam = OptimPIRLS(y, x, Basis, family_name, link_name; Optimizer, maxIter, tol)
    return gam
end
