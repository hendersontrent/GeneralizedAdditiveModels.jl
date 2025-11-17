"""
    GAMFormula(y, covariates)
Holds a structural representation of the GAM formulation.

Usage:
```julia-repl
GAMFormula(y, covariates)
```
Arguments:
- `y` : `Symbol` denoting the response variable.
- `covariates` : `DataFrame` containing the covariate smooth information.
"""
struct GAMFormula
    y::Symbol
    covariates::DataFrame
end

"""
    ParseFormula(formula)
Parse String formulation of GAM model into constituent parts for usage in modelling.

Usage:
```julia-repl
ParseFormula(formula)
```
Arguments:
- `formula` : `String` containing the expression of the model. Continuous covariates are wrapped in s() like `mgcv` in R, where `s()` has 3 parts: name of column, `k`` (integer denoting number of knots), and `degree` (polynomial degree of the spline). An example expression is `"Y ~ s(MPG, k=5, degree=3) + WHT + s(TRL, k=5, degree=2)"`
"""

function ParseFormula(formula::String)

    #--------------------------
    # Extract each variable and 
    # whitespace and plus signs
    #--------------------------

    vars = String.(collect(m.match for m in eachmatch(r"\s*\+?\s*(s\((\w+),\s*k=(\d+),\s*degree=(\d+)\)|(\w+))", formula)))
    lhs = vars[1] # Response variable
    rhs = filter!(e -> e≠lhs, vars) # Covariates

    for c in 1:size(rhs)[1]
        rhs[c] = replace(rhs[c], r"\s" => "")
        rhs[c] = replace(rhs[c], r"\+" => "")
    end

    #------------------------------
    # Extract covariate information
    # for smooths
    #------------------------------

    # Create an empty DataFrame with appropriate column names

    df = DataFrame(variable = Symbol[], k = Int[], degree = Int[], smooth = Bool[])

    # Extracting information from each right-hand side component and add rows to DataFrame

    for component in rhs
        if component[end] == ')'  # s() wrapping present
            component = replace(component, r"s[()]" => "")
            component = replace(component, r"[()]" => "")
            component = replace(component, r"\s+" => "")
            symbol_name = Symbol(split(component, ',')[1])
            k = parse(Int, split(split(component, ',')[2], '=')[2])
            degree = parse(Int, split(split(component, ',')[3], '=')[2])
            smooth = true
        else  # no s() wrapping
            symbol_name = Symbol(component)
            k = 10
            degree = 10  # Set default values when s() wrapping is absent
            smooth = false
        end
        push!(df, (symbol_name, k, degree, smooth))
    end

    outs = GAMFormula(Symbol(lhs), df)
    return outs
end
