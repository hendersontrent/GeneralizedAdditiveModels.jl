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
            k = degree = 0  # Set default values when s() wrapping is absent
            smooth = false
        end
        push!(df, (symbol_name, k, degree, smooth))
    end

    outs = GAMFormula(Symbol(lhs), df)
    return outs
end

"""
    ParseFormula(formula::FormulaTerm)
Parse StatsModels FormulaTerm into GAMFormula structure.

Usage:
```julia-repl
f = @formula(y ~ s(x1, k=10, degree=3) + x2)
ParseFormula(f)
```
Arguments:
- `formula` : `FormulaTerm` from StatsModels.jl @formula macro
"""
function ParseFormula(formula::FormulaTerm)
    # Extract response variable
    y = formula.lhs.sym

    # Process right-hand side terms
    rhs = formula.rhs

    # Create DataFrame to hold covariate information
    df = DataFrame(variable = Symbol[], k = Int[], degree = Int[], smooth = Bool[])

    # Extract terms from the right-hand side
    terms = extract_terms(rhs)

    for term in terms
        if isa(term, SmoothTerm)
            # Smooth term: extract k, degree, and variable name
            push!(df, (term.term.sym, term.k, term.degree, true))
        elseif isa(term, Term)
            # Linear term: add with default k=0, degree=0, smooth=false
            push!(df, (term.sym, 0, 0, false))
        elseif isa(term, InterceptTerm) || isa(term, ConstantTerm)
            # Intercept term - we handle this separately, skip for now
            continue
        else
            @warn "Unsupported term type in formula: $(typeof(term)). Skipping."
        end
    end

    return GAMFormula(y, df)
end

"""
    extract_terms(rhs)
Recursively extract individual terms from the right-hand side of a formula.

Handles different StatsModels term types including tuples, individual terms, and smooth terms.
"""
function extract_terms(rhs)
    terms = []

    if isa(rhs, Tuple)
        # Multiple terms: recursively extract from each
        for term in rhs
            append!(terms, extract_terms(term))
        end
    elseif isa(rhs, SmoothTerm) || isa(rhs, Term)
        # Single term: add directly
        push!(terms, rhs)
    elseif isa(rhs, InterceptTerm) || isa(rhs, ConstantTerm)
        # Intercept/constant: add directly
        push!(terms, rhs)
    elseif isa(rhs, StatsModels.FunctionTerm)
        # Handle FunctionTerm (from @formula macro)
        # Check if it's our s() function
        if rhs.exorig.head == :call && rhs.exorig.args[1] == :s
            # Extract arguments from the function call
            # rhs.exorig.args[2] is the variable name
            # rhs.exorig.args[3] is k (if present)
            # rhs.exorig.args[4] is degree (if present)
            var_sym = rhs.exorig.args[2]
            k = length(rhs.exorig.args) >= 3 ? rhs.exorig.args[3] : 10
            degree = length(rhs.exorig.args) >= 4 ? rhs.exorig.args[4] : 3

            # Create a SmoothTerm
            push!(terms, SmoothTerm(Term(var_sym), k, degree))
        else
            @warn "Unsupported function in formula: $(rhs.exorig.args[1])"
        end
    else
        # Try to handle other StatsModels types
        # For composite terms, try to extract nested terms
        try
            # If it has a .terms field (like CategoricalTerm, InteractionTerm, etc.)
            if hasfield(typeof(rhs), :terms)
                append!(terms, extract_terms(rhs.terms))
            else
                @warn "Unable to extract terms from type $(typeof(rhs))"
            end
        catch e
            @warn "Error extracting terms: $e"
        end
    end

    return terms
end
