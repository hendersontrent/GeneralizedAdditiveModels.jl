#==
    SmoothTerm

Extension of StatsModels.jl for GAM smooth terms.

This module defines custom term types for representing smooth functions in GAM formulas,
allowing syntax like: @formula(y ~ s(x1, k=10, degree=3) + x2)
==#


"""
    SmoothTerm

Represents a smooth spline term in a GAM formula.

# Fields
- `term::Term`: The variable to be smoothed
- `k::Int`: Number of knots for the spline basis (default: 10)
- `degree::Int`: Polynomial degree of the spline (default: 3)
"""
struct SmoothTerm <: AbstractTerm
    term::Term
    k::Int
    degree::Int
end

# Constructor with default values
SmoothTerm(term::Term; k::Int=10, degree::Int=3) = SmoothTerm(term, k, degree)

# Allow creating from a Symbol
SmoothTerm(sym::Symbol; k::Int=10, degree::Int=3) = SmoothTerm(Term(sym), k, degree)

# Pretty printing
Base.show(io::IO, st::SmoothTerm) = print(io, "s($(st.term.sym), k=$(st.k), degree=$(st.degree))")

"""
    s(variable, k=10, degree=3)

Create a smooth spline term for use in GAM formulas.

# Arguments
- `variable`: The variable to be smoothed (Symbol or Term)
- `k`: Number of knots for the spline basis (default: 10)
- `degree`: Polynomial degree of the spline (default: 3)

# Examples
```julia
using GeneralizedAdditiveModels, StatsModels

# Using the @formula macro with smooth terms (positional arguments)
f = @formula(y ~ s(x1, 10, 3) + s(x2, 5, 2) + x3)

# Or define smooth terms before the formula
s1 = s(:x1, 10, 3)
s2 = s(:x2, 5, 2)
# Note: You'll need to use the string formula syntax for pre-defined terms

# Fit a GAM with the formula
model = gam(f, data)
```
"""
# Positional argument versions (for use with @formula macro)
s(term::Term, k::Int=10, degree::Int=3) = SmoothTerm(term, k, degree)
s(sym::Symbol, k::Int=10, degree::Int=3) = SmoothTerm(Term(sym), k, degree)
