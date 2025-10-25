"""
    PIRLS(y, x, sp, Basis, Dist, Link; maxIter, tol)
Fit penalised iterative reweighted least squares algorithm.

Usage:
```julia-repl
PIRLS(y, x, sp, Basis, Dist, Link; maxIter, tol)
```
Arguments:
- `y` : `Vector` containing the response variable.
- `x` : `Array` of input data.
- `sp` : `Float` of the optimised smoothing parameter.
- `Basis` : `AbstractArray` containing the basis matrix.
- `Dist` : Likelihood distribution.
- `Link` : Link function.
- `maxIter` : Maximum number of iterations for algorithm.
- `tol` : Tolerance for solver.
"""
function PIRLS(y, x, sp, Basis, Dist, Link; maxIter = 25, tol = 1e-6)
    
    # Initial Predictions
    n = length(y)

    if Dist[:Name] == "Bernoulli"
        mu = clamp.(y, 1e-6, 1 - 1e-6)
    else
        mu = y
    end

    eta = Link[:Function].(mu)
    
    # Deviance
    logLik = sum(map(x -> logpdf(Dist[:Distribution](x), x), y))
    dev = logLik

    for i in 1:maxIter
        # Compute weights
        a = alpha(y, mu, Dist, Link)
        z = @. Link[:Derivative](mu) * (y - mu) / a + eta
        w = @. a / (Link[:Derivative](mu)^2 * Dist[:V](mu))

        global mod = FitWPS(z, x, sp, Basis, Dist, Link, w)
        eta = mod.Fitted
        mu = Link[:Inverse].(eta)
        if Dist[:Name] == "Bernoulli"
            mu = clamp.(mu, 1e-6, 1-1e-6)
        end
        oldDev = dev
        dev = 2 * (logLik - sum(map((x,y) -> logpdf(Dist[:Distribution](x), y), mu, y)))
        if abs(dev - oldDev) < tol * dev
            break
        end
    end

    #mod.Family = Dist
    mod.Fitted = Link[:Inverse].(mod.Fitted)
    return mod
end

"""
    OptimPIRLS(y, x, Basis, Dist, Link; Optimizer)
Optimise penalised iterative reweighted least squares algorithm.

Usage:
```julia-repl
OptimPIRLS(y, x, Basis, Dist, Link; Optimizer)
```
Arguments:
- `y` : `Vector` containing the response variable.
- `x` : `Array` of input data.
- `Basis` : `AbstractArray` containing the basis matrix.
- `Dist` : Likelihood distribution.
- `Link` : Link function.
- `Optimizer` : Algorithm to use for optimisation. Defaults to `NelderMead()`.
- `maxIter` : Maximum number of iterations for algorithm.
- `tol` : Tolerance for solver.
"""
function OptimPIRLS(y, x, Basis, Dist, Link; Optimizer = NelderMead(), maxIter = 25, tol = 1e-6)

    # Find Optimal Smoothing Params

    res = optimize(
        sp -> PIRLS(y, x, exp.(sp), Basis, Dist, Link; maxIter).Diagnostics[:GCV], 
        zeros(length(x)), 
        Optimizer
    )

    sp = exp.(Optim.minimizer(res))

    # Fit Optimal Model

    return PIRLS(y, x, sp, Basis, Dist, Link; maxIter, tol)
end
