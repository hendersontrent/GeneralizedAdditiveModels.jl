@recipe function plot(mod::GAMData; var=0)
    if typeof(var) != Int
        error("var must be an integer")
    end

    if var != 0
        x = mod.x[var]
        pred = PredictPartial(mod, var)
        ord = sortperm(x)

        return @series begin
            x[ord], pred[ord]
        end
    else
        n = length(mod.x)
        layout := (1, n)
        for partial in 1:n
            x = mod.x[partial]
            pred = PredictPartial(mod, partial)
            ord = sortperm(x)

            @series begin
                subplot := partial
                link := :y
                x[ord], pred[ord]
            end
        end
    end
end

"""
    partialdependenceplot(mod, var)
Draw partial dependence plot.

Usage:
```julia-repl
partialdependenceplot(mod, var)
```

Arguments:
- `mod` : `GAMData` containing the model.
- `var` : `Int` denoting the variable to plot.
"""
@userplot PartialDependencePlot
@recipe function f(p::PartialDependencePlot)
    mod, var = p.args
    if typeof(mod) != GAMData
        error("First argument must be a GAMData object")
    end
    if typeof(var) != Int
        error("Second argument must be an integer")
    end
    x = mod.x[var]
    pred = PredictPartial(mod, var)
    ord = sortperm(x)

    return @series begin
        x[ord], pred[ord]
    end
end

"""
    plotGAM(mod)
Plot GAM.

Usage:
```julia-repl
plotGAM(mod)
```
Arguments:
- `mod` : `GAMData` containing the model.
"""
@userplot plotGAM
@recipe function f(p::plotGAM)
    mod = p.args[1]
    if typeof(mod) != GAMData
        error("Argument must be a GAMData object")
    end
    n = length(mod.x)
    layout := (1, n)
    for partial in 1:n
        x = mod.x[partial]
        pred = PredictPartial(mod, partial)
        ord = sortperm(x)

        @series begin
            subplot := partial
            link := :y
            x[ord], pred[ord]
        end
    end
end
