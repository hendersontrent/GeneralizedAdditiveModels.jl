# Examples
Worked examples using GAM.jl from Generalized Additive Models, Simon N. Wood

## Linear Models
```
using RDatasets
using GAM
using Plots
```
### Trees
```
df=dataset("datasets", "trees")
mod = gam("Volume ~ s(Girth, k=10, degree=3) + s(Height, k=10, degree=3)", df)
```

### Hubble
Model: </br>
y ~ x  
#### Load Hubble Dataset
```
hubble=dataset("gamair", "hubble")
```
#### Fit with GAM
```
hubMod =gam("y ~ x", hubble) 
GAM.summary(hubMod)
#Remove outliers
hubMod1=gam("y ~ x", hubble[Not([3,15]), :])
GAM.summary(hubMod1)
```
#### Plots
```
scatter(hubble.x, hubble.y, label="Measured")
scatter!(hubMod.x, hubMod.Fitted, label="Fitted")
#Plot fitted data vs residuals
scatter(hubMod.Fitted, residuals(hubMod), label="", xlab="fitted values", ylab="residuals")

```