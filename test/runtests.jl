using GAM
using Test
using RDatasets, DataFrames, Plots
using Distributions, Random, Statistics, LinearAlgebra

#-------------------- Run tests -----------------

@testset "GAM.jl Test Suite" begin
    
    # ===========================
    # Core Gaussian GAM Tests
    # ===========================
    @testset "Gaussian GAM (Normal Family)" begin
        df = dataset("datasets", "trees")
        
        @testset "Basic fitting and prediction" begin
            mod = gam("Volume ~ s(Girth, k=10, degree=3) + s(Height, k=10, degree=3)", df)
            
            @test mod isa GAMData
            @test mod.Family[:Name] == "Normal"
            @test mod.Link[:Name] == "Identity"
            @test length(mod.Fitted) == nrow(df)
            
            # Check residuals properties
            residuals = df.Volume .- mod.Fitted
            @test mean(residuals) ≈ 0 atol=0.5
            @test std(residuals) < std(df.Volume)  # Model should explain some variance
        end
        
        @testset "Model diagnostics" begin
            mod = gam("Volume ~ s(Girth, k=8, degree=3) + s(Height, k=8, degree=3)", df)
            
            @test haskey(mod.Diagnostics, :RSS)
            @test haskey(mod.Diagnostics, :EDF)
            @test haskey(mod.Diagnostics, :GCV)
            
            @test mod.Diagnostics[:RSS] > 0
            @test 2 < mod.Diagnostics[:EDF] < 16  # Between minimum and maximum possible
            @test mod.Diagnostics[:GCV] > 0
        end
        
        @testset "Partial predictions" begin
            mod = gam("Volume ~ s(Girth, k=10, degree=3) + s(Height, k=10, degree=3)", df)
            
            # Test partial predictions for each smooth
            for i in 1:length(mod.x)
                partial = GAM.PredictPartial(mod, i)
                @test length(partial) == length(mod.x[i])
                @test !any(isnan.(partial))
                @test !any(isinf.(partial))
            end
        end
        
        @testset "Visualization" begin
            mod = gam("Volume ~ s(Girth, k=10, degree=3) + s(Height, k=10, degree=3)", df)
            p = plotGAM(mod)
            @test p isa Plots.Plot
        end
    end
    
    # ===========================
    # Gamma GAM Tests
    # ===========================
    @testset "Gamma GAM" begin
        df = dataset("datasets", "trees")
        
        @testset "Basic fitting with log link" begin
            mod = gam("Volume ~ s(Girth, k=10, degree=3) + s(Height, k=10, degree=3)", df; 
                     Family = "Gamma", Link = "Log")
            
            @test mod.Family[:Name] == "Gamma"
            @test mod.Link[:Name] == "Log"
            @test all(mod.Fitted .> 0)  # Gamma predictions must be positive
            @test length(mod.Fitted) == nrow(df)
        end
        
        @testset "Convergence and stability" begin
            # Test with different starting values via different maxIter
            mod1 = gam("Volume ~ s(Girth, k=8, degree=3)", df; 
                      Family = "Gamma", Link = "Log", maxIter = 10)
            mod2 = gam("Volume ~ s(Girth, k=8, degree=3)", df; 
                      Family = "Gamma", Link = "Log", maxIter = 50)
            
            # Should converge to similar values
            @test cor(mod1.Fitted, mod2.Fitted) > 0.99
        end
    end
    
    # ===========================
    # Bernoulli GAM Tests
    # ===========================
    @testset "Bernoulli GAM (Logistic)" begin
        Random.seed!(123)  # For reproducibility
        
        @testset "Nonlinear binary classification" begin
            n = 300
            x1 = range(-2, 2, length=n)
            x2 = randn(n)
            
            # Create true nonlinear effect
            f1 = sin.(x1 * π/2)
            f2 = 0.5 * (x2.^2 .- 1)
            eta = f1 + f2
            p_true = 1 ./ (1 .+ exp.(-eta))
            y = rand.(Bernoulli.(p_true))
            
            df = DataFrame(y=y, x1=x1, x2=x2)
            
            mod = gam("y ~ s(x1, k=8, degree=3) + s(x2, k=8, degree=3)", df; 
                     Family = "Bernoulli", Link = "Logit")
            
            @test mod.Family[:Name] == "Bernoulli"
            @test mod.Link[:Name] == "Logit"
            @test all(0 .<= mod.Fitted .<= 1)
            
            # Check classification performance
            predictions = mod.Fitted .> 0.5
            accuracy = mean(predictions .== y)
            @test accuracy > 0.6  # Should do better than random
            
            # Check calibration: average predicted prob should ≈ observed proportion
            @test abs(mean(mod.Fitted) - mean(y)) < 0.1
        end
        
        @testset "Numerical stability with extreme probabilities" begin
            n = 100
            x = range(-5, 5, length=n)
            
            # Create separation - extreme but not complete
            eta = 8 * x
            p = 1 ./ (1 .+ exp.(-eta))
            y = Float64.(p .> 0.5)
            
            df = DataFrame(y=y, x=x)
            
            mod = gam("y ~ s(x, k=5, degree=3)", df; 
                     Family = "Bernoulli", Link = "Logit")
            
            @test !any(isnan.(mod.Fitted))
            @test !any(isinf.(mod.Fitted))
            @test all(0 .<= mod.Fitted .<= 1)
            
            # Should still separate classes well
            @test mean(mod.Fitted[y .== 1]) > 0.9
            @test mean(mod.Fitted[y .== 0]) < 0.1
        end
        
        @testset "Input validation" begin
            n = 50
            x = randn(n)
            
            # Non-binary data should throw error
            y_continuous = randn(n)
            df_invalid = DataFrame(y=y_continuous, x=x)
            @test_throws AssertionError gam("y ~ s(x, k=5, degree=3)", df_invalid; 
                                           Family = "Bernoulli", Link = "Logit")
            
            # Binary data should work
            y_binary = rand([0, 1], n)
            df_valid = DataFrame(y=y_binary, x=x)
            mod = gam("y ~ s(x, k=5, degree=3)", df_valid; 
                     Family = "Bernoulli", Link = "Logit")
            @test mod isa GAMData
        end
        
        @testset "Monotonic relationships" begin
            n = 200
            x = sort(randn(n) * 2)
            
            # Strong monotonic relationship
            eta = 3 * x
            p = 1 ./ (1 .+ exp.(-eta))
            y = rand.(Bernoulli.(p))
            
            df = DataFrame(y=y, x=x)
            mod = gam("y ~ s(x, k=10, degree=3)", df; 
                     Family = "Bernoulli", Link = "Logit")
            
            # Check monotonicity of fitted values
            fitted_sorted = mod.Fitted[sortperm(x)]
            differences = diff(fitted_sorted)
            @test sum(differences .> 0) / length(differences) > 0.95  # Should be mostly increasing
            
            # Check effective degrees of freedom (should be relatively low for smooth monotonic)
            @test 1.5 < mod.Diagnostics[:EDF] < 5
        end
    end
    
    # ===========================
    # Mixed Models Tests
    # ===========================
    @testset "Mixed Linear and Smooth Terms" begin
        
        @testset "Gaussian with mixed terms" begin
            df = dataset("datasets", "trees")
            
            mod = gam("Volume ~ s(Girth, k=8, degree=3) + Height", df)
            
            @test mod isa GAMData
            @test length(mod.x) == 2
            @test length(mod.Basis) == 2
            
            # Linear term should have minimal degrees of freedom
            # (This assumes Height is the second term)
            partial_height = GAM.PredictPartial(mod, 2)
            @test length(unique(diff(partial_height[sortperm(df.Height)]))) < 5  # Should be nearly constant slope
        end
        
        @testset "Bernoulli with mixed terms" begin
            Random.seed!(456)
            n = 500
            
            # Linear and nonlinear effects
            x_linear = randn(n)
            x_smooth = sort(rand(n) * 2 .- 1)
            
            # True model: linear + smooth
            beta_linear = 1.5
            f_smooth = 2 * sin.(3 * π * x_smooth)
            eta = 0.5 .+ beta_linear * x_linear .+ f_smooth
            p = 1 ./ (1 .+ exp.(-eta))
            y = rand.(Bernoulli.(p))
            
            df = DataFrame(y=y, x_linear=x_linear, x_smooth=x_smooth)
            
            mod = gam("y ~ x_linear + s(x_smooth, k=10, degree=3)", df; 
                     Family="Bernoulli", Link="Logit")
            
            @test all(0 .<= mod.Fitted .<= 1)
            
            # Check that linear effect is captured
            partial_linear = GAM.PredictPartial(mod, 1)
            linear_coef = cov(partial_linear, x_linear) / var(x_linear)
            @test abs(linear_coef - beta_linear) < 0.5  # Should be close to true value
            
            # Check that smooth captures nonlinearity
            partial_smooth = GAM.PredictPartial(mod, 2)
            smooth_sorted = partial_smooth[sortperm(x_smooth)]
            
            # Should have multiple turning points (capturing the sine wave)
            second_diff = diff(diff(smooth_sorted))
            sign_changes = sum(diff(sign.(second_diff)) .!= 0)
            @test sign_changes > 2  # Sine wave should have multiple inflection points
        end
    end
    
    # ===========================
    # Poisson GAM Tests
    # ===========================
    @testset "Poisson GAM" begin
        Random.seed!(789)
        n = 200
        
        @testset "Count data modeling" begin
            x1 = randn(n)
            x2 = rand(n) * 3
            
            # Create count data
            eta = 1.5 .+ 0.8 * x1 .+ sin.(2π * x2)
            lambda = exp.(eta)
            lambda = clamp.(lambda, 0.1, 20)  # Keep reasonable for testing
            y = rand.(Poisson.(lambda))
            
            df = DataFrame(y=y, x1=x1, x2=x2)
            
            mod = gam("y ~ s(x1, k=6, degree=3) + s(x2, k=8, degree=3)", df; 
                     Family = "Poisson", Link = "Log")
            
            @test mod.Family[:Name] == "Poisson"
            @test mod.Link[:Name] == "Log"
            @test all(mod.Fitted .>= 0)  # Poisson predictions must be non-negative
            
            # Check mean-variance relationship (Poisson property)
            # Group predictions into bins and check mean ≈ variance
            n_bins = 5
            fitted_quantiles = quantile(mod.Fitted, range(0, 1, length=n_bins+1))
            for i in 1:n_bins
                mask = fitted_quantiles[i] .<= mod.Fitted .<= fitted_quantiles[i+1]
                if sum(mask) > 10
                    observed = y[mask]
                    predicted_mean = mean(mod.Fitted[mask])
                    observed_mean = mean(observed)
                    observed_var = var(observed)
                    # For Poisson, mean ≈ variance
                    @test abs(log(observed_mean + 1) - log(observed_var + 1)) < 1
                end
            end
        end
    end
    
    # ===========================
    # Basis Function Tests
    # ===========================
    @testset "Basis Function Properties" begin
        x = randn(100)
        
        @testset "Uniform basis" begin
            basis = GAM.BuildUniformBasis(x, 10, 3)
            @test length(basis.breakpoints) == 10
            @test basis.order == 3
            @test minimum(basis.breakpoints) ≈ minimum(x)
            @test maximum(basis.breakpoints) ≈ maximum(x)
        end
        
        @testset "Basis matrix properties" begin
            basis = GAM.BuildUniformBasis(x, 8, 3)
            X = GAM.BuildBasisMatrix(basis, x)
            
            @test size(X, 1) == length(x)
            @test size(X, 2) == length(basis)
            @test all(0 .<= X .<= 1)  # B-splines are bounded
            @test all(sum(X, dims=2) .≈ 1)  # Partition of unity
        end
        
        @testset "Penalty matrix" begin
            basis = GAM.BuildUniformBasis(x, 10, 3)
            D = GAM.BuildDifferenceMatrix(basis)
            
            @test size(D, 2) == length(basis)
            @test size(D, 1) == length(basis) - 2  # Second differences
            @test rank(D) == size(D, 1)  # Should be full rank
        end
    end
    
    # ===========================
    # Edge Cases and Robustness
    # ===========================
    @testset "Edge Cases" begin
        
        @testset "Small sample sizes" begin
            n = 20
            x = randn(n)
            y = randn(n)
            df = DataFrame(x=x, y=y)
            
            # Should work with small n but limited knots
            mod = gam("y ~ s(x, k=4, degree=2)", df)
            @test mod isa GAMData
            @test mod.Diagnostics[:EDF] <= 4
        end
        
        @testset "Perfect fit scenario" begin
            # Create data that can be fit perfectly
            x = [1.0, 2.0, 3.0, 4.0, 5.0]
            y = [2.0, 4.0, 6.0, 8.0, 10.0]  # Perfect linear
            df = DataFrame(x=x, y=y)
            
            mod = gam("y ~ s(x, k=3, degree=2)", df)
            @test maximum(abs.(y .- mod.Fitted)) < 0.1  # Should fit nearly perfectly
        end
        
        @testset "Constant response" begin
            n = 50
            x = randn(n)
            y = ones(n) * 5.0  # Constant response
            df = DataFrame(x=x, y=y)
            
            mod = gam("y ~ s(x, k=5, degree=3)", df)
            @test std(mod.Fitted) < 0.1  # Fitted values should be nearly constant
            @test mean(mod.Fitted) ≈ 5.0 atol=0.1
        end
    end
    
    # ===========================
    # Performance Tests
    # ===========================
    @testset "Performance and Scaling" begin
        
        @testset "Large dataset handling" begin
            n = 1000
            x = randn(n)
            y = 2 * sin.(x) + 0.5 * randn(n)
            df = DataFrame(x=x, y=y)
            
            # Should complete in reasonable time
            t = @elapsed mod = gam("y ~ s(x, k=20, degree=3)", df)
            @test t < 30.0  # Should finish within 30 seconds
            @test mod isa GAMData
        end
        
        @testset "Multiple smooths scaling" begin
            n = 200
            p = 5  # Number of predictors
            
            df = DataFrame()
            formula_parts = String[]
            
            for i in 1:p
                df[!, Symbol("x$i")] = randn(n)
                push!(formula_parts, "s(x$i, k=5, degree=3)")
            end
            
            # Response is sum of nonlinear functions
            y = zeros(n)
            for i in 1:p
                y .+= sin.(df[!, Symbol("x$i")])
            end
            y .+= 0.5 * randn(n)
            df.y = y
            
            formula = "y ~ " * join(formula_parts, " + ")
            
            t = @elapsed mod = gam(formula, df)
            @test t < 60.0  # Should handle multiple smooths efficiently
            @test mod isa GAMData
            @test length(mod.Basis) == p
        end
    end
end
