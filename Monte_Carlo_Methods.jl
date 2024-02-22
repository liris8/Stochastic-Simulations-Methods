# Monte Carlo Functions

module Monte_Carlo_Methods

export hit_and_miss_multi, uniform_sampling_multi, importance_sampling, Correlations, calculate_errors

"""
Perform Hit & Miss integration for multivariable functions.

Parameters:
- g: Integrand function.
- a: Initial value in x axis.
- b: Final value in x axis.
- c: Highest value in y axis.
- M: Number of times to repeat the experiment.
- n: Dimension. Default is 1.

Returns:
- I: Estimated integral value.
- σ: Estimated error.
"""
function hit_and_miss_multi(g, a, b, c, M, n = 1)  
    nb = 0
    for i in 1:M
        x = n == 1 ? a + (b-a) * rand(Float64) : a .+ (b-a) .*rand(Float64, n)
        y = c * rand(Float64)
        if g(x) > y
            nb += 1
        end 
    end 
    p = nb/M                    
    I = c*p*(b-a)^n              
    σ = sqrt(p*(1-p)/M)*c*(b-a)^n 
    return I, σ
end

"""
Perform uniform sampling integration for multivariable functions.

Parameters:
- g: Integrand function.
- a: Initial value in x axis.
- b: Final value in x axis.
- M: Number of sampling points.
- n: Dimension. Default is 1.

Returns:
- r: Estimated integral value.
- σ: Estimated error.
"""
function uniform_sampling_multi(g, a, b, M, n = 1)
    r, s = 0, 0
    for i in 1:M
        x = n == 1 ? a + (b-a) * rand(Float64) : a .+ (b-a) .*rand(Float64, n)
        g0 = g(x)
        r += g0
        s += g0^2
    end
    r = r/M
    σ = sqrt((s/M-r^2)/M)
    r = r * (b-a)^n             # Unbiased estimator
    σ = σ * (b-a)^n             # Sigma error
    return r, σ
end

"""
Perform importance sampling integration.

Parameters:
- G: Function for generating samples according to the importance distribution.
- M: Number of sampling points.

Returns:
- r: Estimated integral value.
- σ: Estimated error.
"""
function importance_sampling(G, M)
    r, s = 0, 0 
    
    for i in 1:M
        G0 = G(-log(rand(Float64)))
        r  = r + G0
        s  = s + G0*2
    end
    
    r = r/M                 
    σ = sqrt((s/M-r^2)/M)  
    return r, σ
end

"""
Calculate correlations for a given function.

Parameters:
- G: Function for which correlations are calculated.
- xs: Array of values for the function argument.
- ks: Array of values for the lag parameter.

Returns:
- ρg: Correlation function ρG(k).
- τG: Correlation time.
- τeq: Approximated correlation time.
"""
function Correlations(G, xs, ks)
    G_vals = G.(xs)

    # Correlation function ρG(k)
    # https://juliastats.org/StatsBase.jl/stable/signalcorr/
    ρg = autocor(G_vals, ks[1:2000])

    # Correlation time 
    τG  = sum(ρg)

    # Approximated Correlation time
    τeq = ρg[1]/(1-ρg[1])

    return ρg, τG, τeq
end

"""
Calculate errors in correlation time estimation.

Parameters:
- G: Function for which correlations are calculated.
- num_iterations: Number of iterations.

Returns:
None
"""
function calculate_errors(G, num_iterations)
    results_τG = Float64[]
    results_τeq = Float64[]
    
    for _ in 1:num_iterations
        xs = Metropolis(M, y, δ)
        # G_vals = G.(xs)
        _, τG, τeq = Correlations(G, xs, ks)
        push!(results_τG, τG)
        push!(results_τeq, τeq)
    end

    println("τG: ",  mean(results_τG), " ± ", std(results_τG))
    println("τeq: ", mean(results_τeq), " ± ", std(results_τeq))
end

end
