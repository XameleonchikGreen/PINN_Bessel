# Physical-informed Neural Network
using NeuralPDE, Lux, ModelingToolkit
using Optimization, OptimizationOptimisers
using QuasiMonteCarlo, Random
# Diagram of functions
using Plots, LaTeXStrings
import ModelingToolkit: Interval, infimum, supremum
using Printf
# Saving weights
using JLD2
using Logging
# Solution
using Bessels

disable_logging(Logging.Warn)

@parameters x
@variables y(..)
Dx = Differential(x)
Dxx = Differential(x)^2

# Consts
const lr1 = 4000;                   # Epochs for 0.01 lr
const lr2 = 4000;                   # Epochs for 0.001 lr
const lr3 = 8000;                   # Epochs for 0.0001 lr
const lr4 = 14000;                  # Epochs for 0.00001 lr
const α = 0;                        # Order of Bessel function
pnt = Vector{Vector{Float64}}(undef, 1);
errors = Dict(
    "uniform" => [], 
    "random" => [],
    "quasi-random" => [],
    "adaptive" => []
)
sizes = []

# Struct for training
Base.@kwdef struct MyGrid <: QuasiMonteCarlo.DeterministicSamplingAlgorithm
    R::RandomizationMethod = NoRand()
end

function QuasiMonteCarlo.sample(n::Integer, d::Integer, S::MyGrid, T = Float64)
    global pnt
    randomize(mapreduce(permutedims, vcat, pnt), S.R)
end

# Bessel first derivative
function besselj_der(α, x)
    return α / x * besselj(α, x) - besselj(α + 1, x)
end

# Loading weights
loadata = load("../weights/SIMB_$(α).jld2")

for a in 0.01:0.005:0.04
    println("a = $(a):")
    # Calculating number of points
    arr = collect(0.2:0.2:10.0)
    l = 0
    while(l+1 <= length(arr))
        if l == 0
            x0 = 0.0
        else
            x0 = arr[l]
        end
        x1 = arr[l+1]
        if(abs(besselj_der(α, (x0 + x1) / 2.0)) * (x1 - x0) < a)
            l += 1
        else
            push!(arr, arr[length(arr)])
            for i in (length(arr) - 1):-1:l+2
                arr[i] = arr[i - 1]
            end
            arr[l+1] = (x0 + x1) / 2
        end
    end
    arr = arr ./ 10.0
    n = length(arr)
    push!(sizes, log(1 // n))
    for Grid in ["uniform", "random", "quasi-random", "adaptive"]
        println("    $(Grid):")
        global pnt
        # Bessel ODE
        eq = Dxx(y(x)) * x^2 + Dx(y(x)) * x + (x^2 - α^2) * y(x) ~ 0

        # Boundary conditions
        if(α == 0)
            bcs = [y(0) ~ 1.0,
                Dx(y(0)) ~ 0.0]
        elseif(α == 1)
            bcs = [y(0) ~ 0.0,
                Dx(y(0)) ~ 0.5]
        end
        # Space domain
        domains = [x ∈ Interval(0.0, 10.0)]

        # Neural network
        inner = 16
        chain = Lux.Chain(Dense(1, inner, sin),
                        Dense(inner, inner, sin),
                        Dense(inner, inner, sin),
                        Dense(inner, 1))

        # Discretization
        if(Grid == "uniform")
            step = 1 // n
            pnt[1] = collect(Float64, step:step:1.0)
        elseif(Grid == "random")
            pnt[1] = Random.rand(Float64, n)
        elseif(Grid == "quasi-random")
            pnt[1] = []
            s_0 = 0
            ϕ = Base.MathConstants.golden
            for k in 1:1:n
                push!(pnt[1], s_0 + k*ϕ)
            end
            pnt[1] = mod1.(pnt[1], 1)
        elseif(Grid == "adaptive")
            pnt[1] = arr
        end

        strategy = QuasiRandomTraining(n, sampling_alg = MyGrid(), resampling = false, minibatch = 1)

        discretization = PhysicsInformedNN(chain, strategy, init_params = loadata["params"])
        @named pde_system = PDESystem(eq, bcs, domains, [x], [y(x)])
        sym_prob = NeuralPDE.symbolic_discretize(pde_system, discretization)

        phi = sym_prob.phi

        pde_loss_functions = sym_prob.loss_functions.pde_loss_functions
        bc_loss_functions = sym_prob.loss_functions.bc_loss_functions
        loss_functions = [pde_loss_functions; bc_loss_functions]

        function loss_function(θ, p)
            return sum(map(l -> l(θ), loss_functions))
        end

        lss = 0.0

        # Callback function
        callback = function (p, l)
            lss = l
            return false
        end

        f_ = OptimizationFunction(loss_function, Optimization.AutoZygote())
        prob = Optimization.OptimizationProblem(f_, sym_prob.flat_init_params)

        res = Optimization.solve(prob, OptimizationOptimisers.Adam(0.01); callback = callback, maxiters = lr1)
        prob = remake(prob, u0 = res.u)
        @printf("        after %5d epochs loss is %g \n", lr1, lss)
        res = Optimization.solve(prob, OptimizationOptimisers.Adam(0.001); callback = callback, maxiters = lr2)
        prob = remake(prob, u0 = res.u)
        @printf("        after %5d epochs loss is %g \n", lr1+lr2, lss)
        res = Optimization.solve(prob, OptimizationOptimisers.Adam(0.0001); callback = callback, maxiters = lr3)
        # prob = remake(prob, u0 = res.u)
        # @printf("        after %5d epochs loss is %g \n", lr1+lr2+lr3, lss)
        # res = Optimization.solve(prob, OptimizationOptimisers.Adam(0.00001); callback = callback, maxiters = lr4)

        u_real = [besselj(α, x) for x in pnt[1]]
        u_predict = [first(phi(x, res.u)) for x in pnt[1]]

        error = 0.0
        for i in eachindex(u_real)
            error = max(error, abs(u_real[i] - u_predict[i]))
        end

        @printf("      final loss is %g, final error is %g\n", lss, error)
        push!(errors[Grid], log(error))
    end
end

plot(sizes,
     errors["uniform"],
     label = "Uniform grid",
     size = (1600, 900),
     margin = 10Plots.mm
    )

scatter!(sizes,
         errors["random"],
         label = "Random grid"
        )

scatter!(sizes,
         errors["quasi-random"],
         label = "Quasi-random grid"
        )

scatter!(sizes,
         errors["adaptive"],
         label = "Adaptive grid"
       )

xlabel!(L"\ln\left(\frac{1}{N - 1}\right)")
ylabel!(L"\ln\left(errors\right)")

png("../images/absolute_error/Bessel_$(α)_absoluteError.png")
