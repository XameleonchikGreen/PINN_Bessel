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
# Solution
using Bessels

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
const Grid = "quasi-random";             # "uniform", "random", "quasi-random", "adaptive" grids
const finding_weights = false;      # 'true' for finding new weights and 'false' for using weights from file
pnt = Vector{Vector{Float64}}(undef, 1);

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
const inner = 16
chain = Lux.Chain(Dense(1, inner, sin),
                  Dense(inner, inner, sin),
                  Dense(inner, inner, sin),
                  Dense(inner, 1))

# Discretization
if(Grid == "uniform")
    pnt[1] = collect(Float64, 0.005:0.005:1.0)
elseif(Grid == "random")
    pnt[1] = Random.rand(Float64, 200)
elseif(Grid == "quasi-random")
    pnt[1] = []
    const s_0 = 0
    const ϕ = Base.MathConstants.golden
    for n in 1:1:200
        push!(pnt[1], s_0 + n*ϕ)
    end
    pnt[1] = mod1.(pnt[1], 1)
end
strategy = QuasiRandomTraining(200, sampling_alg = MyGrid(), resampling = false, minibatch = 1)

loadata = Dict{String, Any}("params" => nothing)

if(!finding_weights)
    # Loading weights
    loadata = load("../weights/SIMB_$(α).jld2")
end

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

loss = []

# Callback function
callback = function (p, l)
    push!(loss, log10(l))
    @printf("Step: %5d Current loss is: %g \n", length(loss), l)
    return false
end

f_ = OptimizationFunction(loss_function, Optimization.AutoZygote())
prob = Optimization.OptimizationProblem(f_, sym_prob.flat_init_params)

if(finding_weights)
    # Saving weights
    weights = prob.u0
    save("../weights/SIMB_$(α).jld2", Dict("params" => weights))
end

res = Optimization.solve(prob, OptimizationOptimisers.Adam(0.01); callback = callback, maxiters = lr1)
prob = remake(prob, u0 = res.u)
res = Optimization.solve(prob, OptimizationOptimisers.Adam(0.001); callback = callback, maxiters = lr2)
prob = remake(prob, u0 = res.u)
res = Optimization.solve(prob, OptimizationOptimisers.Adam(0.0001); callback = callback, maxiters = lr3)
prob = remake(prob, u0 = res.u)
res = Optimization.solve(prob, OptimizationOptimisers.Adam(0.00001); callback = callback, maxiters = lr4)

# Creating diagrams
dx = 0.05
xs = [infimum(d.domain):(dx / 10):supremum(d.domain) for d in domains][1]
u_real = [besselj(α, x) for x in xs]
u_predict = [first(phi(x, res.u)) for x in xs]
x_scatter = pnt[1] .* 10;
u_predict_scatter = [first(phi(x, res.u)) for x in x_scatter]
x_plot = collect(xs)

plot(x_plot,
     u_real,
     label = "Bessel function",
     size = (1600, 900),
     margin = 10Plots.mm,
     framestyle = :origin
    )

plot!(x_plot,
      u_predict,
      label = "PINN solution",
     )

plot!(x_scatter,
      u_predict_scatter,
      seriestype = :sticks,
      linecolor = :green,
      linealpha = 0.4,
      label = "",
     )

xlabel!(L"X")
ylabel!(L"Y")
ylims!(-1, 1)
xticks!(0:1:10)

png("../images/$(Grid)/Bessel_$(α)_$(Grid).png")

p1 = plot(LinearIndices(loss),
          loss,
          label = "loss(epochs)",
          size = (1600, 900),
          margin = 10Plots.mm,
          formatter=:plain
         )

vline!([0, lr1, lr1+lr2, lr1+lr2+lr3],
       line = :dash,
       label = ""
      )

annotate!(0+900, 2, text(L"η = 0.01"))
annotate!(lr1+1000, 2, text(L"η = 0.001"))
annotate!(lr1+lr2+1100, 2, text(L"η = 0.0001"))
annotate!(lr1+lr2+lr3+1200, 2, text(L"η = 0.00001"))

xlabel!(L"epochs")
ylabel!(L"log_{10}(loss)")
yticks!(-5:1:3)
xticks!(0:2_000:30_000)

png(p1, "../images/$(Grid)/loss_$(α)_$(Grid).png")
