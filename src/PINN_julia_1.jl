# Physical-informed Neural Network
using NeuralPDE, Lux, ModelingToolkit
using Optimization, OptimizationOptimisers
# Diagram of functions
using Plots, LaTeXStrings
import ModelingToolkit: Interval, infimum, supremum
using Printf
# Solution
using Bessels

@parameters x
@variables y(..)
Dx = Differential(x)
Dxx = Differential(x)^2

α = 1

# Bessel ODE
eq = Dxx(y(x)) * x^2 + Dx(y(x)) * x + (x^2 - α^2) * y(x) ~ 0

# Boundary conditions
bcs = [y(0) ~ 0.0,
       Dx(y(0)) ~ 0.5]
# Space and time domains
domains = [x ∈ Interval(0.0, 10.0)]

# Neural network
inner = 16
chain = Lux.Chain(Dense(1, inner, sin),
                  Dense(inner, inner, sin),
                  Dense(inner, inner, sin),
                  Dense(inner, 1))

# Discretization
strategy = GridTraining(0.05)
loss_type = NonAdaptiveLoss(pde_loss_weights = 1.0, bc_loss_weights = 1.0, additional_loss_weights = 0.0)
discretization = PhysicsInformedNN(chain, strategy, adaptive_loss=loss_type)

@named pde_system = PDESystem(eq, bcs, domains, [x], [y(x)])
prob = discretize(pde_system, discretization)
loss = []

# Callback function
callback = function (p, l)
    push!(loss, log10(l))
    @printf("Step: %5d Current loss is: %g \n", length(loss), l)
    return false
end

res = Optimization.solve(prob, OptimizationOptimisers.Adam(0.01); callback = callback, maxiters = 2000)
prob = remake(prob, u0 = res.u)
res = Optimization.solve(prob, OptimizationOptimisers.Adam(0.001); callback = callback, maxiters = 2000)
prob = remake(prob, u0 = res.u)
res = Optimization.solve(prob, OptimizationOptimisers.Adam(0.0001); callback = callback, maxiters = 2000)
prob = remake(prob, u0 = res.u)
res = Optimization.solve(prob, OptimizationOptimisers.Adam(0.00001); callback = callback, maxiters = 6000)
prob = remake(prob, u0 = res.u)
res = Optimization.solve(prob, OptimizationOptimisers.Adam(0.000001); callback = callback, maxiters = 8000)

phi = discretization.phi

# Creating diagrams
dx = 0.05
xs = [infimum(d.domain):(dx / 10):supremum(d.domain) for d in domains][1]
u_real = [besselj(α, x) for x in xs]
u_predict = [first(phi(x, res.u)) for x in xs]

x_plot = collect(xs)
plot(x_plot,
     u_real,
     label = "Bessel function",
     xlabel = L"X",
     ylabel = L"Y",
     ylims=(-1,1),
     xlims=(0,10))
plot!(x_plot,
      u_predict,
      label = "PINN solution",
      xlabel = L"X",
      ylabel = L"Y",
      ylims=(-1,1),
      xlims=(0,10))
png("../images/Bessel_1.png")

p1 = plot(LinearIndices(loss),
          loss,
          label = "loss(epochs)",
          xlabel = L"epochs",
          ylabel = L"log_{10}(loss)",
          yticks = -5:1:3,
          xticks = 0:1000:20000,
          size = (1200, 800),
          left_margin=10Plots.mm,
          bottom_margin=10Plots.mm,
          formatter=:plain)
png(p1, "../images/loss_1.png")
