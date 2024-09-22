# Physical-informed Neural Network
using NeuralPDE, Lux, ModelingToolkit
using Optimization, OptimizationOptimisers
# Diagram of functions
using Plots
import ModelingToolkit: Interval, infimum, supremum
# Solution
using Bessels

@parameters x
@variables y(..)
Dx = Differential(x)
Dxx = Differential(x)^2

α = 0

# Bessel ODE
eq = Dxx(y(x)) * x^2 + Dx(y(x)) * x + (x^2 - α^2) * y(x) ~ 0

# Boundary conditions
bcs = [y(0) ~ 1.0]
# Space and time domains
domains = [x ∈ Interval(0.0, 10.0)]

# Neural network
inner = 16
chain = Lux.Chain(Dense(1, inner, Lux.σ),
                  Dense(inner, inner, Lux.σ),
                  Dense(inner, inner, Lux.σ),
                  Dense(inner, 1))

# Discretization
strategy = GridTraining(0.05)
discretization = PhysicsInformedNN(chain, strategy)

@named pde_system = PDESystem(eq, bcs, domains, [x], [y(x)])
prob = discretize(pde_system, discretization)

# Callback function
callback = function (p, l)
    println("Current loss is: $l")
    return false
end

res = Optimization.solve(prob, OptimizationOptimisers.Adam(0.01); callback = callback, maxiters = 2000)
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
     xlabel="x",
     ylabel= "y",
     ylims=(-1,1),
     xlims=(0,10))
plot!(x_plot,
      u_predict,
      label = "PINN solution",
      xlabel="x",
      ylabel= "y",
      ylims=(-1,1),
      xlims=(0,10))
png("Bessel.png")
