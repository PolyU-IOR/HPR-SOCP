using JuMP
using HPRSOCP

# This demo builds the SOCP encoded in data/model.cbf directly with JuMP.
#
# The CBF file represents
#
#     minimize    -3x - 5y + z
#     subject to   x + 2y <= 10
#                 3x +  y <= 12
#                 x, y, z >= 0
#                 ||(0.5z - 1, sqrt(2)x, sqrt(2)y)||_2 <= 0.5z + 1.
#
# The SOC constraint is the standard epigraph form for x^2 + y^2 <= z.
# Since
#     (0.5z + 1)^2 - (0.5z - 1)^2 = 2z,
# the cone inequality is equivalent to
#     2x^2 + 2y^2 <= 2z,
# i.e. x^2 + y^2 <= z.

model = Model(HPRSOCP.Optimizer)

set_optimizer_attribute(model, "use_gpu", true)
set_optimizer_attribute(model, "warm_up", true)
set_optimizer_attribute(model, "verbose", true)
set_optimizer_attribute(model, "stoptol", 1e-8)

@variable(model, x >= 0.0)
@variable(model, y >= 0.0)
@variable(model, z >= 0.0)

# Objective coefficients come from the OBJACOORD section of data/model.cbf.
@objective(model, Min, -3.0 * x - 5.0 * y + z)

# The two L- rows in the CBF file are stored internally as lower-bound rows
# after multiplying by -1; in JuMP we write them in their natural <= form.
@constraint(model, x + 2.0 * y <= 10.0)
@constraint(model, 3.0 * x + y <= 12.0)

# The Q row group is a four-dimensional second-order cone:
#     [0.5z + 1, 0.5z - 1, sqrt(2)x, sqrt(2)y] in SOC.
@constraint(model, [0.5 * z + 1.0, 0.5 * z - 1.0, sqrt(2.0) * x, sqrt(2.0) * y] in SecondOrderCone())

optimize!(model)

println("JuMP/HPRSOCP status: ", termination_status(model))
println("JuMP/HPRSOCP objective: ", objective_value(model))
println("JuMP/HPRSOCP solution: x = ", value(x), ", y = ", value(y), ", z = ", value(z))