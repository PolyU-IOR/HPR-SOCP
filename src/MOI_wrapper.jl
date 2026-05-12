# MOI Wrapper for HPRSOCP
# Based on the structure of HPRLP's MOI wrapper

import MathOptInterface as MOI

# Supported scalar sets
const SCALAR_SETS = Union{
    MOI.GreaterThan{Float64},
    MOI.LessThan{Float64},
    MOI.EqualTo{Float64},
    MOI.Interval{Float64},
}

# Generic cache so the wrapper can receive both scalar rows and vector SOC cones.
const OptimizerCache = MOI.Utilities.Model{Float64}

"""
    Optimizer()

Create a new HPRSOCP Optimizer object.

Set optimizer attributes using `MOI.RawOptimizerAttribute` or
`JuMP.set_optimizer_attribute`.

## Example

```julia
using JuMP, HPRSOCP
model = JuMP.Model(HPRSOCP.Optimizer)
set_optimizer_attribute(model, "stoptol", 1e-6)
set_optimizer_attribute(model, "use_gpu", true)
set_optimizer_attribute(model, "device_number", 0)
```
"""
mutable struct Optimizer <: MOI.AbstractOptimizer
    params::HPRSOCP_parameters
    results::Union{Nothing, HPRSOCP_results}
    silent::Bool
    cache::Union{Nothing, OptimizerCache}
    index_map::Union{Nothing, MOI.IndexMap}
    obj_sense::MOI.OptimizationSense  # Track objective sense for result conversion
    variable_to_solver::Vector{Int}
    solver_to_variable::Vector{Int}
    constraint_row_map::Dict{Any,Any}
    constraint_var_map::Dict{Any,Vector{Int}}
    
    function Optimizer()
        return new(
            HPRSOCP_parameters(),
            nothing,
            false,
            nothing,
            nothing,
            MOI.MIN_SENSE,
            Int[],
            Int[],
            Dict{Any,Any}(),
            Dict{Any,Vector{Int}}(),
        )
    end
end

# ====================
#   Utility functions
# ====================

function MOI.default_cache(::Optimizer, ::Type{Float64})
    return MOI.Utilities.UniversalFallback(OptimizerCache())
end

# ====================
#   Empty functions
# ====================

function MOI.is_empty(model::Optimizer)
    return model.results === nothing
end

function MOI.empty!(model::Optimizer)
    model.results = nothing
    model.cache = nothing
    model.index_map = nothing
    model.obj_sense = MOI.MIN_SENSE  # Reset to default
    empty!(model.variable_to_solver)
    empty!(model.solver_to_variable)
    empty!(model.constraint_row_map)
    empty!(model.constraint_var_map)
    return
end

# ====================
#   Solver attributes
# ====================

MOI.get(::Optimizer, ::MOI.SolverName) = "HPRSOCP"

function MOI.get(::Optimizer, ::MOI.SolverVersion)
    return "0.1.0"  # Update this to match your package version
end

# HPRSOCP does not support incremental interface - requires copy_to
MOI.supports_incremental_interface(::Optimizer) = false

# Silent mode
MOI.supports(::Optimizer, ::MOI.Silent) = true

function MOI.set(model::Optimizer, ::MOI.Silent, value::Bool)
    model.silent = value
    return
end

MOI.get(model::Optimizer, ::MOI.Silent) = model.silent

# Time limit
MOI.supports(::Optimizer, ::MOI.TimeLimitSec) = true

function MOI.set(model::Optimizer, ::MOI.TimeLimitSec, value::Real)
    model.params.time_limit = Float64(value)
    return
end

function MOI.set(model::Optimizer, ::MOI.TimeLimitSec, ::Nothing)
    model.params.time_limit = 3600.0  # Default value
    return
end

function MOI.get(model::Optimizer, ::MOI.TimeLimitSec)
    return model.params.time_limit
end

# Number of threads (not supported)
MOI.supports(::Optimizer, ::MOI.NumberOfThreads) = false

# Raw optimizer attributes for HPRSOCP parameters
const SUPPORTED_PARAMETERS = (
    "stoptol",
    "sigma",
    "max_iter",
    "time_limit",
    "check_iter",
    "warm_up",
    "print_frequency",
    "device_number",
    "use_Ruiz_scaling",
    "ruiz_iterations",
    "use_bc_scaling",
    "bc_scaling_norm_type",
    "use_l2_scaling",
    "use_Pock_Chambolle_scaling",
    "soc_block_scaling_strategy",
    "initial_x",
    "initial_y",
    "auto_save",
    "save_filename",
    "verbose",
    "use_gpu",
)

function MOI.supports(::Optimizer, param::MOI.RawOptimizerAttribute)
    return param.name in SUPPORTED_PARAMETERS
end

function MOI.set(model::Optimizer, param::MOI.RawOptimizerAttribute, value)
    name = String(param.name)
    if name == "stoptol"
        model.params.stoptol = Float64(value)
    elseif name == "sigma"
        model.params.sigma = Float64(value)
    elseif name == "max_iter"
        model.params.max_iter = Int(value)
    elseif name == "time_limit"
        model.params.time_limit = Float64(value)
    elseif name == "check_iter"
        model.params.check_iter = Int(value)
    elseif name == "warm_up"
        model.params.warm_up = Bool(value)
    elseif name == "print_frequency"
        model.params.print_frequency = Int(value)
    elseif name == "device_number"
        model.params.device_number = Int32(value)
    elseif name == "use_Ruiz_scaling"
        model.params.use_Ruiz_scaling = Bool(value)
    elseif name == "ruiz_iterations"
        model.params.ruiz_iterations = Int(value)
    elseif name == "use_bc_scaling"
        model.params.use_bc_scaling = Bool(value)
    elseif name == "bc_scaling_norm_type"
        model.params.bc_scaling_norm_type = Symbol(value)
    elseif name == "use_l2_scaling"
        model.params.use_l2_scaling = Bool(value)
    elseif name == "use_Pock_Chambolle_scaling"
        model.params.use_Pock_Chambolle_scaling = Bool(value)
    elseif name == "soc_block_scaling_strategy"
        model.params.soc_block_scaling_strategy = Symbol(value)
    elseif name == "use_gpu"
        model.params.use_gpu = Bool(value)
    elseif name == "initial_x"
        model.params.initial_x = value === nothing ? nothing : Vector{Float64}(value)
    elseif name == "initial_y"
        model.params.initial_y = value === nothing ? nothing : Vector{Float64}(value)
    elseif name == "auto_save"
        model.params.auto_save = Bool(value)
    elseif name == "save_filename"
        model.params.save_filename = String(value)
    elseif name == "verbose"
        model.params.verbose = Bool(value)
    else
        throw(MOI.UnsupportedAttribute(param))
    end
    return
end

function MOI.get(model::Optimizer, param::MOI.RawOptimizerAttribute)
    name = String(param.name)
    if name == "stoptol"
        return model.params.stoptol
    elseif name == "sigma"
        return model.params.sigma
    elseif name == "max_iter"
        return model.params.max_iter
    elseif name == "time_limit"
        return model.params.time_limit
    elseif name == "check_iter"
        return model.params.check_iter
    elseif name == "warm_up"
        return model.params.warm_up
    elseif name == "print_frequency"
        return model.params.print_frequency
    elseif name == "device_number"
        return model.params.device_number
    elseif name == "use_Ruiz_scaling"
        return model.params.use_Ruiz_scaling
    elseif name == "ruiz_iterations"
        return model.params.ruiz_iterations
    elseif name == "use_bc_scaling"
        return model.params.use_bc_scaling
    elseif name == "bc_scaling_norm_type"
        return model.params.bc_scaling_norm_type
    elseif name == "use_l2_scaling"
        return model.params.use_l2_scaling
    elseif name == "use_Pock_Chambolle_scaling"
        return model.params.use_Pock_Chambolle_scaling
    elseif name == "soc_block_scaling_strategy"
        return model.params.soc_block_scaling_strategy
    elseif name == "use_gpu"
        return model.params.use_gpu
    elseif name == "initial_x"
        return model.params.initial_x
    elseif name == "initial_y"
        return model.params.initial_y
    elseif name == "auto_save"
        return model.params.auto_save
    elseif name == "save_filename"
        return model.params.save_filename
    elseif name == "verbose"
        return model.params.verbose
    end
    throw(MOI.UnsupportedAttribute(param))
end

# ========================================
#   Supported constraints and objectives
# ========================================

function MOI.supports_constraint(
    ::Optimizer,
    ::Type{<:Union{MOI.VariableIndex,MOI.ScalarAffineFunction{Float64}}},
    ::Type{<:SCALAR_SETS},
)
    return true
end

function MOI.supports_constraint(
    ::Optimizer,
    ::Type{MOI.VectorAffineFunction{Float64}},
    ::Type{MOI.SecondOrderCone},
)
    return true
end

function MOI.supports_constraint(
    ::Optimizer,
    ::Type{MOI.VectorOfVariables},
    ::Type{MOI.SecondOrderCone},
)
    return true
end

MOI.supports(::Optimizer, ::MOI.ObjectiveSense) = true

# Support linear objective
function MOI.supports(
    ::Optimizer,
    ::MOI.ObjectiveFunction{MOI.ScalarAffineFunction{Float64}},
)
    return true
end

function MOI.supports(
    ::Optimizer,
    ::MOI.ObjectiveFunction{MOI.VariableIndex},
)
    return true
end

# Support quadratic objective
function MOI.supports(
    ::Optimizer,
    ::MOI.ObjectiveFunction{MOI.ScalarQuadraticFunction{Float64}},
)
    return true
end

# =======================
#   `copy_to` function
# =======================

function MOI.copy_to(dest::Optimizer, src::OptimizerCache)
    # This is called with the cache directly
    @assert MOI.is_empty(dest)
    dest.cache = src
    dest.index_map = _index_map(src)
    return dest.index_map
end

function MOI.copy_to(
    dest::Optimizer,
    src::MOI.Utilities.UniversalFallback{OptimizerCache},
)
    # Throw error if there are unsupported constraints
    MOI.Utilities.throw_unsupported(src)
    return MOI.copy_to(dest, src.model)
end

function MOI.copy_to(dest::Optimizer, src::MOI.ModelLike)
    # For general MOI models, copy to cache first
    cache = OptimizerCache()
    index_map = MOI.copy_to(cache, src)
    
    # Copy from cache to optimizer
    MOI.copy_to(dest, cache)
    
    return index_map
end

# Helper function to create index map from cache
function _index_map(src::OptimizerCache)
    index_map = MOI.IndexMap()
    for (i, x) in enumerate(MOI.get(src, MOI.ListOfVariableIndices()))
        index_map[x] = MOI.VariableIndex(i)
    end
    for (F, S) in MOI.get(src, MOI.ListOfConstraintTypesPresent())
        for ci in MOI.get(src, MOI.ListOfConstraintIndices{F,S}())
            index_map[ci] = MOI.ConstraintIndex{F,S}(ci.value)
        end
    end
    return index_map
end

# ===============================
#   Optimize and post-optimize
# ===============================

const MOIRow = Tuple{Vector{Int},Vector{Float64},Float64}

function _moi_soc_variable_blocks(src::OptimizerCache, num_variables::Int)
    used_in_soc_block = falses(num_variables)
    soc_variable_blocks = Tuple{Any,Vector{Int}}[]

    for constraint_index in MOI.get(src, MOI.ListOfConstraintIndices{MOI.VectorOfVariables,MOI.SecondOrderCone}())
        function_data = MOI.get(src, MOI.ConstraintFunction(), constraint_index)
        cone_set = MOI.get(src, MOI.ConstraintSet(), constraint_index)
        variable_indices = [variable.value for variable in function_data.variables]

        length(variable_indices) == MOI.dimension(cone_set) || error("SOC variable constraint dimension mismatch.")
        length(unique(variable_indices)) == length(variable_indices) || error("A variable appears twice in one SOC variable constraint.")
        for variable_index in variable_indices
            1 <= variable_index <= num_variables || error("Invalid variable index in SOC variable constraint.")
            !used_in_soc_block[variable_index] || error("A variable cannot appear in multiple SOC variable constraints.")
            used_in_soc_block[variable_index] = true
        end

        push!(soc_variable_blocks, (constraint_index, variable_indices))
    end

    boxed_variables = [variable_index for variable_index in 1:num_variables if !used_in_soc_block[variable_index]]
    solver_to_variable = copy(boxed_variables)
    for (_, variable_indices) in soc_variable_blocks
        append!(solver_to_variable, variable_indices)
    end

    variable_to_solver = zeros(Int, num_variables)
    for (solver_index, variable_index) in enumerate(solver_to_variable)
        variable_to_solver[variable_index] = solver_index
    end

    soc_var_idx = Int[]
    if isempty(soc_variable_blocks)
        push!(soc_var_idx, num_variables + 1)
    else
        block_start = length(boxed_variables) + 1
        for (_, variable_indices) in soc_variable_blocks
            push!(soc_var_idx, block_start)
            block_start += length(variable_indices)
        end
        push!(soc_var_idx, num_variables + 1)
    end

    constraint_var_map = Dict{Any,Vector{Int}}()
    for (constraint_index, variable_indices) in soc_variable_blocks
        constraint_var_map[constraint_index] = [variable_to_solver[variable_index] for variable_index in variable_indices]
    end

    return variable_to_solver, solver_to_variable, soc_var_idx, constraint_var_map
end

function _moi_objective_data(src::OptimizerCache, num_variables::Int, variable_to_solver::Vector{Int})
    objective_type = MOI.get(src, MOI.ObjectiveFunctionType())
    objective_sense = MOI.get(src, MOI.ObjectiveSense())

    linear_objective = zeros(num_variables)
    objective_constant = 0.0
    q_row_indices = Int[]
    q_col_indices = Int[]
    q_values = Float64[]

    if objective_type == MOI.ScalarQuadraticFunction{Float64}
        objective_function = MOI.get(src, MOI.ObjectiveFunction{MOI.ScalarQuadraticFunction{Float64}}())
        for term in objective_function.affine_terms
            solver_index = variable_to_solver[term.variable.value]
            linear_objective[solver_index] += term.coefficient
        end
        for term in objective_function.quadratic_terms
            row_index = variable_to_solver[term.variable_1.value]
            col_index = variable_to_solver[term.variable_2.value]
            coefficient = term.coefficient
            if row_index == col_index
                push!(q_row_indices, row_index)
                push!(q_col_indices, col_index)
                push!(q_values, coefficient)
            else
                push!(q_row_indices, row_index)
                push!(q_col_indices, col_index)
                push!(q_values, 2.0 * coefficient)
                push!(q_row_indices, col_index)
                push!(q_col_indices, row_index)
                push!(q_values, 2.0 * coefficient)
            end
        end
        objective_constant = objective_function.constant
    elseif objective_type == MOI.ScalarAffineFunction{Float64}
        objective_function = MOI.get(src, MOI.ObjectiveFunction{MOI.ScalarAffineFunction{Float64}}())
        for term in objective_function.terms
            solver_index = variable_to_solver[term.variable.value]
            linear_objective[solver_index] += term.coefficient
        end
        objective_constant = objective_function.constant
    elseif objective_type == MOI.VariableIndex
        objective_variable = MOI.get(src, MOI.ObjectiveFunction{MOI.VariableIndex}())
        linear_objective[variable_to_solver[objective_variable.value]] += 1.0
    else
        error("Unsupported objective function type: $objective_type")
    end

    quadratic_objective = isempty(q_row_indices) ? spzeros(num_variables, num_variables) :
        sparse(q_row_indices, q_col_indices, q_values, num_variables, num_variables)

    if objective_sense == MOI.MAX_SENSE
        quadratic_objective = -quadratic_objective
        linear_objective = -linear_objective
        objective_constant = -objective_constant
    end

    return quadratic_objective, linear_objective, objective_constant, objective_sense
end

function _moi_scalar_affine_row(
    function_data::MOI.ScalarAffineFunction{Float64},
    variable_to_solver::Vector{Int},
    multiplier::Float64,
    rhs_value::Float64,
)
    columns = Int[]
    values = Float64[]
    for term in function_data.terms
        push!(columns, variable_to_solver[term.variable.value])
        push!(values, multiplier * term.coefficient)
    end
    return (columns, values, rhs_value)::MOIRow
end

function _moi_add_scalar_affine_rows!(
    equality_rows::Vector{MOIRow},
    inequality_rows::Vector{MOIRow},
    pending_row_map::Dict{Any,Any},
    constraint_index,
    function_data::MOI.ScalarAffineFunction{Float64},
    set_data,
    variable_to_solver::Vector{Int},
)
    if set_data isa MOI.EqualTo{Float64}
        push!(equality_rows, _moi_scalar_affine_row(function_data, variable_to_solver, 1.0, set_data.value - function_data.constant))
        pending_row_map[constraint_index] = (:eq, length(equality_rows))
    elseif set_data isa MOI.GreaterThan{Float64}
        push!(inequality_rows, _moi_scalar_affine_row(function_data, variable_to_solver, 1.0, set_data.lower - function_data.constant))
        pending_row_map[constraint_index] = (:ineq, length(inequality_rows))
    elseif set_data isa MOI.LessThan{Float64}
        push!(inequality_rows, _moi_scalar_affine_row(function_data, variable_to_solver, -1.0, function_data.constant - set_data.upper))
        pending_row_map[constraint_index] = (:ineq, length(inequality_rows))
    elseif set_data isa MOI.Interval{Float64}
        if set_data.lower == set_data.upper
            push!(equality_rows, _moi_scalar_affine_row(function_data, variable_to_solver, 1.0, set_data.lower - function_data.constant))
            pending_row_map[constraint_index] = (:eq, length(equality_rows))
        else
            push!(inequality_rows, _moi_scalar_affine_row(function_data, variable_to_solver, 1.0, set_data.lower - function_data.constant))
            lower_row = length(inequality_rows)
            push!(inequality_rows, _moi_scalar_affine_row(function_data, variable_to_solver, -1.0, function_data.constant - set_data.upper))
            upper_row = length(inequality_rows)
            pending_row_map[constraint_index] = (:ineq_pair, lower_row, upper_row)
        end
    else
        error("Unsupported scalar affine set: $(typeof(set_data))")
    end

    return
end

function _moi_collect_scalar_affine_rows(src::OptimizerCache, variable_to_solver::Vector{Int})
    equality_rows = MOIRow[]
    inequality_rows = MOIRow[]
    pending_row_map = Dict{Any,Any}()

    for set_type in (MOI.EqualTo{Float64}, MOI.GreaterThan{Float64}, MOI.LessThan{Float64}, MOI.Interval{Float64})
        for constraint_index in MOI.get(src, MOI.ListOfConstraintIndices{MOI.ScalarAffineFunction{Float64},set_type}())
            function_data = MOI.get(src, MOI.ConstraintFunction(), constraint_index)
            set_data = MOI.get(src, MOI.ConstraintSet(), constraint_index)
            _moi_add_scalar_affine_rows!(
                equality_rows,
                inequality_rows,
                pending_row_map,
                constraint_index,
                function_data,
                set_data,
                variable_to_solver,
            )
        end
    end

    return equality_rows, inequality_rows, pending_row_map
end

function _moi_vector_affine_rows(
    function_data::MOI.VectorAffineFunction{Float64},
    cone_set::MOI.SecondOrderCone,
    variable_to_solver::Vector{Int},
)
    dimension = MOI.dimension(cone_set)
    length(function_data.constants) == dimension || error("SOC affine constraint dimension mismatch.")
    rows = MOIRow[]
    for output_index in 1:dimension
        push!(rows, (Int[], Float64[], -function_data.constants[output_index]))
    end
    for term in function_data.terms
        output_index = term.output_index
        1 <= output_index <= dimension || error("Invalid SOC affine output index.")
        scalar_term = term.scalar_term
        push!(rows[output_index][1], variable_to_solver[scalar_term.variable.value])
        push!(rows[output_index][2], scalar_term.coefficient)
    end
    return rows
end

function _moi_collect_soc_affine_rows(src::OptimizerCache, variable_to_solver::Vector{Int}, linear_row_count::Int)
    soc_rows = MOIRow[]
    soc_dimensions = Int[]
    soc_row_map = Dict{Any,Any}()

    for constraint_index in MOI.get(src, MOI.ListOfConstraintIndices{MOI.VectorAffineFunction{Float64},MOI.SecondOrderCone}())
        function_data = MOI.get(src, MOI.ConstraintFunction(), constraint_index)
        cone_set = MOI.get(src, MOI.ConstraintSet(), constraint_index)
        block_rows = _moi_vector_affine_rows(function_data, cone_set, variable_to_solver)
        block_start = linear_row_count + length(soc_rows) + 1
        append!(soc_rows, block_rows)
        push!(soc_dimensions, length(block_rows))
        soc_row_map[constraint_index] = block_start:(block_start + length(block_rows) - 1)
    end

    return soc_rows, soc_dimensions, soc_row_map
end

function _moi_add_sparse_rows!(row_indices::Vector{Int}, col_indices::Vector{Int}, values::Vector{Float64}, rows::Vector{MOIRow}, start_row::Int)
    for (row_offset, row_data) in enumerate(rows)
        row_index = start_row + row_offset - 1
        columns, coefficients, _ = row_data
        for (column_index, coefficient) in zip(columns, coefficients)
            push!(row_indices, row_index)
            push!(col_indices, column_index)
            push!(values, coefficient)
        end
    end
    return
end

function _moi_resolve_row_map(pending_row_map::Dict{Any,Any}, soc_row_map::Dict{Any,Any}, number_eq::Int)
    row_map = Dict{Any,Any}()
    for (constraint_index, row_info) in pending_row_map
        if row_info[1] == :eq
            row_map[constraint_index] = row_info[2]
        elseif row_info[1] == :ineq
            row_map[constraint_index] = number_eq + row_info[2]
        elseif row_info[1] == :ineq_pair
            row_map[constraint_index] = (number_eq + row_info[2], number_eq + row_info[3])
        end
    end
    merge!(row_map, soc_row_map)
    return row_map
end

function _moi_cache_to_socp_model(dest::Optimizer, src::OptimizerCache)
    num_variables = length(MOI.get(src, MOI.ListOfVariableIndices()))
    variable_to_solver, solver_to_variable, soc_var_idx, constraint_var_map =
        _moi_soc_variable_blocks(src, num_variables)

    quadratic_objective, linear_objective, objective_constant, objective_sense =
        _moi_objective_data(src, num_variables, variable_to_solver)

    equality_rows, inequality_rows, pending_row_map = _moi_collect_scalar_affine_rows(src, variable_to_solver)
    number_eq = length(equality_rows)
    number_ineq = length(inequality_rows)
    linear_row_count = number_eq + number_ineq
    soc_rows, soc_dimensions, soc_row_map = _moi_collect_soc_affine_rows(src, variable_to_solver, linear_row_count)

    all_rows = vcat(equality_rows, inequality_rows, soc_rows)
    row_indices = Int[]
    col_indices = Int[]
    values = Float64[]
    _moi_add_sparse_rows!(row_indices, col_indices, values, equality_rows, 1)
    _moi_add_sparse_rows!(row_indices, col_indices, values, inequality_rows, number_eq + 1)
    _moi_add_sparse_rows!(row_indices, col_indices, values, soc_rows, linear_row_count + 1)

    num_rows = length(all_rows)
    constraint_matrix = isempty(row_indices) ? spzeros(num_rows, num_variables) :
        sparse(row_indices, col_indices, values, num_rows, num_variables)
    rhs = [row_data[3] for row_data in all_rows]

    soc_con_idx = Int[]
    if isempty(soc_dimensions)
        push!(soc_con_idx, linear_row_count + 1)
    else
        block_start = linear_row_count + 1
        for dimension in soc_dimensions
            push!(soc_con_idx, block_start)
            block_start += dimension
        end
        push!(soc_con_idx, num_rows + 1)
    end

    lower_bounds = [src.variables.lower[variable_index] for variable_index in solver_to_variable]
    upper_bounds = [src.variables.upper[variable_index] for variable_index in solver_to_variable]

    dest.variable_to_solver = variable_to_solver
    dest.solver_to_variable = solver_to_variable
    dest.constraint_row_map = _moi_resolve_row_map(pending_row_map, soc_row_map, number_eq)
    dest.constraint_var_map = constraint_var_map
    dest.obj_sense = objective_sense

    return build_from_SOCP_data(
        SparseMatrixCSC{Float64,Int32}(quadratic_objective),
        linear_objective,
        SparseMatrixCSC{Float64,Int32}(constraint_matrix),
        rhs,
        soc_con_idx,
        number_eq,
        number_ineq,
        lower_bounds,
        upper_bounds,
        soc_var_idx;
        obj_constant=objective_constant,
        verbose=dest.params.verbose,
    )
end

function MOI.optimize!(dest::Optimizer, src::OptimizerCache)
    # Set verbose: if silent mode is set, disable verbose regardless of parameter
    if dest.silent
        dest.params.verbose = false
    end
    
    model = _moi_cache_to_socp_model(dest, src)
    dest.results = optimize(model, dest.params)
    
    return
end

function MOI.optimize!(model::Optimizer)
    # Extract from stored cache
    if model.cache === nothing
        error("No problem has been loaded. Use JuMP.Model(HPRSOCP.Optimizer) and build your model first.")
    end
    
    MOI.optimize!(model, model.cache)
    return
end

# Solve time
function MOI.get(model::Optimizer, ::MOI.SolveTimeSec)
    if model.results === nothing
        return 0.0
    end
    return model.results.time
end

# Objective value
function MOI.get(model::Optimizer, attr::MOI.ObjectiveValue)
    MOI.check_result_index_bounds(model, attr)
    if model.results === nothing
        error("No results available. Call optimize! first.")
    end
    value = model.results.primal_obj
    # Convert back if this was a maximization problem
    if model.obj_sense == MOI.MAX_SENSE
        value = -value
    end
    return value
end

# Number of variables
function MOI.get(model::Optimizer, ::MOI.NumberOfVariables)
    if model.results === nothing
        return 0
    end
    return length(model.results.x)
end

function _moi_solver_index(model::Optimizer, variable::MOI.VariableIndex)
    return isempty(model.variable_to_solver) ? variable.value : model.variable_to_solver[variable.value]
end

function _moi_variable_primal(model::Optimizer, variable::MOI.VariableIndex)
    return model.results.x[_moi_solver_index(model, variable)]
end

function _moi_variable_z(model::Optimizer, variable::MOI.VariableIndex)
    return model.results.z[_moi_solver_index(model, variable)]
end

function _moi_eval_scalar_affine(model::Optimizer, function_data::MOI.ScalarAffineFunction{Float64})
    value = function_data.constant
    for term in function_data.terms
        value += term.coefficient * _moi_variable_primal(model, term.variable)
    end
    return value
end

function _moi_eval_vector_affine(model::Optimizer, function_data::MOI.VectorAffineFunction{Float64})
    values = copy(function_data.constants)
    for term in function_data.terms
        scalar_term = term.scalar_term
        values[term.output_index] += scalar_term.coefficient * _moi_variable_primal(model, scalar_term.variable)
    end
    return values
end

function _moi_constraint_variable(model::Optimizer, constraint_index)
    if model.cache === nothing
        return MOI.VariableIndex(constraint_index.value)
    end
    return MOI.get(model.cache, MOI.ConstraintFunction(), constraint_index)
end

function _moi_adjust_for_sense(model::Optimizer, value)
    return model.obj_sense == MOI.MAX_SENSE ? -value : value
end

# ===============================
#   Termination and Result Status
# ===============================

function MOI.get(model::Optimizer, ::MOI.TerminationStatus)
    if model.results === nothing
        return MOI.OPTIMIZE_NOT_CALLED
    end
    
    if model.results.status == "OPTIMAL"
        return MOI.OPTIMAL
    elseif model.results.status == "MAX_ITER"
        return MOI.ITERATION_LIMIT
    elseif model.results.status == "TIME_LIMIT"
        return MOI.TIME_LIMIT
    else
        return MOI.OTHER_ERROR
    end
end

function MOI.get(model::Optimizer, ::MOI.RawStatusString)
    if model.results === nothing
        return "OPTIMIZE_NOT_CALLED"
    end
    return model.results.status
end

function MOI.get(model::Optimizer, ::MOI.ResultCount)
    if model.results === nothing
        return 0
    end
    # HPRSOCP always returns a result if it has run
    return model.results.status == "OPTIMAL" ? 1 : 0
end

function MOI.get(model::Optimizer, attr::MOI.PrimalStatus)
    if attr.result_index != 1
        return MOI.NO_SOLUTION
    end
    if model.results === nothing
        return MOI.NO_SOLUTION
    end
    if model.results.status == "OPTIMAL"
        return MOI.FEASIBLE_POINT
    end
    return MOI.UNKNOWN_RESULT_STATUS
end

function MOI.get(model::Optimizer, attr::MOI.DualStatus)
    if attr.result_index != 1
        return MOI.NO_SOLUTION
    end
    if model.results === nothing
        return MOI.NO_SOLUTION
    end
    if model.results.status == "OPTIMAL"
        return MOI.FEASIBLE_POINT
    end
    return MOI.UNKNOWN_RESULT_STATUS
end

# ===================
#   Primal solution
# ===================

function MOI.get(
    model::Optimizer,
    attr::MOI.VariablePrimal,
    x::MOI.VariableIndex,
)
    MOI.check_result_index_bounds(model, attr)
    if model.results === nothing
        error("No results available.")
    end
    return _moi_variable_primal(model, x)
end

function MOI.get(
    model::Optimizer,
    attr::MOI.VariablePrimal,
    xs::Vector{MOI.VariableIndex},
)
    MOI.check_result_index_bounds(model, attr)
    if model.results === nothing
        error("No results available.")
    end
    return [_moi_variable_primal(model, variable) for variable in xs]
end

function MOI.get(
    model::Optimizer,
    attr::MOI.ConstraintPrimal,
    c::MOI.ConstraintIndex{MOI.ScalarAffineFunction{Float64},<:SCALAR_SETS},
)
    MOI.check_result_index_bounds(model, attr)
    if model.results === nothing
        error("No results available.")
    end
    if model.cache === nothing
        return MOI.Utilities.get_fallback(model, attr, c)
    end
    function_data = MOI.get(model.cache, MOI.ConstraintFunction(), c)
    return _moi_eval_scalar_affine(model, function_data)
end

function MOI.get(
    model::Optimizer,
    attr::MOI.ConstraintPrimal,
    c::MOI.ConstraintIndex{MOI.VariableIndex,<:SCALAR_SETS},
)
    MOI.check_result_index_bounds(model, attr)
    if model.results === nothing
        error("No results available.")
    end
    return _moi_variable_primal(model, _moi_constraint_variable(model, c))
end

function MOI.get(
    model::Optimizer,
    attr::MOI.ConstraintPrimal,
    constraint_index::MOI.ConstraintIndex{MOI.VectorAffineFunction{Float64},MOI.SecondOrderCone},
)
    MOI.check_result_index_bounds(model, attr)
    if model.results === nothing
        error("No results available.")
    end
    function_data = MOI.get(model.cache, MOI.ConstraintFunction(), constraint_index)
    return _moi_eval_vector_affine(model, function_data)
end

function MOI.get(
    model::Optimizer,
    attr::MOI.ConstraintPrimal,
    constraint_index::MOI.ConstraintIndex{MOI.VectorOfVariables,MOI.SecondOrderCone},
)
    MOI.check_result_index_bounds(model, attr)
    if model.results === nothing
        error("No results available.")
    end
    function_data = MOI.get(model.cache, MOI.ConstraintFunction(), constraint_index)
    return [_moi_variable_primal(model, variable) for variable in function_data.variables]
end

# =================
#   Dual solution
# =================

function MOI.get(
    model::Optimizer,
    attr::MOI.ConstraintDual,
    c::MOI.ConstraintIndex{MOI.ScalarAffineFunction{Float64},<:SCALAR_SETS},
)
    MOI.check_result_index_bounds(model, attr)
    if model.results === nothing
        error("No results available.")
    end
    row_info = get(model.constraint_row_map, c, c.value)
    if row_info isa Tuple
        dual_value = model.results.y[row_info[1]] - model.results.y[row_info[2]]
    else
        dual_value = model.results.y[row_info]
        if c isa MOI.ConstraintIndex{MOI.ScalarAffineFunction{Float64},MOI.LessThan{Float64}}
            dual_value = -dual_value
        end
    end
    return _moi_adjust_for_sense(model, dual_value)
end

function MOI.get(
    model::Optimizer,
    attr::MOI.ConstraintDual,
    c::MOI.ConstraintIndex{MOI.VariableIndex,MOI.LessThan{Float64}},
)
    MOI.check_result_index_bounds(model, attr)
    if model.results === nothing
        error("No results available.")
    end
    variable = _moi_constraint_variable(model, c)
    # Return reduced cost (should be non-positive for upper bounds)
    dual_val = min(0.0, _moi_variable_z(model, variable))
    return _moi_adjust_for_sense(model, dual_val)
end

function MOI.get(
    model::Optimizer,
    attr::MOI.ConstraintDual,
    c::MOI.ConstraintIndex{MOI.VariableIndex,MOI.GreaterThan{Float64}},
)
    MOI.check_result_index_bounds(model, attr)
    if model.results === nothing
        error("No results available.")
    end
    variable = _moi_constraint_variable(model, c)
    # Return reduced cost (should be non-negative for lower bounds)
    dual_val = max(0.0, _moi_variable_z(model, variable))
    return _moi_adjust_for_sense(model, dual_val)
end

function MOI.get(
    model::Optimizer,
    attr::MOI.ConstraintDual,
    c::MOI.ConstraintIndex{
        MOI.VariableIndex,
        <:Union{MOI.Interval{Float64},MOI.EqualTo{Float64}},
    },
)
    MOI.check_result_index_bounds(model, attr)
    if model.results === nothing
        error("No results available.")
    end
    variable = _moi_constraint_variable(model, c)
    # For interval and equality constraints, return the full reduced cost
    dual_val = _moi_variable_z(model, variable)
    return _moi_adjust_for_sense(model, dual_val)
end

function MOI.get(
    model::Optimizer,
    attr::MOI.ConstraintDual,
    constraint_index::MOI.ConstraintIndex{MOI.VectorAffineFunction{Float64},MOI.SecondOrderCone},
)
    MOI.check_result_index_bounds(model, attr)
    if model.results === nothing
        error("No results available.")
    end
    row_range = model.constraint_row_map[constraint_index]
    return _moi_adjust_for_sense(model, collect(model.results.y[row_range]))
end

function MOI.get(
    model::Optimizer,
    attr::MOI.ConstraintDual,
    constraint_index::MOI.ConstraintIndex{MOI.VectorOfVariables,MOI.SecondOrderCone},
)
    MOI.check_result_index_bounds(model, attr)
    if model.results === nothing
        error("No results available.")
    end
    solver_indices = model.constraint_var_map[constraint_index]
    return _moi_adjust_for_sense(model, [model.results.z[solver_index] for solver_index in solver_indices])
end
