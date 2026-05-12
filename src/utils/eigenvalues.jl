# This file is included by ../utils.jl.

# Eigenvalue Estimation via Power Iteration
# ============================================================================
#
# These functions estimate the largest eigenvalue of matrices A'A and Q using
# the power iteration method. They are used to set algorithm step sizes.
#
# UNIFIED DESIGN VIA MULTIPLE DISPATCH:
# -------------------------------------
# These functions use Julia's multiple dispatch to provide a unified interface
# that automatically selects CPU or GPU implementations based on matrix types:
#   - CPU: SparseMatrixCSC matrices dispatch to CPU implementations
#   - GPU: CuSparseMatrixCSR matrices dispatch to GPU implementations
#
# The implementations use unified operations (unified_norm, unified_dot,
# unified_mul!) which dispatch to device-specific kernels at compile time
# with zero runtime overhead. This is the same pattern used throughout the
# main solver algorithm.
#
# BENEFITS:
# ---------
# 1. **Single Interface**: Algorithm code calls power_iteration_A/Q without
#    needing to know about CPU vs GPU
# 2. **Type Safety**: Compiler ensures correct device is used
# 3. **Zero Overhead**: Multiple dispatch resolves at compile time
# 4. **Maintainability**: Algorithm logic written once, not duplicated
#
# ============================================================================

"""
    power_iteration_A(A, AT, max_iterations=5000, tolerance=1e-4)

Estimate the largest eigenvalue of A'A using power iteration.

Automatically dispatches to CPU or GPU implementation based on matrix types:
- CPU: `A::SparseMatrixCSC`, `AT::SparseMatrixCSC`
- GPU: `A::CuSparseMatrixCSR`, `AT::CuSparseMatrixCSR`

# Arguments
- `A`: Constraint matrix (sparse matrix, CPU or GPU)
- `AT`: Transpose of A (sparse matrix, CPU or GPU)
- `max_iterations::Int`: Maximum number of iterations (default: 5000)
- `tolerance::Float64`: Convergence tolerance (default: 1e-4)

# Returns
- `lambda_max::Float64`: Estimated largest eigenvalue of A'A

# Algorithm
Uses the power iteration method:
1. Start with random vector z
2. Iterate: q = z/‖z‖, compute A'(Aq), λ = q'(A'Aq)
3. Check convergence: ‖A'Aq - λq‖ / (‖A'Aq‖ + λ) < tolerance

# Examples
```julia
# CPU version
A_cpu = sprand(100, 50, 0.1)
AT_cpu = A_cpu'
λ = power_iteration_A(A_cpu, AT_cpu)

# GPU version
A_gpu = CuSparseMatrixCSR(A_cpu)
AT_gpu = CuSparseMatrixCSR(AT_cpu)
λ = power_iteration_A(A_gpu, AT_gpu)
```
"""
function power_iteration_A(ws::HPRSOCP_workspace,
    max_iterations::Int=5000, tolerance::Float64=1e-4)
    A = ws.A
    AT = ws.AT
    seed = 1
    m, n = size(A)

    # Create vectors with appropriate type (Vector for CPU, CuVector for GPU)
    z_init = randn(Random.MersenneTwister(seed), m) .+ 1e-8
    is_gpu = ws isa HPRSOCP_workspace_gpu
    if is_gpu
        spmv_A = ws.spmv_A
        spmv_AT = ws.spmv_AT
    else
        spmv_A = nothing
        spmv_AT = nothing
    end
    z = is_gpu ? CuVector(z_init) : z_init
    q = similar(z)
    ATq = similar(z, n)

    lambda_max = 1.0
    error = 1.0

    for i in 1:max_iterations
        q .= z
        q ./= unified_norm(q)
        # Use preprocessed structures if available (GPU only)
        if spmv_AT !== nothing
            unified_mul!(ATq, AT, q, spmv_AT)
        else
            unified_mul!(ATq, AT, q)
        end
        if spmv_A !== nothing
            unified_mul!(z, A, ATq, spmv_A)
        else
            unified_mul!(z, A, ATq)
        end
        lambda_max = unified_dot(q, z)
        q .= z .- lambda_max .* q
        error = unified_norm(q) / (unified_norm(z) + lambda_max)

        if error < tolerance
            return lambda_max
        end
    end

    println("Power iteration (A) did not converge within the specified tolerance.")
    println("The maximum iteration is ", max_iterations, " and the error is ", error)
    return lambda_max
end

"""
    power_iteration_Q(Q, max_iterations=5000, tolerance=1e-4)

Estimate the largest eigenvalue of Q using power iteration.

Automatically dispatches to CPU or GPU implementation based on Q type:
- CPU: `Q::SparseMatrixCSC` or `Q::AbstractQOperatorCPU`
- GPU: `Q::CuSparseMatrixCSR` or `Q::AbstractQOperator` (GPU operators)

# Arguments
- `Q`: Quadratic term matrix or operator (CPU or GPU)
- `max_iterations::Int`: Maximum number of iterations (default: 5000)
- `tolerance::Float64`: Convergence tolerance (default: 1e-4)

# Returns
- `lambda_max::Float64`: Estimated largest eigenvalue of Q

# Examples
```julia
# CPU sparse matrix
Q_cpu = sprand(100, 100, 0.1)
λ = power_iteration_Q(Q_cpu)

# GPU sparse matrix
Q_gpu = CuSparseMatrixCSR(Q_cpu)
λ = power_iteration_Q(Q_gpu)

# Custom operator (GPU)
op = create_custom_operator_gpu(...)
λ = power_iteration_Q(op)
```
"""
function power_iteration_Q(ws::HPRSOCP_workspace,
    max_iterations::Int=5000, tolerance::Float64=1e-4)
    Q = ws.Q
    seed = 1
    n = get_problem_size(Q)

    # Create vectors with appropriate type based on Q
    z_init = randn(Random.MersenneTwister(seed), n) .+ 1e-8
    is_gpu = ws isa HPRSOCP_workspace_gpu
    if is_gpu
        spmv_Q = ws.spmv_Q
    else
        spmv_Q = nothing
    end
    z = is_gpu ? CuVector(z_init) : z_init
    q = similar(z)

    lambda_max = 1.0
    error = 1.0

    for i in 1:max_iterations
        q .= z
        q ./= unified_norm(q)
        # For sparse matrices, pass spmv_Q if available
        if Q isa CuSparseMatrixCSR
            Qmap!(q, z, Q, spmv_Q)
        elseif Q isa SparseMatrixCSC
            Qmap!(q, z, Q)
        else
            # For operators, they handle preprocessing internally
            Qmap!(q, z, Q)
        end
        lambda_max = unified_dot(q, z)
        q .= z .- lambda_max .* q
        error = unified_norm(q) / (unified_norm(z) + lambda_max)

        if error < tolerance
            return lambda_max
        end
    end

    println("Power iteration (Q) did not converge within the specified tolerance.")
    println("The maximum iteration is ", max_iterations, " and the error is ", error)
    return lambda_max
end

