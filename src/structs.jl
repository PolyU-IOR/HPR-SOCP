# ============================================================================
# CUSPARSE Preprocessing Structures (defined early for operator files)
# ============================================================================

# CUSPARSE SpMV structure for A matrix operations
mutable struct CUSPARSE_spmv_A
    handle::CUDA.CUSPARSE.cusparseHandle_t
    operator::Char
    alpha::Ref{Float64}
    desc_A::CUDA.CUSPARSE.CuSparseMatrixDescriptor
    desc_x_bar::CUDA.CUSPARSE.CuDenseVectorDescriptor
    desc_x_hat::CUDA.CUSPARSE.CuDenseVectorDescriptor
    desc_dx::CUDA.CUSPARSE.CuDenseVectorDescriptor
    desc_tempv::CUDA.CUSPARSE.CuDenseVectorDescriptor  # Added for Amap! generic use
    beta::Ref{Float64}
    desc_Ax::CUDA.CUSPARSE.CuDenseVectorDescriptor
    compute_type::DataType
    alg::CUDA.CUSPARSE.cusparseSpMVAlg_t
    buf::CuArray{UInt8}
end

# CUSPARSE SpMV structure for AT (transpose of A) matrix operations
mutable struct CUSPARSE_spmv_AT
    handle::CUDA.CUSPARSE.cusparseHandle_t
    operator::Char
    alpha::Ref{Float64}
    desc_AT::CUDA.CUSPARSE.CuSparseMatrixDescriptor
    desc_y_bar::CUDA.CUSPARSE.CuDenseVectorDescriptor
    desc_y::CUDA.CUSPARSE.CuDenseVectorDescriptor
    beta::Ref{Float64}
    desc_ATy_bar::CUDA.CUSPARSE.CuDenseVectorDescriptor  # Output for AT*y_bar
    desc_ATy::CUDA.CUSPARSE.CuDenseVectorDescriptor      # Output for AT*y
    compute_type::DataType
    alg::CUDA.CUSPARSE.cusparseSpMVAlg_t
    buf::CuArray{UInt8}
end

# CUSPARSE SpMV structure for Q matrix operations (when Q is a sparse matrix)
mutable struct CUSPARSE_spmv_Q
    handle::CUDA.CUSPARSE.cusparseHandle_t
    operator::Char
    alpha::Ref{Float64}
    desc_Q::CUDA.CUSPARSE.CuSparseMatrixDescriptor
    desc_w::CUDA.CUSPARSE.CuDenseVectorDescriptor
    desc_w_bar::CUDA.CUSPARSE.CuDenseVectorDescriptor  # Added for second Qmap! call
    beta::Ref{Float64}
    desc_Qw::CUDA.CUSPARSE.CuDenseVectorDescriptor
    desc_Qw_bar::CUDA.CUSPARSE.CuDenseVectorDescriptor  # Added for second Qmap! call
    compute_type::DataType
    alg::CUDA.CUSPARSE.cusparseSpMVAlg_t
    buf::CuArray{UInt8}
end

# ============================================================================
# Q Operator Types and Interfaces
# ============================================================================
# Q operators are defined in separate files under Q_operators/
# Custom operator types can add their own files as needed.

include("Q_operators/Q_operator_interface.jl")  # Base interface and abstract types
include("Q_operators/sparse_matrix_operator.jl") # Standard sparse matrix QP

# ============================================================================
# Type Abstractions for Unified CPU/GPU Implementation
# ============================================================================

"""
Abstract base types to enable unified function signatures for both CPU and GPU.
Following the same pattern as the Q operator interface (AbstractQOperator, AbstractQOperatorCPU).
"""

# Abstract workspace type - enables unified algorithm functions
abstract type HPRSOCP_workspace end

# Abstract problem data type - enables unified problem handling
abstract type HPRSOCP_QP_info end

# Abstract scaling info type - enables unified scaling functions
abstract type HPRSOCP_scaling end

# Abstract saved state type - enables unified auto-save functionality
abstract type HPRSOCP_saved_state end

# Type unions for array/vector operations
const GPUOrCPUVector{T} = Union{Vector{T},CuVector{T}}
const GPUOrCPUSparseCSR{T,I} = Union{SparseMatrixCSC{T,I},CuSparseMatrixCSR{T,I}}

# ============================================================================
# Problem Data Structures
# ============================================================================

# This struct stores the problem data.
mutable struct QP_info_cpu <: HPRSOCP_QP_info
    """
        Q::QTypeCPU
            The Q matrix/operator. Can be:
            - SparseMatrixCSC{Float64,Int32}: Standard QP with explicit Q matrix
            - AbstractQOperatorCPU: Operator-based Q supplied by the caller
            Use to_gpu(qp.Q) to transfer to GPU.

        c::Vector{Float64}
            The linear coefficient vector in the objective function.

        A::SparseMatrixCSC{Float64,Int32}
            The constraint matrix in CSC format.

        AT::SparseMatrixCSC{Float64,Int32}
            The transpose of the constraint matrix `A` in CSC format.

        AL::Vector{Float64}
            The lower bounds for the linear constraints.

        AU::Vector{Float64}
            The upper bounds for the linear constraints.

        l::Vector{Float64}
            The lower bounds for the decision variables.

        u::Vector{Float64}
            The upper bounds for the decision variables.

        obj_constant::Float64
            The constant term in the objective function.

        diag_Q::Vector{Float64}
            The diagonal elements of the matrix `Q`.

    Q_is_diag::Bool
        Indicates whether the matrix `Q` is diagonal.
    """
    Q::QTypeCPU  # Sparse matrix or CPU operator
    c::Vector{Float64}
    A::SparseMatrixCSC{Float64,Int32}
    AT::SparseMatrixCSC{Float64,Int32}
    soc_rhs::Vector{Float64}
    soc_rhs_full::Vector{Float64}
    AL::Vector{Float64}
    AU::Vector{Float64}
    SOC_con_idx::Vector{Int}
    number_eq::Int
    number_ineq::Int
    l::Vector{Float64}
    u::Vector{Float64}
    SOC_var_idx::Vector{Int}
    number_lu_x::Int
    obj_constant::Float64
end

# This struct stores the problem data for GPU computations.
# Q can be either a sparse matrix or a caller-supplied Q operator.
mutable struct QP_info_gpu <: HPRSOCP_QP_info
    Q::QType  # Union{CuSparseMatrixCSR{Float64,Int32}, AbstractQOperator}
    c::CuVector{Float64}
    A::CuSparseMatrixCSR{Float64,Int32}
    AT::CuSparseMatrixCSR{Float64,Int32}
    soc_rhs::CuVector{Float64}
    soc_rhs_full::CuVector{Float64}
    AL::CuVector{Float64}
    AU::CuVector{Float64}
    SOC_con_idx::CuVector{Int}
    number_eq::Int
    number_ineq::Int
    l::CuVector{Float64}
    u::CuVector{Float64}
    SOC_var_idx::CuVector{Int}
    number_lu_x::Int
    obj_constant::Float64
end

# This struct stores the scaling information.
mutable struct Scaling_info_cpu <: HPRSOCP_scaling
    l_org::Vector{Float64}
    u_org::Vector{Float64}
    row_norm::Vector{Float64}
    col_norm::Vector{Float64}
    b_scale::Float64
    c_scale::Float64
    norm_b::Float64
    norm_c::Float64
    norm_b_org::Float64
    norm_c_org::Float64
    SOC_con_scale::Vector{Float64}
    SOC_var_scale::Vector{Float64}
end

# This struct stores the scaling information for GPU computations.
mutable struct Scaling_info_gpu <: HPRSOCP_scaling
    l_org::CuVector{Float64}
    u_org::CuVector{Float64}
    row_norm::CuVector{Float64}
    col_norm::CuVector{Float64}
    b_scale::Float64
    c_scale::Float64
    norm_b::Float64
    norm_c::Float64
    norm_b_org::Float64
    norm_c_org::Float64
    SOC_con_scale::CuVector{Float64}
    SOC_var_scale::CuVector{Float64}
end

# Internal safety factor used when inflating power-iteration eigenvalue estimates.
const DEFAULT_EIG_FACTOR = 1.05

# This struct contains parameters for the HPR-SOCP solver.
mutable struct HPRSOCP_parameters
    """
        stoptol::Float64
            Stopping tolerance for the algorithm; determines convergence accuracy.
        sigma::Float64
            Initial penalty parameter used in the algorithm.
        max_iter::Int
            Maximum number of iterations allowed.
        time_limit::Float64
            Maximum allowed runtime in seconds.
        check_iter::Int
            Frequency (in iterations) to check for convergence or perform other checks.
        warm_up::Bool
            If true, enables a warm-up phase before the main algorithm starts.
        print_frequency::Int
            Frequency (in iterations) for printing progress or logging information.
        device_number::Int32
            Identifier for the computational device (e.g., GPU device number 0 1 2 3).
        use_Ruiz_scaling::Bool
            If true, applies Ruiz scaling to the problem data.
        ruiz_iterations::Int
            Number of Ruiz scaling passes.
        use_bc_scaling::Bool
            If true, applies bc scaling.
        bc_scaling_norm_type::Symbol
            Scalar summary used for b/c scaling (`:l2`, `:rms`, or `:linf`).
        use_l2_scaling::Bool
            If true, applies L2-norm based scaling.
        use_Pock_Chambolle_scaling::Bool
            If true, applies Pock-Chambolle scaling to the problem data.
        soc_block_scaling_strategy::Symbol
            Shared SOC block aggregation rule (for example `:hybrid` or `:phase_taper`).
        initial_x::Union{Vector{Float64},Nothing}
            Initial primal solution (default: nothing).
        initial_y::Union{Vector{Float64},Nothing}
            Initial dual solution (default: nothing).
        auto_save::Bool
            Automatically save best x, y, z, w, and sigma during optimization (default: false).
        save_filename::String
            Filename for auto-save HDF5 file (default: "HPRSOCP_autosave.h5").
        verbose::Bool
            Enable verbose output (default: true).
    """
    stoptol::Float64
    sigma::Float64
    max_iter::Int
    time_limit::Float64
    check_iter::Int
    warm_up::Bool
    print_frequency::Int
    device_number::Int32
    # scaling
    use_Ruiz_scaling::Bool
    ruiz_iterations::Int
    use_bc_scaling::Bool
    bc_scaling_norm_type::Symbol
    use_l2_scaling::Bool
    use_Pock_Chambolle_scaling::Bool
    soc_block_scaling_strategy::Symbol
    # warm-start
    initial_x::Union{Vector{Float64},Nothing}
    initial_y::Union{Vector{Float64},Nothing}
    # auto-save
    auto_save::Bool
    save_filename::String
    # verbose output
    verbose::Bool
    # use GPU or CPU
    use_gpu::Bool
    HPRSOCP_parameters() = new(1e-6, -1, typemax(Int32), 3600.0, 150, false, -1, 0, true, 10, true, :l2, false, true, :phase_taper, nothing, nothing, false, "HPRSOCP_autosave.h5", true, true)
end

# This struct stores the residuals and other metrics during the HPR-SOCP algorithm.
mutable struct HPRSOCP_residuals
    is_updated::Bool
    err_Rp_org_bar::Float64
    err_Rp_linear_org_bar::Float64
    err_Rp_soc_org_bar::Float64
    err_Rd_org_bar::Float64
    KKTx_and_gap_org_bar::Float64
    primal_obj_bar::Float64
    rel_gap_bar::Float64
    dual_obj_bar::Float64

    # Define a default constructor
    HPRSOCP_residuals() = new(false, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
end

# This struct stores the results of the HPR-SOCP algorithm.
mutable struct HPRSOCP_results
    iter::Int                # Total number of iterations performed.
    iter_4::Int              # Number of iterations to get 1e-4 (if applicable).
    iter_6::Int              # Number of iterations to get 1e-6 (if applicable).
    time::Float64            # Total computation time (seconds).
    time_4::Float64          # Computation time spent to get 1e-4 (seconds).
    time_6::Float64          # Computation time spent to get 1e-6 (seconds).
    power_time::Float64      # Time spent on eigenvalue estimation (seconds).
    primal_obj::Float64      # Final value of the primal objective function.
    residuals::Float64       # Final value of the residuals.
    gap::Float64             # Final duality gap.
    status::String      # Status or type of output (e.g., "OPTIMAL", "MAX_ITER", "TIME_LIMIT").
    x::Vector{Float64}       # Solution vector for the primal variables.
    y::Vector{Float64}       # Solution vector for the dual variables (equality/inequality constraints).
    z::Vector{Float64}       # Solution vector for the dual variables (bounds).
    w::Vector{Float64}       # Auxiliary variable vector.
    HPRSOCP_results() = new()
end

# ============================================================================
# Saved State Structures
# ============================================================================

# This struct stores the best-so-far state for auto-save feature (CPU version)
mutable struct HPRSOCP_saved_state_cpu <: HPRSOCP_saved_state
    # Best x found so far (CPU)
    save_x::Vector{Float64}

    # Best y found so far (CPU)
    save_y::Vector{Float64}

    # Best z found so far (CPU)
    save_z::Vector{Float64}

    # Best w found so far (CPU)
    save_w::Vector{Float64}

    # Best sigma value
    save_sigma::Float64

    # Iteration when best state was saved
    save_iter::Int

    # Primal residual at best state
    save_err_Rp::Float64

    # Dual residual at best state
    save_err_Rd::Float64

    # Primal objective at best state
    save_primal_obj::Float64

    # Dual objective at best state
    save_dual_obj::Float64

    # Relative gap at best state
    save_rel_gap::Float64

    # Default constructor
    HPRSOCP_saved_state_cpu() = new()
end

# This struct stores the best-so-far state for auto-save feature (GPU version)
mutable struct HPRSOCP_saved_state_gpu <: HPRSOCP_saved_state
    # Best x found so far (GPU)
    save_x::CuVector{Float64}

    # Best y found so far (GPU)
    save_y::CuVector{Float64}

    # Best z found so far (GPU)
    save_z::CuVector{Float64}

    # Best w found so far (GPU)
    save_w::CuVector{Float64}

    # Best sigma value
    save_sigma::Float64

    # Iteration when best state was saved
    save_iter::Int

    # Primal residual at best state
    save_err_Rp::Float64

    # Dual residual at best state
    save_err_Rd::Float64

    # Primal objective at best state
    save_primal_obj::Float64

    # Dual objective at best state
    save_dual_obj::Float64

    # Relative gap at best state
    save_rel_gap::Float64

    # Default constructor
    HPRSOCP_saved_state_gpu() = new()
end

mutable struct SOC_var_fast_paths_gpu
    size3_starts::CuVector{Int32}
    size4_starts::CuVector{Int32}
    size5_starts::CuVector{Int32}
    size8_starts::CuVector{Int32}
    small_starts::CuVector{Int32}
    small_sizes::CuVector{Int32}
    huge_starts::CuVector{Int32}
    huge_sizes::CuVector{Int32}
    huge_block_starts::CuVector{Int32}
    huge_block_offsets::CuVector{Int32}
    huge_block_cone_ids::CuVector{Int32}
    huge_block_ptr::CuVector{Int32}
    huge_partial_sums::CuVector{Float64}
    huge_t_raw::CuVector{Float64}
    huge_proj_t::CuVector{Float64}
    huge_alpha::CuVector{Float64}
    huge_case::CuVector{Int32}
    large_starts::CuVector{Int32}
    large_sizes::CuVector{Int32}
    generic_starts::CuVector{Int32}
    generic_sizes::CuVector{Int32}
    size3_count::Int
    size4_count::Int
    size5_count::Int
    size8_count::Int
    small_count::Int
    huge_count::Int
    huge_total_blocks::Int
    huge_kernel_mode::Symbol
    large_count::Int
    large_max_size::Int
    generic_count::Int
    SOC_var_fast_paths_gpu() = new()
end

mutable struct SOC_con_fast_paths_gpu
    size3_starts::CuVector{Int32}
    size4_starts::CuVector{Int32}
    size5_starts::CuVector{Int32}
    small_starts::CuVector{Int32}
    small_sizes::CuVector{Int32}
    large_starts::CuVector{Int32}
    large_sizes::CuVector{Int32}
    generic_starts::CuVector{Int32}
    generic_sizes::CuVector{Int32}
    size3_count::Int
    size4_count::Int
    size5_count::Int
    small_count::Int
    large_count::Int
    generic_count::Int
    SOC_con_fast_paths_gpu() = new()
end

# This struct stores the workspace for the HPR-SOCP algorithm on the GPU.
mutable struct HPRSOCP_workspace_gpu <: HPRSOCP_workspace
    w::CuVector{Float64}
    w_hat::CuVector{Float64}
    w_bar::CuVector{Float64}
    dw::CuVector{Float64}
    x::CuVector{Float64}
    x_hat::CuVector{Float64}
    x_bar::CuVector{Float64}
    dx::CuVector{Float64}
    y::CuVector{Float64}
    y_hat::CuVector{Float64}
    y_bar::CuVector{Float64}
    dy::CuVector{Float64}
    z_bar::CuVector{Float64}
    Q::QType  # Can be sparse matrix or Q operator
    A::CuSparseMatrixCSR{Float64,Int32}
    AT::CuSparseMatrixCSR{Float64,Int32}
    soc_rhs::CuVector{Float64}
    soc_rhs_full::CuVector{Float64}
    AL::CuVector{Float64}
    AU::CuVector{Float64}
    SOC_con_idx::CuVector{Int}
    SOC_var_idx::CuVector{Int}
    number_eq::Int
    number_ineq::Int
    number_lu_x::Int
    number_SOC_con::Int
    number_SOC_var::Int
    c::CuVector{Float64}
    l::CuVector{Float64}
    u::CuVector{Float64}
    Rp::CuVector{Float64}
    Rd::CuVector{Float64}
    m::Int
    n::Int
    sigma::Float64
    lambda_max_A::Float64
    lambda_max_Q::Float64
    Ax::CuVector{Float64}
    ATy::CuVector{Float64}
    ATy_bar::CuVector{Float64}
    ATdy::CuVector{Float64}
    QATdy::CuVector{Float64}
    s::CuVector{Float64}
    Qw::CuVector{Float64}
    Qw_hat::CuVector{Float64}
    Qw_bar::CuVector{Float64}
    Qx::CuVector{Float64}
    dQw::CuVector{Float64}
    last_x::CuVector{Float64}
    last_y::CuVector{Float64}
    last_Qw::CuVector{Float64}
    last_w::CuVector{Float64}
    last_ATy::CuVector{Float64}
    tempv::CuVector{Float64}
    Q_is_diag::Bool
    diag_Q::CuVector{Float64}
    fact1::CuVector{Float64}
    fact2::CuVector{Float64}
    fact::CuVector{Float64}
    fact_M::CuVector{Float64}
    SOC_norms_temp::CuVector{Float64}
    soc_con_fast_paths::SOC_con_fast_paths_gpu
    soc_var_fast_paths::SOC_var_fast_paths_gpu
    to_check::Bool
    # CUSPARSE SpMV structures for preprocessed matrix operations
    spmv_A::Union{CUSPARSE_spmv_A,Nothing}  # For A matrix operations (nothing if m=0)
    spmv_AT::Union{CUSPARSE_spmv_AT,Nothing}  # For AT matrix operations (nothing if m=0)
    spmv_Q::Union{CUSPARSE_spmv_Q,Nothing}  # For Q matrix operations (nothing if Q is operator)
    # Saved state for auto_save feature
    saved_state::HPRSOCP_saved_state_gpu
    noq_soc_scratch_aty_mode::Symbol
    noC::Bool
    HPRSOCP_workspace_gpu() = new()
end

# This struct stores the workspace for the HPR-SOCP algorithm on the CPU.
mutable struct HPRSOCP_workspace_cpu <: HPRSOCP_workspace
    w::Vector{Float64}
    w_hat::Vector{Float64}
    w_bar::Vector{Float64}
    dw::Vector{Float64}
    x::Vector{Float64}
    x_hat::Vector{Float64}
    x_bar::Vector{Float64}
    dx::Vector{Float64}
    y::Vector{Float64}
    y_hat::Vector{Float64}
    y_bar::Vector{Float64}
    dy::Vector{Float64}
    z_bar::Vector{Float64}
    Q::QTypeCPU  # Can be sparse matrix or CPU Q operator
    A::SparseMatrixCSC{Float64,Int32}
    AT::SparseMatrixCSC{Float64,Int32}
    soc_rhs::Vector{Float64}
    soc_rhs_full::Vector{Float64}
    AL::Vector{Float64}
    AU::Vector{Float64}
    SOC_con_idx::Vector{Int}
    SOC_var_idx::Vector{Int}
    number_eq::Int
    number_ineq::Int
    number_lu_x::Int
    number_SOC_con::Int
    number_SOC_var::Int
    c::Vector{Float64}
    l::Vector{Float64}
    u::Vector{Float64}
    Rp::Vector{Float64}
    Rd::Vector{Float64}
    m::Int
    n::Int
    sigma::Float64
    lambda_max_A::Float64
    lambda_max_Q::Float64
    Ax::Vector{Float64}
    ATy::Vector{Float64}
    ATy_bar::Vector{Float64}
    ATdy::Vector{Float64}
    QATdy::Vector{Float64}
    s::Vector{Float64}
    Qw::Vector{Float64}
    Qw_hat::Vector{Float64}
    Qw_bar::Vector{Float64}
    Qx::Vector{Float64}
    dQw::Vector{Float64}
    last_x::Vector{Float64}
    last_y::Vector{Float64}
    last_Qw::Vector{Float64}
    last_w::Vector{Float64}
    last_ATy::Vector{Float64}
    tempv::Vector{Float64}
    Q_is_diag::Bool
    diag_Q::Vector{Float64}
    fact1::Vector{Float64}
    fact2::Vector{Float64}
    fact::Vector{Float64}
    fact_M::Vector{Float64}
    SOC_norms_temp::Vector{Float64}
    to_check::Bool
    # Saved state for auto_save feature
    saved_state::HPRSOCP_saved_state_cpu
    noq_soc_scratch_aty_mode::Symbol
    noC::Bool
    HPRSOCP_workspace_cpu() = new()
end

# This struct stores the restart information for the HPR-SOCP algorithm.
mutable struct HPRSOCP_restart
    restart_flag::Int
    first_restart::Bool
    last_gap::Float64
    current_gap::Float64
    save_gap::Float64
    inner::Int
    step::Int
    sufficient::Int
    necessary::Int
    long::Int
    ratio::Int
    times::Int

    weighted_norm::Float64
    best_gap::Float64
    best_kkt::Float64
    best_sigma::Float64
    best_iter::Int
    sigma_correction_active::Bool
    sigma_correction_hold::Int
    sigma_correction_dir::Int
    HPRSOCP_restart() = new()
end
