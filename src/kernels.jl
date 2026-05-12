## Q operator mapping functions for implicit Q representations
## 
## Qmap! implementations are now in their respective operator files:
##   - sparse_matrix_operator.jl: Qmap! for CuSparseMatrixCSR
##
## Users can add custom operators by implementing Qmap! in their operator files:
##   @inline function Qmap!(x::CuVector{Float64}, Qx::CuVector{Float64}, Q::MyOperatorGPU)
##       # Your implementation: Qx .= Q * x
##   end

# ============================================================================
# Matrix-Vector Multiplication Wrappers for A and AT
# ============================================================================
# Similar to Qmap!, these wrappers provide a clean interface for A*x and AT*y
# operations, automatically selecting between preprocessed CUSPARSE or standard
# CUSPARSE based on available workspace structures.
# ============================================================================

"""
    Amap!(x, Ax, A, spmv_A)

Compute Ax = A * x using GPU sparse matrix-vector multiplication.
Automatically selects between preprocessed CUSPARSE (if spmv_A available) or standard CUSPARSE.

# Arguments
- `x`: Input vector (CuVector)
- `Ax`: Output vector (CuVector), will contain A*x
- `A`: Sparse matrix in CSR format (CuSparseMatrixCSR)
- `spmv_A`: Optional preprocessed CUSPARSE structure (can be nothing)
"""
@inline function Amap!(x::CuVector{Float64}, Ax::CuVector{Float64}, 
                       A::CuSparseMatrixCSR{Float64,Int32}, 
                       spmv_A::Union{CUSPARSE_spmv_A,Nothing})
    # Standard CUSPARSE operation (preprocessing doesn't help without pre-created descriptors)
    CUDA.CUSPARSE.mv!('N', 1, A, x, 0, Ax, 'O', CUDA.CUSPARSE.CUSPARSE_SPMV_CSR_ALG2)
end

"""
    Amap!(desc_x, desc_Ax, A, spmv_A)

Compute Ax = A * x using preprocessed CUSPARSE with pre-created descriptors (zero allocation).

# Arguments
- `desc_x`: Pre-created CuDenseVectorDescriptor for input vector
- `desc_Ax`: Pre-created CuDenseVectorDescriptor for output vector
- `A`: Sparse matrix in CSR format (CuSparseMatrixCSR)
- `spmv_A`: Preprocessed CUSPARSE structure (must not be nothing)
"""
@inline function Amap!(desc_x::CUDA.CUSPARSE.CuDenseVectorDescriptor, 
                       desc_Ax::CUDA.CUSPARSE.CuDenseVectorDescriptor,
                       A::CuSparseMatrixCSR{Float64,Int32}, 
                       spmv_A::CUSPARSE_spmv_A)
    # Use preprocessed CUSPARSE with pre-created descriptors - zero allocation!
    CUDA.CUSPARSE.cusparseSpMV(spmv_A.handle, spmv_A.operator, spmv_A.alpha,
        spmv_A.desc_A, desc_x, spmv_A.beta, desc_Ax,
        spmv_A.compute_type, spmv_A.alg, spmv_A.buf)
end

"""
    ATmap!(y, ATy, AT, spmv_AT)

Compute ATy = AT * y using GPU sparse matrix-vector multiplication.
Automatically selects between preprocessed CUSPARSE (if spmv_AT available) or standard CUSPARSE.

# Arguments
- `y`: Input vector (CuVector)
- `ATy`: Output vector (CuVector), will contain AT*y
- `AT`: Sparse matrix in CSR format (CuSparseMatrixCSR)
- `spmv_AT`: Optional preprocessed CUSPARSE structure (can be nothing)
"""
@inline function ATmap!(y::CuVector{Float64}, ATy::CuVector{Float64}, 
                        AT::CuSparseMatrixCSR{Float64,Int32}, 
                        spmv_AT::Union{CUSPARSE_spmv_AT,Nothing})
    # Standard CUSPARSE operation
    CUDA.CUSPARSE.mv!('N', 1, AT, y, 0, ATy, 'O', CUDA.CUSPARSE.CUSPARSE_SPMV_CSR_ALG2)
end

"""
    ATmap!(desc_y, desc_ATy, AT, spmv_AT)

Compute ATy = AT * y using preprocessed CUSPARSE with pre-created descriptors (zero allocation).

# Arguments
- `desc_y`: Pre-created CuDenseVectorDescriptor for input vector
- `desc_ATy`: Pre-created CuDenseVectorDescriptor for output vector
- `AT`: Sparse matrix in CSR format (CuSparseMatrixCSR)
- `spmv_AT`: Preprocessed CUSPARSE structure (must not be nothing)
"""
@inline function ATmap!(desc_y::CUDA.CUSPARSE.CuDenseVectorDescriptor, 
                        desc_ATy::CUDA.CUSPARSE.CuDenseVectorDescriptor,
                        AT::CuSparseMatrixCSR{Float64,Int32}, 
                        spmv_AT::CUSPARSE_spmv_AT)
    # Use preprocessed CUSPARSE with pre-created descriptors - zero allocation!
    CUDA.CUSPARSE.cusparseSpMV(spmv_AT.handle, spmv_AT.operator, spmv_AT.alpha,
        spmv_AT.desc_AT, desc_y, spmv_AT.beta, desc_ATy,
        spmv_AT.compute_type, spmv_AT.alg, spmv_AT.buf)
end

# ============================================================================
# Unified Kernel Wrapper Functions (CPU and GPU)
# ============================================================================
# These functions dispatch based on workspace type, following the Q operator pattern.
# GPU versions call CUDA kernels, CPU versions use optimized loops.
# ============================================================================

@inline _scaled_primal_component_to_original(v_bar::Float64, row_norm_i::Float64, b_scale::Float64) =
    v_bar * row_norm_i * b_scale

@inline _scaled_dual_component_to_original(v_bar::Float64, col_norm_i::Float64, c_scale::Float64) =
    v_bar * col_norm_i * c_scale

"""
    compute_Rd!(ws, sc)

Compute the original-coordinate dual residual directly from scaled data already
stored in the workspace. This path uses `Qx`, not `Qw`.
"""
function compute_Rd!(ws::HPRSOCP_workspace, sc::HPRSOCP_scaling, skip_q::Bool=false)
    # Compute A'y if constraints exist
    if ws.m > 0
        unified_mul!(ws.ATdy, ws.AT, ws.y_bar)
    end
    # Convert each scaled dual component to original coordinates directly.
    if skip_q
        _compute_Rd_noQ_impl!(ws, sc)
    else
        _compute_Rd_impl!(ws, sc)
    end
end

function compute_Rd!(ws::HPRSOCP_workspace, qp::HPRSOCP_QP_info, sc::HPRSOCP_scaling)
    skip_q = !has_quadratic_terms(qp.Q)
    if !skip_q
        if isa(qp.Q, CuSparseMatrixCSR{Float64,Int32}) && isa(ws, HPRSOCP_workspace_gpu)
            Qmap!(ws.x_bar, ws.Qx, qp.Q, ws.spmv_Q)
        else
            Qmap!(ws.x_bar, ws.Qx, qp.Q)
        end
    end
    return compute_Rd!(ws, sc, skip_q)
end

# GPU implementation using kernel
function _compute_Rd_impl!(ws::HPRSOCP_workspace_gpu, sc::Scaling_info_gpu)
    threads, blocks = gpu_launch_config(ws.n)
    if threads > 0
        @cuda threads = threads blocks = blocks compute_Rd_kernel!(ws.ATdy, ws.z_bar, ws.c, ws.Qx, ws.Rd, sc.col_norm, sc.c_scale, ws.n)
    end
end

function _compute_Rd_noQ_impl!(ws::HPRSOCP_workspace_gpu, sc::Scaling_info_gpu)
    threads, blocks = gpu_launch_config(ws.n)
    if threads > 0
        @cuda threads = threads blocks = blocks compute_Rd_noQ_kernel!(
            ws.ATdy, ws.z_bar, ws.c, ws.Rd, sc.col_norm, sc.c_scale, ws.n
        )
    end
end

# CPU implementation using loop
function _compute_Rd_impl!(ws::HPRSOCP_workspace_cpu, sc::Scaling_info_cpu)
    Rd = ws.Rd
    ATdy = ws.ATdy
    z_bar = ws.z_bar
    Qx = ws.Qx
    c = ws.c
    col_norm = sc.col_norm

    @simd for i in eachindex(Rd)
        @inbounds begin
            qx_org = _scaled_dual_component_to_original(Qx[i], col_norm[i], sc.c_scale)
            c_org = _scaled_dual_component_to_original(c[i], col_norm[i], sc.c_scale)
            atdy_org = _scaled_dual_component_to_original(ATdy[i], col_norm[i], sc.c_scale)
            z_org = _scaled_dual_component_to_original(z_bar[i], col_norm[i], sc.c_scale)
            Rd[i] = qx_org + c_org - atdy_org - z_org
            Qx[i] = qx_org
            ATdy[i] = atdy_org
        end
    end
end

function _compute_Rd_noQ_impl!(ws::HPRSOCP_workspace_cpu, sc::Scaling_info_cpu)
    Rd = ws.Rd
    ATdy = ws.ATdy
    z_bar = ws.z_bar
    c = ws.c
    col_norm = sc.col_norm

    @simd for i in eachindex(Rd)
        @inbounds begin
            c_org = _scaled_dual_component_to_original(c[i], col_norm[i], sc.c_scale)
            atdy_org = _scaled_dual_component_to_original(ATdy[i], col_norm[i], sc.c_scale)
            z_org = _scaled_dual_component_to_original(z_bar[i], col_norm[i], sc.c_scale)
            Rd[i] = c_org - atdy_org - z_org
            ATdy[i] = atdy_org
        end
    end
end

"""
    compute_Rp!(ws, sc)

Compute the original-coordinate primal residual directly from scaled `A*xbar`.
"""
function compute_Rp!(ws::HPRSOCP_workspace, sc::HPRSOCP_scaling)
    # Compute Ax = A * x_bar
    unified_mul!(ws.Ax, ws.A, ws.x_bar)
    # Convert scaled primal quantities to original coordinates directly.
    _compute_Rp_impl!(ws, sc)
    _compute_soc_Rp_impl!(ws, sc)
end

# GPU implementation using kernel
function _compute_Rp_impl!(ws::HPRSOCP_workspace_gpu, sc::Scaling_info_gpu)
    threads, blocks = gpu_launch_config(ws.m)
    if threads > 0
        @cuda threads = threads blocks = blocks compute_Rp_kernel!(ws.Rp, ws.AL, ws.AU, ws.Ax, sc.row_norm, sc.b_scale, ws.m)
    end
end

# CPU implementation using loop
function _compute_Rp_impl!(ws::HPRSOCP_workspace_cpu, sc::Scaling_info_cpu)
    AL = ws.AL
    AU = ws.AU
    Ax = ws.Ax
    Rp = ws.Rp
    row_norm = sc.row_norm
    b_scale = sc.b_scale

    @simd for i in eachindex(Rp)
        @inbounds begin
            ax_org = _scaled_primal_component_to_original(Ax[i], row_norm[i], b_scale)
            AL_org = _scaled_primal_component_to_original(AL[i], row_norm[i], b_scale)
            AU_org = _scaled_primal_component_to_original(AU[i], row_norm[i], b_scale)
            ax_proj = min(max(ax_org, AL_org), AU_org)
            Rp[i] = ax_proj - ax_org
            Ax[i] = ax_org
        end
    end
end

function _compute_soc_Rp_impl!(ws::HPRSOCP_workspace_gpu, sc::Scaling_info_gpu)
    if ws.number_SOC_con > 0
        threads, blocks = gpu_launch_config(ws.number_SOC_con)
        if threads > 0
            @cuda threads = threads blocks = blocks compute_Rp_SOC_kernel!(
                ws.Rp, ws.soc_rhs, ws.Ax, ws.SOC_con_idx, sc.row_norm, sc.b_scale, ws.number_SOC_con
            )
        end
    end
end

function _compute_soc_Rp_impl!(ws::HPRSOCP_workspace_cpu, sc::Scaling_info_cpu)
    if ws.number_SOC_con == 0
        return
    end

    for i in 1:ws.number_SOC_con
        start_idx = ws.SOC_con_idx[i]
        end_idx = ws.SOC_con_idx[i+1] - 1
        offset = start_idx - ws.SOC_con_idx[1] + 1
        rhs_t = _scaled_primal_component_to_original(ws.soc_rhs[offset], sc.row_norm[start_idx], sc.b_scale)
        t = ws.Ax[start_idx] - rhs_t
        norm_s = 0.0
        for j in (start_idx + 1):end_idx
            rhs_j = _scaled_primal_component_to_original(
                ws.soc_rhs[offset + (j - start_idx)],
                sc.row_norm[j],
                sc.b_scale,
            )
            s_j = ws.Ax[j] - rhs_j
            norm_s += s_j^2
            ws.Rp[j] = s_j
        end
        norm_s = sqrt(norm_s)
        ws.Rp[start_idx] = t

        if norm_s <= -t
            # Already at maximal violation representation; leave as-is.
        elseif norm_s <= t
            for j in start_idx:end_idx
                ws.Rp[j] = 0.0
            end
        else
            fact = (1 + t / norm_s) / 2
            ws.Rp[start_idx] = (norm_s + t) / 2 - t
            for j in (start_idx + 1):end_idx
                ws.Rp[j] = fact * ws.Rp[j] - ws.Rp[j]
            end
        end
    end
end

# ============================================================================
# GPU Kernel Definitions and Launch Configurations
# ============================================================================
#
# This section contains CUDA kernels that execute on the GPU. These kernels
# MUST remain GPU-specific and should NOT be unified with CPU implementations.
#
# The kernels are organized into the following categories:
#
# 1. Unified Standard QP Kernels (lines ~270-690)
#    - unified_update_zxw1_kernel_full!/partial!: Standard QP variable updates
#    - compute_tempv_unified_kernel!: Compute temporary vectors for updates
#    - unified_update_y_kernel_full!/partial!: Dual variable y updates
#    - unified_update_w2_kernel_full!/partial!: Dual variable w updates
#    - Wrapper functions: unified_update_zxw1_gpu!, unified_update_y_gpu!, unified_update_w2_gpu!
#
# 2. LP-Specific Kernels (lines ~690-1000)
#    - unified_update_zx_gpu!: Update for problems with empty Q (LP problems)
#    - unified_update_y_noQ_gpu!: Dual updates for LP problems
#
# 3. Golden Section Search and Factor Updates (lines ~1000-1160)
#    - golden_Q_diag: GPU version of golden section search for sigma tuning
#    - update_Q_factors_kernel!/gpu!: Update scaling factors for diagonal Q
#
# 4. Scaling and Utility Kernels (lines ~1240-1570)
#    - compute_Rd_kernel!/compute_Rp_kernel!: Residual computation kernels
#    - axpby_gpu!: GPU vector operation (y = a*x + b*y)
#    - compute_row/col_max/sum kernels: CSR matrix statistics
#    - scale_* kernels: Various scaling operations on CSR matrices and vectors
#
# Key Design Principles:
# - Each kernel is optimized for GPU parallelism with minimal thread divergence
# - Kernels use CUDA.@fastmath for performance where numerical stability permits
# - Full vs Partial kernel variants control which intermediate values are stored
# - Custom vs cuSPARSE SpMV modes allow flexible performance tuning
# - Val{} type parameters enable compile-time specialization without runtime overhead
#
# These kernels are called from wrapper functions that handle:
# - Thread/block configuration via gpu_launch_config()
# - Device memory management
# - Kernel parameter setup
# - Integration with the overall solver algorithm
#
# NOTE: Do NOT attempt to unify these with CPU implementations. The CPU versions
# use fundamentally different execution models (vectorized loops vs parallel kernels).
# ============================================================================

const DEFAULT_KERNEL_THREADS = 256
const HUGE_SOC_KERNEL_THREADS = 512
# Must remain divisible by both 3 and 4 for cooperative lane subgroup mapping.
const SOC_CON_TINY_COOP_THREADS = 96
const SOC_CON_SIZE5_COOP_THREADS = 160
# Must remain divisible by 5 for SOC-var size-5 cooperative lane subgroup mapping.
const SOC_VAR_SIZE5_COOP_THREADS = 160

@inline function gpu_launch_config(length::Int)
    @assert length >= 0 "kernel launch length must be non-negative"
    if length == 0
        return 0, 0
    end
    threads = min(DEFAULT_KERNEL_THREADS, max(32, 32 * cld(length, 32)))
    blocks = cld(length, threads)
    return threads, blocks
end

## normal z x w1 y w2 kernels (customized and unified)

# Fully unified kernel that handles all variants in a single implementation:
# 
# SpMV Strategy (use_custom_spmv):
#   - true:  Compute Q*w inline (better for small problems, avoids cuSPARSE overhead)
#   - false: Use pre-computed Qw from cuSPARSE (better for large problems)
#
# Q Matrix Structure (Q_is_diag):
#   - true:  Q is diagonal, use element-wise fact1_vec[i] and fact2_vec[i]
#   - false: Q is general, use scalar fact1_scalar and fact2_scalar
#
# Key optimizations:
#   1. Kernel fusion: combines SpMV + update operations in one kernel
#   2. Minimal branching: conditionals are uniform across warps (no divergence)
#   3. Adaptive: can switch strategies based on problem characteristics
#
# Note: tempv computation is separated into compute_tempv_unified_kernel! for clarity
#
# Full version: computes all intermediate values
CUDA.@fastmath @inline function unified_update_zxw1_kernel_full!(::Val{UseCustom}, ::Val{IsDiag},
    dx::CuDeviceVector{Float64},
    rowPtrQ::CuDeviceVector{Int32}, colValQ::CuDeviceVector{Int32}, nzValQ::CuDeviceVector{Float64},
    w_bar::CuDeviceVector{Float64}, w::CuDeviceVector{Float64},
    z_bar::CuDeviceVector{Float64}, x_bar::CuDeviceVector{Float64}, x_hat::CuDeviceVector{Float64},
    last_x::CuDeviceVector{Float64}, x::CuDeviceVector{Float64},
    Qw::CuDeviceVector{Float64}, ATy::CuDeviceVector{Float64}, c::CuDeviceVector{Float64},
    l::CuDeviceVector{Float64}, u::CuDeviceVector{Float64},
    sigma::Float64, fact1_scalar::Float64, fact2_scalar::Float64,
    fact1_vec::CuDeviceVector{Float64}, fact2_vec::CuDeviceVector{Float64},
    Halpern_fact1::Float64, Halpern_fact2::Float64, n::Int) where {UseCustom,IsDiag}
    i = threadIdx().x + (blockDim().x * (blockIdx().x - 1))
    @inbounds if i <= n
        qw_val = if UseCustom
            startQ = rowPtrQ[i]
            stopQ = rowPtrQ[i+1] - 1
            acc = 0.0
            @inbounds for k in startQ:stopQ
                acc += nzValQ[k] * w[colValQ[k]]
            end
            Qw[i] = acc
            acc
        else
            Qw[i]
        end

        atyi = ATy[i]
        c_i = c[i]
        x_i = x[i]
        last_x_i = last_x[i]
        l_i = l[i]
        u_i = u[i]
        w_i = w[i]

        tmp = -qw_val + atyi - c_i
        z_raw = x_i + sigma * tmp
        x_bar_i = min(max(z_raw, l_i), u_i)

        x_hat_i = 2.0 * x_bar_i - x_i
        x_new = muladd(Halpern_fact2, x_hat_i, Halpern_fact1 * last_x_i)

        w_bar_i = if IsDiag
            fact1_i = fact1_vec[i]
            fact2_i = fact2_vec[i]
            muladd(fact1_i, w_i, fact2_i * x_hat_i)
        else
            muladd(fact1_scalar, w_i, fact2_scalar * x_hat_i)
        end

        dx_val = x_bar_i - x_i
        dx[i] = dx_val
        x_bar[i] = x_bar_i
        z_bar[i] = (x_bar_i - z_raw) / sigma
        x[i] = x_new
        x_hat[i] = x_hat_i
        w_bar[i] = w_bar_i
    end
    return
end

# Partial version: skips intermediate writes
CUDA.@fastmath @inline function unified_update_zxw1_kernel_partial!(::Val{UseCustom}, ::Val{IsDiag},
    dx::CuDeviceVector{Float64},
    rowPtrQ::CuDeviceVector{Int32}, colValQ::CuDeviceVector{Int32}, nzValQ::CuDeviceVector{Float64},
    w_bar::CuDeviceVector{Float64}, w::CuDeviceVector{Float64},
    z_bar::CuDeviceVector{Float64}, x_bar::CuDeviceVector{Float64}, x_hat::CuDeviceVector{Float64},
    last_x::CuDeviceVector{Float64}, x::CuDeviceVector{Float64},
    Qw::CuDeviceVector{Float64}, ATy::CuDeviceVector{Float64}, c::CuDeviceVector{Float64},
    l::CuDeviceVector{Float64}, u::CuDeviceVector{Float64},
    sigma::Float64, fact1_scalar::Float64, fact2_scalar::Float64,
    fact1_vec::CuDeviceVector{Float64}, fact2_vec::CuDeviceVector{Float64},
    Halpern_fact1::Float64, Halpern_fact2::Float64, n::Int) where {UseCustom,IsDiag}
    i = threadIdx().x + (blockDim().x * (blockIdx().x - 1))
    @inbounds if i <= n
        qw_val = if UseCustom
            startQ = rowPtrQ[i]
            stopQ = rowPtrQ[i+1] - 1
            acc = 0.0
            @inbounds for k in startQ:stopQ
                acc += nzValQ[k] * w[colValQ[k]]
            end
            Qw[i] = acc
            acc
        else
            Qw[i]
        end

        atyi = ATy[i]
        c_i = c[i]
        x_i = x[i]
        last_x_i = last_x[i]
        l_i = l[i]
        u_i = u[i]
        w_i = w[i]

        tmp = -qw_val + atyi - c_i
        z_raw = x_i + sigma * tmp
        x_bar_i = min(max(z_raw, l_i), u_i)

        x_hat_i = 2.0 * x_bar_i - x_i
        x_new = muladd(Halpern_fact2, x_hat_i, Halpern_fact1 * last_x_i)

        w_bar_i = if IsDiag
            fact1_i = fact1_vec[i]
            fact2_i = fact2_vec[i]
            muladd(fact1_i, w_i, fact2_i * x_hat_i)
        else
            muladd(fact1_scalar, w_i, fact2_scalar * x_hat_i)
        end

        x[i] = x_new
        x_hat[i] = x_hat_i
        w_bar[i] = w_bar_i
    end
    return
end

CUDA.@fastmath @inline function _write_soc_var_projection!(::Val{IsDiag},
    idx::Int,
    projected::Float64,
    z_val::Float64,
    dx::CuDeviceVector{Float64},
    w_bar::CuDeviceVector{Float64},
    w::CuDeviceVector{Float64},
    z_bar::CuDeviceVector{Float64},
    x_bar::CuDeviceVector{Float64},
    x_hat::CuDeviceVector{Float64},
    last_x::CuDeviceVector{Float64},
    x::CuDeviceVector{Float64},
    fact1_scalar::Float64,
    fact2_scalar::Float64,
    fact1_vec::CuDeviceVector{Float64},
    fact2_vec::CuDeviceVector{Float64},
    Halpern_fact1::Float64,
    Halpern_fact2::Float64) where {IsDiag}
    x_i = x[idx]
    x_hat_i = 2.0 * projected - x_i

    x_bar[idx] = projected
    x_hat[idx] = x_hat_i
    dx[idx] = projected - x_i
    z_bar[idx] = z_val
    x[idx] = muladd(Halpern_fact2, x_hat_i, Halpern_fact1 * last_x[idx])
    w_bar[idx] = if IsDiag
        muladd(fact1_vec[idx], w[idx], fact2_vec[idx] * x_hat_i)
    else
        muladd(fact1_scalar, w[idx], fact2_scalar * x_hat_i)
    end
    return
end

CUDA.@fastmath @inline function _write_soc_var_projection_partial!(::Val{IsDiag},
    idx::Int,
    projected::Float64,
    w_bar::CuDeviceVector{Float64},
    w::CuDeviceVector{Float64},
    x_hat::CuDeviceVector{Float64},
    last_x::CuDeviceVector{Float64},
    x::CuDeviceVector{Float64},
    fact1_scalar::Float64,
    fact2_scalar::Float64,
    fact1_vec::CuDeviceVector{Float64},
    fact2_vec::CuDeviceVector{Float64},
    Halpern_fact1::Float64,
    Halpern_fact2::Float64) where {IsDiag}
    x_i = x[idx]
    x_hat_i = 2.0 * projected - x_i

    x_hat[idx] = x_hat_i
    x[idx] = muladd(Halpern_fact2, x_hat_i, Halpern_fact1 * last_x[idx])
    w_bar[idx] = if IsDiag
        muladd(fact1_vec[idx], w[idx], fact2_vec[idx] * x_hat_i)
    else
        muladd(fact1_scalar, w[idx], fact2_scalar * x_hat_i)
    end
    return
end

CUDA.@fastmath @inline function _write_soc_var_projection_noQ!(
    idx::Int,
    projected::Float64,
    z_val::Float64,
    dx::CuDeviceVector{Float64},
    z_bar::CuDeviceVector{Float64},
    x_bar::CuDeviceVector{Float64},
    x_hat::CuDeviceVector{Float64},
    last_x::CuDeviceVector{Float64},
    x::CuDeviceVector{Float64},
    Halpern_fact1::Float64,
    Halpern_fact2::Float64)
    x_i = x[idx]
    x_hat_i = 2.0 * projected - x_i

    x_bar[idx] = projected
    x_hat[idx] = x_hat_i
    dx[idx] = projected - x_i
    z_bar[idx] = z_val
    x[idx] = muladd(Halpern_fact2, x_hat_i, Halpern_fact1 * last_x[idx])
    return
end

CUDA.@fastmath @inline function _write_soc_var_projection_noQ_partial!(
    idx::Int,
    projected::Float64,
    x_hat::CuDeviceVector{Float64},
    last_x::CuDeviceVector{Float64},
    x::CuDeviceVector{Float64},
    Halpern_fact1::Float64,
    Halpern_fact2::Float64)
    x_i = x[idx]
    x_hat_i = 2.0 * projected - x_i

    x_hat[idx] = x_hat_i
    x[idx] = muladd(Halpern_fact2, x_hat_i, Halpern_fact1 * last_x[idx])
    return
end

CUDA.@fastmath @inline function _soc_raw_update(::Val{UseQ},
    idx::Int,
    x::CuDeviceVector{Float64},
    Qw::CuDeviceVector{Float64},
    ATy::CuDeviceVector{Float64},
    c::CuDeviceVector{Float64},
    sigma::Float64) where {UseQ}
    q_term = UseQ ? -Qw[idx] : 0.0
    return x[idx] + sigma * (q_term + ATy[idx] - c[idx])
end

CUDA.@fastmath @inline function _soc_raw_update_noQ(
    idx::Int,
    x::CuDeviceVector{Float64},
    ATy::CuDeviceVector{Float64},
    c::CuDeviceVector{Float64},
    sigma::Float64)
    return x[idx] + sigma * (ATy[idx] - c[idx])
end

CUDA.@fastmath @inline function _soc_raw_update_noQ_custom(
    idx::Int,
    x::CuDeviceVector{Float64},
    rowPtrAT::CuDeviceVector{Int32},
    colValAT::CuDeviceVector{Int32},
    nzValAT::CuDeviceVector{Float64},
    y::CuDeviceVector{Float64},
    c::CuDeviceVector{Float64},
    sigma::Float64)
    startAT = rowPtrAT[idx]
    stopAT = rowPtrAT[idx+1] - 1
    acc = 0.0
    @inbounds for k in startAT:stopAT
        acc += nzValAT[k] * y[colValAT[k]]
    end
    return x[idx] + sigma * (acc - c[idx])
end

CUDA.@fastmath @inline function unified_update_zxw1_SOC_size3_kernel!(::Val{UseQ}, ::Val{IsDiag},
    dx::CuDeviceVector{Float64},
    w_bar::CuDeviceVector{Float64},
    w::CuDeviceVector{Float64},
    z_bar::CuDeviceVector{Float64},
    x_bar::CuDeviceVector{Float64},
    x_hat::CuDeviceVector{Float64},
    last_x::CuDeviceVector{Float64},
    x::CuDeviceVector{Float64},
    Qw::CuDeviceVector{Float64},
    ATy::CuDeviceVector{Float64},
    c::CuDeviceVector{Float64},
    sigma::Float64,
    fact1_scalar::Float64,
    fact2_scalar::Float64,
    fact1_vec::CuDeviceVector{Float64},
    fact2_vec::CuDeviceVector{Float64},
    Halpern_fact1::Float64,
    Halpern_fact2::Float64,
    soc_var_starts::CuDeviceVector{Int32},
    cone_count::Int) where {UseQ,IsDiag}
    i = threadIdx().x + (blockDim().x * (blockIdx().x - 1))
    @inbounds if i <= cone_count
        start_idx = Int(soc_var_starts[i])
        j1 = start_idx + 1
        j2 = start_idx + 2

        t_raw = _soc_raw_update(Val(UseQ), start_idx, x, Qw, ATy, c, sigma)
        z_raw_1 = _soc_raw_update(Val(UseQ), j1, x, Qw, ATy, c, sigma)
        z_raw_2 = _soc_raw_update(Val(UseQ), j2, x, Qw, ATy, c, sigma)
        norm_s = 0.0
        norm_s += z_raw_1^2
        norm_s += z_raw_2^2
        norm_s = sqrt(norm_s)

        if norm_s <= -t_raw
            _write_soc_var_projection!(Val(IsDiag), start_idx, 0.0, -t_raw / sigma,
                dx, w_bar, w, z_bar, x_bar, x_hat, last_x, x,
                fact1_scalar, fact2_scalar, fact1_vec, fact2_vec,
                Halpern_fact1, Halpern_fact2)
            _write_soc_var_projection!(Val(IsDiag), j1, 0.0, -z_raw_1 / sigma,
                dx, w_bar, w, z_bar, x_bar, x_hat, last_x, x,
                fact1_scalar, fact2_scalar, fact1_vec, fact2_vec,
                Halpern_fact1, Halpern_fact2)
            _write_soc_var_projection!(Val(IsDiag), j2, 0.0, -z_raw_2 / sigma,
                dx, w_bar, w, z_bar, x_bar, x_hat, last_x, x,
                fact1_scalar, fact2_scalar, fact1_vec, fact2_vec,
                Halpern_fact1, Halpern_fact2)
        elseif norm_s <= t_raw
            _write_soc_var_projection!(Val(IsDiag), start_idx, t_raw, 0.0,
                dx, w_bar, w, z_bar, x_bar, x_hat, last_x, x,
                fact1_scalar, fact2_scalar, fact1_vec, fact2_vec,
                Halpern_fact1, Halpern_fact2)
            _write_soc_var_projection!(Val(IsDiag), j1, z_raw_1, 0.0,
                dx, w_bar, w, z_bar, x_bar, x_hat, last_x, x,
                fact1_scalar, fact2_scalar, fact1_vec, fact2_vec,
                Halpern_fact1, Halpern_fact2)
            _write_soc_var_projection!(Val(IsDiag), j2, z_raw_2, 0.0,
                dx, w_bar, w, z_bar, x_bar, x_hat, last_x, x,
                fact1_scalar, fact2_scalar, fact1_vec, fact2_vec,
                Halpern_fact1, Halpern_fact2)
        else
            proj_t = (norm_s + t_raw) / 2.0
            alpha = proj_t / norm_s
            proj_1 = alpha * z_raw_1
            proj_2 = alpha * z_raw_2

            _write_soc_var_projection!(Val(IsDiag), start_idx, proj_t, (proj_t - t_raw) / sigma,
                dx, w_bar, w, z_bar, x_bar, x_hat, last_x, x,
                fact1_scalar, fact2_scalar, fact1_vec, fact2_vec,
                Halpern_fact1, Halpern_fact2)
            _write_soc_var_projection!(Val(IsDiag), j1, proj_1, (proj_1 - z_raw_1) / sigma,
                dx, w_bar, w, z_bar, x_bar, x_hat, last_x, x,
                fact1_scalar, fact2_scalar, fact1_vec, fact2_vec,
                Halpern_fact1, Halpern_fact2)
            _write_soc_var_projection!(Val(IsDiag), j2, proj_2, (proj_2 - z_raw_2) / sigma,
                dx, w_bar, w, z_bar, x_bar, x_hat, last_x, x,
                fact1_scalar, fact2_scalar, fact1_vec, fact2_vec,
                Halpern_fact1, Halpern_fact2)
        end
    end
    return
end

CUDA.@fastmath @inline function unified_update_zxw1_SOC_exact_kernel!(::Val{ConeSize}, ::Val{UseQ}, ::Val{IsDiag},
    dx::CuDeviceVector{Float64},
    w_bar::CuDeviceVector{Float64},
    w::CuDeviceVector{Float64},
    z_bar::CuDeviceVector{Float64},
    x_bar::CuDeviceVector{Float64},
    x_hat::CuDeviceVector{Float64},
    last_x::CuDeviceVector{Float64},
    x::CuDeviceVector{Float64},
    Qw::CuDeviceVector{Float64},
    ATy::CuDeviceVector{Float64},
    c::CuDeviceVector{Float64},
    sigma::Float64,
    fact1_scalar::Float64,
    fact2_scalar::Float64,
    fact1_vec::CuDeviceVector{Float64},
    fact2_vec::CuDeviceVector{Float64},
    Halpern_fact1::Float64,
    Halpern_fact2::Float64,
    soc_var_starts::CuDeviceVector{Int32},
    cone_count::Int) where {ConeSize,UseQ,IsDiag}
    i = threadIdx().x + (blockDim().x * (blockIdx().x - 1))
    @inbounds if i <= cone_count
        start_idx = Int(soc_var_starts[i])
        j1 = start_idx + 1
        j2 = start_idx + 2
        j3 = start_idx + 3
        j4 = start_idx + 4
        j5 = start_idx + 5
        j6 = start_idx + 6
        j7 = start_idx + 7

        t_raw = _soc_raw_update(Val(UseQ), start_idx, x, Qw, ATy, c, sigma)
        norm_s = 0.0
        z_raw_1 = 0.0
        z_raw_2 = 0.0
        z_raw_3 = 0.0
        z_raw_4 = 0.0
        z_raw_5 = 0.0
        z_raw_6 = 0.0
        z_raw_7 = 0.0

        if ConeSize > 1
            z_raw_1 = _soc_raw_update(Val(UseQ), j1, x, Qw, ATy, c, sigma)
            norm_s += z_raw_1^2
        end
        if ConeSize > 2
            z_raw_2 = _soc_raw_update(Val(UseQ), j2, x, Qw, ATy, c, sigma)
            norm_s += z_raw_2^2
        end
        if ConeSize > 3
            z_raw_3 = _soc_raw_update(Val(UseQ), j3, x, Qw, ATy, c, sigma)
            norm_s += z_raw_3^2
        end
        if ConeSize > 4
            z_raw_4 = _soc_raw_update(Val(UseQ), j4, x, Qw, ATy, c, sigma)
            norm_s += z_raw_4^2
        end
        if ConeSize > 5
            z_raw_5 = _soc_raw_update(Val(UseQ), j5, x, Qw, ATy, c, sigma)
            norm_s += z_raw_5^2
        end
        if ConeSize > 6
            z_raw_6 = _soc_raw_update(Val(UseQ), j6, x, Qw, ATy, c, sigma)
            norm_s += z_raw_6^2
        end
        if ConeSize > 7
            z_raw_7 = _soc_raw_update(Val(UseQ), j7, x, Qw, ATy, c, sigma)
            norm_s += z_raw_7^2
        end
        norm_s = sqrt(norm_s)

        if norm_s <= -t_raw
            _write_soc_var_projection!(Val(IsDiag), start_idx, 0.0, -t_raw / sigma,
                dx, w_bar, w, z_bar, x_bar, x_hat, last_x, x,
                fact1_scalar, fact2_scalar, fact1_vec, fact2_vec,
                Halpern_fact1, Halpern_fact2)

            if ConeSize > 1
                _write_soc_var_projection!(Val(IsDiag), j1, 0.0, -z_raw_1 / sigma,
                    dx, w_bar, w, z_bar, x_bar, x_hat, last_x, x,
                    fact1_scalar, fact2_scalar, fact1_vec, fact2_vec,
                    Halpern_fact1, Halpern_fact2)
            end
            if ConeSize > 2
                _write_soc_var_projection!(Val(IsDiag), j2, 0.0, -z_raw_2 / sigma,
                    dx, w_bar, w, z_bar, x_bar, x_hat, last_x, x,
                    fact1_scalar, fact2_scalar, fact1_vec, fact2_vec,
                    Halpern_fact1, Halpern_fact2)
            end
            if ConeSize > 3
                _write_soc_var_projection!(Val(IsDiag), j3, 0.0, -z_raw_3 / sigma,
                    dx, w_bar, w, z_bar, x_bar, x_hat, last_x, x,
                    fact1_scalar, fact2_scalar, fact1_vec, fact2_vec,
                    Halpern_fact1, Halpern_fact2)
            end
            if ConeSize > 4
                _write_soc_var_projection!(Val(IsDiag), j4, 0.0, -z_raw_4 / sigma,
                    dx, w_bar, w, z_bar, x_bar, x_hat, last_x, x,
                    fact1_scalar, fact2_scalar, fact1_vec, fact2_vec,
                    Halpern_fact1, Halpern_fact2)
            end
            if ConeSize > 5
                _write_soc_var_projection!(Val(IsDiag), j5, 0.0, -z_raw_5 / sigma,
                    dx, w_bar, w, z_bar, x_bar, x_hat, last_x, x,
                    fact1_scalar, fact2_scalar, fact1_vec, fact2_vec,
                    Halpern_fact1, Halpern_fact2)
            end
            if ConeSize > 6
                _write_soc_var_projection!(Val(IsDiag), j6, 0.0, -z_raw_6 / sigma,
                    dx, w_bar, w, z_bar, x_bar, x_hat, last_x, x,
                    fact1_scalar, fact2_scalar, fact1_vec, fact2_vec,
                    Halpern_fact1, Halpern_fact2)
            end
            if ConeSize > 7
                _write_soc_var_projection!(Val(IsDiag), j7, 0.0, -z_raw_7 / sigma,
                    dx, w_bar, w, z_bar, x_bar, x_hat, last_x, x,
                    fact1_scalar, fact2_scalar, fact1_vec, fact2_vec,
                    Halpern_fact1, Halpern_fact2)
            end
        elseif norm_s <= t_raw
            _write_soc_var_projection!(Val(IsDiag), start_idx, t_raw, 0.0,
                dx, w_bar, w, z_bar, x_bar, x_hat, last_x, x,
                fact1_scalar, fact2_scalar, fact1_vec, fact2_vec,
                Halpern_fact1, Halpern_fact2)

            if ConeSize > 1
                _write_soc_var_projection!(Val(IsDiag), j1, z_raw_1, 0.0,
                    dx, w_bar, w, z_bar, x_bar, x_hat, last_x, x,
                    fact1_scalar, fact2_scalar, fact1_vec, fact2_vec,
                    Halpern_fact1, Halpern_fact2)
            end
            if ConeSize > 2
                _write_soc_var_projection!(Val(IsDiag), j2, z_raw_2, 0.0,
                    dx, w_bar, w, z_bar, x_bar, x_hat, last_x, x,
                    fact1_scalar, fact2_scalar, fact1_vec, fact2_vec,
                    Halpern_fact1, Halpern_fact2)
            end
            if ConeSize > 3
                _write_soc_var_projection!(Val(IsDiag), j3, z_raw_3, 0.0,
                    dx, w_bar, w, z_bar, x_bar, x_hat, last_x, x,
                    fact1_scalar, fact2_scalar, fact1_vec, fact2_vec,
                    Halpern_fact1, Halpern_fact2)
            end
            if ConeSize > 4
                _write_soc_var_projection!(Val(IsDiag), j4, z_raw_4, 0.0,
                    dx, w_bar, w, z_bar, x_bar, x_hat, last_x, x,
                    fact1_scalar, fact2_scalar, fact1_vec, fact2_vec,
                    Halpern_fact1, Halpern_fact2)
            end
            if ConeSize > 5
                _write_soc_var_projection!(Val(IsDiag), j5, z_raw_5, 0.0,
                    dx, w_bar, w, z_bar, x_bar, x_hat, last_x, x,
                    fact1_scalar, fact2_scalar, fact1_vec, fact2_vec,
                    Halpern_fact1, Halpern_fact2)
            end
            if ConeSize > 6
                _write_soc_var_projection!(Val(IsDiag), j6, z_raw_6, 0.0,
                    dx, w_bar, w, z_bar, x_bar, x_hat, last_x, x,
                    fact1_scalar, fact2_scalar, fact1_vec, fact2_vec,
                    Halpern_fact1, Halpern_fact2)
            end
            if ConeSize > 7
                _write_soc_var_projection!(Val(IsDiag), j7, z_raw_7, 0.0,
                    dx, w_bar, w, z_bar, x_bar, x_hat, last_x, x,
                    fact1_scalar, fact2_scalar, fact1_vec, fact2_vec,
                    Halpern_fact1, Halpern_fact2)
            end
        else
            proj_t = (norm_s + t_raw) / 2.0
            alpha = proj_t / norm_s

            _write_soc_var_projection!(Val(IsDiag), start_idx, proj_t, (proj_t - t_raw) / sigma,
                dx, w_bar, w, z_bar, x_bar, x_hat, last_x, x,
                fact1_scalar, fact2_scalar, fact1_vec, fact2_vec,
                Halpern_fact1, Halpern_fact2)

            if ConeSize > 1
                proj_1 = alpha * z_raw_1
                _write_soc_var_projection!(Val(IsDiag), j1, proj_1, (proj_1 - z_raw_1) / sigma,
                    dx, w_bar, w, z_bar, x_bar, x_hat, last_x, x,
                    fact1_scalar, fact2_scalar, fact1_vec, fact2_vec,
                    Halpern_fact1, Halpern_fact2)
            end
            if ConeSize > 2
                proj_2 = alpha * z_raw_2
                _write_soc_var_projection!(Val(IsDiag), j2, proj_2, (proj_2 - z_raw_2) / sigma,
                    dx, w_bar, w, z_bar, x_bar, x_hat, last_x, x,
                    fact1_scalar, fact2_scalar, fact1_vec, fact2_vec,
                    Halpern_fact1, Halpern_fact2)
            end
            if ConeSize > 3
                proj_3 = alpha * z_raw_3
                _write_soc_var_projection!(Val(IsDiag), j3, proj_3, (proj_3 - z_raw_3) / sigma,
                    dx, w_bar, w, z_bar, x_bar, x_hat, last_x, x,
                    fact1_scalar, fact2_scalar, fact1_vec, fact2_vec,
                    Halpern_fact1, Halpern_fact2)
            end
            if ConeSize > 4
                proj_4 = alpha * z_raw_4
                _write_soc_var_projection!(Val(IsDiag), j4, proj_4, (proj_4 - z_raw_4) / sigma,
                    dx, w_bar, w, z_bar, x_bar, x_hat, last_x, x,
                    fact1_scalar, fact2_scalar, fact1_vec, fact2_vec,
                    Halpern_fact1, Halpern_fact2)
            end
            if ConeSize > 5
                proj_5 = alpha * z_raw_5
                _write_soc_var_projection!(Val(IsDiag), j5, proj_5, (proj_5 - z_raw_5) / sigma,
                    dx, w_bar, w, z_bar, x_bar, x_hat, last_x, x,
                    fact1_scalar, fact2_scalar, fact1_vec, fact2_vec,
                    Halpern_fact1, Halpern_fact2)
            end
            if ConeSize > 6
                proj_6 = alpha * z_raw_6
                _write_soc_var_projection!(Val(IsDiag), j6, proj_6, (proj_6 - z_raw_6) / sigma,
                    dx, w_bar, w, z_bar, x_bar, x_hat, last_x, x,
                    fact1_scalar, fact2_scalar, fact1_vec, fact2_vec,
                    Halpern_fact1, Halpern_fact2)
            end
            if ConeSize > 7
                proj_7 = alpha * z_raw_7
                _write_soc_var_projection!(Val(IsDiag), j7, proj_7, (proj_7 - z_raw_7) / sigma,
                    dx, w_bar, w, z_bar, x_bar, x_hat, last_x, x,
                    fact1_scalar, fact2_scalar, fact1_vec, fact2_vec,
                    Halpern_fact1, Halpern_fact2)
            end
        end
    end
    return
end

CUDA.@fastmath @inline function unified_update_zxw1_SOC_generic_kernel!(::Val{UseQ}, ::Val{IsDiag},
    dx::CuDeviceVector{Float64},
    w_bar::CuDeviceVector{Float64},
    w::CuDeviceVector{Float64},
    z_bar::CuDeviceVector{Float64},
    x_bar::CuDeviceVector{Float64},
    x_hat::CuDeviceVector{Float64},
    last_x::CuDeviceVector{Float64},
    x::CuDeviceVector{Float64},
    Qw::CuDeviceVector{Float64},
    ATy::CuDeviceVector{Float64},
    c::CuDeviceVector{Float64},
    sigma::Float64,
    fact1_scalar::Float64,
    fact2_scalar::Float64,
    fact1_vec::CuDeviceVector{Float64},
    fact2_vec::CuDeviceVector{Float64},
    Halpern_fact1::Float64,
    Halpern_fact2::Float64,
    soc_var_starts::CuDeviceVector{Int32},
    soc_var_sizes::CuDeviceVector{Int32},
    cone_count::Int) where {UseQ,IsDiag}
    i = threadIdx().x + (blockDim().x * (blockIdx().x - 1))
    @inbounds if i <= cone_count
        start_idx = Int(soc_var_starts[i])
        cone_size = Int(soc_var_sizes[i])
        end_idx = start_idx + cone_size - 1

        t_raw = _soc_raw_update(Val(UseQ), start_idx, x, Qw, ATy, c, sigma)
        z_bar[start_idx] = t_raw
        norm_s = 0.0
        for j in (start_idx + 1):end_idx
            z_raw_j = _soc_raw_update(Val(UseQ), j, x, Qw, ATy, c, sigma)
            z_bar[j] = z_raw_j
            norm_s += z_raw_j^2
        end
        norm_s = sqrt(norm_s)

        if norm_s <= -t_raw
            _write_soc_var_projection!(Val(IsDiag), start_idx, 0.0, -t_raw / sigma,
                dx, w_bar, w, z_bar, x_bar, x_hat, last_x, x,
                fact1_scalar, fact2_scalar, fact1_vec, fact2_vec,
                Halpern_fact1, Halpern_fact2)

            for j in (start_idx + 1):end_idx
                z_raw_j = z_bar[j]
                _write_soc_var_projection!(Val(IsDiag), j, 0.0, -z_raw_j / sigma,
                    dx, w_bar, w, z_bar, x_bar, x_hat, last_x, x,
                    fact1_scalar, fact2_scalar, fact1_vec, fact2_vec,
                    Halpern_fact1, Halpern_fact2)
            end
        elseif norm_s <= t_raw
            _write_soc_var_projection!(Val(IsDiag), start_idx, t_raw, 0.0,
                dx, w_bar, w, z_bar, x_bar, x_hat, last_x, x,
                fact1_scalar, fact2_scalar, fact1_vec, fact2_vec,
                Halpern_fact1, Halpern_fact2)

            for j in (start_idx + 1):end_idx
                z_raw_j = z_bar[j]
                _write_soc_var_projection!(Val(IsDiag), j, z_raw_j, 0.0,
                    dx, w_bar, w, z_bar, x_bar, x_hat, last_x, x,
                    fact1_scalar, fact2_scalar, fact1_vec, fact2_vec,
                    Halpern_fact1, Halpern_fact2)
            end
        else
            proj_t = (norm_s + t_raw) / 2.0
            alpha = proj_t / norm_s

            _write_soc_var_projection!(Val(IsDiag), start_idx, proj_t, (proj_t - t_raw) / sigma,
                dx, w_bar, w, z_bar, x_bar, x_hat, last_x, x,
                fact1_scalar, fact2_scalar, fact1_vec, fact2_vec,
                Halpern_fact1, Halpern_fact2)

            for j in (start_idx + 1):end_idx
                z_raw_j = z_bar[j]
                proj_j = alpha * z_raw_j
                _write_soc_var_projection!(Val(IsDiag), j, proj_j, (proj_j - z_raw_j) / sigma,
                    dx, w_bar, w, z_bar, x_bar, x_hat, last_x, x,
                    fact1_scalar, fact2_scalar, fact1_vec, fact2_vec,
                    Halpern_fact1, Halpern_fact2)
            end
        end
    end
    return
end

# Large SOC cones use block-cooperative raw-value generation and writeback,
# but keep the norm accumulation in the generic kernel's serial order so the
# long-run solve trajectory stays unchanged.
CUDA.@fastmath function unified_update_zxw1_SOC_large_kernel!(::Val{UseQ}, ::Val{IsDiag},
    dx::CuDeviceVector{Float64},
    w_bar::CuDeviceVector{Float64},
    w::CuDeviceVector{Float64},
    z_bar::CuDeviceVector{Float64},
    x_bar::CuDeviceVector{Float64},
    x_hat::CuDeviceVector{Float64},
    last_x::CuDeviceVector{Float64},
    x::CuDeviceVector{Float64},
    Qw::CuDeviceVector{Float64},
    ATy::CuDeviceVector{Float64},
    c::CuDeviceVector{Float64},
    sigma::Float64,
    fact1_scalar::Float64,
    fact2_scalar::Float64,
    fact1_vec::CuDeviceVector{Float64},
    fact2_vec::CuDeviceVector{Float64},
    Halpern_fact1::Float64,
    Halpern_fact2::Float64,
    soc_var_starts::CuDeviceVector{Int32},
    soc_var_sizes::CuDeviceVector{Int32},
    cone_count::Int) where {UseQ,IsDiag}
    cone_idx = Int(blockIdx().x)
    tid = Int(threadIdx().x)
    block_threads = Int(blockDim().x)

    shared_meta = CuStaticSharedArray(Float64, 3)  # t_raw, proj_t, alpha
    shared_case = CuStaticSharedArray(Int32, 1)

    @inbounds if cone_idx <= cone_count
        start_idx = Int(soc_var_starts[cone_idx])
        cone_size = Int(soc_var_sizes[cone_idx])
        end_idx = start_idx + cone_size - 1

        if tid == 1
            t_raw = _soc_raw_update(Val(UseQ), start_idx, x, Qw, ATy, c, sigma)
            shared_meta[1] = t_raw
            z_bar[start_idx] = t_raw
        end

        for offset in tid:block_threads:(cone_size - 1)
            j = start_idx + offset
            z_bar[j] = _soc_raw_update(Val(UseQ), j, x, Qw, ATy, c, sigma)
        end
        sync_threads()

        if tid == 1
            t_raw = shared_meta[1]
            norm_s = 0.0
            for j in (start_idx + 1):end_idx
                z_raw_j = z_bar[j]
                norm_s += z_raw_j^2
            end
            norm_s = sqrt(norm_s)
            if norm_s <= -t_raw
                shared_case[1] = Int32(0)
                shared_meta[2] = 0.0
                shared_meta[3] = 0.0
            elseif norm_s <= t_raw
                shared_case[1] = Int32(1)
                shared_meta[2] = t_raw
                shared_meta[3] = 1.0
            else
                proj_t = (norm_s + t_raw) / 2.0
                shared_case[1] = Int32(2)
                shared_meta[2] = proj_t
                shared_meta[3] = proj_t / norm_s
            end
        end
        sync_threads()

        t_raw = shared_meta[1]
        proj_t = shared_meta[2]
        alpha = shared_meta[3]
        case_id = shared_case[1]

        if tid == 1
            projected_t = if case_id == 0
                0.0
            elseif case_id == 1
                t_raw
            else
                proj_t
            end
            z_t = if case_id == 0
                -t_raw / sigma
            elseif case_id == 1
                0.0
            else
                (proj_t - t_raw) / sigma
            end
            _write_soc_var_projection!(Val(IsDiag), start_idx, projected_t, z_t,
                dx, w_bar, w, z_bar, x_bar, x_hat, last_x, x,
                fact1_scalar, fact2_scalar, fact1_vec, fact2_vec,
                Halpern_fact1, Halpern_fact2)
        end

        for offset in tid:block_threads:(cone_size - 1)
            j = start_idx + offset
            z_raw_j = z_bar[j]
            projected_j = if case_id == 0
                0.0
            elseif case_id == 1
                z_raw_j
            else
                alpha * z_raw_j
            end
            z_j = if case_id == 0
                -z_raw_j / sigma
            elseif case_id == 1
                0.0
            else
                (projected_j - z_raw_j) / sigma
            end
            _write_soc_var_projection!(Val(IsDiag), j, projected_j, z_j,
                dx, w_bar, w, z_bar, x_bar, x_hat, last_x, x,
                fact1_scalar, fact2_scalar, fact1_vec, fact2_vec,
                Halpern_fact1, Halpern_fact2)
        end
    end
    return
end

CUDA.@fastmath function unified_update_zxw1_SOC_large_kernel_partial!(::Val{UseQ}, ::Val{IsDiag},
    w_bar::CuDeviceVector{Float64},
    w::CuDeviceVector{Float64},
    x_hat::CuDeviceVector{Float64},
    last_x::CuDeviceVector{Float64},
    x::CuDeviceVector{Float64},
    Qw::CuDeviceVector{Float64},
    ATy::CuDeviceVector{Float64},
    c::CuDeviceVector{Float64},
    sigma::Float64,
    fact1_scalar::Float64,
    fact2_scalar::Float64,
    fact1_vec::CuDeviceVector{Float64},
    fact2_vec::CuDeviceVector{Float64},
    Halpern_fact1::Float64,
    Halpern_fact2::Float64,
    soc_var_starts::CuDeviceVector{Int32},
    soc_var_sizes::CuDeviceVector{Int32},
    cone_count::Int) where {UseQ,IsDiag}
    cone_idx = Int(blockIdx().x)
    tid = Int(threadIdx().x)
    block_threads = Int(blockDim().x)

    shared_meta = CuStaticSharedArray(Float64, 3)  # t_raw, proj_t, alpha
    shared_case = CuStaticSharedArray(Int32, 1)

    @inbounds if cone_idx <= cone_count
        start_idx = Int(soc_var_starts[cone_idx])
        cone_size = Int(soc_var_sizes[cone_idx])
        end_idx = start_idx + cone_size - 1

        if tid == 1
            shared_meta[1] = _soc_raw_update(Val(UseQ), start_idx, x, Qw, ATy, c, sigma)
        end

        # Reuse x_hat as temporary raw-value storage before writing the final
        # x_hat values for the projected SOC tail entries.
        for offset in tid:block_threads:(cone_size - 1)
            j = start_idx + offset
            x_hat[j] = _soc_raw_update(Val(UseQ), j, x, Qw, ATy, c, sigma)
        end
        sync_threads()

        if tid == 1
            t_raw = shared_meta[1]
            norm_s = 0.0
            for j in (start_idx + 1):end_idx
                z_raw_j = x_hat[j]
                norm_s += z_raw_j^2
            end
            norm_s = sqrt(norm_s)
            if norm_s <= -t_raw
                shared_case[1] = Int32(0)
                shared_meta[2] = 0.0
                shared_meta[3] = 0.0
            elseif norm_s <= t_raw
                shared_case[1] = Int32(1)
                shared_meta[2] = t_raw
                shared_meta[3] = 1.0
            else
                proj_t = (norm_s + t_raw) / 2.0
                shared_case[1] = Int32(2)
                shared_meta[2] = proj_t
                shared_meta[3] = proj_t / norm_s
            end
        end
        sync_threads()

        t_raw = shared_meta[1]
        proj_t = shared_meta[2]
        alpha = shared_meta[3]
        case_id = shared_case[1]

        if tid == 1
            projected_t = if case_id == 0
                0.0
            elseif case_id == 1
                t_raw
            else
                proj_t
            end
            _write_soc_var_projection_partial!(Val(IsDiag), start_idx, projected_t,
                w_bar, w, x_hat, last_x, x, fact1_scalar, fact2_scalar, fact1_vec, fact2_vec,
                Halpern_fact1, Halpern_fact2)
        end

        for offset in tid:block_threads:(cone_size - 1)
            j = start_idx + offset
            z_raw_j = x_hat[j]
            projected_j = if case_id == 0
                0.0
            elseif case_id == 1
                z_raw_j
            else
                alpha * z_raw_j
            end
            _write_soc_var_projection_partial!(Val(IsDiag), j, projected_j,
                w_bar, w, x_hat, last_x, x, fact1_scalar, fact2_scalar, fact1_vec, fact2_vec,
                Halpern_fact1, Halpern_fact2)
        end
    end
    return
end

@inline function launch_unified_update_zxw1_SOC_exact!(::Val{ConeSize},
    ::Val{UseQ},
    ws::HPRSOCP_workspace_gpu,
    soc_var_starts::CuVector{Int32},
    cone_count::Int,
    fact1_scalar::Float64,
    fact2_scalar::Float64,
    Halpern_fact1::Float64,
    Halpern_fact2::Float64) where {ConeSize,UseQ}
    threads, blocks = gpu_launch_config(cone_count)
    if threads > 0 && cone_count > 0
        @cuda threads = threads blocks = blocks unified_update_zxw1_SOC_exact_kernel!(
            Val(ConeSize), Val(UseQ), Val(ws.Q_is_diag),
            ws.dx, ws.w_bar, ws.w, ws.z_bar, ws.x_bar, ws.x_hat, ws.last_x, ws.x,
            ws.Qw, ws.ATy, ws.c, ws.sigma, fact1_scalar, fact2_scalar, ws.fact1, ws.fact2,
            Halpern_fact1, Halpern_fact2, soc_var_starts, cone_count)
    end
end

@inline function launch_unified_update_zxw1_SOC_size3!(
    ::Val{UseQ},
    ws::HPRSOCP_workspace_gpu,
    soc_var_starts::CuVector{Int32},
    cone_count::Int,
    fact1_scalar::Float64,
    fact2_scalar::Float64,
    Halpern_fact1::Float64,
    Halpern_fact2::Float64) where {UseQ}
    threads, blocks = gpu_launch_config(cone_count)
    if threads > 0 && cone_count > 0
        @cuda threads = threads blocks = blocks unified_update_zxw1_SOC_size3_kernel!(
            Val(UseQ), Val(ws.Q_is_diag),
            ws.dx, ws.w_bar, ws.w, ws.z_bar, ws.x_bar, ws.x_hat, ws.last_x, ws.x,
            ws.Qw, ws.ATy, ws.c, ws.sigma, fact1_scalar, fact2_scalar, ws.fact1, ws.fact2,
            Halpern_fact1, Halpern_fact2, soc_var_starts, cone_count)
    end
end

@inline function launch_unified_update_zxw1_SOC_large!(
    ::Val{UseQ},
    ws::HPRSOCP_workspace_gpu,
    soc_var_starts::CuVector{Int32},
    soc_var_sizes::CuVector{Int32},
    cone_count::Int,
    fact1_scalar::Float64,
    fact2_scalar::Float64,
    Halpern_fact1::Float64,
    Halpern_fact2::Float64) where {UseQ}
    threads = DEFAULT_KERNEL_THREADS
    blocks = cone_count
    if blocks > 0
        @cuda threads = threads blocks = blocks unified_update_zxw1_SOC_large_kernel!(
            Val(UseQ), Val(ws.Q_is_diag),
            ws.dx, ws.w_bar, ws.w, ws.z_bar, ws.x_bar, ws.x_hat, ws.last_x, ws.x,
            ws.Qw, ws.ATy, ws.c, ws.sigma, fact1_scalar, fact2_scalar, ws.fact1, ws.fact2,
            Halpern_fact1, Halpern_fact2, soc_var_starts, soc_var_sizes, cone_count)
    end
end

@inline function launch_unified_update_zxw1_SOC_huge!(
    ::Val{UseQ},
    ws::HPRSOCP_workspace_gpu,
    soc_var_starts::CuVector{Int32},
    soc_var_sizes::CuVector{Int32},
    cone_count::Int,
    fact1_scalar::Float64,
    fact2_scalar::Float64,
    Halpern_fact1::Float64,
    Halpern_fact2::Float64) where {UseQ}
    threads = HUGE_SOC_KERNEL_THREADS
    blocks = cone_count
    if blocks > 0
        @cuda threads = threads blocks = blocks unified_update_zxw1_SOC_large_kernel!(
            Val(UseQ), Val(ws.Q_is_diag),
            ws.dx, ws.w_bar, ws.w, ws.z_bar, ws.x_bar, ws.x_hat, ws.last_x, ws.x,
            ws.Qw, ws.ATy, ws.c, ws.sigma, fact1_scalar, fact2_scalar, ws.fact1, ws.fact2,
            Halpern_fact1, Halpern_fact2, soc_var_starts, soc_var_sizes, cone_count)
    end
end

@inline function launch_unified_update_zxw1_SOC_large_partial!(
    ::Val{UseQ},
    ws::HPRSOCP_workspace_gpu,
    soc_var_starts::CuVector{Int32},
    soc_var_sizes::CuVector{Int32},
    cone_count::Int,
    fact1_scalar::Float64,
    fact2_scalar::Float64,
    Halpern_fact1::Float64,
    Halpern_fact2::Float64) where {UseQ}
    threads = DEFAULT_KERNEL_THREADS
    blocks = cone_count
    if blocks > 0
        @cuda threads = threads blocks = blocks unified_update_zxw1_SOC_large_kernel_partial!(
            Val(UseQ), Val(ws.Q_is_diag),
            ws.w_bar, ws.w, ws.x_hat, ws.last_x, ws.x,
            ws.Qw, ws.ATy, ws.c, ws.sigma, fact1_scalar, fact2_scalar, ws.fact1, ws.fact2,
            Halpern_fact1, Halpern_fact2, soc_var_starts, soc_var_sizes, cone_count)
    end
end

@inline function launch_unified_update_zxw1_SOC_huge_partial!(
    ::Val{UseQ},
    ws::HPRSOCP_workspace_gpu,
    soc_var_starts::CuVector{Int32},
    soc_var_sizes::CuVector{Int32},
    cone_count::Int,
    fact1_scalar::Float64,
    fact2_scalar::Float64,
    Halpern_fact1::Float64,
    Halpern_fact2::Float64) where {UseQ}
    threads = HUGE_SOC_KERNEL_THREADS
    blocks = cone_count
    if blocks > 0
        @cuda threads = threads blocks = blocks unified_update_zxw1_SOC_large_kernel_partial!(
            Val(UseQ), Val(ws.Q_is_diag),
            ws.w_bar, ws.w, ws.x_hat, ws.last_x, ws.x,
            ws.Qw, ws.ATy, ws.c, ws.sigma, fact1_scalar, fact2_scalar, ws.fact1, ws.fact2,
            Halpern_fact1, Halpern_fact2, soc_var_starts, soc_var_sizes, cone_count)
    end
end

@inline function launch_unified_update_zxw1_SOC_generic!(
    ::Val{UseQ},
    ws::HPRSOCP_workspace_gpu,
    soc_var_starts::CuVector{Int32},
    soc_var_sizes::CuVector{Int32},
    cone_count::Int,
    fact1_scalar::Float64,
    fact2_scalar::Float64,
    Halpern_fact1::Float64,
    Halpern_fact2::Float64) where {UseQ}
    threads, blocks = gpu_launch_config(cone_count)
    if threads > 0 && cone_count > 0
        @cuda threads = threads blocks = blocks unified_update_zxw1_SOC_generic_kernel!(
            Val(UseQ), Val(ws.Q_is_diag),
            ws.dx, ws.w_bar, ws.w, ws.z_bar, ws.x_bar, ws.x_hat, ws.last_x, ws.x,
            ws.Qw, ws.ATy, ws.c, ws.sigma, fact1_scalar, fact2_scalar, ws.fact1, ws.fact2,
            Halpern_fact1, Halpern_fact2, soc_var_starts, soc_var_sizes, cone_count)
    end
end

CUDA.@fastmath @inline function unified_update_zxw1_SOC_small_kernel!(::Val{UseQ}, ::Val{IsDiag},
    dx::CuDeviceVector{Float64},
    w_bar::CuDeviceVector{Float64},
    w::CuDeviceVector{Float64},
    z_bar::CuDeviceVector{Float64},
    x_bar::CuDeviceVector{Float64},
    x_hat::CuDeviceVector{Float64},
    last_x::CuDeviceVector{Float64},
    x::CuDeviceVector{Float64},
    Qw::CuDeviceVector{Float64},
    ATy::CuDeviceVector{Float64},
    c::CuDeviceVector{Float64},
    sigma::Float64,
    fact1_scalar::Float64,
    fact2_scalar::Float64,
    fact1_vec::CuDeviceVector{Float64},
    fact2_vec::CuDeviceVector{Float64},
    Halpern_fact1::Float64,
    Halpern_fact2::Float64,
    soc_var_starts::CuDeviceVector{Int32},
    soc_var_sizes::CuDeviceVector{Int32},
    cone_count::Int) where {UseQ,IsDiag}
    i = threadIdx().x + (blockDim().x * (blockIdx().x - 1))
    @inbounds if i <= cone_count
        start_idx = Int(soc_var_starts[i])
        cone_size = Int(soc_var_sizes[i])
        end_idx = start_idx + cone_size - 1

        t_raw = _soc_raw_update(Val(UseQ), start_idx, x, Qw, ATy, c, sigma)
        z_bar[start_idx] = t_raw
        norm_s = 0.0
        for j in (start_idx + 1):end_idx
            z_raw_j = _soc_raw_update(Val(UseQ), j, x, Qw, ATy, c, sigma)
            z_bar[j] = z_raw_j
            norm_s += z_raw_j^2
        end
        norm_s = sqrt(norm_s)

        if norm_s <= -t_raw
            _write_soc_var_projection!(Val(IsDiag), start_idx, 0.0, -t_raw / sigma,
                dx, w_bar, w, z_bar, x_bar, x_hat, last_x, x,
                fact1_scalar, fact2_scalar, fact1_vec, fact2_vec,
                Halpern_fact1, Halpern_fact2)

            for j in (start_idx + 1):end_idx
                z_raw_j = z_bar[j]
                _write_soc_var_projection!(Val(IsDiag), j, 0.0, -z_raw_j / sigma,
                    dx, w_bar, w, z_bar, x_bar, x_hat, last_x, x,
                    fact1_scalar, fact2_scalar, fact1_vec, fact2_vec,
                    Halpern_fact1, Halpern_fact2)
            end
        elseif norm_s <= t_raw
            _write_soc_var_projection!(Val(IsDiag), start_idx, t_raw, 0.0,
                dx, w_bar, w, z_bar, x_bar, x_hat, last_x, x,
                fact1_scalar, fact2_scalar, fact1_vec, fact2_vec,
                Halpern_fact1, Halpern_fact2)

            for j in (start_idx + 1):end_idx
                z_raw_j = z_bar[j]
                _write_soc_var_projection!(Val(IsDiag), j, z_raw_j, 0.0,
                    dx, w_bar, w, z_bar, x_bar, x_hat, last_x, x,
                    fact1_scalar, fact2_scalar, fact1_vec, fact2_vec,
                    Halpern_fact1, Halpern_fact2)
            end
        else
            proj_t = (norm_s + t_raw) / 2.0
            alpha = proj_t / norm_s

            _write_soc_var_projection!(Val(IsDiag), start_idx, proj_t, (proj_t - t_raw) / sigma,
                dx, w_bar, w, z_bar, x_bar, x_hat, last_x, x,
                fact1_scalar, fact2_scalar, fact1_vec, fact2_vec,
                Halpern_fact1, Halpern_fact2)

            for j in (start_idx + 1):end_idx
                z_raw_j = z_bar[j]
                proj_j = alpha * z_raw_j
                _write_soc_var_projection!(Val(IsDiag), j, proj_j, (proj_j - z_raw_j) / sigma,
                    dx, w_bar, w, z_bar, x_bar, x_hat, last_x, x,
                    fact1_scalar, fact2_scalar, fact1_vec, fact2_vec,
                    Halpern_fact1, Halpern_fact2)
            end
        end
    end
    return
end

CUDA.@fastmath @inline function unified_update_zxw1_SOC_small_kernel_partial!(::Val{UseQ}, ::Val{IsDiag},
    w_bar::CuDeviceVector{Float64},
    w::CuDeviceVector{Float64},
    x_hat::CuDeviceVector{Float64},
    last_x::CuDeviceVector{Float64},
    x::CuDeviceVector{Float64},
    Qw::CuDeviceVector{Float64},
    ATy::CuDeviceVector{Float64},
    c::CuDeviceVector{Float64},
    sigma::Float64,
    fact1_scalar::Float64,
    fact2_scalar::Float64,
    fact1_vec::CuDeviceVector{Float64},
    fact2_vec::CuDeviceVector{Float64},
    Halpern_fact1::Float64,
    Halpern_fact2::Float64,
    soc_var_starts::CuDeviceVector{Int32},
    soc_var_sizes::CuDeviceVector{Int32},
    cone_count::Int) where {UseQ,IsDiag}
    i = threadIdx().x + (blockDim().x * (blockIdx().x - 1))
    @inbounds if i <= cone_count
        start_idx = Int(soc_var_starts[i])
        cone_size = Int(soc_var_sizes[i])
        end_idx = start_idx + cone_size - 1

        if cone_size == 3
            j1 = start_idx + 1
            j2 = start_idx + 2

            t_raw = _soc_raw_update(Val(UseQ), start_idx, x, Qw, ATy, c, sigma)
            z_raw_1 = _soc_raw_update(Val(UseQ), j1, x, Qw, ATy, c, sigma)
            z_raw_2 = _soc_raw_update(Val(UseQ), j2, x, Qw, ATy, c, sigma)
            norm_s = sqrt(z_raw_1^2 + z_raw_2^2)

            if norm_s <= -t_raw
                _write_soc_var_projection_partial!(Val(IsDiag), start_idx, 0.0,
                    w_bar, w, x_hat, last_x, x, fact1_scalar, fact2_scalar, fact1_vec, fact2_vec,
                    Halpern_fact1, Halpern_fact2)
                _write_soc_var_projection_partial!(Val(IsDiag), j1, 0.0,
                    w_bar, w, x_hat, last_x, x, fact1_scalar, fact2_scalar, fact1_vec, fact2_vec,
                    Halpern_fact1, Halpern_fact2)
                _write_soc_var_projection_partial!(Val(IsDiag), j2, 0.0,
                    w_bar, w, x_hat, last_x, x, fact1_scalar, fact2_scalar, fact1_vec, fact2_vec,
                    Halpern_fact1, Halpern_fact2)
            elseif norm_s <= t_raw
                _write_soc_var_projection_partial!(Val(IsDiag), start_idx, t_raw,
                    w_bar, w, x_hat, last_x, x, fact1_scalar, fact2_scalar, fact1_vec, fact2_vec,
                    Halpern_fact1, Halpern_fact2)
                _write_soc_var_projection_partial!(Val(IsDiag), j1, z_raw_1,
                    w_bar, w, x_hat, last_x, x, fact1_scalar, fact2_scalar, fact1_vec, fact2_vec,
                    Halpern_fact1, Halpern_fact2)
                _write_soc_var_projection_partial!(Val(IsDiag), j2, z_raw_2,
                    w_bar, w, x_hat, last_x, x, fact1_scalar, fact2_scalar, fact1_vec, fact2_vec,
                    Halpern_fact1, Halpern_fact2)
            else
                proj_t = (norm_s + t_raw) / 2.0
                alpha = proj_t / norm_s
                proj_1 = alpha * z_raw_1
                proj_2 = alpha * z_raw_2

                _write_soc_var_projection_partial!(Val(IsDiag), start_idx, proj_t,
                    w_bar, w, x_hat, last_x, x, fact1_scalar, fact2_scalar, fact1_vec, fact2_vec,
                    Halpern_fact1, Halpern_fact2)
                _write_soc_var_projection_partial!(Val(IsDiag), j1, proj_1,
                    w_bar, w, x_hat, last_x, x, fact1_scalar, fact2_scalar, fact1_vec, fact2_vec,
                    Halpern_fact1, Halpern_fact2)
                _write_soc_var_projection_partial!(Val(IsDiag), j2, proj_2,
                    w_bar, w, x_hat, last_x, x, fact1_scalar, fact2_scalar, fact1_vec, fact2_vec,
                    Halpern_fact1, Halpern_fact2)
            end
            return
        end

        t_raw = _soc_raw_update(Val(UseQ), start_idx, x, Qw, ATy, c, sigma)
        norm_s = 0.0
        for j in (start_idx + 1):end_idx
            z_raw_j = _soc_raw_update(Val(UseQ), j, x, Qw, ATy, c, sigma)
            norm_s += z_raw_j^2
        end
        norm_s = sqrt(norm_s)

        if norm_s <= -t_raw
            _write_soc_var_projection_partial!(Val(IsDiag), start_idx, 0.0,
                w_bar, w, x_hat, last_x, x, fact1_scalar, fact2_scalar, fact1_vec, fact2_vec,
                Halpern_fact1, Halpern_fact2)
            for j in (start_idx + 1):end_idx
                _write_soc_var_projection_partial!(Val(IsDiag), j, 0.0,
                    w_bar, w, x_hat, last_x, x, fact1_scalar, fact2_scalar, fact1_vec, fact2_vec,
                    Halpern_fact1, Halpern_fact2)
            end
        elseif norm_s <= t_raw
            _write_soc_var_projection_partial!(Val(IsDiag), start_idx, t_raw,
                w_bar, w, x_hat, last_x, x, fact1_scalar, fact2_scalar, fact1_vec, fact2_vec,
                Halpern_fact1, Halpern_fact2)
            for j in (start_idx + 1):end_idx
                z_raw_j = _soc_raw_update(Val(UseQ), j, x, Qw, ATy, c, sigma)
                _write_soc_var_projection_partial!(Val(IsDiag), j, z_raw_j,
                    w_bar, w, x_hat, last_x, x, fact1_scalar, fact2_scalar, fact1_vec, fact2_vec,
                    Halpern_fact1, Halpern_fact2)
            end
        else
            proj_t = (norm_s + t_raw) / 2.0
            alpha = proj_t / norm_s

            _write_soc_var_projection_partial!(Val(IsDiag), start_idx, proj_t,
                w_bar, w, x_hat, last_x, x, fact1_scalar, fact2_scalar, fact1_vec, fact2_vec,
                Halpern_fact1, Halpern_fact2)
            for j in (start_idx + 1):end_idx
                z_raw_j = _soc_raw_update(Val(UseQ), j, x, Qw, ATy, c, sigma)
                proj_j = alpha * z_raw_j
                _write_soc_var_projection_partial!(Val(IsDiag), j, proj_j,
                    w_bar, w, x_hat, last_x, x, fact1_scalar, fact2_scalar, fact1_vec, fact2_vec,
                    Halpern_fact1, Halpern_fact2)
            end
        end
    end
    return
end

@inline function launch_unified_update_zxw1_SOC_small!(
    ::Val{UseQ},
    ws::HPRSOCP_workspace_gpu,
    soc_var_starts::CuVector{Int32},
    soc_var_sizes::CuVector{Int32},
    cone_count::Int,
    fact1_scalar::Float64,
    fact2_scalar::Float64,
    Halpern_fact1::Float64,
    Halpern_fact2::Float64) where {UseQ}
    threads, blocks = gpu_launch_config(cone_count)
    if threads > 0 && cone_count > 0
        @cuda threads = threads blocks = blocks unified_update_zxw1_SOC_small_kernel!(
            Val(UseQ), Val(ws.Q_is_diag),
            ws.dx, ws.w_bar, ws.w, ws.z_bar, ws.x_bar, ws.x_hat, ws.last_x, ws.x,
            ws.Qw, ws.ATy, ws.c, ws.sigma, fact1_scalar, fact2_scalar, ws.fact1, ws.fact2,
            Halpern_fact1, Halpern_fact2, soc_var_starts, soc_var_sizes, cone_count)
    end
end

@inline function launch_unified_update_zxw1_SOC_small_partial!(
    ::Val{UseQ},
    ws::HPRSOCP_workspace_gpu,
    soc_var_starts::CuVector{Int32},
    soc_var_sizes::CuVector{Int32},
    cone_count::Int,
    fact1_scalar::Float64,
    fact2_scalar::Float64,
    Halpern_fact1::Float64,
    Halpern_fact2::Float64) where {UseQ}
    threads, blocks = gpu_launch_config(cone_count)
    if threads > 0 && cone_count > 0
        @cuda threads = threads blocks = blocks unified_update_zxw1_SOC_small_kernel_partial!(
            Val(UseQ), Val(ws.Q_is_diag),
            ws.w_bar, ws.w, ws.x_hat, ws.last_x, ws.x,
            ws.Qw, ws.ATy, ws.c, ws.sigma, fact1_scalar, fact2_scalar, ws.fact1, ws.fact2,
            Halpern_fact1, Halpern_fact2, soc_var_starts, soc_var_sizes, cone_count)
    end
end

CUDA.@fastmath @inline function unified_update_zx_noQ_SOC_size3_kernel!(
    dx::CuDeviceVector{Float64},
    z_bar::CuDeviceVector{Float64},
    x_bar::CuDeviceVector{Float64},
    x_hat::CuDeviceVector{Float64},
    last_x::CuDeviceVector{Float64},
    x::CuDeviceVector{Float64},
    ATy::CuDeviceVector{Float64},
    c::CuDeviceVector{Float64},
    sigma::Float64,
    Halpern_fact1::Float64,
    Halpern_fact2::Float64,
    soc_var_starts::CuDeviceVector{Int32},
    cone_count::Int)
    i = threadIdx().x + (blockDim().x * (blockIdx().x - 1))
    @inbounds if i <= cone_count
        start_idx = Int(soc_var_starts[i])
        j1 = start_idx + 1
        j2 = start_idx + 2

        t_raw = _soc_raw_update_noQ(start_idx, x, ATy, c, sigma)
        z_raw_1 = _soc_raw_update_noQ(j1, x, ATy, c, sigma)
        z_raw_2 = _soc_raw_update_noQ(j2, x, ATy, c, sigma)
        norm_s = sqrt(z_raw_1^2 + z_raw_2^2)

        if norm_s <= -t_raw
            _write_soc_var_projection_noQ!(start_idx, 0.0, -t_raw / sigma,
                dx, z_bar, x_bar, x_hat, last_x, x, Halpern_fact1, Halpern_fact2)
            _write_soc_var_projection_noQ!(j1, 0.0, -z_raw_1 / sigma,
                dx, z_bar, x_bar, x_hat, last_x, x, Halpern_fact1, Halpern_fact2)
            _write_soc_var_projection_noQ!(j2, 0.0, -z_raw_2 / sigma,
                dx, z_bar, x_bar, x_hat, last_x, x, Halpern_fact1, Halpern_fact2)
        elseif norm_s <= t_raw
            _write_soc_var_projection_noQ!(start_idx, t_raw, 0.0,
                dx, z_bar, x_bar, x_hat, last_x, x, Halpern_fact1, Halpern_fact2)
            _write_soc_var_projection_noQ!(j1, z_raw_1, 0.0,
                dx, z_bar, x_bar, x_hat, last_x, x, Halpern_fact1, Halpern_fact2)
            _write_soc_var_projection_noQ!(j2, z_raw_2, 0.0,
                dx, z_bar, x_bar, x_hat, last_x, x, Halpern_fact1, Halpern_fact2)
        else
            proj_t = (norm_s + t_raw) / 2.0
            alpha = proj_t / norm_s
            proj_1 = alpha * z_raw_1
            proj_2 = alpha * z_raw_2

            _write_soc_var_projection_noQ!(start_idx, proj_t, (proj_t - t_raw) / sigma,
                dx, z_bar, x_bar, x_hat, last_x, x, Halpern_fact1, Halpern_fact2)
            _write_soc_var_projection_noQ!(j1, proj_1, (proj_1 - z_raw_1) / sigma,
                dx, z_bar, x_bar, x_hat, last_x, x, Halpern_fact1, Halpern_fact2)
            _write_soc_var_projection_noQ!(j2, proj_2, (proj_2 - z_raw_2) / sigma,
                dx, z_bar, x_bar, x_hat, last_x, x, Halpern_fact1, Halpern_fact2)
        end
    end
    return
end

CUDA.@fastmath @inline function unified_update_zx_noQ_SOC_generic_kernel!(
    dx::CuDeviceVector{Float64},
    z_bar::CuDeviceVector{Float64},
    x_bar::CuDeviceVector{Float64},
    x_hat::CuDeviceVector{Float64},
    last_x::CuDeviceVector{Float64},
    x::CuDeviceVector{Float64},
    ATy::CuDeviceVector{Float64},
    c::CuDeviceVector{Float64},
    sigma::Float64,
    Halpern_fact1::Float64,
    Halpern_fact2::Float64,
    soc_var_starts::CuDeviceVector{Int32},
    soc_var_sizes::CuDeviceVector{Int32},
    cone_count::Int)
    i = threadIdx().x + (blockDim().x * (blockIdx().x - 1))
    @inbounds if i <= cone_count
        start_idx = Int(soc_var_starts[i])
        cone_size = Int(soc_var_sizes[i])
        end_idx = start_idx + cone_size - 1

        t_raw = _soc_raw_update_noQ(start_idx, x, ATy, c, sigma)
        z_bar[start_idx] = t_raw
        norm_s = 0.0
        for j in (start_idx + 1):end_idx
            z_raw_j = _soc_raw_update_noQ(j, x, ATy, c, sigma)
            z_bar[j] = z_raw_j
            norm_s += z_raw_j^2
        end
        norm_s = sqrt(norm_s)

        if norm_s <= -t_raw
            _write_soc_var_projection_noQ!(start_idx, 0.0, -t_raw / sigma,
                dx, z_bar, x_bar, x_hat, last_x, x, Halpern_fact1, Halpern_fact2)

            for j in (start_idx + 1):end_idx
                z_raw_j = z_bar[j]
                _write_soc_var_projection_noQ!(j, 0.0, -z_raw_j / sigma,
                    dx, z_bar, x_bar, x_hat, last_x, x, Halpern_fact1, Halpern_fact2)
            end
        elseif norm_s <= t_raw
            _write_soc_var_projection_noQ!(start_idx, t_raw, 0.0,
                dx, z_bar, x_bar, x_hat, last_x, x, Halpern_fact1, Halpern_fact2)

            for j in (start_idx + 1):end_idx
                z_raw_j = z_bar[j]
                _write_soc_var_projection_noQ!(j, z_raw_j, 0.0,
                    dx, z_bar, x_bar, x_hat, last_x, x, Halpern_fact1, Halpern_fact2)
            end
        else
            proj_t = (norm_s + t_raw) / 2.0
            alpha = proj_t / norm_s

            _write_soc_var_projection_noQ!(start_idx, proj_t, (proj_t - t_raw) / sigma,
                dx, z_bar, x_bar, x_hat, last_x, x, Halpern_fact1, Halpern_fact2)

            for j in (start_idx + 1):end_idx
                z_raw_j = z_bar[j]
                proj_j = alpha * z_raw_j
                _write_soc_var_projection_noQ!(j, proj_j, (proj_j - z_raw_j) / sigma,
                    dx, z_bar, x_bar, x_hat, last_x, x, Halpern_fact1, Halpern_fact2)
            end
        end
    end
    return
end

CUDA.@fastmath function unified_update_zx_noQ_SOC_large_kernel!(
    dx::CuDeviceVector{Float64},
    z_bar::CuDeviceVector{Float64},
    x_bar::CuDeviceVector{Float64},
    x_hat::CuDeviceVector{Float64},
    last_x::CuDeviceVector{Float64},
    x::CuDeviceVector{Float64},
    ATy::CuDeviceVector{Float64},
    c::CuDeviceVector{Float64},
    sigma::Float64,
    Halpern_fact1::Float64,
    Halpern_fact2::Float64,
    soc_var_starts::CuDeviceVector{Int32},
    soc_var_sizes::CuDeviceVector{Int32},
    cone_count::Int)
    cone_idx = Int(blockIdx().x)
    tid = Int(threadIdx().x)
    block_threads = Int(blockDim().x)

    shared_meta = CuStaticSharedArray(Float64, 3)  # t_raw, proj_t, alpha
    shared_case = CuStaticSharedArray(Int32, 1)

    @inbounds if cone_idx <= cone_count
        start_idx = Int(soc_var_starts[cone_idx])
        cone_size = Int(soc_var_sizes[cone_idx])
        end_idx = start_idx + cone_size - 1

        if tid == 1
            t_raw = _soc_raw_update_noQ(start_idx, x, ATy, c, sigma)
            shared_meta[1] = t_raw
            z_bar[start_idx] = t_raw
        end

        for offset in tid:block_threads:(cone_size - 1)
            j = start_idx + offset
            z_bar[j] = _soc_raw_update_noQ(j, x, ATy, c, sigma)
        end
        sync_threads()

        if tid == 1
            t_raw = shared_meta[1]
            norm_s = 0.0
            for j in (start_idx + 1):end_idx
                z_raw_j = z_bar[j]
                norm_s += z_raw_j^2
            end
            norm_s = sqrt(norm_s)
            if norm_s <= -t_raw
                shared_case[1] = Int32(0)
                shared_meta[2] = 0.0
                shared_meta[3] = 0.0
            elseif norm_s <= t_raw
                shared_case[1] = Int32(1)
                shared_meta[2] = t_raw
                shared_meta[3] = 1.0
            else
                proj_t = (norm_s + t_raw) / 2.0
                shared_case[1] = Int32(2)
                shared_meta[2] = proj_t
                shared_meta[3] = proj_t / norm_s
            end
        end
        sync_threads()

        t_raw = shared_meta[1]
        proj_t = shared_meta[2]
        alpha = shared_meta[3]
        case_id = shared_case[1]

        if tid == 1
            projected_t = if case_id == 0
                0.0
            elseif case_id == 1
                t_raw
            else
                proj_t
            end
            z_t = if case_id == 0
                -t_raw / sigma
            elseif case_id == 1
                0.0
            else
                (proj_t - t_raw) / sigma
            end
            _write_soc_var_projection_noQ!(start_idx, projected_t, z_t,
                dx, z_bar, x_bar, x_hat, last_x, x, Halpern_fact1, Halpern_fact2)
        end

        for offset in tid:block_threads:(cone_size - 1)
            j = start_idx + offset
            z_raw_j = z_bar[j]
            projected_j = if case_id == 0
                0.0
            elseif case_id == 1
                z_raw_j
            else
                alpha * z_raw_j
            end
            z_j = if case_id == 0
                -z_raw_j / sigma
            elseif case_id == 1
                0.0
            else
                (projected_j - z_raw_j) / sigma
            end
            _write_soc_var_projection_noQ!(j, projected_j, z_j,
                dx, z_bar, x_bar, x_hat, last_x, x, Halpern_fact1, Halpern_fact2)
        end
    end
    return
end

CUDA.@fastmath function unified_update_zx_noQ_SOC_large_kernel_partial!(::Val{UseCustom},
    z_bar::CuDeviceVector{Float64},
    x_hat::CuDeviceVector{Float64},
    last_x::CuDeviceVector{Float64},
    x::CuDeviceVector{Float64},
    rowPtrAT::CuDeviceVector{Int32},
    colValAT::CuDeviceVector{Int32},
    nzValAT::CuDeviceVector{Float64},
    y::CuDeviceVector{Float64},
    ATy::CuDeviceVector{Float64},
    c::CuDeviceVector{Float64},
    sigma::Float64,
    Halpern_fact1::Float64,
    Halpern_fact2::Float64,
    soc_var_starts::CuDeviceVector{Int32},
    soc_var_sizes::CuDeviceVector{Int32},
    cone_count::Int) where {UseCustom}
    cone_idx = Int(blockIdx().x)
    tid = Int(threadIdx().x)
    block_threads = Int(blockDim().x)

    shared_meta = CuStaticSharedArray(Float64, 3)  # t_raw, proj_t, alpha
    shared_case = CuStaticSharedArray(Int32, 1)
    shared_sums = CuStaticSharedArray(Float64, DEFAULT_KERNEL_THREADS)

    @inbounds if cone_idx <= cone_count
        start_idx = Int(soc_var_starts[cone_idx])
        cone_size = Int(soc_var_sizes[cone_idx])

        if tid == 1
            t_raw = UseCustom ?
                _soc_raw_update_noQ_custom(start_idx, x, rowPtrAT, colValAT, nzValAT, y, c, sigma) :
                _soc_raw_update_noQ(start_idx, x, ATy, c, sigma)
            shared_meta[1] = t_raw
            z_bar[start_idx] = t_raw
        end

        # Use x_hat as temporary raw-value storage before overwriting it with
        # the final x_hat output for the projected SOC tail entries.
        for offset in tid:block_threads:(cone_size - 1)
            j = start_idx + offset
            x_hat[j] = UseCustom ?
                _soc_raw_update_noQ_custom(j, x, rowPtrAT, colValAT, nzValAT, y, c, sigma) :
                _soc_raw_update_noQ(j, x, ATy, c, sigma)
        end
        sync_threads()

        local_sum = 0.0
        for offset in tid:block_threads:(cone_size - 1)
            j = start_idx + offset
            z_raw_j = x_hat[j]
            local_sum += z_raw_j^2
        end
        shared_sums[tid] = local_sum
        sync_threads()

        # Tree reduction assumes a power-of-two block size (DEFAULT_KERNEL_THREADS=256).
        stride = block_threads ÷ 2
        while stride > 0
            if tid <= stride
                shared_sums[tid] += shared_sums[tid + stride]
            end
            sync_threads()
            stride ÷= 2
        end

        if tid == 1
            t_raw = shared_meta[1]
            norm_s = sqrt(shared_sums[1])
            if norm_s <= -t_raw
                shared_case[1] = Int32(0)
                shared_meta[2] = 0.0
                shared_meta[3] = 0.0
            elseif norm_s <= t_raw
                shared_case[1] = Int32(1)
                shared_meta[2] = t_raw
                shared_meta[3] = 1.0
            else
                proj_t = (norm_s + t_raw) / 2.0
                shared_case[1] = Int32(2)
                shared_meta[2] = proj_t
                shared_meta[3] = proj_t / norm_s
            end
        end
        sync_threads()

        t_raw = shared_meta[1]
        proj_t = shared_meta[2]
        alpha = shared_meta[3]
        case_id = shared_case[1]

        if tid == 1
            projected_t = if case_id == 0
                0.0
            elseif case_id == 1
                t_raw
            else
                proj_t
            end
            z_bar[start_idx] = if case_id == 0
                -t_raw / sigma
            elseif case_id == 1
                0.0
            else
                (proj_t - t_raw) / sigma
            end
            _write_soc_var_projection_noQ_partial!(
                start_idx, projected_t, x_hat, last_x, x, Halpern_fact1, Halpern_fact2
            )
        end

        for offset in tid:block_threads:(cone_size - 1)
            j = start_idx + offset
            z_raw_j = x_hat[j]
            projected_j = if case_id == 0
                0.0
            elseif case_id == 1
                z_raw_j
            else
                alpha * z_raw_j
            end
            z_bar[j] = if case_id == 0
                -z_raw_j / sigma
            elseif case_id == 1
                0.0
            else
                (projected_j - z_raw_j) / sigma
            end
            _write_soc_var_projection_noQ_partial!(
                j, projected_j, x_hat, last_x, x, Halpern_fact1, Halpern_fact2
            )
        end
    end
    return
end

CUDA.@fastmath function unified_update_zx_noQ_SOC_large_kernel_partial_legacy!(
    z_bar::CuDeviceVector{Float64},
    x_hat::CuDeviceVector{Float64},
    last_x::CuDeviceVector{Float64},
    x::CuDeviceVector{Float64},
    ATy::CuDeviceVector{Float64},
    c::CuDeviceVector{Float64},
    sigma::Float64,
    Halpern_fact1::Float64,
    Halpern_fact2::Float64,
    soc_var_starts::CuDeviceVector{Int32},
    soc_var_sizes::CuDeviceVector{Int32},
    cone_count::Int)
    cone_idx = Int(blockIdx().x)
    tid = Int(threadIdx().x)
    block_threads = Int(blockDim().x)

    shared_meta = CuStaticSharedArray(Float64, 3)  # t_raw, proj_t, alpha
    shared_case = CuStaticSharedArray(Int32, 1)

    @inbounds if cone_idx <= cone_count
        start_idx = Int(soc_var_starts[cone_idx])
        cone_size = Int(soc_var_sizes[cone_idx])
        end_idx = start_idx + cone_size - 1

        if tid == 1
            t_raw = _soc_raw_update_noQ(start_idx, x, ATy, c, sigma)
            shared_meta[1] = t_raw
            z_bar[start_idx] = t_raw
        end

        for offset in tid:block_threads:(cone_size - 1)
            j = start_idx + offset
            x_hat[j] = _soc_raw_update_noQ(j, x, ATy, c, sigma)
        end
        sync_threads()

        if tid == 1
            t_raw = shared_meta[1]
            norm_s = 0.0
            for j in (start_idx + 1):end_idx
                z_raw_j = x_hat[j]
                norm_s += z_raw_j^2
            end
            norm_s = sqrt(norm_s)
            if norm_s <= -t_raw
                shared_case[1] = Int32(0)
                shared_meta[2] = 0.0
                shared_meta[3] = 0.0
            elseif norm_s <= t_raw
                shared_case[1] = Int32(1)
                shared_meta[2] = t_raw
                shared_meta[3] = 1.0
            else
                proj_t = (norm_s + t_raw) / 2.0
                shared_case[1] = Int32(2)
                shared_meta[2] = proj_t
                shared_meta[3] = proj_t / norm_s
            end
        end
        sync_threads()

        t_raw = shared_meta[1]
        proj_t = shared_meta[2]
        alpha = shared_meta[3]
        case_id = shared_case[1]

        if tid == 1
            projected_t = if case_id == 0
                0.0
            elseif case_id == 1
                t_raw
            else
                proj_t
            end
            z_bar[start_idx] = if case_id == 0
                -t_raw / sigma
            elseif case_id == 1
                0.0
            else
                (proj_t - t_raw) / sigma
            end
            _write_soc_var_projection_noQ_partial!(
                start_idx, projected_t, x_hat, last_x, x, Halpern_fact1, Halpern_fact2
            )
        end

        for offset in tid:block_threads:(cone_size - 1)
            j = start_idx + offset
            z_raw_j = x_hat[j]
            projected_j = if case_id == 0
                0.0
            elseif case_id == 1
                z_raw_j
            else
                alpha * z_raw_j
            end
            z_bar[j] = if case_id == 0
                -z_raw_j / sigma
            elseif case_id == 1
                0.0
            else
                (projected_j - z_raw_j) / sigma
            end
            _write_soc_var_projection_noQ_partial!(
                j, projected_j, x_hat, last_x, x, Halpern_fact1, Halpern_fact2
            )
        end
    end
    return
end

CUDA.@fastmath function unified_update_zx_noQ_SOC_huge_kernel!(
    dx::CuDeviceVector{Float64},
    z_bar::CuDeviceVector{Float64},
    x_bar::CuDeviceVector{Float64},
    x_hat::CuDeviceVector{Float64},
    last_x::CuDeviceVector{Float64},
    x::CuDeviceVector{Float64},
    ATy::CuDeviceVector{Float64},
    c::CuDeviceVector{Float64},
    sigma::Float64,
    Halpern_fact1::Float64,
    Halpern_fact2::Float64,
    soc_var_starts::CuDeviceVector{Int32},
    soc_var_sizes::CuDeviceVector{Int32},
    cone_count::Int)
    cone_idx = Int(blockIdx().x)
    tid = Int(threadIdx().x)
    block_threads = Int(blockDim().x)

    shared_meta = CuStaticSharedArray(Float64, 3)  # t_raw, proj_t, alpha
    shared_case = CuStaticSharedArray(Int32, 1)
    shared_sums = CuStaticSharedArray(Float64, HUGE_SOC_KERNEL_THREADS)

    @inbounds if cone_idx <= cone_count
        start_idx = Int(soc_var_starts[cone_idx])
        cone_size = Int(soc_var_sizes[cone_idx])

        if tid == 1
            t_raw = _soc_raw_update_noQ(start_idx, x, ATy, c, sigma)
            shared_meta[1] = t_raw
            z_bar[start_idx] = t_raw
        end

        local_sum = 0.0
        for offset in tid:block_threads:(cone_size - 1)
            j = start_idx + offset
            z_raw_j = _soc_raw_update_noQ(j, x, ATy, c, sigma)
            z_bar[j] = z_raw_j
            local_sum += z_raw_j^2
        end
        shared_sums[tid] = local_sum
        sync_threads()

        stride = block_threads ÷ 2
        while stride >= 1
            if tid <= stride
                shared_sums[tid] += shared_sums[tid + stride]
            end
            sync_threads()
            stride ÷= 2
        end

        if tid == 1
            t_raw = shared_meta[1]
            norm_s = sqrt(shared_sums[1])
            if norm_s <= -t_raw
                shared_case[1] = Int32(0)
                shared_meta[2] = 0.0
                shared_meta[3] = 0.0
            elseif norm_s <= t_raw
                shared_case[1] = Int32(1)
                shared_meta[2] = t_raw
                shared_meta[3] = 1.0
            else
                proj_t = (norm_s + t_raw) / 2.0
                shared_case[1] = Int32(2)
                shared_meta[2] = proj_t
                shared_meta[3] = proj_t / norm_s
            end
        end
        sync_threads()

        t_raw = shared_meta[1]
        proj_t = shared_meta[2]
        alpha = shared_meta[3]
        case_id = shared_case[1]

        if tid == 1
            projected_t = if case_id == 0
                0.0
            elseif case_id == 1
                t_raw
            else
                proj_t
            end
            z_t = if case_id == 0
                -t_raw / sigma
            elseif case_id == 1
                0.0
            else
                (proj_t - t_raw) / sigma
            end
            _write_soc_var_projection_noQ!(start_idx, projected_t, z_t,
                dx, z_bar, x_bar, x_hat, last_x, x, Halpern_fact1, Halpern_fact2)
        end

        for offset in tid:block_threads:(cone_size - 1)
            j = start_idx + offset
            z_raw_j = z_bar[j]
            projected_j = if case_id == 0
                0.0
            elseif case_id == 1
                z_raw_j
            else
                alpha * z_raw_j
            end
            z_j = if case_id == 0
                -z_raw_j / sigma
            elseif case_id == 1
                0.0
            else
                (projected_j - z_raw_j) / sigma
            end
            _write_soc_var_projection_noQ!(j, projected_j, z_j,
                dx, z_bar, x_bar, x_hat, last_x, x, Halpern_fact1, Halpern_fact2)
        end
    end
    return
end

CUDA.@fastmath function unified_update_zx_noQ_SOC_huge_kernel_partial!(
    z_bar::CuDeviceVector{Float64},
    x_hat::CuDeviceVector{Float64},
    last_x::CuDeviceVector{Float64},
    x::CuDeviceVector{Float64},
    ATy::CuDeviceVector{Float64},
    c::CuDeviceVector{Float64},
    sigma::Float64,
    Halpern_fact1::Float64,
    Halpern_fact2::Float64,
    soc_var_starts::CuDeviceVector{Int32},
    soc_var_sizes::CuDeviceVector{Int32},
    cone_count::Int)
    cone_idx = Int(blockIdx().x)
    tid = Int(threadIdx().x)
    block_threads = Int(blockDim().x)

    shared_meta = CuStaticSharedArray(Float64, 3)  # t_raw, proj_t, alpha
    shared_case = CuStaticSharedArray(Int32, 1)
    shared_sums = CuStaticSharedArray(Float64, HUGE_SOC_KERNEL_THREADS)

    @inbounds if cone_idx <= cone_count
        start_idx = Int(soc_var_starts[cone_idx])
        cone_size = Int(soc_var_sizes[cone_idx])

        if tid == 1
            t_raw = _soc_raw_update_noQ(start_idx, x, ATy, c, sigma)
            shared_meta[1] = t_raw
            z_bar[start_idx] = t_raw
        end

        local_sum = 0.0
        # Use x_hat as temporary raw-value storage before overwriting it with
        # the final x_hat output for the projected SOC tail entries.
        for offset in tid:block_threads:(cone_size - 1)
            j = start_idx + offset
            z_raw_j = _soc_raw_update_noQ(j, x, ATy, c, sigma)
            x_hat[j] = z_raw_j
            local_sum += z_raw_j^2
        end
        shared_sums[tid] = local_sum
        sync_threads()

        stride = block_threads ÷ 2
        while stride >= 1
            if tid <= stride
                shared_sums[tid] += shared_sums[tid + stride]
            end
            sync_threads()
            stride ÷= 2
        end

        if tid == 1
            t_raw = shared_meta[1]
            norm_s = sqrt(shared_sums[1])
            if norm_s <= -t_raw
                shared_case[1] = Int32(0)
                shared_meta[2] = 0.0
                shared_meta[3] = 0.0
            elseif norm_s <= t_raw
                shared_case[1] = Int32(1)
                shared_meta[2] = t_raw
                shared_meta[3] = 1.0
            else
                proj_t = (norm_s + t_raw) / 2.0
                shared_case[1] = Int32(2)
                shared_meta[2] = proj_t
                shared_meta[3] = proj_t / norm_s
            end
        end
        sync_threads()

        t_raw = shared_meta[1]
        proj_t = shared_meta[2]
        alpha = shared_meta[3]
        case_id = shared_case[1]

        if tid == 1
            projected_t = if case_id == 0
                0.0
            elseif case_id == 1
                t_raw
            else
                proj_t
            end
            z_bar[start_idx] = if case_id == 0
                -t_raw / sigma
            elseif case_id == 1
                0.0
            else
                (proj_t - t_raw) / sigma
            end
            _write_soc_var_projection_noQ_partial!(
                start_idx, projected_t, x_hat, last_x, x, Halpern_fact1, Halpern_fact2
            )
        end

        for offset in tid:block_threads:(cone_size - 1)
            j = start_idx + offset
            z_raw_j = x_hat[j]
            projected_j = if case_id == 0
                0.0
            elseif case_id == 1
                z_raw_j
            else
                alpha * z_raw_j
            end
            z_bar[j] = if case_id == 0
                -z_raw_j / sigma
            elseif case_id == 1
                0.0
            else
                (projected_j - z_raw_j) / sigma
            end
            _write_soc_var_projection_noQ_partial!(
                j, projected_j, x_hat, last_x, x, Halpern_fact1, Halpern_fact2
            )
        end
    end
    return
end

CUDA.@fastmath function unified_update_zx_noQ_SOC_huge_cooperative_kernel!(
    dx::CuDeviceVector{Float64},
    z_bar::CuDeviceVector{Float64},
    x_bar::CuDeviceVector{Float64},
    x_hat::CuDeviceVector{Float64},
    last_x::CuDeviceVector{Float64},
    x::CuDeviceVector{Float64},
    ATy::CuDeviceVector{Float64},
    c::CuDeviceVector{Float64},
    sigma::Float64,
    Halpern_fact1::Float64,
    Halpern_fact2::Float64,
    huge_sizes::CuDeviceVector{Int32},
    huge_block_starts::CuDeviceVector{Int32},
    huge_block_offsets::CuDeviceVector{Int32},
    huge_block_cone_ids::CuDeviceVector{Int32},
    huge_block_ptr::CuDeviceVector{Int32},
    huge_partial_sums::CuDeviceVector{Float64},
    huge_t_raw::CuDeviceVector{Float64},
    huge_proj_t::CuDeviceVector{Float64},
    huge_alpha::CuDeviceVector{Float64},
    huge_case::CuDeviceVector{Int32},
    total_blocks::Int)
    block_id = Int(blockIdx().x)
    tid = Int(threadIdx().x)
    block_threads = Int(blockDim().x)
    grid = CUDA.CG.this_grid()
    shared_sums = CuStaticSharedArray(Float64, HUGE_SOC_KERNEL_THREADS)
    tail_raw_cached = 0.0
    has_tail = false

    @inbounds if block_id <= total_blocks
        cone_id = Int(huge_block_cone_ids[block_id])
        start_idx = Int(huge_block_starts[block_id])
        tail_offset = Int(huge_block_offsets[block_id])
        cone_size = Int(huge_sizes[cone_id])

        if tail_offset == 0 && tid == 1
            huge_t_raw[cone_id] = _soc_raw_update_noQ(start_idx, x, ATy, c, sigma)
        end

        local_sum = 0.0
        tail_pos = tail_offset + tid
        if tail_pos <= cone_size - 1
            j = start_idx + tail_pos
            tail_raw_cached = _soc_raw_update_noQ(j, x, ATy, c, sigma)
            has_tail = true
            local_sum = tail_raw_cached^2
        end
        shared_sums[tid] = local_sum
        sync_threads()

        stride = block_threads ÷ 2
        while stride >= 1
            if tid <= stride
                shared_sums[tid] += shared_sums[tid + stride]
            end
            sync_threads()
            stride ÷= 2
        end

        if tid == 1
            huge_partial_sums[block_id] = shared_sums[1]
        end
    end

    CUDA.CG.sync(grid)

    @inbounds if block_id <= total_blocks
        cone_id = Int(huge_block_cone_ids[block_id])
        if block_id == Int(huge_block_ptr[cone_id])
            block_start = Int(huge_block_ptr[cone_id])
            block_stop = Int(huge_block_ptr[cone_id + 1]) - 1
            local_sum = 0.0
            if tid <= DEFAULT_KERNEL_THREADS
                partial_idx = block_start + tid - 1
                while partial_idx <= block_stop
                    local_sum += huge_partial_sums[partial_idx]
                    partial_idx += DEFAULT_KERNEL_THREADS
                end
            end
            shared_sums[tid] = tid <= DEFAULT_KERNEL_THREADS ? local_sum : 0.0
            sync_threads()

            stride = DEFAULT_KERNEL_THREADS ÷ 2
            while stride >= 1
                if tid <= stride
                    shared_sums[tid] += shared_sums[tid + stride]
                end
                sync_threads()
                stride ÷= 2
            end

            if tid == 1
                t_raw = huge_t_raw[cone_id]
                norm_s = sqrt(shared_sums[1])
                if norm_s <= -t_raw
                    huge_case[cone_id] = Int32(0)
                    huge_proj_t[cone_id] = 0.0
                    huge_alpha[cone_id] = 0.0
                elseif norm_s <= t_raw
                    huge_case[cone_id] = Int32(1)
                    huge_proj_t[cone_id] = t_raw
                    huge_alpha[cone_id] = 1.0
                else
                    proj_t = (norm_s + t_raw) / 2.0
                    huge_case[cone_id] = Int32(2)
                    huge_proj_t[cone_id] = proj_t
                    huge_alpha[cone_id] = proj_t / norm_s
                end
            end
        end
    end

    CUDA.CG.sync(grid)

    @inbounds if block_id <= total_blocks
        cone_id = Int(huge_block_cone_ids[block_id])
        start_idx = Int(huge_block_starts[block_id])
        tail_offset = Int(huge_block_offsets[block_id])
        cone_size = Int(huge_sizes[cone_id])
        t_raw = huge_t_raw[cone_id]
        proj_t = huge_proj_t[cone_id]
        alpha = huge_alpha[cone_id]
        case_id = huge_case[cone_id]

        if tail_offset == 0 && tid == 1
            projected_t = if case_id == 0
                0.0
            elseif case_id == 1
                t_raw
            else
                proj_t
            end
            z_t = if case_id == 0
                -t_raw / sigma
            elseif case_id == 1
                0.0
            else
                (proj_t - t_raw) / sigma
            end
            _write_soc_var_projection_noQ!(start_idx, projected_t, z_t,
                dx, z_bar, x_bar, x_hat, last_x, x, Halpern_fact1, Halpern_fact2)
        end

        tail_pos = tail_offset + tid
        if has_tail
            j = start_idx + tail_pos
            z_raw_j = tail_raw_cached
            projected_j = if case_id == 0
                0.0
            elseif case_id == 1
                z_raw_j
            else
                alpha * z_raw_j
            end
            z_j = if case_id == 0
                -z_raw_j / sigma
            elseif case_id == 1
                0.0
            else
                (projected_j - z_raw_j) / sigma
            end
            _write_soc_var_projection_noQ!(j, projected_j, z_j,
                dx, z_bar, x_bar, x_hat, last_x, x, Halpern_fact1, Halpern_fact2)
        end
    end
    return
end

CUDA.@fastmath function unified_update_zx_noQ_SOC_huge_cooperative_kernel_partial!(::Val{UseCustom},
    z_bar::CuDeviceVector{Float64},
    x_hat::CuDeviceVector{Float64},
    last_x::CuDeviceVector{Float64},
    x::CuDeviceVector{Float64},
    rowPtrAT::CuDeviceVector{Int32},
    colValAT::CuDeviceVector{Int32},
    nzValAT::CuDeviceVector{Float64},
    y::CuDeviceVector{Float64},
    ATy::CuDeviceVector{Float64},
    c::CuDeviceVector{Float64},
    sigma::Float64,
    Halpern_fact1::Float64,
    Halpern_fact2::Float64,
    huge_sizes::CuDeviceVector{Int32},
    huge_block_starts::CuDeviceVector{Int32},
    huge_block_offsets::CuDeviceVector{Int32},
    huge_block_cone_ids::CuDeviceVector{Int32},
    huge_block_ptr::CuDeviceVector{Int32},
    huge_partial_sums::CuDeviceVector{Float64},
    huge_t_raw::CuDeviceVector{Float64},
    huge_proj_t::CuDeviceVector{Float64},
    huge_alpha::CuDeviceVector{Float64},
    huge_case::CuDeviceVector{Int32},
    total_blocks::Int) where {UseCustom}
    block_id = Int(blockIdx().x)
    tid = Int(threadIdx().x)
    block_threads = Int(blockDim().x)
    grid = CUDA.CG.this_grid()
    shared_sums = CuStaticSharedArray(Float64, HUGE_SOC_KERNEL_THREADS)
    tail_raw_cached = 0.0
    has_tail = false

    @inbounds if block_id <= total_blocks
        cone_id = Int(huge_block_cone_ids[block_id])
        start_idx = Int(huge_block_starts[block_id])
        tail_offset = Int(huge_block_offsets[block_id])
        cone_size = Int(huge_sizes[cone_id])

        if tail_offset == 0 && tid == 1
            huge_t_raw[cone_id] = UseCustom ?
                _soc_raw_update_noQ_custom(start_idx, x, rowPtrAT, colValAT, nzValAT, y, c, sigma) :
                _soc_raw_update_noQ(start_idx, x, ATy, c, sigma)
        end

        local_sum = 0.0
        tail_pos = tail_offset + tid
        if tail_pos <= cone_size - 1
            j = start_idx + tail_pos
            tail_raw_cached = UseCustom ?
                _soc_raw_update_noQ_custom(j, x, rowPtrAT, colValAT, nzValAT, y, c, sigma) :
                _soc_raw_update_noQ(j, x, ATy, c, sigma)
            has_tail = true
            local_sum = tail_raw_cached^2
        end
        shared_sums[tid] = local_sum
        sync_threads()

        stride = block_threads ÷ 2
        while stride >= 1
            if tid <= stride
                shared_sums[tid] += shared_sums[tid + stride]
            end
            sync_threads()
            stride ÷= 2
        end

        if tid == 1
            huge_partial_sums[block_id] = shared_sums[1]
        end
    end

    CUDA.CG.sync(grid)

    @inbounds if block_id <= total_blocks
        cone_id = Int(huge_block_cone_ids[block_id])
        if block_id == Int(huge_block_ptr[cone_id])
            block_start = Int(huge_block_ptr[cone_id])
            block_stop = Int(huge_block_ptr[cone_id + 1]) - 1
            local_sum = 0.0
            if tid <= DEFAULT_KERNEL_THREADS
                partial_idx = block_start + tid - 1
                while partial_idx <= block_stop
                    local_sum += huge_partial_sums[partial_idx]
                    partial_idx += DEFAULT_KERNEL_THREADS
                end
            end
            shared_sums[tid] = tid <= DEFAULT_KERNEL_THREADS ? local_sum : 0.0
            sync_threads()

            stride = DEFAULT_KERNEL_THREADS ÷ 2
            while stride >= 1
                if tid <= stride
                    shared_sums[tid] += shared_sums[tid + stride]
                end
                sync_threads()
                stride ÷= 2
            end

            if tid == 1
                t_raw = huge_t_raw[cone_id]
                norm_s = sqrt(shared_sums[1])
                if norm_s <= -t_raw
                    huge_case[cone_id] = Int32(0)
                    huge_proj_t[cone_id] = 0.0
                    huge_alpha[cone_id] = 0.0
                elseif norm_s <= t_raw
                    huge_case[cone_id] = Int32(1)
                    huge_proj_t[cone_id] = t_raw
                    huge_alpha[cone_id] = 1.0
                else
                    proj_t = (norm_s + t_raw) / 2.0
                    huge_case[cone_id] = Int32(2)
                    huge_proj_t[cone_id] = proj_t
                    huge_alpha[cone_id] = proj_t / norm_s
                end
            end
        end
    end

    CUDA.CG.sync(grid)

    @inbounds if block_id <= total_blocks
        cone_id = Int(huge_block_cone_ids[block_id])
        start_idx = Int(huge_block_starts[block_id])
        tail_offset = Int(huge_block_offsets[block_id])
        cone_size = Int(huge_sizes[cone_id])
        t_raw = huge_t_raw[cone_id]
        proj_t = huge_proj_t[cone_id]
        alpha = huge_alpha[cone_id]
        case_id = huge_case[cone_id]

        if tail_offset == 0 && tid == 1
            projected_t = if case_id == 0
                0.0
            elseif case_id == 1
                t_raw
            else
                proj_t
            end
            z_bar[start_idx] = if case_id == 0
                -t_raw / sigma
            elseif case_id == 1
                0.0
            else
                (proj_t - t_raw) / sigma
            end
            _write_soc_var_projection_noQ_partial!(
                start_idx, projected_t, x_hat, last_x, x, Halpern_fact1, Halpern_fact2
            )
        end

        tail_pos = tail_offset + tid
        if has_tail
            j = start_idx + tail_pos
            z_raw_j = tail_raw_cached
            projected_j = if case_id == 0
                0.0
            elseif case_id == 1
                z_raw_j
            else
                alpha * z_raw_j
            end
            z_bar[j] = if case_id == 0
                -z_raw_j / sigma
            elseif case_id == 1
                0.0
            else
                (projected_j - z_raw_j) / sigma
            end
            _write_soc_var_projection_noQ_partial!(
                j, projected_j, x_hat, last_x, x, Halpern_fact1, Halpern_fact2
            )
        end
    end
    return
end

CUDA.@fastmath function prepare_unified_update_zx_noQ_SOC_huge_segmented_kernel!(
    huge_partial_sums::CuDeviceVector{Float64},
    huge_t_raw::CuDeviceVector{Float64},
    x::CuDeviceVector{Float64},
    ATy::CuDeviceVector{Float64},
    c::CuDeviceVector{Float64},
    sigma::Float64,
    huge_sizes::CuDeviceVector{Int32},
    huge_block_starts::CuDeviceVector{Int32},
    huge_block_offsets::CuDeviceVector{Int32},
    huge_block_cone_ids::CuDeviceVector{Int32},
    total_blocks::Int)
    block_id = Int(blockIdx().x)
    tid = Int(threadIdx().x)
    block_threads = Int(blockDim().x)
    shared_sums = CuStaticSharedArray(Float64, HUGE_SOC_KERNEL_THREADS)

    @inbounds if block_id <= total_blocks
        cone_id = Int(huge_block_cone_ids[block_id])
        start_idx = Int(huge_block_starts[block_id])
        tail_offset = Int(huge_block_offsets[block_id])
        cone_size = Int(huge_sizes[cone_id])

        if tail_offset == 0 && tid == 1
            huge_t_raw[cone_id] = _soc_raw_update_noQ(start_idx, x, ATy, c, sigma)
        end

        local_sum = 0.0
        tail_pos = tail_offset + tid
        if tail_pos <= cone_size - 1
            j = start_idx + tail_pos
            z_raw_j = _soc_raw_update_noQ(j, x, ATy, c, sigma)
            local_sum = z_raw_j^2
        end
        shared_sums[tid] = local_sum
        sync_threads()

        stride = block_threads ÷ 2
        while stride >= 1
            if tid <= stride
                shared_sums[tid] += shared_sums[tid + stride]
            end
            sync_threads()
            stride ÷= 2
        end

        if tid == 1
            huge_partial_sums[block_id] = shared_sums[1]
        end
    end
    return
end

CUDA.@fastmath function apply_unified_update_zx_noQ_SOC_huge_segmented_kernel!(
    dx::CuDeviceVector{Float64},
    z_bar::CuDeviceVector{Float64},
    x_bar::CuDeviceVector{Float64},
    x_hat::CuDeviceVector{Float64},
    last_x::CuDeviceVector{Float64},
    x::CuDeviceVector{Float64},
    ATy::CuDeviceVector{Float64},
    c::CuDeviceVector{Float64},
    sigma::Float64,
    Halpern_fact1::Float64,
    Halpern_fact2::Float64,
    huge_sizes::CuDeviceVector{Int32},
    huge_block_starts::CuDeviceVector{Int32},
    huge_block_offsets::CuDeviceVector{Int32},
    huge_block_cone_ids::CuDeviceVector{Int32},
    huge_t_raw::CuDeviceVector{Float64},
    huge_proj_t::CuDeviceVector{Float64},
    huge_alpha::CuDeviceVector{Float64},
    huge_case::CuDeviceVector{Int32},
    total_blocks::Int)
    block_id = Int(blockIdx().x)
    tid = Int(threadIdx().x)

    @inbounds if block_id <= total_blocks
        cone_id = Int(huge_block_cone_ids[block_id])
        start_idx = Int(huge_block_starts[block_id])
        tail_offset = Int(huge_block_offsets[block_id])
        cone_size = Int(huge_sizes[cone_id])
        t_raw = huge_t_raw[cone_id]
        proj_t = huge_proj_t[cone_id]
        alpha = huge_alpha[cone_id]
        case_id = huge_case[cone_id]

        if tail_offset == 0 && tid == 1
            projected_t = if case_id == 0
                0.0
            elseif case_id == 1
                t_raw
            else
                proj_t
            end
            z_t = if case_id == 0
                -t_raw / sigma
            elseif case_id == 1
                0.0
            else
                (proj_t - t_raw) / sigma
            end
            _write_soc_var_projection_noQ!(start_idx, projected_t, z_t,
                dx, z_bar, x_bar, x_hat, last_x, x, Halpern_fact1, Halpern_fact2)
        end

        tail_pos = tail_offset + tid
        if tail_pos <= cone_size - 1
            j = start_idx + tail_pos
            z_raw_j = _soc_raw_update_noQ(j, x, ATy, c, sigma)
            projected_j = if case_id == 0
                0.0
            elseif case_id == 1
                z_raw_j
            else
                alpha * z_raw_j
            end
            z_j = if case_id == 0
                -z_raw_j / sigma
            elseif case_id == 1
                0.0
            else
                (projected_j - z_raw_j) / sigma
            end
            _write_soc_var_projection_noQ!(j, projected_j, z_j,
                dx, z_bar, x_bar, x_hat, last_x, x, Halpern_fact1, Halpern_fact2)
        end
    end
    return
end

CUDA.@fastmath function prepare_unified_update_zx_noQ_SOC_huge_segmented_kernel_partial!(::Val{UseCustom},
    huge_partial_sums::CuDeviceVector{Float64},
    huge_t_raw::CuDeviceVector{Float64},
    x::CuDeviceVector{Float64},
    rowPtrAT::CuDeviceVector{Int32},
    colValAT::CuDeviceVector{Int32},
    nzValAT::CuDeviceVector{Float64},
    y::CuDeviceVector{Float64},
    ATy::CuDeviceVector{Float64},
    c::CuDeviceVector{Float64},
    sigma::Float64,
    huge_sizes::CuDeviceVector{Int32},
    huge_block_starts::CuDeviceVector{Int32},
    huge_block_offsets::CuDeviceVector{Int32},
    huge_block_cone_ids::CuDeviceVector{Int32},
    total_blocks::Int) where {UseCustom}
    block_id = Int(blockIdx().x)
    tid = Int(threadIdx().x)
    block_threads = Int(blockDim().x)
    shared_sums = CuStaticSharedArray(Float64, HUGE_SOC_KERNEL_THREADS)

    @inbounds if block_id <= total_blocks
        cone_id = Int(huge_block_cone_ids[block_id])
        start_idx = Int(huge_block_starts[block_id])
        tail_offset = Int(huge_block_offsets[block_id])
        cone_size = Int(huge_sizes[cone_id])

        if tail_offset == 0 && tid == 1
            huge_t_raw[cone_id] = UseCustom ?
                _soc_raw_update_noQ_custom(start_idx, x, rowPtrAT, colValAT, nzValAT, y, c, sigma) :
                _soc_raw_update_noQ(start_idx, x, ATy, c, sigma)
        end

        local_sum = 0.0
        tail_pos = tail_offset + tid
        if tail_pos <= cone_size - 1
            j = start_idx + tail_pos
            z_raw_j = UseCustom ?
                _soc_raw_update_noQ_custom(j, x, rowPtrAT, colValAT, nzValAT, y, c, sigma) :
                _soc_raw_update_noQ(j, x, ATy, c, sigma)
            local_sum = z_raw_j^2
        end
        shared_sums[tid] = local_sum
        sync_threads()

        stride = block_threads ÷ 2
        while stride >= 1
            if tid <= stride
                shared_sums[tid] += shared_sums[tid + stride]
            end
            sync_threads()
            stride ÷= 2
        end

        if tid == 1
            huge_partial_sums[block_id] = shared_sums[1]
        end
    end
    return
end

CUDA.@fastmath function apply_unified_update_zx_noQ_SOC_huge_segmented_kernel_partial!(::Val{UseCustom},
    z_bar::CuDeviceVector{Float64},
    x_hat::CuDeviceVector{Float64},
    last_x::CuDeviceVector{Float64},
    x::CuDeviceVector{Float64},
    rowPtrAT::CuDeviceVector{Int32},
    colValAT::CuDeviceVector{Int32},
    nzValAT::CuDeviceVector{Float64},
    y::CuDeviceVector{Float64},
    ATy::CuDeviceVector{Float64},
    c::CuDeviceVector{Float64},
    sigma::Float64,
    Halpern_fact1::Float64,
    Halpern_fact2::Float64,
    huge_sizes::CuDeviceVector{Int32},
    huge_block_starts::CuDeviceVector{Int32},
    huge_block_offsets::CuDeviceVector{Int32},
    huge_block_cone_ids::CuDeviceVector{Int32},
    huge_t_raw::CuDeviceVector{Float64},
    huge_proj_t::CuDeviceVector{Float64},
    huge_alpha::CuDeviceVector{Float64},
    huge_case::CuDeviceVector{Int32},
    total_blocks::Int) where {UseCustom}
    block_id = Int(blockIdx().x)
    tid = Int(threadIdx().x)

    @inbounds if block_id <= total_blocks
        cone_id = Int(huge_block_cone_ids[block_id])
        start_idx = Int(huge_block_starts[block_id])
        tail_offset = Int(huge_block_offsets[block_id])
        cone_size = Int(huge_sizes[cone_id])
        t_raw = huge_t_raw[cone_id]
        proj_t = huge_proj_t[cone_id]
        alpha = huge_alpha[cone_id]
        case_id = huge_case[cone_id]

        if tail_offset == 0 && tid == 1
            projected_t = if case_id == 0
                0.0
            elseif case_id == 1
                t_raw
            else
                proj_t
            end
            z_bar[start_idx] = if case_id == 0
                -t_raw / sigma
            elseif case_id == 1
                0.0
            else
                (proj_t - t_raw) / sigma
            end
            _write_soc_var_projection_noQ_partial!(
                start_idx, projected_t, x_hat, last_x, x, Halpern_fact1, Halpern_fact2
            )
        end

        tail_pos = tail_offset + tid
        if tail_pos <= cone_size - 1
            j = start_idx + tail_pos
            z_raw_j = UseCustom ?
                _soc_raw_update_noQ_custom(j, x, rowPtrAT, colValAT, nzValAT, y, c, sigma) :
                _soc_raw_update_noQ(j, x, ATy, c, sigma)
            projected_j = if case_id == 0
                0.0
            elseif case_id == 1
                z_raw_j
            else
                alpha * z_raw_j
            end
            z_bar[j] = if case_id == 0
                -z_raw_j / sigma
            elseif case_id == 1
                0.0
            else
                (projected_j - z_raw_j) / sigma
            end
            _write_soc_var_projection_noQ_partial!(
                j, projected_j, x_hat, last_x, x, Halpern_fact1, Halpern_fact2
            )
        end
    end
    return
end

CUDA.@fastmath function prepare_unified_update_zx_noQ_SOC_huge_kernel!(
    z_bar::CuDeviceVector{Float64},
    huge_partial_sums::CuDeviceVector{Float64},
    huge_t_raw::CuDeviceVector{Float64},
    x::CuDeviceVector{Float64},
    ATy::CuDeviceVector{Float64},
    c::CuDeviceVector{Float64},
    sigma::Float64,
    huge_sizes::CuDeviceVector{Int32},
    huge_block_starts::CuDeviceVector{Int32},
    huge_block_offsets::CuDeviceVector{Int32},
    huge_block_cone_ids::CuDeviceVector{Int32},
    total_blocks::Int)
    block_id = Int(blockIdx().x)
    tid = Int(threadIdx().x)
    block_threads = Int(blockDim().x)
    shared_sums = CuStaticSharedArray(Float64, HUGE_SOC_KERNEL_THREADS)

    @inbounds if block_id <= total_blocks
        cone_id = Int(huge_block_cone_ids[block_id])
        start_idx = Int(huge_block_starts[block_id])
        tail_offset = Int(huge_block_offsets[block_id])
        cone_size = Int(huge_sizes[cone_id])

        if tail_offset == 0 && tid == 1
            t_raw = _soc_raw_update_noQ(start_idx, x, ATy, c, sigma)
            huge_t_raw[cone_id] = t_raw
            z_bar[start_idx] = t_raw
        end

        local_sum = 0.0
        tail_pos = tail_offset + tid
        if tail_pos <= cone_size - 1
            j = start_idx + tail_pos
            z_raw_j = _soc_raw_update_noQ(j, x, ATy, c, sigma)
            z_bar[j] = z_raw_j
            local_sum = z_raw_j^2
        end
        shared_sums[tid] = local_sum
        sync_threads()

        stride = block_threads ÷ 2
        while stride >= 1
            if tid <= stride
                shared_sums[tid] += shared_sums[tid + stride]
            end
            sync_threads()
            stride ÷= 2
        end

        if tid == 1
            huge_partial_sums[block_id] = shared_sums[1]
        end
    end
    return
end

CUDA.@fastmath function finalize_unified_update_zx_noQ_SOC_huge_kernel!(
    huge_proj_t::CuDeviceVector{Float64},
    huge_alpha::CuDeviceVector{Float64},
    huge_case::CuDeviceVector{Int32},
    huge_partial_sums::CuDeviceVector{Float64},
    huge_t_raw::CuDeviceVector{Float64},
    huge_block_ptr::CuDeviceVector{Int32},
    cone_count::Int)
    cone_id = Int(blockIdx().x)
    tid = Int(threadIdx().x)
    block_threads = Int(blockDim().x)
    shared_sums = CuStaticSharedArray(Float64, DEFAULT_KERNEL_THREADS)

    @inbounds if cone_id <= cone_count
        block_start = Int(huge_block_ptr[cone_id])
        block_stop = Int(huge_block_ptr[cone_id + 1]) - 1
        local_sum = 0.0
        partial_idx = block_start + tid - 1
        while partial_idx <= block_stop
            local_sum += huge_partial_sums[partial_idx]
            partial_idx += block_threads
        end
        shared_sums[tid] = local_sum
        sync_threads()

        stride = block_threads ÷ 2
        while stride >= 1
            if tid <= stride
                shared_sums[tid] += shared_sums[tid + stride]
            end
            sync_threads()
            stride ÷= 2
        end

        if tid == 1
            t_raw = huge_t_raw[cone_id]
            norm_s = sqrt(shared_sums[1])
            if norm_s <= -t_raw
                huge_case[cone_id] = Int32(0)
                huge_proj_t[cone_id] = 0.0
                huge_alpha[cone_id] = 0.0
            elseif norm_s <= t_raw
                huge_case[cone_id] = Int32(1)
                huge_proj_t[cone_id] = t_raw
                huge_alpha[cone_id] = 1.0
            else
                proj_t = (norm_s + t_raw) / 2.0
                huge_case[cone_id] = Int32(2)
                huge_proj_t[cone_id] = proj_t
                huge_alpha[cone_id] = proj_t / norm_s
            end
        end
    end
    return
end

CUDA.@fastmath function apply_unified_update_zx_noQ_SOC_huge_kernel!(
    dx::CuDeviceVector{Float64},
    z_bar::CuDeviceVector{Float64},
    x_bar::CuDeviceVector{Float64},
    x_hat::CuDeviceVector{Float64},
    last_x::CuDeviceVector{Float64},
    x::CuDeviceVector{Float64},
    sigma::Float64,
    Halpern_fact1::Float64,
    Halpern_fact2::Float64,
    huge_sizes::CuDeviceVector{Int32},
    huge_block_starts::CuDeviceVector{Int32},
    huge_block_offsets::CuDeviceVector{Int32},
    huge_block_cone_ids::CuDeviceVector{Int32},
    huge_t_raw::CuDeviceVector{Float64},
    huge_proj_t::CuDeviceVector{Float64},
    huge_alpha::CuDeviceVector{Float64},
    huge_case::CuDeviceVector{Int32},
    total_blocks::Int)
    block_id = Int(blockIdx().x)
    tid = Int(threadIdx().x)

    @inbounds if block_id <= total_blocks
        cone_id = Int(huge_block_cone_ids[block_id])
        start_idx = Int(huge_block_starts[block_id])
        tail_offset = Int(huge_block_offsets[block_id])
        cone_size = Int(huge_sizes[cone_id])
        t_raw = huge_t_raw[cone_id]
        proj_t = huge_proj_t[cone_id]
        alpha = huge_alpha[cone_id]
        case_id = huge_case[cone_id]

        if tail_offset == 0 && tid == 1
            projected_t = if case_id == 0
                0.0
            elseif case_id == 1
                t_raw
            else
                proj_t
            end
            z_t = if case_id == 0
                -t_raw / sigma
            elseif case_id == 1
                0.0
            else
                (proj_t - t_raw) / sigma
            end
            _write_soc_var_projection_noQ!(start_idx, projected_t, z_t,
                dx, z_bar, x_bar, x_hat, last_x, x, Halpern_fact1, Halpern_fact2)
        end

        tail_pos = tail_offset + tid
        if tail_pos <= cone_size - 1
            j = start_idx + tail_pos
            z_raw_j = z_bar[j]
            projected_j = if case_id == 0
                0.0
            elseif case_id == 1
                z_raw_j
            else
                alpha * z_raw_j
            end
            z_j = if case_id == 0
                -z_raw_j / sigma
            elseif case_id == 1
                0.0
            else
                (projected_j - z_raw_j) / sigma
            end
            _write_soc_var_projection_noQ!(j, projected_j, z_j,
                dx, z_bar, x_bar, x_hat, last_x, x, Halpern_fact1, Halpern_fact2)
        end
    end
    return
end

CUDA.@fastmath function prepare_unified_update_zx_noQ_SOC_huge_kernel_partial!(::Val{UseCustom},
    z_bar::CuDeviceVector{Float64},
    x_hat::CuDeviceVector{Float64},
    huge_partial_sums::CuDeviceVector{Float64},
    huge_t_raw::CuDeviceVector{Float64},
    x::CuDeviceVector{Float64},
    rowPtrAT::CuDeviceVector{Int32},
    colValAT::CuDeviceVector{Int32},
    nzValAT::CuDeviceVector{Float64},
    y::CuDeviceVector{Float64},
    ATy::CuDeviceVector{Float64},
    c::CuDeviceVector{Float64},
    sigma::Float64,
    huge_sizes::CuDeviceVector{Int32},
    huge_block_starts::CuDeviceVector{Int32},
    huge_block_offsets::CuDeviceVector{Int32},
    huge_block_cone_ids::CuDeviceVector{Int32},
    total_blocks::Int) where {UseCustom}
    block_id = Int(blockIdx().x)
    tid = Int(threadIdx().x)
    block_threads = Int(blockDim().x)
    shared_sums = CuStaticSharedArray(Float64, HUGE_SOC_KERNEL_THREADS)

    @inbounds if block_id <= total_blocks
        cone_id = Int(huge_block_cone_ids[block_id])
        start_idx = Int(huge_block_starts[block_id])
        tail_offset = Int(huge_block_offsets[block_id])
        cone_size = Int(huge_sizes[cone_id])

        if tail_offset == 0 && tid == 1
            t_raw = UseCustom ?
                _soc_raw_update_noQ_custom(start_idx, x, rowPtrAT, colValAT, nzValAT, y, c, sigma) :
                _soc_raw_update_noQ(start_idx, x, ATy, c, sigma)
            huge_t_raw[cone_id] = t_raw
            z_bar[start_idx] = t_raw
        end

        local_sum = 0.0
        tail_pos = tail_offset + tid
        if tail_pos <= cone_size - 1
            j = start_idx + tail_pos
            z_raw_j = UseCustom ?
                _soc_raw_update_noQ_custom(j, x, rowPtrAT, colValAT, nzValAT, y, c, sigma) :
                _soc_raw_update_noQ(j, x, ATy, c, sigma)
            x_hat[j] = z_raw_j
            local_sum = z_raw_j^2
        end
        shared_sums[tid] = local_sum
        sync_threads()

        stride = block_threads ÷ 2
        while stride >= 1
            if tid <= stride
                shared_sums[tid] += shared_sums[tid + stride]
            end
            sync_threads()
            stride ÷= 2
        end

        if tid == 1
            huge_partial_sums[block_id] = shared_sums[1]
        end
    end
    return
end

CUDA.@fastmath function finalize_unified_update_zx_noQ_SOC_huge_kernel_partial!(
    huge_proj_t::CuDeviceVector{Float64},
    huge_alpha::CuDeviceVector{Float64},
    huge_case::CuDeviceVector{Int32},
    huge_partial_sums::CuDeviceVector{Float64},
    huge_t_raw::CuDeviceVector{Float64},
    huge_block_ptr::CuDeviceVector{Int32},
    cone_count::Int)
    cone_id = Int(blockIdx().x)
    tid = Int(threadIdx().x)
    block_threads = Int(blockDim().x)
    shared_sums = CuStaticSharedArray(Float64, DEFAULT_KERNEL_THREADS)

    @inbounds if cone_id <= cone_count
        block_start = Int(huge_block_ptr[cone_id])
        block_stop = Int(huge_block_ptr[cone_id + 1]) - 1
        local_sum = 0.0
        partial_idx = block_start + tid - 1
        while partial_idx <= block_stop
            local_sum += huge_partial_sums[partial_idx]
            partial_idx += block_threads
        end
        shared_sums[tid] = local_sum
        sync_threads()

        stride = block_threads ÷ 2
        while stride >= 1
            if tid <= stride
                shared_sums[tid] += shared_sums[tid + stride]
            end
            sync_threads()
            stride ÷= 2
        end

        if tid == 1
            t_raw = huge_t_raw[cone_id]
            norm_s = sqrt(shared_sums[1])
            if norm_s <= -t_raw
                huge_case[cone_id] = Int32(0)
                huge_proj_t[cone_id] = 0.0
                huge_alpha[cone_id] = 0.0
            elseif norm_s <= t_raw
                huge_case[cone_id] = Int32(1)
                huge_proj_t[cone_id] = t_raw
                huge_alpha[cone_id] = 1.0
            else
                proj_t = (norm_s + t_raw) / 2.0
                huge_case[cone_id] = Int32(2)
                huge_proj_t[cone_id] = proj_t
                huge_alpha[cone_id] = proj_t / norm_s
            end
        end
    end
    return
end

CUDA.@fastmath function apply_unified_update_zx_noQ_SOC_huge_kernel_partial!(
    z_bar::CuDeviceVector{Float64},
    x_hat::CuDeviceVector{Float64},
    last_x::CuDeviceVector{Float64},
    x::CuDeviceVector{Float64},
    sigma::Float64,
    Halpern_fact1::Float64,
    Halpern_fact2::Float64,
    huge_sizes::CuDeviceVector{Int32},
    huge_block_starts::CuDeviceVector{Int32},
    huge_block_offsets::CuDeviceVector{Int32},
    huge_block_cone_ids::CuDeviceVector{Int32},
    huge_t_raw::CuDeviceVector{Float64},
    huge_proj_t::CuDeviceVector{Float64},
    huge_alpha::CuDeviceVector{Float64},
    huge_case::CuDeviceVector{Int32},
    total_blocks::Int)
    block_id = Int(blockIdx().x)
    tid = Int(threadIdx().x)

    @inbounds if block_id <= total_blocks
        cone_id = Int(huge_block_cone_ids[block_id])
        start_idx = Int(huge_block_starts[block_id])
        tail_offset = Int(huge_block_offsets[block_id])
        cone_size = Int(huge_sizes[cone_id])
        t_raw = huge_t_raw[cone_id]
        proj_t = huge_proj_t[cone_id]
        alpha = huge_alpha[cone_id]
        case_id = huge_case[cone_id]

        if tail_offset == 0 && tid == 1
            projected_t = if case_id == 0
                0.0
            elseif case_id == 1
                t_raw
            else
                proj_t
            end
            z_bar[start_idx] = if case_id == 0
                -t_raw / sigma
            elseif case_id == 1
                0.0
            else
                (proj_t - t_raw) / sigma
            end
            _write_soc_var_projection_noQ_partial!(
                start_idx, projected_t, x_hat, last_x, x, Halpern_fact1, Halpern_fact2
            )
        end

        tail_pos = tail_offset + tid
        if tail_pos <= cone_size - 1
            j = start_idx + tail_pos
            z_raw_j = x_hat[j]
            projected_j = if case_id == 0
                0.0
            elseif case_id == 1
                z_raw_j
            else
                alpha * z_raw_j
            end
            z_bar[j] = if case_id == 0
                -z_raw_j / sigma
            elseif case_id == 1
                0.0
            else
                (projected_j - z_raw_j) / sigma
            end
            _write_soc_var_projection_noQ_partial!(
                j, projected_j, x_hat, last_x, x, Halpern_fact1, Halpern_fact2
            )
        end
    end
    return
end

@inline function launch_unified_update_zx_noQ_SOC_size3!(
    ws::HPRSOCP_workspace_gpu,
    soc_var_starts::CuVector{Int32},
    cone_count::Int,
    Halpern_fact1::Float64,
    Halpern_fact2::Float64)
    threads, blocks = gpu_launch_config(cone_count)
    if threads > 0 && cone_count > 0
        @cuda threads = threads blocks = blocks unified_update_zx_noQ_SOC_size3_kernel!(
            ws.dx, ws.z_bar, ws.x_bar, ws.x_hat, ws.last_x, ws.x, ws.ATy, ws.c, ws.sigma,
            Halpern_fact1, Halpern_fact2, soc_var_starts, cone_count)
    end
end

CUDA.@fastmath function unified_update_zx_noQ_SOC_size5_cooperative_kernel_partial!(::Val{UseCustom},
    z_bar::CuDeviceVector{Float64},
    x_hat::CuDeviceVector{Float64},
    last_x::CuDeviceVector{Float64},
    x::CuDeviceVector{Float64},
    rowPtrAT::CuDeviceVector{Int32},
    colValAT::CuDeviceVector{Int32},
    nzValAT::CuDeviceVector{Float64},
    y::CuDeviceVector{Float64},
    ATy::CuDeviceVector{Float64},
    c::CuDeviceVector{Float64},
    sigma::Float64,
    Halpern_fact1::Float64,
    Halpern_fact2::Float64,
    soc_var_starts::CuDeviceVector{Int32},
    cone_count::Int) where {UseCustom}
    tid = Int(threadIdx().x)
    tid0 = tid - 1
    cone_in_block = tid0 ÷ 5
    lane = tid0 % 5
    cones_per_block = SOC_VAR_SIZE5_COOP_THREADS ÷ 5
    cone_idx = (Int(blockIdx().x) - 1) * cones_per_block + cone_in_block + 1

    shared_raw = CuStaticSharedArray(Float64, SOC_VAR_SIZE5_COOP_THREADS)
    active = cone_idx <= cone_count
    raw = 0.0
    j = 0

    if active
        start_idx = Int(soc_var_starts[cone_idx])
        j = start_idx + lane
        raw = UseCustom ?
            _soc_raw_update_noQ_custom(j, x, rowPtrAT, colValAT, nzValAT, y, c, sigma) :
            _soc_raw_update_noQ(j, x, ATy, c, sigma)
    end
    shared_raw[tid] = raw
    sync_threads()

    @inbounds if active
        base = cone_in_block * 5
        t_raw = shared_raw[base + 1]
        s_1 = shared_raw[base + 2]
        s_2 = shared_raw[base + 3]
        s_3 = shared_raw[base + 4]
        s_4 = shared_raw[base + 5]
        norm_s = sqrt(s_1^2 + s_2^2 + s_3^2 + s_4^2)

        if norm_s <= -t_raw
            projected = 0.0
        elseif norm_s <= t_raw
            projected = lane == 0 ? t_raw : raw
        else
            proj_t = (norm_s + t_raw) / 2.0
            alpha = proj_t / norm_s
            projected = lane == 0 ? proj_t : alpha * raw
        end

        z_bar[j] = (projected - raw) / sigma
        _write_soc_var_projection_noQ_partial!(
            j, projected, x_hat, last_x, x, Halpern_fact1, Halpern_fact2
        )
    end
    return
end

@inline function launch_unified_update_zx_noQ_SOC_size5_cooperative!(
    ws::HPRSOCP_workspace_gpu,
    soc_var_starts::CuVector{Int32},
    cone_count::Int,
    Halpern_fact1::Float64,
    Halpern_fact2::Float64)
    launch_unified_update_zx_noQ_SOC_size5_cooperative!(
        Val(false), ws, soc_var_starts, cone_count, Halpern_fact1, Halpern_fact2)
end

@inline function launch_unified_update_zx_noQ_SOC_size5_cooperative!(::Val{UseCustom},
    ws::HPRSOCP_workspace_gpu,
    soc_var_starts::CuVector{Int32},
    cone_count::Int,
    Halpern_fact1::Float64,
    Halpern_fact2::Float64) where {UseCustom}
    threads = SOC_VAR_SIZE5_COOP_THREADS
    cones_per_block = threads ÷ 5
    blocks = cld(cone_count, cones_per_block)
    if blocks > 0
        @cuda threads = threads blocks = blocks unified_update_zx_noQ_SOC_size5_cooperative_kernel_partial!(
            Val(UseCustom),
            ws.z_bar, ws.x_hat, ws.last_x, ws.x,
            ws.AT.rowPtr, ws.AT.colVal, ws.AT.nzVal, ws.y, ws.ATy, ws.c, ws.sigma,
            Halpern_fact1, Halpern_fact2, soc_var_starts, cone_count)
    end
end

CUDA.@fastmath @inline function unified_update_zx_noQ_SOC_small_kernel!(
    dx::CuDeviceVector{Float64},
    z_bar::CuDeviceVector{Float64},
    x_bar::CuDeviceVector{Float64},
    x_hat::CuDeviceVector{Float64},
    last_x::CuDeviceVector{Float64},
    x::CuDeviceVector{Float64},
    ATy::CuDeviceVector{Float64},
    c::CuDeviceVector{Float64},
    sigma::Float64,
    Halpern_fact1::Float64,
    Halpern_fact2::Float64,
    soc_var_starts::CuDeviceVector{Int32},
    soc_var_sizes::CuDeviceVector{Int32},
    cone_count::Int)
    i = threadIdx().x + (blockDim().x * (blockIdx().x - 1))
    @inbounds if i <= cone_count
        start_idx = Int(soc_var_starts[i])
        cone_size = Int(soc_var_sizes[i])
        end_idx = start_idx + cone_size - 1

        if cone_size == 3
            j1 = start_idx + 1
            j2 = start_idx + 2

            t_raw = _soc_raw_update_noQ(start_idx, x, ATy, c, sigma)
            z_raw_1 = _soc_raw_update_noQ(j1, x, ATy, c, sigma)
            z_raw_2 = _soc_raw_update_noQ(j2, x, ATy, c, sigma)
            norm_s = sqrt(z_raw_1^2 + z_raw_2^2)

            if norm_s <= -t_raw
                _write_soc_var_projection_noQ!(start_idx, 0.0, -t_raw / sigma,
                    dx, z_bar, x_bar, x_hat, last_x, x, Halpern_fact1, Halpern_fact2)
                _write_soc_var_projection_noQ!(j1, 0.0, -z_raw_1 / sigma,
                    dx, z_bar, x_bar, x_hat, last_x, x, Halpern_fact1, Halpern_fact2)
                _write_soc_var_projection_noQ!(j2, 0.0, -z_raw_2 / sigma,
                    dx, z_bar, x_bar, x_hat, last_x, x, Halpern_fact1, Halpern_fact2)
            elseif norm_s <= t_raw
                _write_soc_var_projection_noQ!(start_idx, t_raw, 0.0,
                    dx, z_bar, x_bar, x_hat, last_x, x, Halpern_fact1, Halpern_fact2)
                _write_soc_var_projection_noQ!(j1, z_raw_1, 0.0,
                    dx, z_bar, x_bar, x_hat, last_x, x, Halpern_fact1, Halpern_fact2)
                _write_soc_var_projection_noQ!(j2, z_raw_2, 0.0,
                    dx, z_bar, x_bar, x_hat, last_x, x, Halpern_fact1, Halpern_fact2)
            else
                proj_t = (norm_s + t_raw) / 2.0
                alpha = proj_t / norm_s
                proj_1 = alpha * z_raw_1
                proj_2 = alpha * z_raw_2

                _write_soc_var_projection_noQ!(start_idx, proj_t, (proj_t - t_raw) / sigma,
                    dx, z_bar, x_bar, x_hat, last_x, x, Halpern_fact1, Halpern_fact2)
                _write_soc_var_projection_noQ!(j1, proj_1, (proj_1 - z_raw_1) / sigma,
                    dx, z_bar, x_bar, x_hat, last_x, x, Halpern_fact1, Halpern_fact2)
                _write_soc_var_projection_noQ!(j2, proj_2, (proj_2 - z_raw_2) / sigma,
                    dx, z_bar, x_bar, x_hat, last_x, x, Halpern_fact1, Halpern_fact2)
            end
            return
        end

        t_raw = _soc_raw_update_noQ(start_idx, x, ATy, c, sigma)
        z_bar[start_idx] = t_raw
        norm_s = 0.0
        for j in (start_idx + 1):end_idx
            z_raw_j = _soc_raw_update_noQ(j, x, ATy, c, sigma)
            z_bar[j] = z_raw_j
            norm_s += z_raw_j^2
        end
        norm_s = sqrt(norm_s)

        if norm_s <= -t_raw
            _write_soc_var_projection_noQ!(start_idx, 0.0, -t_raw / sigma,
                dx, z_bar, x_bar, x_hat, last_x, x, Halpern_fact1, Halpern_fact2)

            for j in (start_idx + 1):end_idx
                z_raw_j = z_bar[j]
                _write_soc_var_projection_noQ!(j, 0.0, -z_raw_j / sigma,
                    dx, z_bar, x_bar, x_hat, last_x, x, Halpern_fact1, Halpern_fact2)
            end
        elseif norm_s <= t_raw
            _write_soc_var_projection_noQ!(start_idx, t_raw, 0.0,
                dx, z_bar, x_bar, x_hat, last_x, x, Halpern_fact1, Halpern_fact2)

            for j in (start_idx + 1):end_idx
                z_raw_j = z_bar[j]
                _write_soc_var_projection_noQ!(j, z_raw_j, 0.0,
                    dx, z_bar, x_bar, x_hat, last_x, x, Halpern_fact1, Halpern_fact2)
            end
        else
            proj_t = (norm_s + t_raw) / 2.0
            alpha = proj_t / norm_s

            _write_soc_var_projection_noQ!(start_idx, proj_t, (proj_t - t_raw) / sigma,
                dx, z_bar, x_bar, x_hat, last_x, x, Halpern_fact1, Halpern_fact2)

            for j in (start_idx + 1):end_idx
                z_raw_j = z_bar[j]
                proj_j = alpha * z_raw_j
                _write_soc_var_projection_noQ!(j, proj_j, (proj_j - z_raw_j) / sigma,
                    dx, z_bar, x_bar, x_hat, last_x, x, Halpern_fact1, Halpern_fact2)
            end
        end
    end
    return
end

CUDA.@fastmath @inline function unified_update_zx_noQ_SOC_small_kernel_partial!(::Val{UseCustom},
    z_bar::CuDeviceVector{Float64},
    x_hat::CuDeviceVector{Float64},
    last_x::CuDeviceVector{Float64},
    x::CuDeviceVector{Float64},
    rowPtrAT::CuDeviceVector{Int32},
    colValAT::CuDeviceVector{Int32},
    nzValAT::CuDeviceVector{Float64},
    y::CuDeviceVector{Float64},
    ATy::CuDeviceVector{Float64},
    c::CuDeviceVector{Float64},
    sigma::Float64,
    Halpern_fact1::Float64,
    Halpern_fact2::Float64,
    soc_var_starts::CuDeviceVector{Int32},
    soc_var_sizes::CuDeviceVector{Int32},
    cone_count::Int) where {UseCustom}
    i = threadIdx().x + (blockDim().x * (blockIdx().x - 1))
    @inbounds if i <= cone_count
        start_idx = Int(soc_var_starts[i])
        cone_size = Int(soc_var_sizes[i])
        end_idx = start_idx + cone_size - 1

        if cone_size == 3
            j1 = start_idx + 1
            j2 = start_idx + 2

            t_raw = UseCustom ?
                _soc_raw_update_noQ_custom(start_idx, x, rowPtrAT, colValAT, nzValAT, y, c, sigma) :
                _soc_raw_update_noQ(start_idx, x, ATy, c, sigma)
            z_raw_1 = UseCustom ?
                _soc_raw_update_noQ_custom(j1, x, rowPtrAT, colValAT, nzValAT, y, c, sigma) :
                _soc_raw_update_noQ(j1, x, ATy, c, sigma)
            z_raw_2 = UseCustom ?
                _soc_raw_update_noQ_custom(j2, x, rowPtrAT, colValAT, nzValAT, y, c, sigma) :
                _soc_raw_update_noQ(j2, x, ATy, c, sigma)
            norm_s = sqrt(z_raw_1^2 + z_raw_2^2)

            if norm_s <= -t_raw
                z_bar[start_idx] = -t_raw / sigma
                z_bar[j1] = -z_raw_1 / sigma
                z_bar[j2] = -z_raw_2 / sigma
                _write_soc_var_projection_noQ_partial!(
                    start_idx, 0.0, x_hat, last_x, x, Halpern_fact1, Halpern_fact2
                )
                _write_soc_var_projection_noQ_partial!(
                    j1, 0.0, x_hat, last_x, x, Halpern_fact1, Halpern_fact2
                )
                _write_soc_var_projection_noQ_partial!(
                    j2, 0.0, x_hat, last_x, x, Halpern_fact1, Halpern_fact2
                )
            elseif norm_s <= t_raw
                z_bar[start_idx] = 0.0
                z_bar[j1] = 0.0
                z_bar[j2] = 0.0
                _write_soc_var_projection_noQ_partial!(
                    start_idx, t_raw, x_hat, last_x, x, Halpern_fact1, Halpern_fact2
                )
                _write_soc_var_projection_noQ_partial!(
                    j1, z_raw_1, x_hat, last_x, x, Halpern_fact1, Halpern_fact2
                )
                _write_soc_var_projection_noQ_partial!(
                    j2, z_raw_2, x_hat, last_x, x, Halpern_fact1, Halpern_fact2
                )
            else
                proj_t = (norm_s + t_raw) / 2.0
                alpha = proj_t / norm_s
                proj_1 = alpha * z_raw_1
                proj_2 = alpha * z_raw_2

                z_bar[start_idx] = (proj_t - t_raw) / sigma
                z_bar[j1] = (proj_1 - z_raw_1) / sigma
                z_bar[j2] = (proj_2 - z_raw_2) / sigma
                _write_soc_var_projection_noQ_partial!(
                    start_idx, proj_t, x_hat, last_x, x, Halpern_fact1, Halpern_fact2
                )
                _write_soc_var_projection_noQ_partial!(
                    j1, proj_1, x_hat, last_x, x, Halpern_fact1, Halpern_fact2
                )
                _write_soc_var_projection_noQ_partial!(
                    j2, proj_2, x_hat, last_x, x, Halpern_fact1, Halpern_fact2
                )
            end
            return
        end

        t_raw = UseCustom ?
            _soc_raw_update_noQ_custom(start_idx, x, rowPtrAT, colValAT, nzValAT, y, c, sigma) :
            _soc_raw_update_noQ(start_idx, x, ATy, c, sigma)
        z_bar[start_idx] = t_raw
        norm_s = 0.0
        for j in (start_idx + 1):end_idx
            z_raw_j = UseCustom ?
                _soc_raw_update_noQ_custom(j, x, rowPtrAT, colValAT, nzValAT, y, c, sigma) :
                _soc_raw_update_noQ(j, x, ATy, c, sigma)
            z_bar[j] = z_raw_j
            norm_s += z_raw_j^2
        end
        norm_s = sqrt(norm_s)

        if norm_s <= -t_raw
            z_bar[start_idx] = -t_raw / sigma
            _write_soc_var_projection_noQ_partial!(
                start_idx, 0.0, x_hat, last_x, x, Halpern_fact1, Halpern_fact2
            )

            for j in (start_idx + 1):end_idx
                z_raw_j = z_bar[j]
                z_bar[j] = -z_raw_j / sigma
                _write_soc_var_projection_noQ_partial!(
                    j, 0.0, x_hat, last_x, x, Halpern_fact1, Halpern_fact2
                )
            end
        elseif norm_s <= t_raw
            z_bar[start_idx] = 0.0
            _write_soc_var_projection_noQ_partial!(
                start_idx, t_raw, x_hat, last_x, x, Halpern_fact1, Halpern_fact2
            )

            for j in (start_idx + 1):end_idx
                z_raw_j = z_bar[j]
                z_bar[j] = 0.0
                _write_soc_var_projection_noQ_partial!(
                    j, z_raw_j, x_hat, last_x, x, Halpern_fact1, Halpern_fact2
                )
            end
        else
            proj_t = (norm_s + t_raw) / 2.0
            alpha = proj_t / norm_s

            z_bar[start_idx] = (proj_t - t_raw) / sigma
            _write_soc_var_projection_noQ_partial!(
                start_idx, proj_t, x_hat, last_x, x, Halpern_fact1, Halpern_fact2
            )

            for j in (start_idx + 1):end_idx
                z_raw_j = z_bar[j]
                proj_j = alpha * z_raw_j
                z_bar[j] = (proj_j - z_raw_j) / sigma
                _write_soc_var_projection_noQ_partial!(
                    j, proj_j, x_hat, last_x, x, Halpern_fact1, Halpern_fact2
                )
            end
        end
    end
    return
end

@inline function launch_unified_update_zx_noQ_SOC_small!(
    ws::HPRSOCP_workspace_gpu,
    soc_var_starts::CuVector{Int32},
    soc_var_sizes::CuVector{Int32},
    cone_count::Int,
    Halpern_fact1::Float64,
    Halpern_fact2::Float64)
    threads, blocks = gpu_launch_config(cone_count)
    if threads > 0 && cone_count > 0
        @cuda threads = threads blocks = blocks unified_update_zx_noQ_SOC_small_kernel!(
            ws.dx, ws.z_bar, ws.x_bar, ws.x_hat, ws.last_x, ws.x, ws.ATy, ws.c, ws.sigma,
            Halpern_fact1, Halpern_fact2, soc_var_starts, soc_var_sizes, cone_count)
    end
end

@inline function launch_unified_update_zx_noQ_SOC_small_partial!(
    ws::HPRSOCP_workspace_gpu,
    soc_var_starts::CuVector{Int32},
    soc_var_sizes::CuVector{Int32},
    cone_count::Int,
    Halpern_fact1::Float64,
    Halpern_fact2::Float64)
    launch_unified_update_zx_noQ_SOC_small_partial!(
        Val(false), ws, soc_var_starts, soc_var_sizes, cone_count, Halpern_fact1, Halpern_fact2)
end

@inline function launch_unified_update_zx_noQ_SOC_small_partial!(::Val{UseCustom},
    ws::HPRSOCP_workspace_gpu,
    soc_var_starts::CuVector{Int32},
    soc_var_sizes::CuVector{Int32},
    cone_count::Int,
    Halpern_fact1::Float64,
    Halpern_fact2::Float64) where {UseCustom}
    threads, blocks = gpu_launch_config(cone_count)
    if threads > 0 && cone_count > 0
        @cuda threads = threads blocks = blocks unified_update_zx_noQ_SOC_small_kernel_partial!(
            Val(UseCustom),
            ws.z_bar, ws.x_hat, ws.last_x, ws.x,
            ws.AT.rowPtr, ws.AT.colVal, ws.AT.nzVal, ws.y, ws.ATy, ws.c, ws.sigma,
            Halpern_fact1, Halpern_fact2, soc_var_starts, soc_var_sizes, cone_count)
    end
end

@inline function launch_unified_update_zx_noQ_SOC_generic!(
    ws::HPRSOCP_workspace_gpu,
    soc_var_starts::CuVector{Int32},
    soc_var_sizes::CuVector{Int32},
    cone_count::Int,
    Halpern_fact1::Float64,
    Halpern_fact2::Float64)
    threads, blocks = gpu_launch_config(cone_count)
    if threads > 0 && cone_count > 0
        @cuda threads = threads blocks = blocks unified_update_zx_noQ_SOC_generic_kernel!(
            ws.dx, ws.z_bar, ws.x_bar, ws.x_hat, ws.last_x, ws.x, ws.ATy, ws.c, ws.sigma,
            Halpern_fact1, Halpern_fact2, soc_var_starts, soc_var_sizes, cone_count)
    end
end

@inline function launch_unified_update_zx_noQ_SOC_large!(
    ws::HPRSOCP_workspace_gpu,
    soc_var_starts::CuVector{Int32},
    soc_var_sizes::CuVector{Int32},
    cone_count::Int,
    Halpern_fact1::Float64,
    Halpern_fact2::Float64)
    threads = DEFAULT_KERNEL_THREADS
    blocks = cone_count
    if blocks > 0
        @cuda threads = threads blocks = blocks unified_update_zx_noQ_SOC_large_kernel!(
            ws.dx, ws.z_bar, ws.x_bar, ws.x_hat, ws.last_x, ws.x, ws.ATy, ws.c, ws.sigma,
            Halpern_fact1, Halpern_fact2, soc_var_starts, soc_var_sizes, cone_count)
    end
end

@inline function launch_unified_update_zx_noQ_SOC_huge_staged!(
    ws::HPRSOCP_workspace_gpu,
    soc_var_starts::CuVector{Int32},
    soc_var_sizes::CuVector{Int32},
    cone_count::Int,
    Halpern_fact1::Float64,
    Halpern_fact2::Float64)
    fast_paths = ws.soc_var_fast_paths
    threads = HUGE_SOC_KERNEL_THREADS
    blocks = fast_paths.huge_total_blocks
    if blocks > 0
        @cuda threads = threads blocks = blocks prepare_unified_update_zx_noQ_SOC_huge_kernel!(
            ws.z_bar, fast_paths.huge_partial_sums, fast_paths.huge_t_raw,
            ws.x, ws.ATy, ws.c, ws.sigma,
            soc_var_sizes, fast_paths.huge_block_starts,
            fast_paths.huge_block_offsets, fast_paths.huge_block_cone_ids, blocks)
        @cuda threads = DEFAULT_KERNEL_THREADS blocks = cone_count finalize_unified_update_zx_noQ_SOC_huge_kernel!(
            fast_paths.huge_proj_t, fast_paths.huge_alpha, fast_paths.huge_case,
            fast_paths.huge_partial_sums, fast_paths.huge_t_raw,
            fast_paths.huge_block_ptr, cone_count)
        @cuda threads = threads blocks = blocks apply_unified_update_zx_noQ_SOC_huge_kernel!(
            ws.dx, ws.z_bar, ws.x_bar, ws.x_hat, ws.last_x, ws.x, ws.sigma,
            Halpern_fact1, Halpern_fact2, soc_var_sizes,
            fast_paths.huge_block_starts, fast_paths.huge_block_offsets,
            fast_paths.huge_block_cone_ids, fast_paths.huge_t_raw,
            fast_paths.huge_proj_t, fast_paths.huge_alpha, fast_paths.huge_case, blocks)
    end
end

@inline function launch_unified_update_zx_noQ_SOC_huge_cooperative!(
    ws::HPRSOCP_workspace_gpu,
    soc_var_starts::CuVector{Int32},
    soc_var_sizes::CuVector{Int32},
    cone_count::Int,
    Halpern_fact1::Float64,
    Halpern_fact2::Float64)
    fast_paths = ws.soc_var_fast_paths
    threads = HUGE_SOC_KERNEL_THREADS
    blocks = fast_paths.huge_total_blocks
    if blocks > 0
        @cuda cooperative=true threads = threads blocks = blocks unified_update_zx_noQ_SOC_huge_cooperative_kernel!(
            ws.dx, ws.z_bar, ws.x_bar, ws.x_hat, ws.last_x, ws.x, ws.ATy, ws.c, ws.sigma,
            Halpern_fact1, Halpern_fact2, soc_var_sizes,
            fast_paths.huge_block_starts, fast_paths.huge_block_offsets,
            fast_paths.huge_block_cone_ids, fast_paths.huge_block_ptr,
            fast_paths.huge_partial_sums, fast_paths.huge_t_raw, fast_paths.huge_proj_t,
            fast_paths.huge_alpha, fast_paths.huge_case, blocks)
    end
end

@inline function launch_unified_update_zx_noQ_SOC_huge_segmented!(
    ws::HPRSOCP_workspace_gpu,
    soc_var_starts::CuVector{Int32},
    soc_var_sizes::CuVector{Int32},
    cone_count::Int,
    Halpern_fact1::Float64,
    Halpern_fact2::Float64)
    fast_paths = ws.soc_var_fast_paths
    threads = HUGE_SOC_KERNEL_THREADS
    blocks = fast_paths.huge_total_blocks
    if blocks > 0
        @cuda threads = threads blocks = blocks prepare_unified_update_zx_noQ_SOC_huge_segmented_kernel!(
            fast_paths.huge_partial_sums, fast_paths.huge_t_raw,
            ws.x, ws.ATy, ws.c, ws.sigma,
            soc_var_sizes, fast_paths.huge_block_starts,
            fast_paths.huge_block_offsets, fast_paths.huge_block_cone_ids, blocks)
        @cuda threads = DEFAULT_KERNEL_THREADS blocks = cone_count finalize_unified_update_zx_noQ_SOC_huge_kernel!(
            fast_paths.huge_proj_t, fast_paths.huge_alpha, fast_paths.huge_case,
            fast_paths.huge_partial_sums, fast_paths.huge_t_raw,
            fast_paths.huge_block_ptr, cone_count)
        @cuda threads = threads blocks = blocks apply_unified_update_zx_noQ_SOC_huge_segmented_kernel!(
            ws.dx, ws.z_bar, ws.x_bar, ws.x_hat, ws.last_x, ws.x,
            ws.ATy, ws.c, ws.sigma, Halpern_fact1, Halpern_fact2, soc_var_sizes,
            fast_paths.huge_block_starts, fast_paths.huge_block_offsets,
            fast_paths.huge_block_cone_ids, fast_paths.huge_t_raw,
            fast_paths.huge_proj_t, fast_paths.huge_alpha, fast_paths.huge_case, blocks)
    end
end

@inline function launch_unified_update_zx_noQ_SOC_huge!(
    ws::HPRSOCP_workspace_gpu,
    soc_var_starts::CuVector{Int32},
    soc_var_sizes::CuVector{Int32},
    cone_count::Int,
    Halpern_fact1::Float64,
    Halpern_fact2::Float64)
    if ws.soc_var_fast_paths.huge_kernel_mode == :cooperative
        launch_unified_update_zx_noQ_SOC_huge_cooperative!(
            ws, soc_var_starts, soc_var_sizes, cone_count, Halpern_fact1, Halpern_fact2)
    elseif ws.soc_var_fast_paths.huge_kernel_mode == :segmented
        launch_unified_update_zx_noQ_SOC_huge_segmented!(
            ws, soc_var_starts, soc_var_sizes, cone_count, Halpern_fact1, Halpern_fact2)
    else
        launch_unified_update_zx_noQ_SOC_huge_staged!(
            ws, soc_var_starts, soc_var_sizes, cone_count, Halpern_fact1, Halpern_fact2)
    end
end

@inline function launch_unified_update_zx_noQ_SOC_large_partial!(
    ws::HPRSOCP_workspace_gpu,
    soc_var_starts::CuVector{Int32},
    soc_var_sizes::CuVector{Int32},
    cone_count::Int,
    Halpern_fact1::Float64,
    Halpern_fact2::Float64)
    launch_unified_update_zx_noQ_SOC_large_partial!(
        Val(false), ws, soc_var_starts, soc_var_sizes, cone_count, Halpern_fact1, Halpern_fact2)
end

@inline function launch_unified_update_zx_noQ_SOC_large_partial!(::Val{UseCustom},
    ws::HPRSOCP_workspace_gpu,
    soc_var_starts::CuVector{Int32},
    soc_var_sizes::CuVector{Int32},
    cone_count::Int,
    Halpern_fact1::Float64,
    Halpern_fact2::Float64) where {UseCustom}
    threads = DEFAULT_KERNEL_THREADS
    blocks = cone_count
    if blocks > 0
        if UseCustom
            @cuda threads = threads blocks = blocks unified_update_zx_noQ_SOC_large_kernel_partial!(
                Val(true),
                ws.z_bar, ws.x_hat, ws.last_x, ws.x,
                ws.AT.rowPtr, ws.AT.colVal, ws.AT.nzVal, ws.y, ws.ATy, ws.c, ws.sigma,
                Halpern_fact1, Halpern_fact2, soc_var_starts, soc_var_sizes, cone_count)
        else
            @cuda threads = threads blocks = blocks unified_update_zx_noQ_SOC_large_kernel_partial_legacy!(
                ws.z_bar, ws.x_hat, ws.last_x, ws.x, ws.ATy, ws.c, ws.sigma,
                Halpern_fact1, Halpern_fact2, soc_var_starts, soc_var_sizes, cone_count)
        end
    end
end

@inline function launch_unified_update_zx_noQ_SOC_large_cooperative_partial!(
    ws::HPRSOCP_workspace_gpu,
    soc_var_starts::CuVector{Int32},
    soc_var_sizes::CuVector{Int32},
    cone_count::Int,
    Halpern_fact1::Float64,
    Halpern_fact2::Float64)
    launch_unified_update_zx_noQ_SOC_large_cooperative_partial!(
        Val(false), ws, soc_var_starts, soc_var_sizes, cone_count, Halpern_fact1, Halpern_fact2)
end

@inline function launch_unified_update_zx_noQ_SOC_large_cooperative_partial!(::Val{UseCustom},
    ws::HPRSOCP_workspace_gpu,
    soc_var_starts::CuVector{Int32},
    soc_var_sizes::CuVector{Int32},
    cone_count::Int,
    Halpern_fact1::Float64,
    Halpern_fact2::Float64) where {UseCustom}
    threads = DEFAULT_KERNEL_THREADS
    blocks = cone_count
    if blocks > 0
        @cuda threads = threads blocks = blocks unified_update_zx_noQ_SOC_large_kernel_partial!(
            Val(UseCustom),
            ws.z_bar, ws.x_hat, ws.last_x, ws.x,
            ws.AT.rowPtr, ws.AT.colVal, ws.AT.nzVal, ws.y, ws.ATy, ws.c, ws.sigma,
            Halpern_fact1, Halpern_fact2, soc_var_starts, soc_var_sizes, cone_count)
    end
end

@inline function launch_unified_update_zx_noQ_SOC_huge_partial_staged!(
    ws::HPRSOCP_workspace_gpu,
    soc_var_starts::CuVector{Int32},
    soc_var_sizes::CuVector{Int32},
    cone_count::Int,
    Halpern_fact1::Float64,
    Halpern_fact2::Float64)
    launch_unified_update_zx_noQ_SOC_huge_partial_staged!(
        Val(false), ws, soc_var_starts, soc_var_sizes, cone_count, Halpern_fact1, Halpern_fact2)
end

@inline function launch_unified_update_zx_noQ_SOC_huge_partial_staged!(::Val{UseCustom},
    ws::HPRSOCP_workspace_gpu,
    soc_var_starts::CuVector{Int32},
    soc_var_sizes::CuVector{Int32},
    cone_count::Int,
    Halpern_fact1::Float64,
    Halpern_fact2::Float64) where {UseCustom}
    fast_paths = ws.soc_var_fast_paths
    threads = HUGE_SOC_KERNEL_THREADS
    blocks = fast_paths.huge_total_blocks
    if blocks > 0
        @cuda threads = threads blocks = blocks prepare_unified_update_zx_noQ_SOC_huge_kernel_partial!(
            Val(UseCustom),
            ws.z_bar, ws.x_hat, fast_paths.huge_partial_sums, fast_paths.huge_t_raw,
            ws.x, ws.AT.rowPtr, ws.AT.colVal, ws.AT.nzVal, ws.y, ws.ATy, ws.c, ws.sigma,
            soc_var_sizes, fast_paths.huge_block_starts,
            fast_paths.huge_block_offsets, fast_paths.huge_block_cone_ids, blocks)
        @cuda threads = DEFAULT_KERNEL_THREADS blocks = cone_count finalize_unified_update_zx_noQ_SOC_huge_kernel_partial!(
            fast_paths.huge_proj_t, fast_paths.huge_alpha, fast_paths.huge_case,
            fast_paths.huge_partial_sums, fast_paths.huge_t_raw,
            fast_paths.huge_block_ptr, cone_count)
        @cuda threads = threads blocks = blocks apply_unified_update_zx_noQ_SOC_huge_kernel_partial!(
            ws.z_bar, ws.x_hat, ws.last_x, ws.x, ws.sigma,
            Halpern_fact1, Halpern_fact2, soc_var_sizes,
            fast_paths.huge_block_starts, fast_paths.huge_block_offsets,
            fast_paths.huge_block_cone_ids, fast_paths.huge_t_raw,
            fast_paths.huge_proj_t, fast_paths.huge_alpha, fast_paths.huge_case, blocks)
    end
end

@inline function launch_unified_update_zx_noQ_SOC_huge_cooperative_partial!(
    ws::HPRSOCP_workspace_gpu,
    soc_var_starts::CuVector{Int32},
    soc_var_sizes::CuVector{Int32},
    cone_count::Int,
    Halpern_fact1::Float64,
    Halpern_fact2::Float64)
    launch_unified_update_zx_noQ_SOC_huge_cooperative_partial!(
        Val(false), ws, soc_var_starts, soc_var_sizes, cone_count, Halpern_fact1, Halpern_fact2)
end

@inline function launch_unified_update_zx_noQ_SOC_huge_cooperative_partial!(::Val{UseCustom},
    ws::HPRSOCP_workspace_gpu,
    soc_var_starts::CuVector{Int32},
    soc_var_sizes::CuVector{Int32},
    cone_count::Int,
    Halpern_fact1::Float64,
    Halpern_fact2::Float64) where {UseCustom}
    fast_paths = ws.soc_var_fast_paths
    threads = HUGE_SOC_KERNEL_THREADS
    blocks = fast_paths.huge_total_blocks
    if blocks > 0
        @cuda cooperative=true threads = threads blocks = blocks unified_update_zx_noQ_SOC_huge_cooperative_kernel_partial!(
            Val(UseCustom),
            ws.z_bar, ws.x_hat, ws.last_x, ws.x,
            ws.AT.rowPtr, ws.AT.colVal, ws.AT.nzVal, ws.y, ws.ATy, ws.c, ws.sigma,
            Halpern_fact1, Halpern_fact2, soc_var_sizes,
            fast_paths.huge_block_starts, fast_paths.huge_block_offsets,
            fast_paths.huge_block_cone_ids, fast_paths.huge_block_ptr,
            fast_paths.huge_partial_sums, fast_paths.huge_t_raw, fast_paths.huge_proj_t,
            fast_paths.huge_alpha, fast_paths.huge_case, blocks)
    end
end

@inline function launch_unified_update_zx_noQ_SOC_huge_segmented_partial!(
    ws::HPRSOCP_workspace_gpu,
    soc_var_starts::CuVector{Int32},
    soc_var_sizes::CuVector{Int32},
    cone_count::Int,
    Halpern_fact1::Float64,
    Halpern_fact2::Float64)
    launch_unified_update_zx_noQ_SOC_huge_segmented_partial!(
        Val(false), ws, soc_var_starts, soc_var_sizes, cone_count, Halpern_fact1, Halpern_fact2)
end

@inline function launch_unified_update_zx_noQ_SOC_huge_segmented_partial!(::Val{UseCustom},
    ws::HPRSOCP_workspace_gpu,
    soc_var_starts::CuVector{Int32},
    soc_var_sizes::CuVector{Int32},
    cone_count::Int,
    Halpern_fact1::Float64,
    Halpern_fact2::Float64) where {UseCustom}
    fast_paths = ws.soc_var_fast_paths
    threads = HUGE_SOC_KERNEL_THREADS
    blocks = fast_paths.huge_total_blocks
    if blocks > 0
        @cuda threads = threads blocks = blocks prepare_unified_update_zx_noQ_SOC_huge_segmented_kernel_partial!(
            Val(UseCustom),
            fast_paths.huge_partial_sums, fast_paths.huge_t_raw,
            ws.x, ws.AT.rowPtr, ws.AT.colVal, ws.AT.nzVal, ws.y, ws.ATy, ws.c, ws.sigma,
            soc_var_sizes, fast_paths.huge_block_starts,
            fast_paths.huge_block_offsets, fast_paths.huge_block_cone_ids, blocks)
        @cuda threads = DEFAULT_KERNEL_THREADS blocks = cone_count finalize_unified_update_zx_noQ_SOC_huge_kernel_partial!(
            fast_paths.huge_proj_t, fast_paths.huge_alpha, fast_paths.huge_case,
            fast_paths.huge_partial_sums, fast_paths.huge_t_raw,
            fast_paths.huge_block_ptr, cone_count)
        @cuda threads = threads blocks = blocks apply_unified_update_zx_noQ_SOC_huge_segmented_kernel_partial!(
            Val(UseCustom),
            ws.z_bar, ws.x_hat, ws.last_x, ws.x,
            ws.AT.rowPtr, ws.AT.colVal, ws.AT.nzVal, ws.y, ws.ATy, ws.c, ws.sigma,
            Halpern_fact1, Halpern_fact2, soc_var_sizes,
            fast_paths.huge_block_starts, fast_paths.huge_block_offsets,
            fast_paths.huge_block_cone_ids, fast_paths.huge_t_raw,
            fast_paths.huge_proj_t, fast_paths.huge_alpha, fast_paths.huge_case, blocks)
    end
end

@inline function launch_unified_update_zx_noQ_SOC_huge_partial!(
    ws::HPRSOCP_workspace_gpu,
    soc_var_starts::CuVector{Int32},
    soc_var_sizes::CuVector{Int32},
    cone_count::Int,
    Halpern_fact1::Float64,
    Halpern_fact2::Float64)
    launch_unified_update_zx_noQ_SOC_huge_partial!(
        Val(false), ws, soc_var_starts, soc_var_sizes, cone_count, Halpern_fact1, Halpern_fact2)
end

@inline function launch_unified_update_zx_noQ_SOC_huge_partial!(::Val{UseCustom},
    ws::HPRSOCP_workspace_gpu,
    soc_var_starts::CuVector{Int32},
    soc_var_sizes::CuVector{Int32},
    cone_count::Int,
    Halpern_fact1::Float64,
    Halpern_fact2::Float64) where {UseCustom}
    if ws.soc_var_fast_paths.huge_kernel_mode == :cooperative
        launch_unified_update_zx_noQ_SOC_huge_cooperative_partial!(
            Val(UseCustom), ws, soc_var_starts, soc_var_sizes, cone_count, Halpern_fact1, Halpern_fact2)
    elseif ws.soc_var_fast_paths.huge_kernel_mode == :segmented
        launch_unified_update_zx_noQ_SOC_huge_segmented_partial!(
            Val(UseCustom), ws, soc_var_starts, soc_var_sizes, cone_count, Halpern_fact1, Halpern_fact2)
    else
        launch_unified_update_zx_noQ_SOC_huge_partial_staged!(
            Val(UseCustom), ws, soc_var_starts, soc_var_sizes, cone_count, Halpern_fact1, Halpern_fact2)
    end
end

# Full version: computes all intermediate values
CUDA.@fastmath @inline function unified_update_zxw_kernel_full!(::Val{UseCustom}, ::Val{IsDiag},
    y::CuDeviceVector{Float64},
    last_w::CuDeviceVector{Float64}, dw::CuDeviceVector{Float64}, dx::CuDeviceVector{Float64},
    rowPtrAT::CuDeviceVector{Int32}, colValAT::CuDeviceVector{Int32}, nzValAT::CuDeviceVector{Float64},
    w_bar::CuDeviceVector{Float64}, w::CuDeviceVector{Float64},
    z_bar::CuDeviceVector{Float64}, x_bar::CuDeviceVector{Float64}, x_hat::CuDeviceVector{Float64},
    last_x::CuDeviceVector{Float64}, x::CuDeviceVector{Float64},
    Qw::CuDeviceVector{Float64}, ATy::CuDeviceVector{Float64}, c::CuDeviceVector{Float64},
    l::CuDeviceVector{Float64}, u::CuDeviceVector{Float64},
    sigma::Float64, fact1_scalar::Float64, fact2_scalar::Float64,
    fact1_vec::CuDeviceVector{Float64}, fact2_vec::CuDeviceVector{Float64},
    Halpern_fact1::Float64, Halpern_fact2::Float64, n::Int) where {UseCustom,IsDiag}
    i = threadIdx().x + (blockDim().x * (blockIdx().x - 1))
    @inbounds if i <= n
        ATy_val = if UseCustom
            startAT = rowPtrAT[i]
            stopAT = rowPtrAT[i+1] - 1
            acc = 0.0
            @inbounds for k in startAT:stopAT
                acc += nzValAT[k] * y[colValAT[k]]
            end
            ATy[i] = acc
            acc
        else
            ATy[i]
        end

        atyi = ATy_val
        qw_i = Qw[i]
        c_i = c[i]
        x_i = x[i]
        last_x_i = last_x[i]
        last_w_i = last_w[i]
        l_i = l[i]
        u_i = u[i]
        w_i = w[i]

        tmp = -qw_i + atyi - c_i
        z_raw = x_i + sigma * tmp
        x_bar_i = min(max(z_raw, l_i), u_i)

        x_hat_i = 2.0 * x_bar_i - x_i
        x_new = muladd(Halpern_fact2, x_hat_i, Halpern_fact1 * last_x_i)

        w_bar_i = if IsDiag
            fact1_i = fact1_vec[i]
            fact2_i = fact2_vec[i]
            muladd(fact1_i, w_i, fact2_i * x_hat_i)
        else
            muladd(fact1_scalar, w_i, fact2_scalar * x_hat_i)
        end

        w_hat_i = 2.0 * w_bar_i - w_i
        w_new = muladd(Halpern_fact2, w_hat_i, Halpern_fact1 * last_w_i)

        dx_val = x_bar_i - x_i
        dx[i] = dx_val
        x_bar[i] = x_bar_i
        z_bar[i] = (x_bar_i - z_raw) / sigma
        x[i] = x_new
        x_hat[i] = x_hat_i
        w_bar[i] = w_bar_i
        w[i] = w_new
        dw[i] = w_bar_i - w_i
    end
    return
end

# Partial version: skips intermediate writes
CUDA.@fastmath @inline function unified_update_zxw_kernel_partial!(::Val{UseCustom}, ::Val{IsDiag},
    y::CuDeviceVector{Float64},
    last_w::CuDeviceVector{Float64}, dw::CuDeviceVector{Float64}, dx::CuDeviceVector{Float64},
    rowPtrAT::CuDeviceVector{Int32}, colValAT::CuDeviceVector{Int32}, nzValAT::CuDeviceVector{Float64},
    w_bar::CuDeviceVector{Float64}, w::CuDeviceVector{Float64},
    z_bar::CuDeviceVector{Float64}, x_bar::CuDeviceVector{Float64}, x_hat::CuDeviceVector{Float64},
    last_x::CuDeviceVector{Float64}, x::CuDeviceVector{Float64},
    Qw::CuDeviceVector{Float64}, ATy::CuDeviceVector{Float64}, c::CuDeviceVector{Float64},
    l::CuDeviceVector{Float64}, u::CuDeviceVector{Float64},
    sigma::Float64, fact1_scalar::Float64, fact2_scalar::Float64,
    fact1_vec::CuDeviceVector{Float64}, fact2_vec::CuDeviceVector{Float64},
    Halpern_fact1::Float64, Halpern_fact2::Float64, n::Int) where {UseCustom,IsDiag}
    i = threadIdx().x + (blockDim().x * (blockIdx().x - 1))
    @inbounds if i <= n
        ATy_val = if UseCustom
            startAT = rowPtrAT[i]
            stopAT = rowPtrAT[i+1] - 1
            acc = 0.0
            @inbounds for k in startAT:stopAT
                acc += nzValAT[k] * y[colValAT[k]]
            end
            ATy[i] = acc
            acc
        else
            ATy[i]
        end

        atyi = ATy_val
        qw_i = Qw[i]
        c_i = c[i]
        x_i = x[i]
        last_x_i = last_x[i]
        last_w_i = last_w[i]
        l_i = l[i]
        u_i = u[i]
        w_i = w[i]

        tmp = -qw_i + atyi - c_i
        z_raw = x_i + sigma * tmp
        x_bar_i = min(max(z_raw, l_i), u_i)

        x_hat_i = 2.0 * x_bar_i - x_i
        x_new = muladd(Halpern_fact2, x_hat_i, Halpern_fact1 * last_x_i)

        w_bar_i = if IsDiag
            fact1_i = fact1_vec[i]
            fact2_i = fact2_vec[i]
            muladd(fact1_i, w_i, fact2_i * x_hat_i)
        else
            muladd(fact1_scalar, w_i, fact2_scalar * x_hat_i)
        end
        w_hat_i = 2.0 * w_bar_i - w_i
        w_new = muladd(Halpern_fact2, w_hat_i, Halpern_fact1 * last_w_i)

        x[i] = x_new
        x_hat[i] = x_hat_i
        w_bar[i] = w_bar_i
        w[i] = w_new
    end
    return
end

# Unified tempv computation kernel
# Computes: tempv = x_hat + sigma * (Qw - Qw_bar)
# where Qw_bar can be computed inline (custom) or pre-computed (cuSPARSE)
CUDA.@fastmath @inline function compute_tempv_unified_kernel!(::Val{UseCustom},
    tempv::CuDeviceVector{Float64},
    rowPtrQ::CuDeviceVector{Int32},
    colValQ::CuDeviceVector{Int32},
    nzValQ::CuDeviceVector{Float64},
    w_bar::CuDeviceVector{Float64},
    x_hat::CuDeviceVector{Float64},
    Qw::CuDeviceVector{Float64},
    Qw_bar::CuDeviceVector{Float64},
    sigma::Float64,
    n::Int) where {UseCustom}
    i = threadIdx().x + (blockDim().x * (blockIdx().x - 1))
    @inbounds if i <= n
        if UseCustom
            startQ = rowPtrQ[i]
            stopQ = rowPtrQ[i+1] - 1
            acc = 0.0
            @inbounds for k in startQ:stopQ
                acc += nzValQ[k] * w_bar[colValQ[k]]
            end
            x_hat_i = x_hat[i]
            qw_i = Qw[i]
            tempv[i] = x_hat_i + sigma * (qw_i - acc)
        else
            x_hat_i = x_hat[i]
            qw_i = Qw[i]
            qw_bar_i = Qw_bar[i]
            tempv[i] = x_hat_i + sigma * (qw_i - qw_bar_i)
        end
    end
    return
end

# Unified tempv computation kernel
# Computes: tempv = x_hat + sigma * (Qw - Qw_bar)
# where Qw_bar can be computed inline (custom) or pre-computed (cuSPARSE)
CUDA.@fastmath @inline function compute_tempv_zxwy_unified_kernel!(::Val{UseCustom},
    Halpern_fact1::Float64, Halpern_fact2::Float64,
    last_Qw::CuDeviceVector{Float64},
    tempv::CuDeviceVector{Float64},
    rowPtrQ::CuDeviceVector{Int32},
    colValQ::CuDeviceVector{Int32},
    nzValQ::CuDeviceVector{Float64},
    w_bar::CuDeviceVector{Float64},
    x_hat::CuDeviceVector{Float64},
    Qw::CuDeviceVector{Float64},
    Qw_bar::CuDeviceVector{Float64},
    sigma::Float64,
    n::Int) where {UseCustom}
    i = threadIdx().x + (blockDim().x * (blockIdx().x - 1))
    @inbounds if i <= n
        if UseCustom
            startQ = rowPtrQ[i]
            stopQ = rowPtrQ[i+1] - 1
            acc = 0.0
            @inbounds for k in startQ:stopQ
                acc += nzValQ[k] * w_bar[colValQ[k]]
            end
            x_hat_i = x_hat[i]
            qw_i = Qw[i]
            qw_hat_i = 2.0 * acc - qw_i
            Qw_bar[i] = acc
            Qw[i] = muladd(Halpern_fact2, qw_hat_i, Halpern_fact1 * last_Qw[i])
            tempv[i] = x_hat_i + sigma * (qw_i - acc)
        else
            x_hat_i = x_hat[i]
            qw_i = Qw[i]
            qw_bar_i = Qw_bar[i]
            qw_hat_i = 2.0 * qw_bar_i - qw_i
            Qw[i] = muladd(Halpern_fact2, qw_hat_i, Halpern_fact1 * last_Qw[i])
            tempv[i] = x_hat_i + sigma * (qw_i - qw_bar_i)
        end
    end
    return
end

# Unified update_y kernel
# Combines A*tempv computation with y update in a single kernel
#
# SpMV Strategy (use_custom_spmv):
#   - true:  Compute A*tempv inline (better for small problems)
#   - false: Use pre-computed Ax from cuSPARSE (better for large problems)
#
# Key optimization: Fuses A*tempv SpMV with y update to reduce memory traffic
#
# Full version: computes all intermediate values
CUDA.@fastmath @inline function unified_update_y_kernel_full!(::Val{UseCustom},
    dy::CuDeviceVector{Float64},
    rowPtrA::CuDeviceVector{Int32},
    colValA::CuDeviceVector{Int32},
    nzValA::CuDeviceVector{Float64},
    tempv::CuDeviceVector{Float64},
    Ax::CuDeviceVector{Float64},
    y_bar::CuDeviceVector{Float64},
    y::CuDeviceVector{Float64},
    last_y::CuDeviceVector{Float64},
    s::CuDeviceVector{Float64},
    AL::CuDeviceVector{Float64},
    AU::CuDeviceVector{Float64},
    fact1::Float64,
    fact2::Float64,
    Halpern_fact1::Float64,
    Halpern_fact2::Float64,
    m::Int) where {UseCustom}
    i = threadIdx().x + (blockDim().x * (blockIdx().x - 1))
    @inbounds if i <= m
        Ax_val = if UseCustom
            startA = rowPtrA[i]
            stopA = rowPtrA[i+1] - 1
            acc = 0.0
            @inbounds for k in startA:stopA
                acc += nzValA[k] * tempv[colValA[k]]
            end
            Ax[i] = acc
            acc
        else
            Ax[i]
        end

        y_i = y[i]
        last_y_i = last_y[i]
        AL_i = AL[i]
        AU_i = AU[i]

        s_raw = Ax_val - fact1 * y_i
        s_proj = min(max(s_raw, AL_i), AU_i)
        # Original ternary correction: (s_raw < AL_i) ? (AL_i - s_raw) : ((s_raw > AU_i) ? (AU_i - s_raw) : 0.0)
        corr = s_proj - s_raw
        y_bar_i = fact2 * corr
        y_new = Halpern_fact1 * last_y_i + Halpern_fact2 * (2.0 * y_bar_i - y_i)

        s[i] = s_proj
        dy_i = y_bar_i - y_i
        dy[i] = dy_i
        y_bar[i] = y_bar_i
        y[i] = y_new
    end
    return
end

# Partial version: skips intermediate writes
CUDA.@fastmath @inline function unified_update_y_kernel_partial!(::Val{UseCustom},
    dy::CuDeviceVector{Float64},
    rowPtrA::CuDeviceVector{Int32},
    colValA::CuDeviceVector{Int32},
    nzValA::CuDeviceVector{Float64},
    tempv::CuDeviceVector{Float64},
    Ax::CuDeviceVector{Float64},
    y_bar::CuDeviceVector{Float64},
    y::CuDeviceVector{Float64},
    last_y::CuDeviceVector{Float64},
    s::CuDeviceVector{Float64},
    AL::CuDeviceVector{Float64},
    AU::CuDeviceVector{Float64},
    fact1::Float64,
    fact2::Float64,
    Halpern_fact1::Float64,
    Halpern_fact2::Float64,
    m::Int) where {UseCustom}
    i = threadIdx().x + (blockDim().x * (blockIdx().x - 1))
    @inbounds if i <= m
        Ax_val = if UseCustom
            startA = rowPtrA[i]
            stopA = rowPtrA[i+1] - 1
            acc = 0.0
            @inbounds for k in startA:stopA
                acc += nzValA[k] * tempv[colValA[k]]
            end
            acc
        else
            Ax[i]
        end

        y_i = y[i]
        last_y_i = last_y[i]
        AL_i = AL[i]
        AU_i = AU[i]

        s_raw = Ax_val - fact1 * y_i
        s_proj = min(max(s_raw, AL_i), AU_i)
        corr = s_proj - s_raw
        y_bar_i = fact2 * corr
        y_new = Halpern_fact1 * last_y_i + Halpern_fact2 * (2.0 * y_bar_i - y_i)

        y_bar[i] = y_bar_i
        y[i] = y_new
    end
    return
end

CUDA.@fastmath @inline function unified_update_y_SOC_size3_kernel_full!(
    dy::CuDeviceVector{Float64},
    y_bar::CuDeviceVector{Float64},
    y::CuDeviceVector{Float64},
    last_y::CuDeviceVector{Float64},
    s::CuDeviceVector{Float64},
    soc_rhs::CuDeviceVector{Float64},
    Ax::CuDeviceVector{Float64},
    fact1::Float64,
    fact2::Float64,
    Halpern_fact1::Float64,
    Halpern_fact2::Float64,
    soc_con_first::Int32,
    soc_con_starts::CuDeviceVector{Int32},
    cone_count::Int)
    i = threadIdx().x + (blockDim().x * (blockIdx().x - 1))
    @inbounds if i <= cone_count
        start_idx = Int(soc_con_starts[i])
        j1 = start_idx + 1
        j2 = start_idx + 2
        offset = start_idx - Int(soc_con_first) + 1

        y_t = y[start_idx]
        y_1 = y[j1]
        y_2 = y[j2]
        t_raw = Ax[start_idx] - soc_rhs[offset] - fact1 * y_t
        s_raw_1 = Ax[j1] - soc_rhs[offset + 1] - fact1 * y_1
        s_raw_2 = Ax[j2] - soc_rhs[offset + 2] - fact1 * y_2

        s[start_idx] = t_raw
        s[j1] = s_raw_1
        s[j2] = s_raw_2

        norm_s = 0.0
        norm_s += s_raw_1^2
        norm_s += s_raw_2^2
        norm_s = sqrt(norm_s)

        if norm_s <= -t_raw
            y_bar_t = fact2 * (-t_raw)
            y_bar_1 = fact2 * (-s_raw_1)
            y_bar_2 = fact2 * (-s_raw_2)

            s[start_idx] = 0.0
            s[j1] = 0.0
            s[j2] = 0.0
        elseif norm_s <= t_raw
            y_bar_t = 0.0
            y_bar_1 = 0.0
            y_bar_2 = 0.0
        else
            proj_t = (norm_s + t_raw) / 2.0
            alpha = proj_t / norm_s
            proj_1 = alpha * s_raw_1
            proj_2 = alpha * s_raw_2

            y_bar_t = fact2 * (proj_t - t_raw)
            y_bar_1 = fact2 * (proj_1 - s_raw_1)
            y_bar_2 = fact2 * (proj_2 - s_raw_2)

            s[start_idx] = proj_t
            s[j1] = proj_1
            s[j2] = proj_2
        end

        dy_t = y_bar_t - y_t
        dy_1 = y_bar_1 - y_1
        dy_2 = y_bar_2 - y_2

        dy[start_idx] = dy_t
        dy[j1] = dy_1
        dy[j2] = dy_2
        y_bar[start_idx] = y_bar_t
        y_bar[j1] = y_bar_1
        y_bar[j2] = y_bar_2
        y[start_idx] = Halpern_fact1 * last_y[start_idx] + Halpern_fact2 * (2.0 * y_bar_t - y_t)
        y[j1] = Halpern_fact1 * last_y[j1] + Halpern_fact2 * (2.0 * y_bar_1 - y_1)
        y[j2] = Halpern_fact1 * last_y[j2] + Halpern_fact2 * (2.0 * y_bar_2 - y_2)
    end
    return
end

CUDA.@fastmath @inline function unified_update_y_SOC_size3_kernel_partial!(
    y_bar::CuDeviceVector{Float64},
    y::CuDeviceVector{Float64},
    last_y::CuDeviceVector{Float64},
    soc_rhs::CuDeviceVector{Float64},
    Ax::CuDeviceVector{Float64},
    fact1::Float64,
    fact2::Float64,
    Halpern_fact1::Float64,
    Halpern_fact2::Float64,
    soc_con_first::Int32,
    soc_con_starts::CuDeviceVector{Int32},
    cone_count::Int)
    i = threadIdx().x + (blockDim().x * (blockIdx().x - 1))
    @inbounds if i <= cone_count
        start_idx = Int(soc_con_starts[i])
        j1 = start_idx + 1
        j2 = start_idx + 2
        offset = start_idx - Int(soc_con_first) + 1

        y_t = y[start_idx]
        y_1 = y[j1]
        y_2 = y[j2]
        t_raw = Ax[start_idx] - soc_rhs[offset] - fact1 * y_t
        s_raw_1 = Ax[j1] - soc_rhs[offset + 1] - fact1 * y_1
        s_raw_2 = Ax[j2] - soc_rhs[offset + 2] - fact1 * y_2

        norm_s = 0.0
        norm_s += s_raw_1^2
        norm_s += s_raw_2^2
        norm_s = sqrt(norm_s)

        if norm_s <= -t_raw
            y_bar_t = fact2 * (-t_raw)
            y_bar_1 = fact2 * (-s_raw_1)
            y_bar_2 = fact2 * (-s_raw_2)
        elseif norm_s <= t_raw
            y_bar_t = 0.0
            y_bar_1 = 0.0
            y_bar_2 = 0.0
        else
            proj_t = (norm_s + t_raw) / 2.0
            alpha = proj_t / norm_s
            y_bar_t = fact2 * (proj_t - t_raw)
            y_bar_1 = fact2 * (alpha * s_raw_1 - s_raw_1)
            y_bar_2 = fact2 * (alpha * s_raw_2 - s_raw_2)
        end

        y_bar[start_idx] = y_bar_t
        y_bar[j1] = y_bar_1
        y_bar[j2] = y_bar_2
        y[start_idx] = Halpern_fact1 * last_y[start_idx] + Halpern_fact2 * (2.0 * y_bar_t - y_t)
        y[j1] = Halpern_fact1 * last_y[j1] + Halpern_fact2 * (2.0 * y_bar_1 - y_1)
        y[j2] = Halpern_fact1 * last_y[j2] + Halpern_fact2 * (2.0 * y_bar_2 - y_2)
    end
    return
end

CUDA.@fastmath function unified_update_y_SOC_large_kernel_full!(
    dy::CuDeviceVector{Float64},
    y_bar::CuDeviceVector{Float64},
    y::CuDeviceVector{Float64},
    last_y::CuDeviceVector{Float64},
    s::CuDeviceVector{Float64},
    soc_rhs::CuDeviceVector{Float64},
    Ax::CuDeviceVector{Float64},
    fact1::Float64,
    fact2::Float64,
    Halpern_fact1::Float64,
    Halpern_fact2::Float64,
    soc_con_first::Int32,
    soc_con_starts::CuDeviceVector{Int32},
    soc_con_sizes::CuDeviceVector{Int32},
    cone_count::Int)
    cone_idx = Int(blockIdx().x)
    tid = Int(threadIdx().x)
    block_threads = Int(blockDim().x)

    shared_meta = CuStaticSharedArray(Float64, 3)  # t_raw, projected_t, alpha

    @inbounds if cone_idx <= cone_count
        start_idx = Int(soc_con_starts[cone_idx])
        cone_size = Int(soc_con_sizes[cone_idx])
        end_idx = start_idx + cone_size - 1
        offset = start_idx - Int(soc_con_first) + 1

        if tid == 1
            t_raw = Ax[start_idx] - soc_rhs[offset] - fact1 * y[start_idx]
            shared_meta[1] = t_raw
            s[start_idx] = t_raw
        end

        for cone_offset in tid:block_threads:(cone_size - 1)
            j = start_idx + cone_offset
            s[j] = Ax[j] - soc_rhs[offset + cone_offset] - fact1 * y[j]
        end
        sync_threads()

        if tid == 1
            t_raw = shared_meta[1]
            norm_s = 0.0
            for j in (start_idx + 1):end_idx
                s_raw_j = s[j]
                norm_s += s_raw_j^2
            end
            norm_s = sqrt(norm_s)
            if norm_s <= -t_raw
                shared_meta[2] = 0.0
                shared_meta[3] = 0.0
            elseif norm_s <= t_raw
                shared_meta[2] = t_raw
                shared_meta[3] = 1.0
            else
                proj_t = (norm_s + t_raw) / 2.0
                shared_meta[2] = proj_t
                shared_meta[3] = proj_t / norm_s
            end
        end
        sync_threads()

        t_raw = shared_meta[1]
        projected_t = shared_meta[2]
        alpha = shared_meta[3]

        if tid == 1
            y_start = y[start_idx]
            y_bar_t = fact2 * (projected_t - t_raw)
            s[start_idx] = projected_t
            dy[start_idx] = y_bar_t - y_start
            y_bar[start_idx] = y_bar_t
            y[start_idx] = Halpern_fact1 * last_y[start_idx] + Halpern_fact2 * (2.0 * y_bar_t - y_start)
        end

        for cone_offset in tid:block_threads:(cone_size - 1)
            j = start_idx + cone_offset
            s_raw_j = s[j]
            projected_j = alpha * s_raw_j
            y_j = y[j]
            y_bar_j = fact2 * (projected_j - s_raw_j)
            s[j] = projected_j
            dy[j] = y_bar_j - y_j
            y_bar[j] = y_bar_j
            y[j] = Halpern_fact1 * last_y[j] + Halpern_fact2 * (2.0 * y_bar_j - y_j)
        end
    end
    return
end

CUDA.@fastmath function unified_update_y_SOC_large_kernel_partial!(
    y_bar::CuDeviceVector{Float64},
    y::CuDeviceVector{Float64},
    last_y::CuDeviceVector{Float64},
    soc_rhs::CuDeviceVector{Float64},
    Ax::CuDeviceVector{Float64},
    fact1::Float64,
    fact2::Float64,
    Halpern_fact1::Float64,
    Halpern_fact2::Float64,
    soc_con_first::Int32,
    soc_con_starts::CuDeviceVector{Int32},
    soc_con_sizes::CuDeviceVector{Int32},
    cone_count::Int)
    cone_idx = Int(blockIdx().x)
    tid = Int(threadIdx().x)
    block_threads = Int(blockDim().x)

    shared_meta = CuStaticSharedArray(Float64, 3)  # t_raw, projected_t, alpha

    @inbounds if cone_idx <= cone_count
        start_idx = Int(soc_con_starts[cone_idx])
        cone_size = Int(soc_con_sizes[cone_idx])
        end_idx = start_idx + cone_size - 1
        offset = start_idx - Int(soc_con_first) + 1

        if tid == 1
            shared_meta[1] = Ax[start_idx] - soc_rhs[offset] - fact1 * y[start_idx]
        end

        for cone_offset in tid:block_threads:(cone_size - 1)
            j = start_idx + cone_offset
            y_bar[j] = Ax[j] - soc_rhs[offset + cone_offset] - fact1 * y[j]
        end
        sync_threads()

        if tid == 1
            t_raw = shared_meta[1]
            norm_s = 0.0
            for j in (start_idx + 1):end_idx
                s_raw_j = y_bar[j]
                norm_s += s_raw_j^2
            end
            norm_s = sqrt(norm_s)
            if norm_s <= -t_raw
                shared_meta[2] = 0.0
                shared_meta[3] = 0.0
            elseif norm_s <= t_raw
                shared_meta[2] = t_raw
                shared_meta[3] = 1.0
            else
                proj_t = (norm_s + t_raw) / 2.0
                shared_meta[2] = proj_t
                shared_meta[3] = proj_t / norm_s
            end
        end
        sync_threads()

        t_raw = shared_meta[1]
        projected_t = shared_meta[2]
        alpha = shared_meta[3]

        if tid == 1
            y_start = y[start_idx]
            y_bar_t = fact2 * (projected_t - t_raw)
            y_bar[start_idx] = y_bar_t
            y[start_idx] = Halpern_fact1 * last_y[start_idx] + Halpern_fact2 * (2.0 * y_bar_t - y_start)
        end

        for cone_offset in tid:block_threads:(cone_size - 1)
            j = start_idx + cone_offset
            s_raw_j = y_bar[j]
            projected_j = alpha * s_raw_j
            y_j = y[j]
            y_bar_j = fact2 * (projected_j - s_raw_j)
            y_bar[j] = y_bar_j
            y[j] = Halpern_fact1 * last_y[j] + Halpern_fact2 * (2.0 * y_bar_j - y_j)
        end
    end
    return
end

CUDA.@fastmath @inline function unified_update_y_SOC_kernel_full!(
    dy::CuDeviceVector{Float64},
    y_bar::CuDeviceVector{Float64},
    y::CuDeviceVector{Float64},
    last_y::CuDeviceVector{Float64},
    s::CuDeviceVector{Float64},
    soc_rhs::CuDeviceVector{Float64},
    Ax::CuDeviceVector{Float64},
    fact1::Float64,
    fact2::Float64,
    Halpern_fact1::Float64,
    Halpern_fact2::Float64,
    soc_con_first::Int32,
    soc_con_starts::CuDeviceVector{Int32},
    soc_con_sizes::CuDeviceVector{Int32},
    cone_count::Int)
    i = threadIdx().x + (blockDim().x * (blockIdx().x - 1))
    @inbounds if i <= cone_count
        start_idx = Int(soc_con_starts[i])
        cone_size = Int(soc_con_sizes[i])
        end_idx = start_idx + cone_size - 1
        offset = start_idx - Int(soc_con_first) + 1

        t_raw = Ax[start_idx] - soc_rhs[offset] - fact1 * y[start_idx]
        s[start_idx] = t_raw
        norm_s = 0.0
        for j in (start_idx + 1):end_idx
            s_raw_j = Ax[j] - soc_rhs[offset + (j - start_idx)] - fact1 * y[j]
            s[j] = s_raw_j
            norm_s += s_raw_j^2
        end
        norm_s = sqrt(norm_s)

        if norm_s <= -t_raw
            s[start_idx] = 0.0
            y_bar[start_idx] = fact2 * (-t_raw)
            dy[start_idx] = y_bar[start_idx] - y[start_idx]
            y[start_idx] = Halpern_fact1 * last_y[start_idx] + Halpern_fact2 * (2.0 * y_bar[start_idx] - y[start_idx])

            for j in (start_idx + 1):end_idx
                s_raw_j = s[j]
                s[j] = 0.0
                y_bar[j] = fact2 * (-s_raw_j)
                dy[j] = y_bar[j] - y[j]
                y[j] = Halpern_fact1 * last_y[j] + Halpern_fact2 * (2.0 * y_bar[j] - y[j])
            end
        elseif norm_s <= t_raw
            y_bar[start_idx] = 0.0
            dy[start_idx] = -y[start_idx]
            y[start_idx] = Halpern_fact1 * last_y[start_idx] + Halpern_fact2 * (-y[start_idx])

            for j in (start_idx + 1):end_idx
                y_bar[j] = 0.0
                dy[j] = -y[j]
                y[j] = Halpern_fact1 * last_y[j] + Halpern_fact2 * (-y[j])
            end
        else
            proj_t = (norm_s + t_raw) / 2.0
            alpha = proj_t / norm_s

            s[start_idx] = proj_t
            y_bar[start_idx] = fact2 * (proj_t - t_raw)
            dy[start_idx] = y_bar[start_idx] - y[start_idx]
            y[start_idx] = Halpern_fact1 * last_y[start_idx] + Halpern_fact2 * (2.0 * y_bar[start_idx] - y[start_idx])

            for j in (start_idx + 1):end_idx
                s_raw_j = s[j]
                proj_j = alpha * s_raw_j
                s[j] = proj_j
                y_bar[j] = fact2 * (proj_j - s_raw_j)
                dy[j] = y_bar[j] - y[j]
                y[j] = Halpern_fact1 * last_y[j] + Halpern_fact2 * (2.0 * y_bar[j] - y[j])
            end
        end
    end
    return
end

@inline function launch_unified_update_y_SOC_large!(
    ws::HPRSOCP_workspace_gpu,
    soc_con_first::Int32,
    soc_con_starts::CuVector{Int32},
    soc_con_sizes::CuVector{Int32},
    cone_count::Int,
    fact1::Float64,
    fact2::Float64,
    Halpern_fact1::Float64,
    Halpern_fact2::Float64)
    threads = DEFAULT_KERNEL_THREADS
    blocks = cone_count
    if blocks > 0
        if ws.to_check
            @cuda threads = threads blocks = blocks unified_update_y_SOC_large_kernel_full!(
                ws.dy, ws.y_bar, ws.y, ws.last_y, ws.s, ws.soc_rhs, ws.Ax,
                fact1, fact2, Halpern_fact1, Halpern_fact2, soc_con_first,
                soc_con_starts, soc_con_sizes, cone_count)
        else
            @cuda threads = threads blocks = blocks unified_update_y_SOC_large_kernel_partial!(
                ws.y_bar, ws.y, ws.last_y, ws.soc_rhs, ws.Ax,
                fact1, fact2, Halpern_fact1, Halpern_fact2, soc_con_first,
                soc_con_starts, soc_con_sizes, cone_count)
        end
    end
end

CUDA.@fastmath @inline function unified_update_y_SOC_kernel_partial!(
    y_bar::CuDeviceVector{Float64},
    y::CuDeviceVector{Float64},
    last_y::CuDeviceVector{Float64},
    soc_rhs::CuDeviceVector{Float64},
    Ax::CuDeviceVector{Float64},
    fact1::Float64,
    fact2::Float64,
    Halpern_fact1::Float64,
    Halpern_fact2::Float64,
    soc_con_first::Int32,
    soc_con_starts::CuDeviceVector{Int32},
    soc_con_sizes::CuDeviceVector{Int32},
    cone_count::Int)
    i = threadIdx().x + (blockDim().x * (blockIdx().x - 1))
    @inbounds if i <= cone_count
        start_idx = Int(soc_con_starts[i])
        cone_size = Int(soc_con_sizes[i])
        end_idx = start_idx + cone_size - 1
        offset = start_idx - Int(soc_con_first) + 1

        t_raw = Ax[start_idx] - soc_rhs[offset] - fact1 * y[start_idx]
        norm_s = 0.0
        for j in (start_idx + 1):end_idx
            s_raw_j = Ax[j] - soc_rhs[offset + (j - start_idx)] - fact1 * y[j]
            norm_s += s_raw_j^2
        end
        norm_s = sqrt(norm_s)

        if norm_s <= -t_raw
            y_start = y[start_idx]
            y_bar_start = fact2 * (-t_raw)
            y_bar[start_idx] = y_bar_start
            y[start_idx] = Halpern_fact1 * last_y[start_idx] + Halpern_fact2 * (2.0 * y_bar_start - y_start)

            for j in (start_idx + 1):end_idx
                s_raw_j = Ax[j] - soc_rhs[offset + (j - start_idx)] - fact1 * y[j]
                y_j = y[j]
                y_bar_j = fact2 * (-s_raw_j)
                y_bar[j] = y_bar_j
                y[j] = Halpern_fact1 * last_y[j] + Halpern_fact2 * (2.0 * y_bar_j - y_j)
            end
        elseif norm_s <= t_raw
            y_start = y[start_idx]
            y_bar[start_idx] = 0.0
            y[start_idx] = Halpern_fact1 * last_y[start_idx] + Halpern_fact2 * (-y_start)

            for j in (start_idx + 1):end_idx
                y_j = y[j]
                y_bar[j] = 0.0
                y[j] = Halpern_fact1 * last_y[j] + Halpern_fact2 * (-y_j)
            end
        else
            proj_t = (norm_s + t_raw) / 2.0
            alpha = proj_t / norm_s

            y_start = y[start_idx]
            y_bar_start = fact2 * (proj_t - t_raw)
            y_bar[start_idx] = y_bar_start
            y[start_idx] = Halpern_fact1 * last_y[start_idx] + Halpern_fact2 * (2.0 * y_bar_start - y_start)

            for j in (start_idx + 1):end_idx
                s_raw_j = Ax[j] - soc_rhs[offset + (j - start_idx)] - fact1 * y[j]
                proj_j = alpha * s_raw_j
                y_j = y[j]
                y_bar_j = fact2 * (proj_j - s_raw_j)
                y_bar[j] = y_bar_j
                y[j] = Halpern_fact1 * last_y[j] + Halpern_fact2 * (2.0 * y_bar_j - y_j)
            end
        end
    end
    return
end

CUDA.@fastmath @inline function unified_update_y_SOC_small_kernel_full!(
    dy::CuDeviceVector{Float64},
    y_bar::CuDeviceVector{Float64},
    y::CuDeviceVector{Float64},
    last_y::CuDeviceVector{Float64},
    s::CuDeviceVector{Float64},
    soc_rhs::CuDeviceVector{Float64},
    Ax::CuDeviceVector{Float64},
    fact1::Float64,
    fact2::Float64,
    Halpern_fact1::Float64,
    Halpern_fact2::Float64,
    soc_con_first::Int32,
    soc_con_starts::CuDeviceVector{Int32},
    soc_con_sizes::CuDeviceVector{Int32},
    cone_count::Int)
    i = threadIdx().x + (blockDim().x * (blockIdx().x - 1))
    @inbounds if i <= cone_count
        start_idx = Int(soc_con_starts[i])
        cone_size = Int(soc_con_sizes[i])
        end_idx = start_idx + cone_size - 1
        offset = start_idx - Int(soc_con_first) + 1

        t_raw = Ax[start_idx] - soc_rhs[offset] - fact1 * y[start_idx]
        s[start_idx] = t_raw
        norm_s = 0.0
        for j in (start_idx + 1):end_idx
            s_raw_j = Ax[j] - soc_rhs[offset + (j - start_idx)] - fact1 * y[j]
            s[j] = s_raw_j
            norm_s += s_raw_j^2
        end
        norm_s = sqrt(norm_s)

        if norm_s <= -t_raw
            s[start_idx] = 0.0
            y_bar[start_idx] = fact2 * (-t_raw)
            dy[start_idx] = y_bar[start_idx] - y[start_idx]
            y[start_idx] = Halpern_fact1 * last_y[start_idx] + Halpern_fact2 * (2.0 * y_bar[start_idx] - y[start_idx])

            for j in (start_idx + 1):end_idx
                s_raw_j = s[j]
                s[j] = 0.0
                y_bar[j] = fact2 * (-s_raw_j)
                dy[j] = y_bar[j] - y[j]
                y[j] = Halpern_fact1 * last_y[j] + Halpern_fact2 * (2.0 * y_bar[j] - y[j])
            end
        elseif norm_s <= t_raw
            y_bar[start_idx] = 0.0
            dy[start_idx] = -y[start_idx]
            y[start_idx] = Halpern_fact1 * last_y[start_idx] + Halpern_fact2 * (-y[start_idx])

            for j in (start_idx + 1):end_idx
                y_bar[j] = 0.0
                dy[j] = -y[j]
                y[j] = Halpern_fact1 * last_y[j] + Halpern_fact2 * (-y[j])
            end
        else
            proj_t = (norm_s + t_raw) / 2.0
            alpha = proj_t / norm_s

            s[start_idx] = proj_t
            y_bar[start_idx] = fact2 * (proj_t - t_raw)
            dy[start_idx] = y_bar[start_idx] - y[start_idx]
            y[start_idx] = Halpern_fact1 * last_y[start_idx] + Halpern_fact2 * (2.0 * y_bar[start_idx] - y[start_idx])

            for j in (start_idx + 1):end_idx
                s_raw_j = s[j]
                proj_j = alpha * s_raw_j
                s[j] = proj_j
                y_bar[j] = fact2 * (proj_j - s_raw_j)
                dy[j] = y_bar[j] - y[j]
                y[j] = Halpern_fact1 * last_y[j] + Halpern_fact2 * (2.0 * y_bar[j] - y[j])
            end
        end
    end
    return
end

CUDA.@fastmath @inline function unified_update_y_SOC_small_kernel_partial!(
    y_bar::CuDeviceVector{Float64},
    y::CuDeviceVector{Float64},
    last_y::CuDeviceVector{Float64},
    soc_rhs::CuDeviceVector{Float64},
    Ax::CuDeviceVector{Float64},
    fact1::Float64,
    fact2::Float64,
    Halpern_fact1::Float64,
    Halpern_fact2::Float64,
    soc_con_first::Int32,
    soc_con_starts::CuDeviceVector{Int32},
    soc_con_sizes::CuDeviceVector{Int32},
    cone_count::Int)
    i = threadIdx().x + (blockDim().x * (blockIdx().x - 1))
    @inbounds if i <= cone_count
        start_idx = Int(soc_con_starts[i])
        cone_size = Int(soc_con_sizes[i])
        end_idx = start_idx + cone_size - 1
        offset = start_idx - Int(soc_con_first) + 1

        t_raw = Ax[start_idx] - soc_rhs[offset] - fact1 * y[start_idx]
        norm_s = 0.0
        for j in (start_idx + 1):end_idx
            s_raw_j = Ax[j] - soc_rhs[offset + (j - start_idx)] - fact1 * y[j]
            norm_s += s_raw_j^2
        end
        norm_s = sqrt(norm_s)

        if norm_s <= -t_raw
            y_start = y[start_idx]
            y_bar_start = fact2 * (-t_raw)
            y_bar[start_idx] = y_bar_start
            y[start_idx] = Halpern_fact1 * last_y[start_idx] + Halpern_fact2 * (2.0 * y_bar_start - y_start)

            for j in (start_idx + 1):end_idx
                s_raw_j = Ax[j] - soc_rhs[offset + (j - start_idx)] - fact1 * y[j]
                y_j = y[j]
                y_bar_j = fact2 * (-s_raw_j)
                y_bar[j] = y_bar_j
                y[j] = Halpern_fact1 * last_y[j] + Halpern_fact2 * (2.0 * y_bar_j - y_j)
            end
        elseif norm_s <= t_raw
            y_start = y[start_idx]
            y_bar[start_idx] = 0.0
            y[start_idx] = Halpern_fact1 * last_y[start_idx] + Halpern_fact2 * (-y_start)

            for j in (start_idx + 1):end_idx
                y_j = y[j]
                y_bar[j] = 0.0
                y[j] = Halpern_fact1 * last_y[j] + Halpern_fact2 * (-y_j)
            end
        else
            proj_t = (norm_s + t_raw) / 2.0
            alpha = proj_t / norm_s

            y_start = y[start_idx]
            y_bar_start = fact2 * (proj_t - t_raw)
            y_bar[start_idx] = y_bar_start
            y[start_idx] = Halpern_fact1 * last_y[start_idx] + Halpern_fact2 * (2.0 * y_bar_start - y_start)

            for j in (start_idx + 1):end_idx
                s_raw_j = Ax[j] - soc_rhs[offset + (j - start_idx)] - fact1 * y[j]
                proj_j = alpha * s_raw_j
                y_j = y[j]
                y_bar_j = fact2 * (proj_j - s_raw_j)
                y_bar[j] = y_bar_j
                y[j] = Halpern_fact1 * last_y[j] + Halpern_fact2 * (2.0 * y_bar_j - y_j)
            end
        end
    end
    return
end

CUDA.@fastmath function unified_update_y_SOC_size3_cooperative_kernel_partial!(
    y_bar::CuDeviceVector{Float64},
    y::CuDeviceVector{Float64},
    last_y::CuDeviceVector{Float64},
    soc_rhs::CuDeviceVector{Float64},
    Ax::CuDeviceVector{Float64},
    fact1::Float64,
    fact2::Float64,
    Halpern_fact1::Float64,
    Halpern_fact2::Float64,
    soc_con_first::Int32,
    soc_con_starts::CuDeviceVector{Int32},
    cone_count::Int)
    tid = Int(threadIdx().x)
    tid0 = tid - 1
    cone_in_block = tid0 ÷ 3
    lane = tid0 % 3
    cones_per_block = SOC_CON_TINY_COOP_THREADS ÷ 3
    cone_idx = (Int(blockIdx().x) - 1) * cones_per_block + cone_in_block + 1

    shared_raw = CuStaticSharedArray(Float64, SOC_CON_TINY_COOP_THREADS)
    active = cone_idx <= cone_count
    raw = 0.0
    y_j = 0.0
    j = 0

    if active
        start_idx = Int(soc_con_starts[cone_idx])
        j = start_idx + lane
        offset = start_idx - Int(soc_con_first) + 1
        y_j = y[j]
        raw = Ax[j] - soc_rhs[offset + lane] - fact1 * y_j
    end
    shared_raw[tid] = raw
    sync_threads()

    @inbounds if active
        base = cone_in_block * 3
        t_raw = shared_raw[base + 1]
        s_1 = shared_raw[base + 2]
        s_2 = shared_raw[base + 3]
        norm_s = sqrt(s_1^2 + s_2^2)

        if norm_s <= -t_raw
            projected = 0.0
        elseif norm_s <= t_raw
            projected = lane == 0 ? t_raw : raw
        else
            proj_t = (norm_s + t_raw) / 2.0
            alpha = proj_t / norm_s
            projected = lane == 0 ? proj_t : alpha * raw
        end

        y_bar_j = fact2 * (projected - raw)
        y_bar[j] = y_bar_j
        y[j] = Halpern_fact1 * last_y[j] + Halpern_fact2 * (2.0 * y_bar_j - y_j)
    end
    return
end

CUDA.@fastmath function unified_update_y_SOC_size4_cooperative_kernel_partial!(
    y_bar::CuDeviceVector{Float64},
    y::CuDeviceVector{Float64},
    last_y::CuDeviceVector{Float64},
    soc_rhs::CuDeviceVector{Float64},
    Ax::CuDeviceVector{Float64},
    fact1::Float64,
    fact2::Float64,
    Halpern_fact1::Float64,
    Halpern_fact2::Float64,
    soc_con_first::Int32,
    soc_con_starts::CuDeviceVector{Int32},
    cone_count::Int)
    tid = Int(threadIdx().x)
    tid0 = tid - 1
    cone_in_block = tid0 ÷ 4
    lane = tid0 % 4
    cones_per_block = SOC_CON_TINY_COOP_THREADS ÷ 4
    cone_idx = (Int(blockIdx().x) - 1) * cones_per_block + cone_in_block + 1

    shared_raw = CuStaticSharedArray(Float64, SOC_CON_TINY_COOP_THREADS)
    active = cone_idx <= cone_count
    raw = 0.0
    y_j = 0.0
    j = 0

    if active
        start_idx = Int(soc_con_starts[cone_idx])
        j = start_idx + lane
        offset = start_idx - Int(soc_con_first) + 1
        y_j = y[j]
        raw = Ax[j] - soc_rhs[offset + lane] - fact1 * y_j
    end
    shared_raw[tid] = raw
    sync_threads()

    @inbounds if active
        base = cone_in_block * 4
        t_raw = shared_raw[base + 1]
        s_1 = shared_raw[base + 2]
        s_2 = shared_raw[base + 3]
        s_3 = shared_raw[base + 4]
        norm_s = sqrt(s_1^2 + s_2^2 + s_3^2)

        if norm_s <= -t_raw
            projected = 0.0
        elseif norm_s <= t_raw
            projected = lane == 0 ? t_raw : raw
        else
            proj_t = (norm_s + t_raw) / 2.0
            alpha = proj_t / norm_s
            projected = lane == 0 ? proj_t : alpha * raw
        end

        y_bar_j = fact2 * (projected - raw)
        y_bar[j] = y_bar_j
        y[j] = Halpern_fact1 * last_y[j] + Halpern_fact2 * (2.0 * y_bar_j - y_j)
    end
    return
end

CUDA.@fastmath function unified_update_y_SOC_size5_cooperative_kernel_partial!(
    y_bar::CuDeviceVector{Float64},
    y::CuDeviceVector{Float64},
    last_y::CuDeviceVector{Float64},
    soc_rhs::CuDeviceVector{Float64},
    Ax::CuDeviceVector{Float64},
    fact1::Float64,
    fact2::Float64,
    Halpern_fact1::Float64,
    Halpern_fact2::Float64,
    soc_con_first::Int32,
    soc_con_starts::CuDeviceVector{Int32},
    cone_count::Int)
    tid = Int(threadIdx().x)
    tid0 = tid - 1
    cone_in_block = tid0 ÷ 5
    lane = tid0 % 5
    cones_per_block = SOC_CON_SIZE5_COOP_THREADS ÷ 5
    cone_idx = (Int(blockIdx().x) - 1) * cones_per_block + cone_in_block + 1

    shared_raw = CuStaticSharedArray(Float64, SOC_CON_SIZE5_COOP_THREADS)
    active = cone_idx <= cone_count
    raw = 0.0
    y_j = 0.0
    j = 0

    if active
        start_idx = Int(soc_con_starts[cone_idx])
        j = start_idx + lane
        offset = start_idx - Int(soc_con_first) + 1
        y_j = y[j]
        raw = Ax[j] - soc_rhs[offset + lane] - fact1 * y_j
    end
    shared_raw[tid] = raw
    sync_threads()

    @inbounds if active
        base = cone_in_block * 5
        t_raw = shared_raw[base + 1]
        s_1 = shared_raw[base + 2]
        s_2 = shared_raw[base + 3]
        s_3 = shared_raw[base + 4]
        s_4 = shared_raw[base + 5]
        norm_s = sqrt(s_1^2 + s_2^2 + s_3^2 + s_4^2)

        if norm_s <= -t_raw
            projected = 0.0
        elseif norm_s <= t_raw
            projected = lane == 0 ? t_raw : raw
        else
            proj_t = (norm_s + t_raw) / 2.0
            alpha = proj_t / norm_s
            projected = lane == 0 ? proj_t : alpha * raw
        end

        y_bar_j = fact2 * (projected - raw)
        y_bar[j] = y_bar_j
        y[j] = Halpern_fact1 * last_y[j] + Halpern_fact2 * (2.0 * y_bar_j - y_j)
    end
    return
end

@inline function launch_unified_update_y_SOC_size3_cooperative!(
    ws::HPRSOCP_workspace_gpu,
    soc_con_first::Int32,
    soc_con_starts::CuVector{Int32},
    cone_count::Int,
    fact1::Float64,
    fact2::Float64,
    Halpern_fact1::Float64,
    Halpern_fact2::Float64)
    threads = SOC_CON_TINY_COOP_THREADS
    cones_per_block = threads ÷ 3
    blocks = cld(cone_count, cones_per_block)
    if blocks > 0
        @cuda threads = threads blocks = blocks unified_update_y_SOC_size3_cooperative_kernel_partial!(
            ws.y_bar, ws.y, ws.last_y, ws.soc_rhs, ws.Ax,
            fact1, fact2, Halpern_fact1, Halpern_fact2, soc_con_first,
            soc_con_starts, cone_count)
    end
end

@inline function launch_unified_update_y_SOC_size4_cooperative!(
    ws::HPRSOCP_workspace_gpu,
    soc_con_first::Int32,
    soc_con_starts::CuVector{Int32},
    cone_count::Int,
    fact1::Float64,
    fact2::Float64,
    Halpern_fact1::Float64,
    Halpern_fact2::Float64)
    threads = SOC_CON_TINY_COOP_THREADS
    cones_per_block = threads ÷ 4
    blocks = cld(cone_count, cones_per_block)
    if blocks > 0
        @cuda threads = threads blocks = blocks unified_update_y_SOC_size4_cooperative_kernel_partial!(
            ws.y_bar, ws.y, ws.last_y, ws.soc_rhs, ws.Ax,
            fact1, fact2, Halpern_fact1, Halpern_fact2, soc_con_first,
            soc_con_starts, cone_count)
    end
end

@inline function launch_unified_update_y_SOC_size5_cooperative!(
    ws::HPRSOCP_workspace_gpu,
    soc_con_first::Int32,
    soc_con_starts::CuVector{Int32},
    cone_count::Int,
    fact1::Float64,
    fact2::Float64,
    Halpern_fact1::Float64,
    Halpern_fact2::Float64)
    threads = SOC_CON_SIZE5_COOP_THREADS
    cones_per_block = threads ÷ 5
    blocks = cld(cone_count, cones_per_block)
    if blocks > 0
        @cuda threads = threads blocks = blocks unified_update_y_SOC_size5_cooperative_kernel_partial!(
            ws.y_bar, ws.y, ws.last_y, ws.soc_rhs, ws.Ax,
            fact1, fact2, Halpern_fact1, Halpern_fact2, soc_con_first,
            soc_con_starts, cone_count)
    end
end

@inline function launch_unified_update_y_SOC_small!(
    ws::HPRSOCP_workspace_gpu,
    soc_con_first::Int32,
    soc_con_starts::CuVector{Int32},
    soc_con_sizes::CuVector{Int32},
    cone_count::Int,
    fact1::Float64,
    fact2::Float64,
    Halpern_fact1::Float64,
    Halpern_fact2::Float64)
    threads, blocks = gpu_launch_config(cone_count)
    if threads > 0 && cone_count > 0
        if ws.to_check
            @cuda threads = threads blocks = blocks unified_update_y_SOC_small_kernel_full!(
                ws.dy, ws.y_bar, ws.y, ws.last_y, ws.s, ws.soc_rhs, ws.Ax,
                fact1, fact2, Halpern_fact1, Halpern_fact2, soc_con_first,
                soc_con_starts, soc_con_sizes, cone_count)
        else
            @cuda threads = threads blocks = blocks unified_update_y_SOC_small_kernel_partial!(
                ws.y_bar, ws.y, ws.last_y, ws.soc_rhs, ws.Ax,
                fact1, fact2, Halpern_fact1, Halpern_fact2, soc_con_first,
                soc_con_starts, soc_con_sizes, cone_count)
        end
    end
end

# Unified update_w2 kernel
# Combines AT*y_bar computation with w2 update in a single kernel
#
# SpMV Strategy (use_custom_spmv):
#   - true:  Compute AT*y_bar inline (better for small problems)
#   - false: Use pre-computed ATy_bar from cuSPARSE (better for large problems)
#
# Q Matrix Structure (Q_is_diag):
#   - true:  Q is diagonal, use element-wise fact_vec[i]
#   - false: Q is general, use scalar fact_scalar
#
# Key optimization: Fuses AT*y_bar SpMV with w2 update to reduce memory traffic
#
# Full version: computes all intermediate values
CUDA.@fastmath @inline function unified_update_w2_kernel_full!(::Val{UseCustom}, ::Val{IsDiag},
    dw::CuDeviceVector{Float64},
    ATdy::CuDeviceVector{Float64},
    rowPtrAT::CuDeviceVector{Int32},
    colValAT::CuDeviceVector{Int32},
    nzValAT::CuDeviceVector{Float64},
    y_bar::CuDeviceVector{Float64},
    ATy_bar::CuDeviceVector{Float64},
    w::CuDeviceVector{Float64},
    w_bar::CuDeviceVector{Float64},
    last_w::CuDeviceVector{Float64},
    last_ATy::CuDeviceVector{Float64},
    ATy::CuDeviceVector{Float64},
    fact_scalar::Float64,
    fact_vec::CuDeviceVector{Float64},
    Halpern_fact1::Float64,
    Halpern_fact2::Float64,
    n::Int) where {UseCustom,IsDiag}
    i = threadIdx().x + (blockDim().x * (blockIdx().x - 1))
    @inbounds if i <= n
        ATy_bar_val = if UseCustom
            startAT = rowPtrAT[i]
            stopAT = rowPtrAT[i+1] - 1
            acc = 0.0
            @inbounds for k in startAT:stopAT
                acc += nzValAT[k] * y_bar[colValAT[k]]
            end
            ATy_bar[i] = acc
            acc
        else
            ATy_bar[i]
        end

        fact = if IsDiag
            fact_vec[i]
        else
            fact_scalar
        end

        w_i = w[i]
        w_bar_i = w_bar[i]
        ATy_i = ATy[i]
        last_w_i = last_w[i]
        last_ATy_i = last_ATy[i]

        w_bar_new = w_bar_i + fact * (ATy_bar_val - ATy_i)
        w_new = Halpern_fact1 * last_w_i + Halpern_fact2 * (2.0 * w_bar_new - w_i)
        ATy_new = Halpern_fact1 * last_ATy_i + Halpern_fact2 * (2.0 * ATy_bar_val - ATy_i)

        w[i] = w_new
        ATy[i] = ATy_new
        w_bar[i] = w_bar_new
        dw_i = w_bar_new - w_i
        ATdy_i = ATy_bar_val - ATy_i
        dw[i] = dw_i
        ATdy[i] = ATdy_i
    end
    return
end

# Partial version: skips intermediate writes
CUDA.@fastmath @inline function unified_update_w2_kernel_partial!(::Val{UseCustom}, ::Val{IsDiag},
    dw::CuDeviceVector{Float64},
    ATdy::CuDeviceVector{Float64},
    rowPtrAT::CuDeviceVector{Int32},
    colValAT::CuDeviceVector{Int32},
    nzValAT::CuDeviceVector{Float64},
    y_bar::CuDeviceVector{Float64},
    ATy_bar::CuDeviceVector{Float64},
    w::CuDeviceVector{Float64},
    w_bar::CuDeviceVector{Float64},
    last_w::CuDeviceVector{Float64},
    last_ATy::CuDeviceVector{Float64},
    ATy::CuDeviceVector{Float64},
    fact_scalar::Float64,
    fact_vec::CuDeviceVector{Float64},
    Halpern_fact1::Float64,
    Halpern_fact2::Float64,
    n::Int) where {UseCustom,IsDiag}
    i = threadIdx().x + (blockDim().x * (blockIdx().x - 1))
    @inbounds if i <= n
        ATy_bar_val = if UseCustom
            startAT = rowPtrAT[i]
            stopAT = rowPtrAT[i+1] - 1
            acc = 0.0
            @inbounds for k in startAT:stopAT
                acc += nzValAT[k] * y_bar[colValAT[k]]
            end
            acc
        else
            ATy_bar[i]
        end

        fact = if IsDiag
            fact_vec[i]
        else
            fact_scalar
        end

        w_i = w[i]
        w_bar_i = w_bar[i]
        ATy_i = ATy[i]
        last_w_i = last_w[i]
        last_ATy_i = last_ATy[i]

        w_bar_new = w_bar_i + fact * (ATy_bar_val - ATy_i)
        w_new = Halpern_fact1 * last_w_i + Halpern_fact2 * (2.0 * w_bar_new - w_i)
        ATy_new = Halpern_fact1 * last_ATy_i + Halpern_fact2 * (2.0 * ATy_bar_val - ATy_i)

        w[i] = w_new
        ATy[i] = ATy_new
    end
    return
end

# Unified wrapper for update_zxw1 that handles both custom and cuSPARSE SpMV, and regular/diagonal Q
function unified_update_zxw_gpu!(ws::HPRSOCP_workspace_gpu, qp::QP_info_gpu,
    Halpern_fact1::Float64, Halpern_fact2::Float64)
    # Prepare scalar factors for regular Q
    fact2_scalar = 1.0 / (1.0 + ws.sigma * ws.lambda_max_Q)
    fact1_scalar = 1.0 - fact2_scalar

    # Pre-computed diagonal scaling factors (ignored when Q is not diagonal)
    fact1_vec = ws.fact1
    fact2_vec = ws.fact2

    # Get Q matrix structure (use dummy vectors for operators)
    if isa(qp.Q, CuSparseMatrixCSR)
        rowPtrQ, colValQ, nzValQ = qp.Q.rowPtr, qp.Q.colVal, qp.Q.nzVal
    else
        rowPtrQ, colValQ, nzValQ = ws.A.rowPtr, ws.A.colVal, ws.A.nzVal
    end

    if ws.spmv_AT !== nothing
        ATmap!(ws.spmv_AT.desc_y, ws.spmv_AT.desc_ATy, ws.AT, ws.spmv_AT)
    else
        ATmap!(ws.y, ws.ATy, ws.AT, ws.spmv_AT)
    end

    # Step 1: Update z, x, w1
    threads, blocks = gpu_launch_config(ws.n)
    if threads > 0
        # Choose kernel based on to_check flag - no recompilation overhead
        if ws.to_check
            @cuda threads = threads blocks = blocks unified_update_zxw_kernel_full!(
                Val(false), Val(ws.Q_is_diag),
                ws.y,
                ws.last_w, ws.dw, ws.dx, ws.AT.rowPtr, ws.AT.colVal, ws.AT.nzVal,
                ws.w_bar, ws.w, ws.z_bar, ws.x_bar, ws.x_hat, ws.last_x, ws.x, ws.Qw, ws.ATy, ws.c,
                ws.l, ws.u, ws.sigma, fact1_scalar, fact2_scalar, fact1_vec, fact2_vec,
                Halpern_fact1, Halpern_fact2, ws.n)
        else
            @cuda threads = threads blocks = blocks unified_update_zxw_kernel_partial!(
                Val(false), Val(ws.Q_is_diag),
                ws.y,
                ws.last_w, ws.dw, ws.dx, ws.AT.rowPtr, ws.AT.colVal, ws.AT.nzVal,
                ws.w_bar, ws.w, ws.z_bar, ws.x_bar, ws.x_hat, ws.last_x, ws.x, ws.Qw, ws.ATy, ws.c,
                ws.l, ws.u, ws.sigma, fact1_scalar, fact2_scalar, fact1_vec, fact2_vec,
                Halpern_fact1, Halpern_fact2, ws.n)
        end
    end

    if isa(qp.Q, CuSparseMatrixCSR{Float64,Int32})
        Qmap!(ws.w_bar, ws.Qw_bar, qp.Q, ws.spmv_Q)
    else
        Qmap!(ws.w_bar, ws.Qw_bar, qp.Q)
    end

    if threads > 0
        @cuda threads = threads blocks = blocks compute_tempv_zxwy_unified_kernel!(
            Val(false),
            Halpern_fact1, Halpern_fact2,
            ws.last_Qw,
            ws.tempv, rowPtrQ, colValQ, nzValQ,
            ws.w_bar, ws.x_hat, ws.Qw, ws.Qw_bar, ws.sigma, ws.n)
    end
end

# Unified wrapper that handles all cases: regular/diagonal Q, custom/cuSPARSE SpMV
# Unified wrapper for update_zxw1 that handles both custom and cuSPARSE SpMV, and regular/diagonal Q
function unified_update_zxw1_gpu!(ws::HPRSOCP_workspace_gpu, qp::QP_info_gpu,
    Halpern_fact1::Float64, Halpern_fact2::Float64)
    skip_q_work = !has_quadratic_terms(qp.Q)

    # Prepare scalar factors for regular Q
    fact2_scalar = skip_q_work ? 1.0 : 1.0 / (1.0 + ws.sigma * ws.lambda_max_Q)
    fact1_scalar = 1.0 - fact2_scalar

    # Pre-computed diagonal scaling factors (ignored when Q is not diagonal)
    fact1_vec = ws.fact1
    fact2_vec = ws.fact2

    # Get Q matrix structure (use dummy vectors for operators)
    if isa(qp.Q, CuSparseMatrixCSR)
        rowPtrQ, colValQ, nzValQ = qp.Q.rowPtr, qp.Q.colVal, qp.Q.nzVal
    else
        rowPtrQ, colValQ, nzValQ = ws.A.rowPtr, ws.A.colVal, ws.A.nzVal
    end

    if !skip_q_work
        if isa(qp.Q, CuSparseMatrixCSR{Float64,Int32}) && ws.spmv_Q !== nothing
            Qmap!(ws.spmv_Q.desc_w, ws.spmv_Q.desc_Qw, qp.Q, ws.spmv_Q)
        elseif isa(qp.Q, CuSparseMatrixCSR{Float64,Int32})
            Qmap!(ws.w, ws.Qw, qp.Q, ws.spmv_Q)
        else
            Qmap!(ws.w, ws.Qw, qp.Q)
        end
    end

    # Step 1: Update z, x, w1
    linear_n = ws.number_SOC_var > 0 ? ws.number_lu_x : ws.n
    threads, blocks = gpu_launch_config(linear_n)
    if threads > 0 && linear_n > 0
        # Choose kernel based on to_check flag - no recompilation overhead
        if ws.to_check
            @cuda threads = threads blocks = blocks unified_update_zxw1_kernel_full!(
                Val(false), Val(ws.Q_is_diag),
                ws.dx, rowPtrQ, colValQ, nzValQ,
                ws.w_bar, ws.w, ws.z_bar, ws.x_bar, ws.x_hat, ws.last_x, ws.x, ws.Qw, ws.ATy, ws.c,
                ws.l, ws.u, ws.sigma, fact1_scalar, fact2_scalar, fact1_vec, fact2_vec,
                Halpern_fact1, Halpern_fact2, linear_n)
        else
            @cuda threads = threads blocks = blocks unified_update_zxw1_kernel_partial!(
                Val(false), Val(ws.Q_is_diag),
                ws.dx, rowPtrQ, colValQ, nzValQ,
                ws.w_bar, ws.w, ws.z_bar, ws.x_bar, ws.x_hat, ws.last_x, ws.x, ws.Qw, ws.ATy, ws.c,
                ws.l, ws.u, ws.sigma, fact1_scalar, fact2_scalar, fact1_vec, fact2_vec,
                Halpern_fact1, Halpern_fact2, linear_n)
        end
    end

    if ws.number_SOC_var > 0
        fast_paths = ws.soc_var_fast_paths
        use_q_val = skip_q_work ? Val(false) : Val(true)
        use_partial_soc_projection = !ws.to_check && skip_q_work

        if fast_paths.small_count > 0
            if use_partial_soc_projection
                launch_unified_update_zxw1_SOC_small_partial!(
                    use_q_val, ws, fast_paths.small_starts, fast_paths.small_sizes, fast_paths.small_count,
                    fact1_scalar, fact2_scalar, Halpern_fact1, Halpern_fact2)
            else
                launch_unified_update_zxw1_SOC_small!(
                    use_q_val, ws, fast_paths.small_starts, fast_paths.small_sizes, fast_paths.small_count,
                    fact1_scalar, fact2_scalar, Halpern_fact1, Halpern_fact2)
            end
        end
        if fast_paths.huge_count > 0
            if use_partial_soc_projection
                launch_unified_update_zxw1_SOC_huge_partial!(
                    use_q_val, ws, fast_paths.huge_starts, fast_paths.huge_sizes, fast_paths.huge_count,
                    fact1_scalar, fact2_scalar, Halpern_fact1, Halpern_fact2)
            else
                launch_unified_update_zxw1_SOC_huge!(
                    use_q_val, ws, fast_paths.huge_starts, fast_paths.huge_sizes, fast_paths.huge_count,
                    fact1_scalar, fact2_scalar, Halpern_fact1, Halpern_fact2)
            end
        end
        if fast_paths.large_count > 0
            if use_partial_soc_projection
                launch_unified_update_zxw1_SOC_large_partial!(
                    use_q_val, ws, fast_paths.large_starts, fast_paths.large_sizes, fast_paths.large_count,
                    fact1_scalar, fact2_scalar, Halpern_fact1, Halpern_fact2)
            else
                launch_unified_update_zxw1_SOC_large!(
                    use_q_val, ws, fast_paths.large_starts, fast_paths.large_sizes, fast_paths.large_count,
                    fact1_scalar, fact2_scalar, Halpern_fact1, Halpern_fact2)
            end
        end
    end

    # Step 2: Compute tempv for subsequent use in update_y
    if skip_q_work
        copyto!(ws.tempv, ws.x_hat)
    else
        if isa(qp.Q, CuSparseMatrixCSR{Float64,Int32}) && ws.spmv_Q !== nothing
            Qmap!(ws.spmv_Q.desc_w_bar, ws.spmv_Q.desc_Qw_bar, qp.Q, ws.spmv_Q)
        elseif isa(qp.Q, CuSparseMatrixCSR{Float64,Int32})
            Qmap!(ws.w_bar, ws.Qw_bar, qp.Q, ws.spmv_Q)
        else
            Qmap!(ws.w_bar, ws.Qw_bar, qp.Q)
        end

        tempv_threads, tempv_blocks = gpu_launch_config(ws.n)
        if tempv_threads > 0
            @cuda threads = tempv_threads blocks = tempv_blocks compute_tempv_unified_kernel!(
                Val(false),
                ws.tempv, rowPtrQ, colValQ, nzValQ,
                ws.w_bar, ws.x_hat, ws.Qw, ws.Qw_bar, ws.sigma, ws.n)
        end
    end
end

# Unified wrapper for update_y that handles both custom and cuSPARSE SpMV
# Unified wrapper for update_y that handles both custom and cuSPARSE SpMV
function unified_update_y_gpu!(ws::HPRSOCP_workspace_gpu, Halpern_fact1::Float64,
    Halpern_fact2::Float64)
    fact1 = ws.lambda_max_A * ws.sigma
    fact2 = 1.0 / fact1

    if ws.spmv_A !== nothing
        Amap!(ws.spmv_A.desc_tempv, ws.spmv_A.desc_Ax, ws.A, ws.spmv_A)
    else
        Amap!(ws.tempv, ws.Ax, ws.A, ws.spmv_A)
    end

    linear_m = ws.number_linear_con
    if linear_m > 0
        threads_A, blocks_A = gpu_launch_config(linear_m)
        if threads_A > 0
            # Choose kernel based on to_check flag - no recompilation overhead
            if ws.to_check
                @cuda threads = threads_A blocks = blocks_A unified_update_y_kernel_full!(
                    Val(false),
                    ws.dy, ws.A.rowPtr, ws.A.colVal, ws.A.nzVal,
                    ws.tempv, ws.Ax, ws.y_bar, ws.y, ws.last_y, ws.s, ws.AL, ws.AU,
                    fact1, fact2, Halpern_fact1, Halpern_fact2, linear_m)
            else
                @cuda threads = threads_A blocks = blocks_A unified_update_y_kernel_partial!(
                    Val(false),
                    ws.dy, ws.A.rowPtr, ws.A.colVal, ws.A.nzVal,
                    ws.tempv, ws.Ax, ws.y_bar, ws.y, ws.last_y, ws.s, ws.AL, ws.AU,
                    fact1, fact2, Halpern_fact1, Halpern_fact2, linear_m)
            end
        end
    end

    if ws.number_SOC_con > 0
        fast_paths = ws.soc_con_fast_paths
        soc_con_first = Int32(ws.number_linear_con + 1)
        if fast_paths.small_count > 0
            launch_unified_update_y_SOC_small!(
                ws, soc_con_first, fast_paths.small_starts, fast_paths.small_sizes, fast_paths.small_count,
                fact1, fact2, Halpern_fact1, Halpern_fact2)
        end
        if fast_paths.large_count > 0
            launch_unified_update_y_SOC_large!(
                ws, soc_con_first, fast_paths.large_starts, fast_paths.large_sizes,
                fast_paths.large_count, fact1, fact2, Halpern_fact1, Halpern_fact2)
        end
    end
end

function unified_update_zx_noQ_soc_gpu!(ws::HPRSOCP_workspace_gpu,
    Halpern_fact1::Float64, Halpern_fact2::Float64)
    if ws.m > 0
        if ws.spmv_AT !== nothing
            ATmap!(ws.spmv_AT.desc_y, ws.spmv_AT.desc_ATy, ws.AT, ws.spmv_AT)
        else
            ATmap!(ws.y, ws.ATy, ws.AT, ws.spmv_AT)
        end
    elseif ws.n > 0
        fill!(ws.ATy, 0.0)
    end

    linear_n = ws.number_SOC_var > 0 ? ws.number_lu_x : ws.n
    fused_size3_partial = false
    if ws.number_SOC_var > 0 && !ws.to_check
        fast_paths = ws.soc_var_fast_paths
        if fast_paths.size3_count > 0 && fast_paths.size3_count == fast_paths.small_count
            total_n = linear_n + fast_paths.size3_count
            threads, blocks = gpu_launch_config(total_n)
            if threads > 0 && total_n > 0
                # Fuse the boxed no-Q update with pure size-3 SOC cones to remove one launch.
                @cuda threads = threads blocks = blocks unified_update_zx_noQ_linear_size3_kernel_partial!(
                    Val(false),
                    ws.dx, ws.AT.rowPtr, ws.AT.colVal, ws.AT.nzVal,
                    ws.y, ws.ATy, ws.z_bar, ws.x_bar, ws.x_hat, ws.x, ws.last_x,
                    ws.c, ws.l, ws.u, ws.sigma,
                    Halpern_fact1, Halpern_fact2, fast_paths.size3_starts,
                    linear_n, fast_paths.size3_count)
            end
            fused_size3_partial = true
        else
            threads, blocks = gpu_launch_config(linear_n)
            if threads > 0 && linear_n > 0
                @cuda threads = threads blocks = blocks unified_update_zx_kernel_partial!(
                    Val(false),
                    ws.dx, ws.AT.rowPtr, ws.AT.colVal, ws.AT.nzVal,
                    ws.y, ws.ATy, ws.z_bar, ws.x_bar, ws.x_hat, ws.x, ws.last_x,
                    ws.c, ws.l, ws.u, ws.sigma,
                    Halpern_fact1, Halpern_fact2, linear_n)
            end
        end
    else
        threads, blocks = gpu_launch_config(linear_n)
        if threads > 0 && linear_n > 0
            if ws.to_check
                @cuda threads = threads blocks = blocks unified_update_zx_kernel_full!(
                    Val(false),
                    ws.dx, ws.AT.rowPtr, ws.AT.colVal, ws.AT.nzVal,
                    ws.y, ws.ATy, ws.z_bar, ws.x_bar, ws.x_hat, ws.x, ws.last_x,
                    ws.c, ws.l, ws.u, ws.sigma,
                    Halpern_fact1, Halpern_fact2, linear_n)
            else
                @cuda threads = threads blocks = blocks unified_update_zx_kernel_partial!(
                    Val(false),
                    ws.dx, ws.AT.rowPtr, ws.AT.colVal, ws.AT.nzVal,
                    ws.y, ws.ATy, ws.z_bar, ws.x_bar, ws.x_hat, ws.x, ws.last_x,
                    ws.c, ws.l, ws.u, ws.sigma,
                    Halpern_fact1, Halpern_fact2, linear_n)
            end
        end
    end

    if ws.number_SOC_var > 0
        fast_paths = ws.soc_var_fast_paths
        if fast_paths.small_count > 0 && !fused_size3_partial
            if ws.to_check
                launch_unified_update_zx_noQ_SOC_small!(
                    ws, fast_paths.small_starts, fast_paths.small_sizes, fast_paths.small_count,
                    Halpern_fact1, Halpern_fact2)
            elseif use_soc_var_cooperative_size5(fast_paths)
                launch_unified_update_zx_noQ_SOC_size5_cooperative!(
                    Val(false),
                    ws, fast_paths.size5_starts, fast_paths.size5_count,
                    Halpern_fact1, Halpern_fact2)
            else
                launch_unified_update_zx_noQ_SOC_small_partial!(
                    Val(false),
                    ws, fast_paths.small_starts, fast_paths.small_sizes, fast_paths.small_count,
                    Halpern_fact1, Halpern_fact2)
            end
        end
        if fast_paths.huge_count > 0
            if ws.to_check
                launch_unified_update_zx_noQ_SOC_huge!(
                    ws, fast_paths.huge_starts, fast_paths.huge_sizes, fast_paths.huge_count,
                    Halpern_fact1, Halpern_fact2)
            else
                launch_unified_update_zx_noQ_SOC_huge_partial!(
                    Val(false),
                    ws, fast_paths.huge_starts, fast_paths.huge_sizes, fast_paths.huge_count,
                    Halpern_fact1, Halpern_fact2)
            end
        end
        if fast_paths.large_count > 0
            if ws.to_check
                launch_unified_update_zx_noQ_SOC_large!(
                    ws, fast_paths.large_starts, fast_paths.large_sizes, fast_paths.large_count,
                    Halpern_fact1, Halpern_fact2)
            elseif use_soc_var_cooperative_large_partial(fast_paths)
                launch_unified_update_zx_noQ_SOC_large_cooperative_partial!(
                    Val(false),
                    ws, fast_paths.large_starts, fast_paths.large_sizes, fast_paths.large_count,
                    Halpern_fact1, Halpern_fact2)
            else
                launch_unified_update_zx_noQ_SOC_large_partial!(
                    Val(false),
                    ws, fast_paths.large_starts, fast_paths.large_sizes, fast_paths.large_count,
                    Halpern_fact1, Halpern_fact2)
            end
        end
    end
end

@inline use_soc_var_cooperative_size5(fast_paths) =
    fast_paths.size5_count == fast_paths.small_count && fast_paths.size5_count >= 4096

@inline use_soc_var_cooperative_large_partial(fast_paths) =
    fast_paths.large_count <= 8 && fast_paths.large_max_size >= 96

@inline use_soc_con_cooperative_size3(fast_paths) =
    fast_paths.size3_count == fast_paths.small_count && fast_paths.size3_count >= 2048

@inline use_soc_con_cooperative_size4(fast_paths) =
    fast_paths.size4_count == fast_paths.small_count && fast_paths.size4_count >= 2048

@inline use_soc_con_cooperative_size5(fast_paths) =
    fast_paths.size5_count == fast_paths.small_count && fast_paths.size5_count >= 2048

function unified_update_y_noQ_soc_gpu!(ws::HPRSOCP_workspace_gpu, Halpern_fact1::Float64,
    Halpern_fact2::Float64)
    fact1 = ws.lambda_max_A * ws.sigma
    fact2 = 1.0 / fact1

    if ws.spmv_A !== nothing
        Amap!(ws.spmv_A.desc_x_hat, ws.spmv_A.desc_Ax, ws.A, ws.spmv_A)
    else
        Amap!(ws.x_hat, ws.Ax, ws.A, ws.spmv_A)
    end

    linear_m = ws.number_linear_con
    if linear_m > 0
        threads_A, blocks_A = gpu_launch_config(linear_m)
        if threads_A > 0
            if ws.to_check
                @cuda threads = threads_A blocks = blocks_A unified_update_y_kernel_full!(
                    Val(false),
                    ws.dy, ws.A.rowPtr, ws.A.colVal, ws.A.nzVal,
                    ws.x_hat, ws.Ax, ws.y_bar, ws.y, ws.last_y, ws.s, ws.AL, ws.AU,
                    fact1, fact2, Halpern_fact1, Halpern_fact2, linear_m)
            else
                @cuda threads = threads_A blocks = blocks_A unified_update_y_kernel_partial!(
                    Val(false),
                    ws.dy, ws.A.rowPtr, ws.A.colVal, ws.A.nzVal,
                    ws.x_hat, ws.Ax, ws.y_bar, ws.y, ws.last_y, ws.s, ws.AL, ws.AU,
                    fact1, fact2, Halpern_fact1, Halpern_fact2, linear_m)
            end
        end
    end

    if ws.number_SOC_con > 0
        fast_paths = ws.soc_con_fast_paths
        soc_con_first = Int32(ws.number_linear_con + 1)
        if fast_paths.small_count > 0
            if !ws.to_check && use_soc_con_cooperative_size3(fast_paths)
                launch_unified_update_y_SOC_size3_cooperative!(
                    ws, soc_con_first, fast_paths.size3_starts, fast_paths.size3_count,
                    fact1, fact2, Halpern_fact1, Halpern_fact2)
            elseif !ws.to_check && use_soc_con_cooperative_size4(fast_paths)
                launch_unified_update_y_SOC_size4_cooperative!(
                    ws, soc_con_first, fast_paths.size4_starts, fast_paths.size4_count,
                    fact1, fact2, Halpern_fact1, Halpern_fact2)
            elseif !ws.to_check && use_soc_con_cooperative_size5(fast_paths)
                launch_unified_update_y_SOC_size5_cooperative!(
                    ws, soc_con_first, fast_paths.size5_starts, fast_paths.size5_count,
                    fact1, fact2, Halpern_fact1, Halpern_fact2)
            else
                launch_unified_update_y_SOC_small!(
                    ws, soc_con_first, fast_paths.small_starts, fast_paths.small_sizes, fast_paths.small_count,
                    fact1, fact2, Halpern_fact1, Halpern_fact2)
            end
        end
        if fast_paths.large_count > 0
            launch_unified_update_y_SOC_large!(
                ws, soc_con_first, fast_paths.large_starts, fast_paths.large_sizes,
                fast_paths.large_count, fact1, fact2, Halpern_fact1, Halpern_fact2)
        end
    end
end

# Unified wrapper for update_w2 that handles both custom and cuSPARSE SpMV, and regular/diagonal Q
# Unified wrapper for update_w2 that handles both custom and cuSPARSE SpMV, and regular/diagonal Q
function unified_update_w2_gpu!(ws::HPRSOCP_workspace_gpu, Halpern_fact1::Float64,
    Halpern_fact2::Float64)
    # Prepare scalar factor for regular Q
    fact_scalar = ws.sigma / (1.0 + ws.sigma * ws.lambda_max_Q)

    # Pre-computed vector factor for diagonal Q (ignored otherwise)
    fact_vec = ws.fact

    if ws.spmv_AT !== nothing
        ATmap!(ws.spmv_AT.desc_y_bar, ws.spmv_AT.desc_ATy_bar, ws.AT, ws.spmv_AT)
    else
        ATmap!(ws.y_bar, ws.ATy_bar, ws.AT, ws.spmv_AT)
    end

    threads, blocks = gpu_launch_config(ws.n)
    if threads > 0
        # Choose kernel based on to_check flag - no recompilation overhead
        if ws.to_check
            @cuda threads = threads blocks = blocks unified_update_w2_kernel_full!(
                Val(false), Val(ws.Q_is_diag),
                ws.dw, ws.ATdy, ws.AT.rowPtr, ws.AT.colVal, ws.AT.nzVal,
                ws.y_bar, ws.ATy_bar, ws.w, ws.w_bar, ws.last_w, ws.last_ATy, ws.ATy,
                fact_scalar, fact_vec, Halpern_fact1, Halpern_fact2, ws.n)
        else
            @cuda threads = threads blocks = blocks unified_update_w2_kernel_partial!(
                Val(false), Val(ws.Q_is_diag),
                ws.dw, ws.ATdy, ws.AT.rowPtr, ws.AT.colVal, ws.AT.nzVal,
                ws.y_bar, ws.ATy_bar, ws.w, ws.w_bar, ws.last_w, ws.last_ATy, ws.ATy,
                fact_scalar, fact_vec, Halpern_fact1, Halpern_fact2, ws.n)
        end
    end
end

## Unified kernels for empty Q case (Q.nzVal has length 0 - linear program)

# Unified update_zx kernel - handles both custom inline AT*y and cuSPARSE
# Full version: computes all intermediate values
CUDA.@fastmath @inline function unified_update_zx_kernel_full!(::Val{UseCustom},
    dx::CuDeviceVector{Float64},
    rowPtrAT::CuDeviceVector{Int32},
    colValAT::CuDeviceVector{Int32},
    nzValAT::CuDeviceVector{Float64},
    y::CuDeviceVector{Float64},
    ATy::CuDeviceVector{Float64},
    z_bar::CuDeviceVector{Float64},
    x_bar::CuDeviceVector{Float64},
    x_hat::CuDeviceVector{Float64},
    x::CuDeviceVector{Float64},
    last_x::CuDeviceVector{Float64},
    c::CuDeviceVector{Float64},
    l::CuDeviceVector{Float64},
    u::CuDeviceVector{Float64},
    sigma::Float64,
    Halpern_fact1::Float64,
    Halpern_fact2::Float64,
    n::Int) where {UseCustom}

    i = threadIdx().x + (blockDim().x * (blockIdx().x - 1))
    @inbounds if i <= n
        ATy_val = if UseCustom
            startAT = rowPtrAT[i]
            stopAT = rowPtrAT[i+1] - 1
            acc = 0.0
            @inbounds for k in startAT:stopAT
                acc += nzValAT[k] * y[colValAT[k]]
            end
            acc
        else
            ATy[i]
        end

        x_i = x[i]
        last_x_i = last_x[i]
        l_i = l[i]
        u_i = u[i]
        c_i = c[i]

        tmp = ATy_val - c_i
        z_raw = x_i + sigma * tmp
        x_bar_i = min(max(z_raw, l_i), u_i)
        x_hat_i = 2.0 * x_bar_i - x_i
        dx_val = x_bar_i - x_i
        z_bar_i = (x_bar_i - z_raw) / sigma
        x_new = Halpern_fact1 * last_x_i + Halpern_fact2 * x_hat_i

        dx[i] = dx_val
        x_bar[i] = x_bar_i
        z_bar[i] = z_bar_i
        x[i] = x_new
        x_hat[i] = x_hat_i
    end
    return
end

# Partial version: skips intermediate writes
CUDA.@fastmath @inline function unified_update_zx_kernel_partial!(::Val{UseCustom},
    dx::CuDeviceVector{Float64},
    rowPtrAT::CuDeviceVector{Int32},
    colValAT::CuDeviceVector{Int32},
    nzValAT::CuDeviceVector{Float64},
    y::CuDeviceVector{Float64},
    ATy::CuDeviceVector{Float64},
    z_bar::CuDeviceVector{Float64},
    x_bar::CuDeviceVector{Float64},
    x_hat::CuDeviceVector{Float64},
    x::CuDeviceVector{Float64},
    last_x::CuDeviceVector{Float64},
    c::CuDeviceVector{Float64},
    l::CuDeviceVector{Float64},
    u::CuDeviceVector{Float64},
    sigma::Float64,
    Halpern_fact1::Float64,
    Halpern_fact2::Float64,
    n::Int) where {UseCustom}

    i = threadIdx().x + (blockDim().x * (blockIdx().x - 1))
    @inbounds if i <= n
        ATy_val = if UseCustom
            startAT = rowPtrAT[i]
            stopAT = rowPtrAT[i+1] - 1
            acc = 0.0
            @inbounds for k in startAT:stopAT
                acc += nzValAT[k] * y[colValAT[k]]
            end
            acc
        else
            ATy[i]
        end

        x_i = x[i]
        last_x_i = last_x[i]
        l_i = l[i]
        u_i = u[i]
        c_i = c[i]

        tmp = ATy_val - c_i
        z_raw = x_i + sigma * tmp
        x_bar_i = min(max(z_raw, l_i), u_i)
        x_hat_i = 2.0 * x_bar_i - x_i
        x_new = Halpern_fact1 * last_x_i + Halpern_fact2 * x_hat_i

        x[i] = x_new
        x_hat[i] = x_hat_i
    end
    return
end

CUDA.@fastmath @inline function unified_update_zx_noQ_linear_size3_kernel_partial!(::Val{UseCustom},
    dx::CuDeviceVector{Float64},
    rowPtrAT::CuDeviceVector{Int32},
    colValAT::CuDeviceVector{Int32},
    nzValAT::CuDeviceVector{Float64},
    y::CuDeviceVector{Float64},
    ATy::CuDeviceVector{Float64},
    z_bar::CuDeviceVector{Float64},
    x_bar::CuDeviceVector{Float64},
    x_hat::CuDeviceVector{Float64},
    x::CuDeviceVector{Float64},
    last_x::CuDeviceVector{Float64},
    c::CuDeviceVector{Float64},
    l::CuDeviceVector{Float64},
    u::CuDeviceVector{Float64},
    sigma::Float64,
    Halpern_fact1::Float64,
    Halpern_fact2::Float64,
    size3_starts::CuDeviceVector{Int32},
    linear_n::Int,
    size3_count::Int) where {UseCustom}

    i = threadIdx().x + (blockDim().x * (blockIdx().x - 1))
    total_n = linear_n + size3_count

    @inbounds if i <= total_n
        if i <= linear_n
            ATy_val = if UseCustom
                startAT = rowPtrAT[i]
                stopAT = rowPtrAT[i+1] - 1
                acc = 0.0
                @inbounds for k in startAT:stopAT
                    acc += nzValAT[k] * y[colValAT[k]]
                end
                acc
            else
                ATy[i]
            end

            x_i = x[i]
            last_x_i = last_x[i]
            l_i = l[i]
            u_i = u[i]
            c_i = c[i]

            tmp = ATy_val - c_i
            z_raw = x_i + sigma * tmp
            x_bar_i = min(max(z_raw, l_i), u_i)
            x_hat_i = 2.0 * x_bar_i - x_i
            x_new = Halpern_fact1 * last_x_i + Halpern_fact2 * x_hat_i

            x[i] = x_new
            x_hat[i] = x_hat_i
        else
            cone_idx = i - linear_n
            start_idx = Int(size3_starts[cone_idx])
            j1 = start_idx + 1
            j2 = start_idx + 2

            t_raw = UseCustom ?
                _soc_raw_update_noQ_custom(start_idx, x, rowPtrAT, colValAT, nzValAT, y, c, sigma) :
                _soc_raw_update_noQ(start_idx, x, ATy, c, sigma)
            z_raw_1 = UseCustom ?
                _soc_raw_update_noQ_custom(j1, x, rowPtrAT, colValAT, nzValAT, y, c, sigma) :
                _soc_raw_update_noQ(j1, x, ATy, c, sigma)
            z_raw_2 = UseCustom ?
                _soc_raw_update_noQ_custom(j2, x, rowPtrAT, colValAT, nzValAT, y, c, sigma) :
                _soc_raw_update_noQ(j2, x, ATy, c, sigma)
            norm_s = sqrt(z_raw_1^2 + z_raw_2^2)

            if norm_s <= -t_raw
                z_bar[start_idx] = -t_raw / sigma
                z_bar[j1] = -z_raw_1 / sigma
                z_bar[j2] = -z_raw_2 / sigma
                _write_soc_var_projection_noQ_partial!(
                    start_idx, 0.0, x_hat, last_x, x, Halpern_fact1, Halpern_fact2
                )
                _write_soc_var_projection_noQ_partial!(
                    j1, 0.0, x_hat, last_x, x, Halpern_fact1, Halpern_fact2
                )
                _write_soc_var_projection_noQ_partial!(
                    j2, 0.0, x_hat, last_x, x, Halpern_fact1, Halpern_fact2
                )
            elseif norm_s <= t_raw
                z_bar[start_idx] = 0.0
                z_bar[j1] = 0.0
                z_bar[j2] = 0.0
                _write_soc_var_projection_noQ_partial!(
                    start_idx, t_raw, x_hat, last_x, x, Halpern_fact1, Halpern_fact2
                )
                _write_soc_var_projection_noQ_partial!(
                    j1, z_raw_1, x_hat, last_x, x, Halpern_fact1, Halpern_fact2
                )
                _write_soc_var_projection_noQ_partial!(
                    j2, z_raw_2, x_hat, last_x, x, Halpern_fact1, Halpern_fact2
                )
            else
                proj_t = (norm_s + t_raw) / 2.0
                alpha = proj_t / norm_s
                proj_1 = alpha * z_raw_1
                proj_2 = alpha * z_raw_2

                z_bar[start_idx] = (proj_t - t_raw) / sigma
                z_bar[j1] = (proj_1 - z_raw_1) / sigma
                z_bar[j2] = (proj_2 - z_raw_2) / sigma
                _write_soc_var_projection_noQ_partial!(
                    start_idx, proj_t, x_hat, last_x, x, Halpern_fact1, Halpern_fact2
                )
                _write_soc_var_projection_noQ_partial!(
                    j1, proj_1, x_hat, last_x, x, Halpern_fact1, Halpern_fact2
                )
                _write_soc_var_projection_noQ_partial!(
                    j2, proj_2, x_hat, last_x, x, Halpern_fact1, Halpern_fact2
                )
            end
        end
    end
    return
end

# Unified wrapper for update_zx
function unified_update_zx_gpu!(ws::HPRSOCP_workspace_gpu, Halpern_fact1::Float64, Halpern_fact2::Float64)
    if ws.spmv_AT !== nothing
        ATmap!(ws.spmv_AT.desc_y, ws.spmv_AT.desc_ATy, ws.AT, ws.spmv_AT)
    else
        ATmap!(ws.y, ws.ATy, ws.AT, ws.spmv_AT)
    end

    threads, blocks = gpu_launch_config(ws.n)
    if threads > 0
        # Choose kernel based on to_check flag - no recompilation overhead
        if ws.to_check
            @cuda threads = threads blocks = blocks unified_update_zx_kernel_full!(
                Val(false),
                ws.dx, ws.AT.rowPtr, ws.AT.colVal, ws.AT.nzVal,
                ws.y, ws.ATy, ws.z_bar, ws.x_bar, ws.x_hat, ws.x, ws.last_x,
                ws.c, ws.l, ws.u, ws.sigma,
                Halpern_fact1, Halpern_fact2, ws.n)
        else
            @cuda threads = threads blocks = blocks unified_update_zx_kernel_partial!(
                Val(false),
                ws.dx, ws.AT.rowPtr, ws.AT.colVal, ws.AT.nzVal,
                ws.y, ws.ATy, ws.z_bar, ws.x_bar, ws.x_hat, ws.x, ws.last_x,
                ws.c, ws.l, ws.u, ws.sigma,
                Halpern_fact1, Halpern_fact2, ws.n)
        end
    end
end

# Unified update_y_noQ - uses x_hat directly instead of tempv
function unified_update_y_noQ_gpu!(ws::HPRSOCP_workspace_gpu, Halpern_fact1::Float64, Halpern_fact2::Float64)
    fact1 = ws.lambda_max_A * ws.sigma
    fact2 = 1.0 / fact1

    if ws.spmv_A !== nothing
        Amap!(ws.spmv_A.desc_x_hat, ws.spmv_A.desc_Ax, ws.A, ws.spmv_A)
    else
        Amap!(ws.x_hat, ws.Ax, ws.A, ws.spmv_A)
    end

    if ws.m > 0
        threads_A, blocks_A = gpu_launch_config(ws.m)
        if threads_A > 0
            # Reuse unified_update_y_kernel but pass x_hat instead of tempv
            # Choose kernel based on to_check flag - no recompilation overhead
            if ws.to_check
                @cuda threads = threads_A blocks = blocks_A unified_update_y_kernel_full!(
                    Val(false),
                    ws.dy, ws.A.rowPtr, ws.A.colVal, ws.A.nzVal,
                    ws.x_hat, ws.Ax, ws.y_bar, ws.y, ws.last_y, ws.s, ws.AL, ws.AU,
                    fact1, fact2, Halpern_fact1, Halpern_fact2, ws.m)
            else
                @cuda threads = threads_A blocks = blocks_A unified_update_y_kernel_partial!(
                    Val(false),
                    ws.dy, ws.A.rowPtr, ws.A.colVal, ws.A.nzVal,
                    ws.x_hat, ws.Ax, ws.y_bar, ws.y, ws.last_y, ws.s, ws.AL, ws.AU,
                    fact1, fact2, Halpern_fact1, Halpern_fact2, ws.m)
            end
        end
    end
end


CUDA.@fastmath @inline function cust_compute_r2_kernel!(rowPtrQ::CuDeviceVector{Int32}, colValQ::CuDeviceVector{Int32}, nzValQ::CuDeviceVector{Float64},
    w_bar::CuDeviceVector{Float64}, Qw::CuDeviceVector{Float64},
    sigma::Float64, x_hat::CuDeviceVector{Float64},
    tempv::CuDeviceVector{Float64}, n::Int)
    i = threadIdx().x + (blockDim().x * (blockIdx().x - 1))
    @inbounds if i <= n
        startQ = rowPtrQ[i]
        stopQ = rowPtrQ[i+1] - 1
        qr1 = 0.0
        @inbounds for k in startQ:stopQ
            qr1 += nzValQ[k] * w_bar[colValQ[k]]
        end
        tempv[i] = x_hat[i] + sigma * (Qw[i] - qr1)
    end
    return
end


## kernels used to update sigma

@inline function f_dev(x, a, b, c, d)
    return a * x + b / x + c * x^2 / (1 + d * x)
end

function golden(
    a_p::Float64, b_p::Float64, c_p::Float64, d_p::Float64;
    lo::Float64=eps(Float64),
    hi::Float64=1e12,
    tol::Float64=1e-12,
    maxiter::Int=200
)
    # golden ratio constant
    φ = (sqrt(5.0) - 1.0) / 2.0
    a = lo
    b = hi
    c = b - φ * (b - a)
    d = a + φ * (b - a)
    f_c = f_dev(c, a_p, b_p, c_p, d_p)
    f_d = f_dev(d, a_p, b_p, c_p, d_p)

    for i in 1:maxiter
        if f_d < f_c
            a, c, f_c = c, d, f_d
            d = a + φ * (b - a)
            f_d = f_dev(d, a_p, b_p, c_p, d_p)
        else
            b, d, f_d = d, c, f_c
            c = b - φ * (b - a)
            f_c = f_dev(c, a_p, b_p, c_p, d_p)
        end
        if (b - a) < tol
            break
        end
    end

    x_sol = 0.5 * (a + b)
    return x_sol
end



# Golden-section search for minimizing 
# f(x) = a*x + b/x + x^2 * dot(c, (I + x*Q) \ d)
# GPU‑enabled golden‐section search for 
# f(x) = a*x + b/x + x^2 * dot(c, d ./ (1 + x*Q))
function golden_Q_diag(a::Float64, b::Float64, Q::CuArray{Float64}, c::CuArray{Float64}, d::CuArray{Float64}, tempv::CuArray{Float64};
    lo::Float64=eps(Float64),
    hi::Float64=1e12,
    tol::Float64=1e-12,
    maxiter::Int=200)
    φ = (sqrt(5.0) - 1.0) / 2.0

    # Objective using GPU operations, reusing tempv
    function f_gpu(x)
        @. tempv = d / (1.0 + x * Q)
        return a * x + b / x + x^2 * dot(c, tempv)
    end

    # Initialize bracket
    x1 = hi - φ * (hi - lo)
    x2 = lo + φ * (hi - lo)
    f1 = f_gpu(x1)
    f2 = f_gpu(x2)

    # Main golden‐section loop
    iter = 0
    while abs(hi - lo) > tol * max(1.0, abs(lo)) && iter < maxiter
        if f1 > f2
            lo = x1
            x1, f1 = x2, f2
            x2 = lo + φ * (hi - lo)
            f2 = f_gpu(x2)
        else
            hi = x2
            x2, f2 = x1, f1
            x1 = hi - φ * (hi - lo)
            f1 = f_gpu(x1)
        end
        iter += 1
    end

    return (lo + hi) / 2
end

#############################
# CUDA kernel to update all four factors in one pass
#############################
function update_Q_factors_kernel!(
    fact2::CuDeviceVector{Float64},
    fact::CuDeviceVector{Float64},
    fact1::CuDeviceVector{Float64},
    fact_M::CuDeviceVector{Float64},
    diag_Q::CuDeviceVector{Float64},
    sigma::Float64,
    s2::Float64,
    N::Int
)
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    if i <= N
        v = diag_Q[i]
        t2 = 1.0 / (1.0 + sigma * v)
        fact2[i] = t2
        fact[i] = sigma * t2
        fact1[i] = sigma * v * t2
        fact_M[i] = s2 * t2
    end
    return
end

#############################
# High-level wrapper to launch the above kernel
#############################
function update_Q_factors_gpu!(
    fact2::CuVector{Float64},
    fact::CuVector{Float64},
    fact1::CuVector{Float64},
    fact_M::CuVector{Float64},
    diag_Q::CuVector{Float64},
    sigma::Float64
)
    N = length(diag_Q)
    s2 = sigma * sigma
    threads = 256
    blocks = cld(N, threads)
    @cuda threads = threads blocks = blocks update_Q_factors_kernel!(
        fact2, fact, fact1, fact_M,
        diag_Q, sigma, s2, N
    )
    return
end

#############################
# CPU version of golden_Q_diag
#############################
function golden_Q_diag_cpu(a::Float64, b::Float64, Q::Vector{Float64}, c::Vector{Float64}, d::Vector{Float64}, tempv::Vector{Float64};
    lo::Float64=eps(Float64),
    hi::Float64=1e12,
    tol::Float64=1e-12,
    maxiter::Int=200)
    φ = (sqrt(5.0) - 1.0) / 2.0

    # Objective using CPU operations, reusing tempv
    function f_cpu(x)
        @. tempv = d / (1.0 + x * Q)
        return a * x + b / x + x^2 * dot(c, tempv)
    end

    # Initialize bracket
    x1 = hi - φ * (hi - lo)
    x2 = lo + φ * (hi - lo)
    f1 = f_cpu(x1)
    f2 = f_cpu(x2)

    # Main golden‐section loop
    iter = 0
    while abs(hi - lo) > tol * max(1.0, abs(lo)) && iter < maxiter
        if f1 > f2
            lo = x1
            x1, f1 = x2, f2
            x2 = lo + φ * (hi - lo)
            f2 = f_cpu(x2)
        else
            hi = x2
            x2, f2 = x1, f1
            x1 = hi - φ * (hi - lo)
            f1 = f_cpu(x1)
        end
        iter += 1
    end

    return (lo + hi) / 2
end

#############################
# CPU version of update_Q_factors
#############################
function update_Q_factors_cpu!(
    fact2::Vector{Float64},
    fact::Vector{Float64},
    fact1::Vector{Float64},
    fact_M::Vector{Float64},
    diag_Q::Vector{Float64},
    sigma::Float64
)
    N = length(diag_Q)
    s2 = sigma * sigma
    for i in 1:N
        v = diag_Q[i]
        t2 = 1.0 / (1.0 + sigma * v)
        fact2[i] = t2
        fact[i] = sigma * t2
        fact1[i] = sigma * v * t2
        fact_M[i] = s2 * t2
    end
    return
end

## kernels to compute residuals

CUDA.@fastmath @inline function compute_Rd_kernel!(ATy::CuDeviceVector{Float64},
    z::CuDeviceVector{Float64},
    c::CuDeviceVector{Float64},
    Qx::CuDeviceVector{Float64},
    Rd::CuDeviceVector{Float64},
    col_norm::CuDeviceVector{Float64},
    c_scale::Float64,
    n::Int)
    i = threadIdx().x + (blockDim().x * (blockIdx().x - 0x1))
    @inbounds if i <= n
        qx_i = Qx[i]
        c_i = c[i]
        atyi = ATy[i]
        z_i = z[i]
        qx_org = _scaled_dual_component_to_original(qx_i, col_norm[i], c_scale)
        c_org = _scaled_dual_component_to_original(c_i, col_norm[i], c_scale)
        atyi_org = _scaled_dual_component_to_original(atyi, col_norm[i], c_scale)
        z_org = _scaled_dual_component_to_original(z_i, col_norm[i], c_scale)
        rd_i = qx_org + c_org - atyi_org - z_org
        Rd[i] = rd_i
        Qx[i] = qx_org
        ATy[i] = atyi_org
    end
    return
end

CUDA.@fastmath @inline function compute_Rd_noQ_kernel!(ATy::CuDeviceVector{Float64},
    z::CuDeviceVector{Float64},
    c::CuDeviceVector{Float64},
    Rd::CuDeviceVector{Float64},
    col_norm::CuDeviceVector{Float64},
    c_scale::Float64,
    n::Int)
    i = threadIdx().x + (blockDim().x * (blockIdx().x - 0x1))
    @inbounds if i <= n
        c_i = c[i]
        atyi = ATy[i]
        z_i = z[i]
        c_org = _scaled_dual_component_to_original(c_i, col_norm[i], c_scale)
        atyi_org = _scaled_dual_component_to_original(atyi, col_norm[i], c_scale)
        z_org = _scaled_dual_component_to_original(z_i, col_norm[i], c_scale)
        Rd[i] = c_org - atyi_org - z_org
        ATy[i] = atyi_org
    end
    return
end

function compute_Rd_gpu!(ws::HPRSOCP_workspace_gpu, sc::Scaling_info_gpu)
    # Call unified version
    compute_Rd!(ws, sc)
end

function compute_Rd_gpu!(ws::HPRSOCP_workspace_gpu, qp::QP_info_gpu, sc::Scaling_info_gpu)
    compute_Rd!(ws, qp, sc)
end

CUDA.@fastmath @inline function compute_Rp_kernel!(Rp::CuDeviceVector{Float64},
    AL::CuDeviceVector{Float64},
    AU::CuDeviceVector{Float64},
    Ax::CuDeviceVector{Float64},
    row_norm::CuDeviceVector{Float64},
    b_scale::Float64,
    m::Int)
    i = threadIdx().x + (blockDim().x * (blockIdx().x - 0x1))
    @inbounds if i <= m
        ax_org = _scaled_primal_component_to_original(Ax[i], row_norm[i], b_scale)
        AL_org = _scaled_primal_component_to_original(AL[i], row_norm[i], b_scale)
        AU_org = _scaled_primal_component_to_original(AU[i], row_norm[i], b_scale)
        ax_proj = min(max(ax_org, AL_org), AU_org)
        Rp[i] = ax_proj - ax_org
        Ax[i] = ax_org
    end
    return
end

CUDA.@fastmath @inline function compute_Rp_SOC_kernel!(Rp::CuDeviceVector{Float64},
    soc_rhs::CuDeviceVector{Float64},
    Ax::CuDeviceVector{Float64},
    SOC_con_idx::CuDeviceVector{Int},
    row_norm::CuDeviceVector{Float64},
    b_scale::Float64,
    number_SOC_con::Int)
    i = threadIdx().x + (blockDim().x * (blockIdx().x - 0x1))
    @inbounds if i <= number_SOC_con
        start_idx = SOC_con_idx[i]
        end_idx = SOC_con_idx[i+1] - 1
        offset = start_idx - SOC_con_idx[1] + 1
        rhs_t = _scaled_primal_component_to_original(soc_rhs[offset], row_norm[start_idx], b_scale)
        t = Ax[start_idx] - rhs_t
        norm_s = 0.0
        Rp[start_idx] = t
        for j in (start_idx + 1):end_idx
            rhs_j = _scaled_primal_component_to_original(
                soc_rhs[offset + (j - start_idx)],
                row_norm[j],
                b_scale,
            )
            s_j = Ax[j] - rhs_j
            Rp[j] = s_j
            norm_s += s_j^2
        end
        norm_s = sqrt(norm_s)

        if norm_s <= -t
            # leave Rp as-is
        elseif norm_s <= t
            for j in start_idx:end_idx
                Rp[j] = 0.0
            end
        else
            fact = (1 + t / norm_s) / 2
            Rp[start_idx] = (norm_s + t) / 2 - t
            for j in (start_idx + 1):end_idx
                Rp[j] = fact * Rp[j] - Rp[j]
            end
        end
    end
    return
end

function compute_Rp_gpu!(ws::HPRSOCP_workspace_gpu, sc::Scaling_info_gpu)
    # Call unified version
    compute_Rp!(ws, sc)
end

function compute_Rp_gpu!(ws::HPRSOCP_workspace_gpu, sc::Scaling_info_gpu, qp::QP_info_gpu)
    compute_Rp!(ws, sc)
end

CUDA.@fastmath @inline function compute_err_lu_kernel!(dx::CuDeviceVector{Float64},
    x::CuDeviceVector{Float64},
    l::CuDeviceVector{Float64},
    u::CuDeviceVector{Float64},
    col_norm::CuDeviceVector{Float64},
    b_scale::Float64,
    n::Int)
    i = threadIdx().x + (blockDim().x * (blockIdx().x - 1))
    @inbounds if i <= n
        x_i = x[i]
        l_i = l[i]
        u_i = u[i]
        lower_violation = max(l_i - x_i, 0.0)
        upper_violation = max(x_i - u_i, 0.0)
        # Original ternary: (x_i < l_i) ? (l_i - x_i) : ((x_i > u_i) ? (x_i - u_i) : 0.0)
        corr = lower_violation + upper_violation
        dx[i] = corr * (b_scale / col_norm[i])
    end
    return
end

CUDA.@fastmath @inline function compute_err_soc_kernel!(dx::CuDeviceVector{Float64},
    x::CuDeviceVector{Float64},
    SOC_var_idx::CuDeviceVector{Int},
    col_norm::CuDeviceVector{Float64},
    b_scale::Float64,
    number_SOC_var::Int)
    i = threadIdx().x + (blockDim().x * (blockIdx().x - 0x1))
    @inbounds if i <= number_SOC_var
        start_idx = SOC_var_idx[i]
        end_idx = SOC_var_idx[i+1] - 1
        t = x[start_idx]
        norm_s = 0.0
        dx[start_idx] = t
        for j in (start_idx + 1):end_idx
            dx[j] = x[j]
            norm_s += x[j]^2
        end
        norm_s = sqrt(norm_s)

        if norm_s <= -t
            # leave dx as-is
        elseif norm_s <= t
            for j in start_idx:end_idx
                dx[j] = 0.0
            end
        else
            fact = (1 + t / norm_s) / 2
            dx[start_idx] = (norm_s + t) / 2 - t
            for j in (start_idx + 1):end_idx
                dx[j] = fact * dx[j] - dx[j]
            end
        end

        for j in start_idx:end_idx
            dx[j] *= b_scale / col_norm[j]
        end
    end
    return
end

CUDA.@fastmath @inline function compute_soc_dual_support_violation_kernel!(
    block_violation::CuDeviceVector{Float64},
    y::CuDeviceVector{Float64},
    SOC_con_idx::CuDeviceVector{I},
    number_SOC_con::Int,
) where {I<:Integer}
    i = threadIdx().x + (blockDim().x * (blockIdx().x - 0x1))
    @inbounds if i <= number_SOC_con
        start_idx = SOC_con_idx[i]
        end_idx = SOC_con_idx[i + 1] - 1
        t = y[start_idx]
        norm_s_sq = 0.0
        for j in (start_idx + 1):end_idx
            norm_s_sq += y[j]^2
        end
        block_violation[i] = max(0.0, sqrt(norm_s_sq) - t)
    end
    return
end

CUDA.@fastmath @inline function axpby_kernel!(a::Float64, x::CuDeviceVector{Float64},
    b::Float64, y::CuDeviceVector{Float64},
    z::CuDeviceVector{Float64}, n::Int)
    i = threadIdx().x + (blockDim().x * (blockIdx().x - 0x1))
    @inbounds if i <= n
        z[i] = muladd(a, x[i], b * y[i])
    end
    return
end

function axpby_gpu!(a::Float64, x::CuArray{Float64},
    b::Float64, y::CuArray{Float64},
    z::CuArray{Float64}, n::Int)
    threads, blocks = gpu_launch_config(n)
    if threads > 0
        @cuda threads = threads blocks = blocks axpby_kernel!(a, x, b, y, z, n)
    end
end

# GPU kernels for scaling operations

# Kernel to compute row-wise maximum of absolute values for CSR matrix
function compute_row_max_abs_kernel!(rowPtr::CuDeviceVector{Int32},
    nzVal::CuDeviceVector{Float64},
    row_norm::CuDeviceVector{Float64},
    m::Int)
    i = threadIdx().x + (blockDim().x * (blockIdx().x - 1))
    if i <= m
        @inbounds begin
            start_idx = rowPtr[i]
            end_idx = rowPtr[i+1] - 1
            max_val = 0.0
            for k in start_idx:end_idx
                val = abs(nzVal[k])
                max_val = max(max_val, val)
            end
            row_norm[i] = clamp(max_val > 0.0 ? sqrt(max_val) : 1.0, 1e-4, 1e4)
        end
    end
    return
end

# Kernel to compute column-wise maximum of absolute values for CSR matrix (operates on AT in CSR format)
function compute_col_max_abs_kernel!(rowPtr::CuDeviceVector{Int32},
    nzVal::CuDeviceVector{Float64},
    col_norm::CuDeviceVector{Float64},
    n::Int)
    i = threadIdx().x + (blockDim().x * (blockIdx().x - 1))
    if i <= n
        @inbounds begin
            start_idx = rowPtr[i]
            end_idx = rowPtr[i+1] - 1
            max_val = 0.0
            for k in start_idx:end_idx
                val = abs(nzVal[k])
                max_val = max(max_val, val)
            end
            col_norm[i] = clamp(max_val > 0.0 ? sqrt(max_val) : 1.0, 1e-4, 1e4)
        end
    end
    return
end

# Kernel to compute row-wise sum of absolute values for CSR matrix
function compute_row_sum_abs_kernel!(rowPtr::CuDeviceVector{Int32},
    nzVal::CuDeviceVector{Float64},
    row_norm::CuDeviceVector{Float64},
    m::Int)
    i = threadIdx().x + (blockDim().x * (blockIdx().x - 1))
    if i <= m
        @inbounds begin
            start_idx = rowPtr[i]
            end_idx = rowPtr[i+1] - 1
            sum_val = 0.0
            for k in start_idx:end_idx
                sum_val += abs(nzVal[k])
            end
            row_norm[i] = clamp(sum_val > 0.0 ? sqrt(sum_val) : 1.0, 1e-4, 1e4)
        end
    end
    return
end

# Kernel to compute column-wise sum of absolute values for CSR matrix (operates on AT in CSR format)
function compute_col_sum_abs_kernel!(rowPtr::CuDeviceVector{Int32},
    nzVal::CuDeviceVector{Float64},
    col_norm::CuDeviceVector{Float64},
    n::Int)
    i = threadIdx().x + (blockDim().x * (blockIdx().x - 1))
    if i <= n
        @inbounds begin
            start_idx = rowPtr[i]
            end_idx = rowPtr[i+1] - 1
            sum_val = 0.0
            for k in start_idx:end_idx
                sum_val += abs(nzVal[k])
            end
            col_norm[i] = clamp(sum_val > 0.0 ? sqrt(sum_val) : 1.0, 1e-4, 1e4)
        end
    end
    return
end

# Kernel to compute row-wise sum of absolute values for CSR matrix without post-processing
function compute_row_abs_sum_kernel!(rowPtr::CuDeviceVector{Int32},
    nzVal::CuDeviceVector{Float64},
    row_sum::CuDeviceVector{Float64},
    m::Int)
    i = threadIdx().x + (blockDim().x * (blockIdx().x - 1))
    if i <= m
        @inbounds begin
            start_idx = rowPtr[i]
            end_idx = rowPtr[i+1] - 1
            sum_val = 0.0
            for k in start_idx:end_idx
                sum_val += abs(nzVal[k])
            end
            row_sum[i] = sum_val
        end
    end
    return
end

# Kernel to compute one clipped SOC scaling factor per cone block and broadcast it to the block rows
function compute_soc_block_gamma_factors_kernel!(row_sum::CuDeviceVector{Float64},
    soc_con_idx::CuDeviceVector{I},
    soc_row_factor::CuDeviceVector{Float64},
    num_blocks::Int) where {I<:Integer}
    i = threadIdx().x + (blockDim().x * (blockIdx().x - 1))
    if i <= num_blocks
        @inbounds begin
            start_idx = soc_con_idx[i]
            end_idx = soc_con_idx[i+1] - 1
            block_norm = 0.0
            for j in start_idx:end_idx
                block_norm = max(block_norm, row_sum[j])
            end
            gamma = clamp(1.0 / max(1.0, block_norm), 1e-4, 1e4)
            for j in start_idx:end_idx
                soc_row_factor[j] = gamma
            end
        end
    end
    return
end

function apply_soc_block_scaling_kernel!(temp_norms::CuDeviceVector{Float64},
    cone_idx::CuDeviceVector{I},
    strategy_code::Int32,
    phase_code::Int32,
    location_code::Int32,
    cone_count::Int32) where {I<:Integer}
    i = threadIdx().x + (blockDim().x * (blockIdx().x - 1))
    if i <= cone_count
        @inbounds begin
            start_idx = Int(cone_idx[i])
            end_idx = Int(cone_idx[i+1]) - 1
            block_len = end_idx - start_idx + 1

            if block_len > 0
                block_max = 0.0
                sum_sq = 0.0
                needs_rms = strategy_code != SOC_SCALE_STRATEGY_MAX

                for j in start_idx:end_idx
                    val = temp_norms[j]
                    block_max = max(block_max, val)
                    if needs_rms
                        sum_sq = muladd(val, val, sum_sq)
                    end
                end

                rms = needs_rms ? sqrt(sum_sq / block_len) : block_max
                block_scale = _soc_block_scale_value_scalar(
                    block_len,
                    block_max,
                    rms,
                    strategy_code,
                    phase_code,
                    location_code,
                    cone_count,
                )

                for j in start_idx:end_idx
                    temp_norms[j] = block_scale
                end
            end
        end
    end
    return
end

function apply_soc_block_scaling_phase_taper_kernel!(temp_norms::CuDeviceVector{Float64},
    cone_idx::CuDeviceVector{I},
    phase_code::Int32,
    cone_count::Int32) where {I<:Integer}
    i = threadIdx().x + (blockDim().x * (blockIdx().x - 1))
    if i <= cone_count
        @inbounds begin
            start_idx = Int(cone_idx[i])
            end_idx = Int(cone_idx[i+1]) - 1
            block_len = end_idx - start_idx + 1

            if block_len > 0
                block_max = 0.0
                sum_sq = 0.0
                for j in start_idx:end_idx
                    val = temp_norms[j]
                    block_max = max(block_max, val)
                    sum_sq = muladd(val, val, sum_sq)
                end

                rms = sqrt(sum_sq / block_len)
                block_scale = if phase_code == SOC_SCALE_PHASE_RUIZ
                    block_len <= ADAPTIVE_SOC_BLOCK_THRESHOLD ? block_max : rms
                elseif phase_code == SOC_SCALE_PHASE_POCK
                    block_len <= ADAPTIVE_SOC_BLOCK_THRESHOLD ? rms :
                    sqrt(block_max * rms)
                else
                    sqrt(block_max * rms)
                end

                for j in start_idx:end_idx
                    temp_norms[j] = block_scale
                end
            end
        end
    end
    return
end

function apply_soc_block_max_scaling_kernel!(temp_norms::CuDeviceVector{Float64},
    cone_idx::CuDeviceVector{I},
    num_blocks::Int) where {I<:Integer}
    i = threadIdx().x + (blockDim().x * (blockIdx().x - 1))
    if i <= num_blocks
        @inbounds begin
            start_idx = cone_idx[i]
            end_idx = cone_idx[i+1] - 1
            block_max = 0.0
            for j in start_idx:end_idx
                block_max = max(block_max, temp_norms[j])
            end
            for j in start_idx:end_idx
                temp_norms[j] = block_max
            end
        end
    end
    return
end

function disable_soc_row_scaling_kernel!(temp_norms::CuDeviceVector{Float64},
    cone_idx::CuDeviceVector{I},
    num_blocks::Int) where {I<:Integer}
    i = threadIdx().x + (blockDim().x * (blockIdx().x - 1))
    if i <= num_blocks
        @inbounds begin
            start_idx = cone_idx[i]
            end_idx = cone_idx[i+1] - 1
            for j in start_idx:end_idx
                temp_norms[j] = 1.0
            end
        end
    end
    return
end

function finalize_soc_scalar_scales_kernel!(block_scale::CuDeviceVector{Float64},
    temp_norms::CuDeviceVector{Float64},
    cone_idx::CuDeviceVector{I},
    num_blocks::Int,
    scaling_factor::Float64) where {I<:Integer}
    i = threadIdx().x + (blockDim().x * (blockIdx().x - 1))
    if i <= num_blocks
        @inbounds begin
            start_idx = cone_idx[i]
            end_idx = cone_idx[i+1] - 1
            block_max = 0.0
            for j in start_idx:end_idx
                block_max = max(block_max, temp_norms[j])
            end
            block_scale[i] = 1.0 / (block_max * scaling_factor)
        end
    end
    return
end

# Kernel to compute row-wise maximum of absolute values including Q diagonal
function compute_row_max_abs_with_Q_kernel!(A_rowPtr::CuDeviceVector{Int32},
    A_nzVal::CuDeviceVector{Float64},
    Q_rowPtr::CuDeviceVector{Int32},
    Q_nzVal::CuDeviceVector{Float64},
    row_norm::CuDeviceVector{Float64},
    n::Int)
    i = threadIdx().x + (blockDim().x * (blockIdx().x - 1))
    if i <= n
        @inbounds begin
            # Compute max for Q (column-wise, using rowPtr since Q is also in CSR)
            start_idx = Q_rowPtr[i]
            end_idx = Q_rowPtr[i+1] - 1
            max_val_Q = 0.0
            for k in start_idx:end_idx
                val = abs(Q_nzVal[k])
                max_val_Q = max(max_val_Q, val)
            end

            # Compute max for A (column-wise, using AT stored in CSR)
            start_idx = A_rowPtr[i]
            end_idx = A_rowPtr[i+1] - 1
            max_val_A = 0.0
            for k in start_idx:end_idx
                val = abs(A_nzVal[k])
                max_val_A = max(max_val_A, val)
            end

            max_val = max(max_val_Q, max_val_A)
            row_norm[i] = clamp(max_val > 0.0 ? sqrt(max_val) : 1.0, 1e-4, 1e4)
        end
    end
    return
end

# Kernel to compute column-wise sum including Q diagonal
function compute_col_sum_abs_with_Q_kernel!(A_rowPtr::CuDeviceVector{Int32},
    A_nzVal::CuDeviceVector{Float64},
    Q_rowPtr::CuDeviceVector{Int32},
    Q_nzVal::CuDeviceVector{Float64},
    col_norm::CuDeviceVector{Float64},
    n::Int)
    i = threadIdx().x + (blockDim().x * (blockIdx().x - 1))
    if i <= n
        @inbounds begin
            # Compute sum for Q (column-wise)
            start_idx = Q_rowPtr[i]
            end_idx = Q_rowPtr[i+1] - 1
            sum_val_Q = 0.0
            for k in start_idx:end_idx
                sum_val_Q += abs(Q_nzVal[k])
            end

            # Compute sum for A (column-wise, using AT)
            start_idx = A_rowPtr[i]
            end_idx = A_rowPtr[i+1] - 1
            sum_val_A = 0.0
            for k in start_idx:end_idx
                sum_val_A += abs(A_nzVal[k])
            end

            sum_val = sum_val_Q + sum_val_A
            col_norm[i] = clamp(sum_val > 0.0 ? sqrt(sum_val) : 1.0, 1e-4, 1e4)
        end
    end
    return
end

# Kernel to check if a CSR matrix is diagonal
# For each row i, check if it has exactly one non-zero at column i
function check_diagonal_kernel!(rowPtr::CuDeviceVector{Int32},
    colVal::CuDeviceVector{Int32},
    is_diag::CuDeviceVector{Bool},
    n::Int)
    i = threadIdx().x + (blockDim().x * (blockIdx().x - 1))
    if i <= n
        @inbounds begin
            start_idx = rowPtr[i]
            end_idx = rowPtr[i+1] - 1
            nnz_in_row = end_idx - start_idx + 1

            # Check: exactly one non-zero AND it's at column index i
            if nnz_in_row == 1
                col_idx = colVal[start_idx]
                is_diag[i] = (col_idx == i)
            elseif nnz_in_row == 0
                # Empty row is considered diagonal (zero on diagonal)
                is_diag[i] = true
            else
                # More than one non-zero: not diagonal
                is_diag[i] = false
            end
        end
    end
    return
end

# Kernel to extract diagonal elements from CSR matrix
function extract_diagonal_csr_kernel!(rowPtr::CuDeviceVector{Int32},
    colVal::CuDeviceVector{Int32},
    nzVal::CuDeviceVector{Float64},
    diag::CuDeviceVector{Float64},
    n::Int)
    i = threadIdx().x + (blockDim().x * (blockIdx().x - 1))
    if i <= n
        @inbounds begin
            start_idx = rowPtr[i]
            end_idx = rowPtr[i+1] - 1

            # Search for diagonal element in this row
            diag[i] = 0.0
            for k in start_idx:end_idx
                if colVal[k] == i
                    diag[i] = nzVal[k]
                    break
                end
            end
        end
    end
    return
end

# Kernel to scale rows of CSR matrix by 1.0 / row_scale
function scale_rows_csr_kernel!(rowPtr::CuDeviceVector{Int32},
    nzVal::CuDeviceVector{Float64},
    row_scale::CuDeviceVector{Float64},
    m::Int)
    i = threadIdx().x + (blockDim().x * (blockIdx().x - 1))
    if i <= m
        @inbounds begin
            scale = 1.0 / row_scale[i]
            start_idx = rowPtr[i]
            end_idx = rowPtr[i+1] - 1
            for k in start_idx:end_idx
                nzVal[k] *= scale
            end
        end
    end
    return
end

# Kernel to scale columns of CSR matrix by column indices
function scale_csr_cols_kernel!(rowPtr::CuDeviceVector{Int32},
    colVal::CuDeviceVector{Int32},
    nzVal::CuDeviceVector{Float64},
    col_scale::CuDeviceVector{Float64},
    m::Int)
    i = threadIdx().x + (blockDim().x * (blockIdx().x - 1))
    if i <= m
        @inbounds begin
            start_idx = rowPtr[i]
            end_idx = rowPtr[i+1] - 1
            for k in start_idx:end_idx
                col_idx = colVal[k]
                nzVal[k] /= col_scale[col_idx]
            end
        end
    end
    return
end

# Kernel to scale CSR matrix entries by both row and column factors in one pass.
# The arithmetic order matches the separate row-then-column kernels.
function scale_csr_row_col_kernel!(rowPtr::CuDeviceVector{Int32},
    colVal::CuDeviceVector{Int32},
    nzVal::CuDeviceVector{Float64},
    row_scale::CuDeviceVector{Float64},
    col_scale::CuDeviceVector{Float64},
    m::Int)
    i = threadIdx().x + (blockDim().x * (blockIdx().x - 1))
    if i <= m
        @inbounds begin
            row_inv = 1.0 / row_scale[i]
            start_idx = rowPtr[i]
            end_idx = rowPtr[i+1] - 1
            for k in start_idx:end_idx
                val = nzVal[k]
                val *= row_inv
                val /= col_scale[colVal[k]]
                nzVal[k] = val
            end
        end
    end
    return
end

# Kernel to scale a vector by another vector element-wise (v[i] /= scale[i])
function scale_vector_div_kernel!(v::CuDeviceVector{Float64},
    scale::CuDeviceVector{Float64},
    n::Int)
    i = threadIdx().x + (blockDim().x * (blockIdx().x - 1))
    if i <= n
        @inbounds v[i] /= scale[i]
    end
    return
end

# Kernel to scale two vectors by another vector element-wise in one launch.
function scale_two_vectors_div_kernel!(v1::CuDeviceVector{Float64},
    v2::CuDeviceVector{Float64},
    scale::CuDeviceVector{Float64},
    n::Int)
    i = threadIdx().x + (blockDim().x * (blockIdx().x - 1))
    if i <= n
        @inbounds begin
            factor = scale[i]
            v1[i] /= factor
            v2[i] /= factor
        end
    end
    return
end

# Kernel to scale a vector by another vector element-wise (v[i] *= scale[i])
function scale_vector_mul_kernel!(v::CuDeviceVector{Float64},
    scale::CuDeviceVector{Float64},
    n::Int)
    i = threadIdx().x + (blockDim().x * (blockIdx().x - 1))
    if i <= n
        @inbounds v[i] *= scale[i]
    end
    return
end

# Kernel to scale two vectors by another vector element-wise in one launch.
function scale_two_vectors_mul_kernel!(v1::CuDeviceVector{Float64},
    v2::CuDeviceVector{Float64},
    scale::CuDeviceVector{Float64},
    n::Int)
    i = threadIdx().x + (blockDim().x * (blockIdx().x - 1))
    if i <= n
        @inbounds begin
            factor = scale[i]
            v1[i] *= factor
            v2[i] *= factor
        end
    end
    return
end

# Kernel to scale a vector by a scalar (v[i] /= scalar)
function scale_vector_scalar_div_kernel!(v::CuDeviceVector{Float64},
    scalar::Float64,
    n::Int)
    i = threadIdx().x + (blockDim().x * (blockIdx().x - 1))
    if i <= n
        @inbounds v[i] /= scalar
    end
    return
end

# Kernel to scale two vectors by the same scalar in one launch.
function scale_two_vectors_scalar_div_kernel!(v1::CuDeviceVector{Float64},
    v2::CuDeviceVector{Float64},
    scalar::Float64,
    n::Int)
    i = threadIdx().x + (blockDim().x * (blockIdx().x - 1))
    if i <= n
        @inbounds begin
            v1[i] /= scalar
            v2[i] /= scalar
        end
    end
    return
end

# Kernel to scale a vector by a scalar (v[i] *= scalar)
function scale_vector_scalar_mul_kernel!(v::CuDeviceVector{Float64},
    scalar::Float64,
    n::Int)
    i = threadIdx().x + (blockDim().x * (blockIdx().x - 1))
    if i <= n
        @inbounds v[i] *= scalar
    end
    return
end

# ============================================================================
# CPU Loop-Based Update Functions
# ============================================================================
#
# This section contains CPU implementations that mirror the GPU kernel logic.
# These functions MUST remain separate from GPU kernels due to fundamentally
# different execution models.
#
# GPU Kernels vs CPU Loops:
# - GPU: Massively parallel execution across thousands of threads
# - CPU: Vectorized serial/SIMD loops optimized for cache locality
#
# The CPU functions are organized to match GPU kernel categories:
#
# 1. Standard QP Updates (lines ~1576-1770)
#    - update_zxw1_cpu!: CPU version of unified_update_zxw1_kernel
#      Mirrors GPU logic but uses @simd loops instead of parallel threads
#      Computes z, x, w updates with box projection for standard QP
#
# 2. Unified LP Updates (lines ~1766-1930)
#    - unified_update_zx_cpu!: CPU version for problems with empty Q (LP)
#    - unified_update_y_noQ_cpu!: CPU dual updates for LP problems
#      These mirror the GPU LP kernels but use CPU-optimized loops
#
# 3. Standard Update Subroutines (lines ~1928-2090)
#    - update_y_cpu!: CPU version of unified_update_y_kernel
#      Dual variable y updates with projection
#    - update_w2_cpu!: CPU version of unified_update_w2_kernel
#      Dual variable w updates with AT*y_bar computation
#
# Key CPU Optimization Techniques:
# - @simd macro: Enables SIMD vectorization for compatible loops
# - @inbounds: Removes bounds checking for performance (use carefully)
# - @fastmath: Relaxes floating-point semantics for speed (use where safe)
# - Loop fusion: Combines multiple operations in single pass
# - Cache-friendly memory access patterns
#
# Relationship to GPU Kernels:
# - Algorithmic logic is identical to GPU kernels
# - Implementation differs to exploit CPU architecture:
#   * Sequential iteration vs parallel GPU threads
#   * CPU SIMD vs GPU warps
#   * Cache hierarchy vs GPU shared memory
# - Both produce bit-identical numerical results (within floating-point tolerance)
#
# Design Rationale:
# - Keeping separate implementations allows architecture-specific optimization
# - CPU code can use standard Julia broadcasting and SIMD
# - GPU code uses CUDA-specific features (shared memory, warp primitives)
# - Attempting to unify would sacrifice performance on both platforms
#
# Calling Convention:
# - These functions are called through device-agnostic wrappers in algorithm.jl
# - Dispatch on workspace type (HPRSOCP_workspace_cpu) selects CPU versions
# - Parameters and semantics match GPU versions for consistency
#
# NOTE: When modifying algorithmic logic, changes must be synchronized between
# GPU kernels and CPU loops to maintain numerical equivalence.
# ============================================================================

# CPU version of unified_update_zxw for noC case
# Updates z, x, and w in one pass, computing Qw inline and tempv for later use
function unified_update_zxw_cpu!(ws::HPRSOCP_workspace_cpu,
    qp::QP_info_cpu,
    Halpern_fact1::Float64,
    Halpern_fact2::Float64)

    # Compute Qw (maps w to Qw)
    # Qmap!(ws.w, ws.Qw, ws.Q)
    # Compute AT*y for next iteration
    mul!(ws.ATy, ws.AT, ws.y)

    # Determine Q type and compute factors
    Q_is_diag = ws.Q_is_diag
    fact2_scalar = 1.0 / (1.0 + ws.sigma * ws.lambda_max_Q)
    fact1_scalar = 1.0 - fact2_scalar

    x = ws.x
    x_bar = ws.x_bar
    z_bar = ws.z_bar
    x_hat = ws.x_hat
    dx = ws.dx
    w = ws.w
    w_bar = ws.w_bar
    dw = ws.dw
    last_x = ws.last_x
    last_w = ws.last_w
    Qw = ws.Qw
    ATy = ws.ATy
    c = ws.c
    l = ws.l
    u = ws.u
    sigma = ws.sigma
    fact1_vec = ws.fact1
    fact2_vec = ws.fact2

    if ws.to_check
        @simd for i in eachindex(x)
            @inbounds begin
                qw_val = Qw[i]
                atyi = ATy[i]
                c_i = c[i]
                x_i = x[i]
                last_x_i = last_x[i]
                last_w_i = last_w[i]
                l_i = l[i]
                u_i = u[i]
                w_i = w[i]

                tmp = -qw_val + atyi - c_i
                z_raw = x_i + sigma * tmp
                x_bar_i = min(max(z_raw, l_i), u_i)

                x_hat_i = 2.0 * x_bar_i - x_i
                x_new = muladd(Halpern_fact2, x_hat_i, Halpern_fact1 * last_x_i)

                w_bar_i = if Q_is_diag
                    muladd(fact1_vec[i], w_i, fact2_vec[i] * x_hat_i)
                else
                    muladd(fact1_scalar, w_i, fact2_scalar * x_hat_i)
                end
                w_hat_i = 2.0 * w_bar_i - w_i
                w_new = muladd(Halpern_fact2, w_hat_i, Halpern_fact1 * last_w_i)

                dx[i] = x_bar_i - x_i
                x_bar[i] = x_bar_i
                z_bar[i] = (x_bar_i - z_raw) / sigma
                x[i] = x_new
                x_hat[i] = x_hat_i
                w_bar[i] = w_bar_i
                w[i] = w_new
                dw[i] = w_bar_i - w_i
            end
        end
    else
        @simd for i in eachindex(x)
            @inbounds begin
                qw_val = Qw[i]
                atyi = ATy[i]
                c_i = c[i]
                x_i = x[i]
                last_x_i = last_x[i]
                last_w_i = last_w[i]
                l_i = l[i]
                u_i = u[i]
                w_i = w[i]

                tmp = -qw_val + atyi - c_i
                z_raw = x_i + sigma * tmp
                x_bar_i = min(max(z_raw, l_i), u_i)

                x_hat_i = 2.0 * x_bar_i - x_i
                x_new = muladd(Halpern_fact2, x_hat_i, Halpern_fact1 * last_x_i)

                w_bar_i = if Q_is_diag
                    muladd(fact1_vec[i], w_i, fact2_vec[i] * x_hat_i)
                else
                    muladd(fact1_scalar, w_i, fact2_scalar * x_hat_i)
                end
                w_hat_i = 2.0 * w_bar_i - w_i
                w_new = muladd(Halpern_fact2, w_hat_i, Halpern_fact1 * last_w_i)

                x[i] = x_new
                x_hat[i] = x_hat_i
                w_bar[i] = w_bar_i
                w[i] = w_new
            end
        end
    end

    # Compute Qw_bar for tempv computation
    Qmap!(ws.w_bar, ws.Qw_bar, ws.Q)

    # Compute tempv = x_hat + sigma * (Qw - Qw_bar) and update Qw with Halpern averaging
    @simd for i in eachindex(ws.tempv)
        @inbounds begin
            qw_i = ws.Qw[i]
            qw_bar_i = ws.Qw_bar[i]
            qw_hat_i = 2.0 * qw_bar_i - qw_i
            ws.Qw[i] = muladd(Halpern_fact2, qw_hat_i, Halpern_fact1 * ws.last_Qw[i])
            ws.tempv[i] = ws.x_hat[i] + sigma * (qw_i - qw_bar_i)
        end
    end
end

# CPU version of unified_update_y for noC case
# Uses tempv computed by unified_update_zxw_cpu! instead of recomputing it
function unified_update_y_cpu!(ws::HPRSOCP_workspace_cpu,
    Halpern_fact1::Float64,
    Halpern_fact2::Float64)

    if ws.m == 0
        return
    end

    # Compute A * tempv (tempv already computed in unified_update_zxw_cpu!)
    mul!(ws.Ax, ws.A, ws.tempv)

    fact1 = ws.lambda_max_A * ws.sigma
    fact2 = 1.0 / fact1

    y = ws.y
    y_bar = ws.y_bar
    dy = ws.dy
    last_y = ws.last_y
    s = ws.s
    Ax = ws.Ax
    AL = ws.AL
    AU = ws.AU

    if ws.to_check
        @simd for i in eachindex(y)
            @inbounds begin
                yi = y[i]
                s_raw = Ax[i] - fact1 * yi
                s_proj = min(max(s_raw, AL[i]), AU[i])
                corr = s_proj - s_raw
                yb = fact2 * corr
                yh = 2.0 * yb - yi
                y_new = muladd(Halpern_fact2, yh, Halpern_fact1 * last_y[i])

                s[i] = s_proj
                dy[i] = yb - yi
                y_bar[i] = yb
                y[i] = y_new
            end
        end
    else
        @simd for i in eachindex(y)
            @inbounds begin
                yi = y[i]
                s_raw = Ax[i] - fact1 * yi
                s_proj = min(max(s_raw, AL[i]), AU[i])
                corr = s_proj - s_raw
                yb = fact2 * corr
                yh = 2.0 * yb - yi
                y_new = muladd(Halpern_fact2, yh, Halpern_fact1 * last_y[i])

                y_bar[i] = yb
                y[i] = y_new
            end
        end
    end
end

# CPU version of update_zxw for standard QP
function update_zxw1_cpu!(ws::HPRSOCP_workspace_cpu,
    qp::QP_info_cpu,
    Halpern_fact1::Float64,
    Halpern_fact2::Float64)
    # Compute Qw
    Qmap!(ws.w, ws.Qw, ws.Q)

    # Determine Q type and compute factors
    Q_is_diag = ws.Q_is_diag
    # Prepare scalar factors for regular Q
    fact2_scalar = 1.0 / (1.0 + ws.sigma * ws.lambda_max_Q)
    fact1_scalar = 1.0 - fact2_scalar

    x = ws.x
    x_bar = ws.x_bar
    z_bar = ws.z_bar
    x_hat = ws.x_hat
    dx = ws.dx
    w = ws.w
    w_bar = ws.w_bar
    dw = ws.dw
    last_x = ws.last_x
    last_w = ws.last_w
    Qw = ws.Qw
    ATy = ws.ATy
    c = ws.c
    l = ws.l
    u = ws.u
    sigma = ws.sigma
    fact1_vec = ws.fact1
    fact2_vec = ws.fact2
    linear_n = ws.number_SOC_var > 0 ? ws.number_lu_x : length(x)

    if ws.to_check
        @simd for i in 1:linear_n
            @inbounds begin
                qw_val = Qw[i]
                atyi = ATy[i]
                c_i = c[i]
                x_i = x[i]
                last_x_i = last_x[i]
                l_i = l[i]
                u_i = u[i]
                w_i = w[i]

                tmp = -qw_val + atyi - c_i
                z_raw = x_i + sigma * tmp
                x_bar_i = min(max(z_raw, l_i), u_i)

                x_hat_i = 2.0 * x_bar_i - x_i
                x_new = muladd(Halpern_fact2, x_hat_i, Halpern_fact1 * last_x_i)

                w_bar_i = if Q_is_diag
                    muladd(fact1_vec[i], w_i, fact2_vec[i] * x_hat_i)
                else
                    muladd(fact1_scalar, w_i, fact2_scalar * x_hat_i)
                end

                dx[i] = x_bar_i - x_i
                x_bar[i] = x_bar_i
                z_bar[i] = (x_bar_i - z_raw) / sigma
                x[i] = x_new
                x_hat[i] = x_hat_i
                w_bar[i] = w_bar_i
            end
        end
    else
        @simd for i in 1:linear_n
            @inbounds begin
                qw_val = Qw[i]
                atyi = ATy[i]
                c_i = c[i]
                x_i = x[i]
                last_x_i = last_x[i]
                l_i = l[i]
                u_i = u[i]
                w_i = w[i]

                tmp = -qw_val + atyi - c_i
                z_raw = x_i + sigma * tmp
                x_bar_i = min(max(z_raw, l_i), u_i)

                x_hat_i = 2.0 * x_bar_i - x_i
                x_new = muladd(Halpern_fact2, x_hat_i, Halpern_fact1 * last_x_i)

                w_bar_i = if Q_is_diag
                    muladd(fact1_vec[i], w_i, fact2_vec[i] * x_hat_i)
                else
                    muladd(fact1_scalar, w_i, fact2_scalar * x_hat_i)
                end

                x[i] = x_new
                x_hat[i] = x_hat_i
                w_bar[i] = w_bar_i
            end
        end
    end

    for soc_idx in 1:ws.number_SOC_var
        start_idx = ws.SOC_var_idx[soc_idx]
        end_idx = ws.SOC_var_idx[soc_idx + 1] - 1

        z_raw_t = x[start_idx] + sigma * (-Qw[start_idx] + ATy[start_idx] - c[start_idx])
        z_bar[start_idx] = z_raw_t
        norm_s = 0.0
        for j in (start_idx + 1):end_idx
            z_raw_j = x[j] + sigma * (-Qw[j] + ATy[j] - c[j])
            z_bar[j] = z_raw_j
            norm_s += z_raw_j^2
        end
        norm_s = sqrt(norm_s)

        if norm_s <= -z_raw_t
            x_bar[start_idx] = 0.0
            dx[start_idx] = -x[start_idx]
            x_hat[start_idx] = -x[start_idx]
            z_bar[start_idx] = -z_raw_t / sigma
            x[start_idx] = muladd(Halpern_fact2, x_hat[start_idx], Halpern_fact1 * last_x[start_idx])
            w_bar[start_idx] = Q_is_diag ?
                               muladd(fact1_vec[start_idx], w[start_idx], fact2_vec[start_idx] * x_hat[start_idx]) :
                               muladd(fact1_scalar, w[start_idx], fact2_scalar * x_hat[start_idx])

            for j in (start_idx + 1):end_idx
                x_bar[j] = 0.0
                dx[j] = -x[j]
                x_hat[j] = -x[j]
                z_bar[j] = -z_bar[j] / sigma
                x[j] = muladd(Halpern_fact2, x_hat[j], Halpern_fact1 * last_x[j])
                w_bar[j] = Q_is_diag ? muladd(fact1_vec[j], w[j], fact2_vec[j] * x_hat[j]) :
                           muladd(fact1_scalar, w[j], fact2_scalar * x_hat[j])
            end
        elseif norm_s <= z_raw_t
            x_bar[start_idx] = z_raw_t
            dx[start_idx] = z_raw_t - x[start_idx]
            x_hat[start_idx] = 2.0 * z_raw_t - x[start_idx]
            z_bar[start_idx] = 0.0
            x[start_idx] = muladd(Halpern_fact2, x_hat[start_idx], Halpern_fact1 * last_x[start_idx])
            w_bar[start_idx] = Q_is_diag ?
                               muladd(fact1_vec[start_idx], w[start_idx], fact2_vec[start_idx] * x_hat[start_idx]) :
                               muladd(fact1_scalar, w[start_idx], fact2_scalar * x_hat[start_idx])

            for j in (start_idx + 1):end_idx
                z_raw_j = z_bar[j]
                x_bar[j] = z_raw_j
                dx[j] = z_raw_j - x[j]
                x_hat[j] = 2.0 * z_raw_j - x[j]
                z_bar[j] = 0.0
                x[j] = muladd(Halpern_fact2, x_hat[j], Halpern_fact1 * last_x[j])
                w_bar[j] = Q_is_diag ? muladd(fact1_vec[j], w[j], fact2_vec[j] * x_hat[j]) :
                           muladd(fact1_scalar, w[j], fact2_scalar * x_hat[j])
            end
        else
            proj_t = (norm_s + z_raw_t) / 2.0
            alpha = proj_t / norm_s

            x_bar[start_idx] = proj_t
            dx[start_idx] = proj_t - x[start_idx]
            x_hat[start_idx] = 2.0 * proj_t - x[start_idx]
            z_bar[start_idx] = (proj_t - z_raw_t) / sigma
            x[start_idx] = muladd(Halpern_fact2, x_hat[start_idx], Halpern_fact1 * last_x[start_idx])
            w_bar[start_idx] = Q_is_diag ?
                               muladd(fact1_vec[start_idx], w[start_idx], fact2_vec[start_idx] * x_hat[start_idx]) :
                               muladd(fact1_scalar, w[start_idx], fact2_scalar * x_hat[start_idx])

            for j in (start_idx + 1):end_idx
                z_raw_j = z_bar[j]
                proj_j = alpha * z_raw_j
                x_bar[j] = proj_j
                dx[j] = proj_j - x[j]
                x_hat[j] = 2.0 * proj_j - x[j]
                z_bar[j] = (proj_j - z_raw_j) / sigma
                x[j] = muladd(Halpern_fact2, x_hat[j], Halpern_fact1 * last_x[j])
                w_bar[j] = Q_is_diag ? muladd(fact1_vec[j], w[j], fact2_vec[j] * x_hat[j]) :
                           muladd(fact1_scalar, w[j], fact2_scalar * x_hat[j])
            end
        end
    end
end

# ============================================================================
# CPU Functions for noQ case (empty Q matrix - linear program / SOCP)
# ============================================================================

# CPU version of unified_update_zx - for noQ case (empty Q)
# Updates z and x variables without Q matrix operations, including SOC projections
function unified_update_zx_cpu!(ws::HPRSOCP_workspace_cpu,
    Halpern_fact1::Float64,
    Halpern_fact2::Float64)

    # Compute AT*y
    mul!(ws.ATy, ws.AT, ws.y)

    x = ws.x
    x_bar = ws.x_bar
    x_hat = ws.x_hat
    z_bar = ws.z_bar
    dx = ws.dx
    last_x = ws.last_x
    ATy = ws.ATy
    c = ws.c
    l = ws.l
    u = ws.u
    sigma = ws.sigma
    linear_n = ws.number_SOC_var > 0 ? ws.number_lu_x : ws.n

    if ws.to_check
        @simd for i in 1:linear_n
            @inbounds begin
                x_i = x[i]
                last_x_i = last_x[i]
                ATy_i = ATy[i]
                c_i = c[i]
                l_i = l[i]
                u_i = u[i]

                # Compute z_raw = x + sigma * (AT*y - c)
                tmp = ATy_i - c_i
                z_raw = x_i + sigma * tmp

                # Project onto bounds [l, u]
                x_bar_i = min(max(z_raw, l_i), u_i)

                # Compute x_hat = 2*x_bar - x (for Peaceman-Rachford)
                x_hat_i = 2.0 * x_bar_i - x_i

                # Compute z_bar (dual variable for bounds)
                z_bar_i = (x_bar_i - z_raw) / sigma

                # Halpern averaging: x_new = alpha*last_x + (1-alpha)*x_hat
                x_new = muladd(Halpern_fact2, x_hat_i, Halpern_fact1 * last_x_i)

                # Store results
                dx[i] = x_bar_i - x_i
                x_bar[i] = x_bar_i
                z_bar[i] = z_bar_i
                x[i] = x_new
                x_hat[i] = x_hat_i
            end
        end
    else
        @simd for i in 1:linear_n
            @inbounds begin
                x_i = x[i]
                last_x_i = last_x[i]
                ATy_i = ATy[i]
                c_i = c[i]
                l_i = l[i]
                u_i = u[i]

                tmp = ATy_i - c_i
                z_raw = x_i + sigma * tmp
                x_bar_i = min(max(z_raw, l_i), u_i)
                x_hat_i = 2.0 * x_bar_i - x_i
                x_new = muladd(Halpern_fact2, x_hat_i, Halpern_fact1 * last_x_i)

                x[i] = x_new
                x_hat[i] = x_hat_i
            end
        end
    end

    for soc_idx in 1:ws.number_SOC_var
        start_idx = ws.SOC_var_idx[soc_idx]
        end_idx = ws.SOC_var_idx[soc_idx + 1] - 1

        z_raw_t = x[start_idx] + sigma * (ATy[start_idx] - c[start_idx])
        z_bar[start_idx] = z_raw_t
        norm_s = 0.0
        for j in (start_idx + 1):end_idx
            z_raw_j = x[j] + sigma * (ATy[j] - c[j])
            z_bar[j] = z_raw_j
            norm_s += z_raw_j^2
        end
        norm_s = sqrt(norm_s)

        if norm_s <= -z_raw_t
            x_bar[start_idx] = 0.0
            dx[start_idx] = -x[start_idx]
            x_hat[start_idx] = -x[start_idx]
            z_bar[start_idx] = -z_raw_t / sigma
            x[start_idx] = muladd(Halpern_fact2, x_hat[start_idx], Halpern_fact1 * last_x[start_idx])

            for j in (start_idx + 1):end_idx
                x_bar[j] = 0.0
                dx[j] = -x[j]
                x_hat[j] = -x[j]
                z_bar[j] = -z_bar[j] / sigma
                x[j] = muladd(Halpern_fact2, x_hat[j], Halpern_fact1 * last_x[j])
            end
        elseif norm_s <= z_raw_t
            x_bar[start_idx] = z_raw_t
            dx[start_idx] = z_raw_t - x[start_idx]
            x_hat[start_idx] = 2.0 * z_raw_t - x[start_idx]
            z_bar[start_idx] = 0.0
            x[start_idx] = muladd(Halpern_fact2, x_hat[start_idx], Halpern_fact1 * last_x[start_idx])

            for j in (start_idx + 1):end_idx
                z_raw_j = z_bar[j]
                x_bar[j] = z_raw_j
                dx[j] = z_raw_j - x[j]
                x_hat[j] = 2.0 * z_raw_j - x[j]
                z_bar[j] = 0.0
                x[j] = muladd(Halpern_fact2, x_hat[j], Halpern_fact1 * last_x[j])
            end
        else
            proj_t = (norm_s + z_raw_t) / 2.0
            alpha = proj_t / norm_s

            x_bar[start_idx] = proj_t
            dx[start_idx] = proj_t - x[start_idx]
            x_hat[start_idx] = 2.0 * proj_t - x[start_idx]
            z_bar[start_idx] = (proj_t - z_raw_t) / sigma
            x[start_idx] = muladd(Halpern_fact2, x_hat[start_idx], Halpern_fact1 * last_x[start_idx])

            for j in (start_idx + 1):end_idx
                z_raw_j = z_bar[j]
                proj_j = alpha * z_raw_j
                x_bar[j] = proj_j
                dx[j] = proj_j - x[j]
                x_hat[j] = 2.0 * proj_j - x[j]
                z_bar[j] = (proj_j - z_raw_j) / sigma
                x[j] = muladd(Halpern_fact2, x_hat[j], Halpern_fact1 * last_x[j])
            end
        end
    end
end

# CPU version of unified_update_y_noQ - for noQ case (empty Q)
# Updates y variable using x_hat directly (no tempv needed since Q is empty),
# including SOC constraint projections
function unified_update_y_noQ_cpu!(ws::HPRSOCP_workspace_cpu,
    Halpern_fact1::Float64,
    Halpern_fact2::Float64)

    if ws.m == 0
        return
    end

    # Compute A*x_hat (no Q corrections needed for noQ case)
    mul!(ws.Ax, ws.A, ws.x_hat)

    fact1 = ws.lambda_max_A * ws.sigma
    fact2 = 1.0 / fact1

    y = ws.y
    y_bar = ws.y_bar
    dy = ws.dy
    last_y = ws.last_y
    s = ws.s
    Ax = ws.Ax
    AL = ws.AL
    AU = ws.AU
    linear_m = ws.number_linear_con

    if ws.to_check
        @simd for i in 1:linear_m
            @inbounds begin
                y_i = y[i]
                last_y_i = last_y[i]
                Ax_i = Ax[i]
                AL_i = AL[i]
                AU_i = AU[i]

                # Compute s_raw = Ax - fact1*y
                s_raw = Ax_i - fact1 * y_i

                # Project onto constraint bounds [AL, AU]
                s_proj = min(max(s_raw, AL_i), AU_i)

                # Compute correction
                corr = s_proj - s_raw

                # Compute y_bar
                y_bar_i = fact2 * corr

                # Compute y_hat = 2*y_bar - y
                y_hat_i = 2.0 * y_bar_i - y_i

                # Halpern averaging: y_new = alpha*last_y + (1-alpha)*y_hat
                y_new = muladd(Halpern_fact2, y_hat_i, Halpern_fact1 * last_y_i)

                # Store results
                s[i] = s_proj
                dy[i] = y_bar_i - y_i
                y_bar[i] = y_bar_i
                y[i] = y_new
            end
        end
    else
        @simd for i in 1:linear_m
            @inbounds begin
                y_i = y[i]
                last_y_i = last_y[i]
                Ax_i = Ax[i]
                AL_i = AL[i]
                AU_i = AU[i]

                s_raw = Ax_i - fact1 * y_i
                s_proj = min(max(s_raw, AL_i), AU_i)
                corr = s_proj - s_raw
                y_bar_i = fact2 * corr
                y_hat_i = 2.0 * y_bar_i - y_i
                y_new = muladd(Halpern_fact2, y_hat_i, Halpern_fact1 * last_y_i)
                y[i] = y_new
            end
        end
    end

    for soc_idx in 1:ws.number_SOC_con
        start_idx = ws.SOC_con_idx[soc_idx]
        end_idx = ws.SOC_con_idx[soc_idx + 1] - 1
        offset = start_idx - ws.SOC_con_idx[1] + 1

        s_raw_t = Ax[start_idx] - ws.soc_rhs[offset] - fact1 * y[start_idx]
        s[start_idx] = s_raw_t
        norm_s = 0.0
        for j in (start_idx + 1):end_idx
            s_raw_j = Ax[j] - ws.soc_rhs[offset + (j - start_idx)] - fact1 * y[j]
            s[j] = s_raw_j
            norm_s += s_raw_j^2
        end
        norm_s = sqrt(norm_s)

        if norm_s <= -s_raw_t
            s[start_idx] = 0.0
            y_bar[start_idx] = fact2 * (-s_raw_t)
            dy[start_idx] = y_bar[start_idx] - y[start_idx]
            y[start_idx] = muladd(Halpern_fact2, 2.0 * y_bar[start_idx] - y[start_idx], Halpern_fact1 * last_y[start_idx])

            for j in (start_idx + 1):end_idx
                s_raw_j = s[j]
                s[j] = 0.0
                y_bar[j] = fact2 * (-s_raw_j)
                dy[j] = y_bar[j] - y[j]
                y[j] = muladd(Halpern_fact2, 2.0 * y_bar[j] - y[j], Halpern_fact1 * last_y[j])
            end
        elseif norm_s <= s_raw_t
            y_bar[start_idx] = 0.0
            dy[start_idx] = -y[start_idx]
            y[start_idx] = muladd(Halpern_fact2, -y[start_idx], Halpern_fact1 * last_y[start_idx])

            for j in (start_idx + 1):end_idx
                y_bar[j] = 0.0
                dy[j] = -y[j]
                y[j] = muladd(Halpern_fact2, -y[j], Halpern_fact1 * last_y[j])
            end
        else
            proj_t = (norm_s + s_raw_t) / 2.0
            alpha = proj_t / norm_s

            s[start_idx] = proj_t
            y_bar[start_idx] = fact2 * (proj_t - s_raw_t)
            dy[start_idx] = y_bar[start_idx] - y[start_idx]
            y[start_idx] = muladd(Halpern_fact2, 2.0 * y_bar[start_idx] - y[start_idx], Halpern_fact1 * last_y[start_idx])

            for j in (start_idx + 1):end_idx
                s_raw_j = s[j]
                proj_j = alpha * s_raw_j
                s[j] = proj_j
                y_bar[j] = fact2 * (proj_j - s_raw_j)
                dy[j] = y_bar[j] - y[j]
                y[j] = muladd(Halpern_fact2, 2.0 * y_bar[j] - y[j], Halpern_fact1 * last_y[j])
            end
        end
    end
end

# ============================================================================
# CPU Update Functions with Q Matrix (Standard QP Updates)
# ============================================================================
#
# These functions handle updates for problems with non-empty Q matrices.
# They mirror the GPU unified_update_y and unified_update_w2 kernels.
#
# - update_y_cpu!: Computes dual variable y updates with A*tempv computation
#   where tempv = x_hat + sigma*(Qw - Qw_bar). Mirrors unified_update_y_kernel.
#
# - update_w2_cpu!: Computes dual variable w updates with AT*y_bar computation.
#   Mirrors unified_update_w2_kernel.
#
# Both functions use:
# - @simd loops for vectorization
# - Broadcasting for vector operations (.= syntax)
# - mul! for efficient matrix-vector products
# - Conditional execution based on ws.to_check flag
# ============================================================================

# CPU version of update_y with Q matrix
# Mirrors unified_update_y_kernel - computes y updates for standard QP
function update_y_cpu!(ws::HPRSOCP_workspace_cpu,
    qp::QP_info_cpu,
    Halpern_fact1::Float64,
    Halpern_fact2::Float64)
    if ws.m == 0
        return
    end

    # Compute Qw_hat
    Qmap!(ws.w_bar, ws.Qw_bar, ws.Q)

    # Compute A * (x_hat + sigma * (Qw - Qw_bar))
    ws.tempv .= ws.x_hat .+ ws.sigma .* (ws.Qw - ws.Qw_bar)
    mul!(ws.Ax, ws.A, ws.tempv)

    fact1 = ws.lambda_max_A * ws.sigma
    fact2 = 1.0 / fact1

    y = ws.y
    y_bar = ws.y_bar
    y_hat = ws.y_hat
    AL = ws.AL
    AU = ws.AU
    Ax = ws.Ax
    last_y = ws.last_y
    dy = ws.dy
    s = ws.s
    linear_m = ws.number_linear_con

    if ws.to_check
        @simd for i in 1:linear_m
            @inbounds begin
                yi = y[i]
                s_raw = Ax[i] - fact1 * yi
                s_proj = min(max(s_raw, AL[i]), AU[i])
                corr = s_proj - s_raw
                yb = fact2 * corr
                yh = 2.0 * yb - yi
                y_new = muladd(Halpern_fact2, yh, Halpern_fact1 * last_y[i])

                s[i] = s_proj
                dy[i] = yb - yi
                y_bar[i] = yb
                y[i] = y_new
            end
        end
    else
        @simd for i in 1:linear_m
            @inbounds begin
                yi = y[i]
                s_raw = Ax[i] - fact1 * yi
                s_proj = min(max(s_raw, AL[i]), AU[i])
                corr = s_proj - s_raw
                yb = fact2 * corr
                yh = 2.0 * yb - yi
                y_new = muladd(Halpern_fact2, yh, Halpern_fact1 * last_y[i])

                y_bar[i] = yb
                y[i] = y_new
            end
        end
    end

    for soc_idx in 1:ws.number_SOC_con
        start_idx = ws.SOC_con_idx[soc_idx]
        end_idx = ws.SOC_con_idx[soc_idx + 1] - 1
        offset = start_idx - ws.SOC_con_idx[1] + 1

        s_raw_t = Ax[start_idx] - ws.soc_rhs[offset] - fact1 * y[start_idx]
        s[start_idx] = s_raw_t
        norm_s = 0.0
        for j in (start_idx + 1):end_idx
            s_raw_j = Ax[j] - ws.soc_rhs[offset + (j - start_idx)] - fact1 * y[j]
            s[j] = s_raw_j
            norm_s += s_raw_j^2
        end
        norm_s = sqrt(norm_s)

        if norm_s <= -s_raw_t
            s[start_idx] = 0.0
            y_bar[start_idx] = fact2 * (-s_raw_t)
            dy[start_idx] = y_bar[start_idx] - y[start_idx]
            y[start_idx] = muladd(Halpern_fact2, 2.0 * y_bar[start_idx] - y[start_idx], Halpern_fact1 * last_y[start_idx])

            for j in (start_idx + 1):end_idx
                s_raw_j = s[j]
                s[j] = 0.0
                y_bar[j] = fact2 * (-s_raw_j)
                dy[j] = y_bar[j] - y[j]
                y[j] = muladd(Halpern_fact2, 2.0 * y_bar[j] - y[j], Halpern_fact1 * last_y[j])
            end
        elseif norm_s <= s_raw_t
            y_bar[start_idx] = 0.0
            dy[start_idx] = -y[start_idx]
            y[start_idx] = muladd(Halpern_fact2, -y[start_idx], Halpern_fact1 * last_y[start_idx])

            for j in (start_idx + 1):end_idx
                y_bar[j] = 0.0
                dy[j] = -y[j]
                y[j] = muladd(Halpern_fact2, -y[j], Halpern_fact1 * last_y[j])
            end
        else
            proj_t = (norm_s + s_raw_t) / 2.0
            alpha = proj_t / norm_s

            s[start_idx] = proj_t
            y_bar[start_idx] = fact2 * (proj_t - s_raw_t)
            dy[start_idx] = y_bar[start_idx] - y[start_idx]
            y[start_idx] = muladd(Halpern_fact2, 2.0 * y_bar[start_idx] - y[start_idx], Halpern_fact1 * last_y[start_idx])

            for j in (start_idx + 1):end_idx
                s_raw_j = s[j]
                proj_j = alpha * s_raw_j
                s[j] = proj_j
                y_bar[j] = fact2 * (proj_j - s_raw_j)
                dy[j] = y_bar[j] - y[j]
                y[j] = muladd(Halpern_fact2, 2.0 * y_bar[j] - y[j], Halpern_fact1 * last_y[j])
            end
        end
    end
end

# CPU version of update_w2 - completes the w update using y_bar
# Mirrors unified_update_w2_kernel - computes AT*y_bar and updates w
function update_w2_cpu!(ws::HPRSOCP_workspace_cpu,
    qp::QP_info_cpu,
    Halpern_fact1::Float64,
    Halpern_fact2::Float64)
    # Compute ATy_bar
    mul!(ws.ATy_bar, ws.AT, ws.y_bar)

    # Determine Q type
    Q_is_diag = ws.Q_is_diag

    # Determine the fact scalar for w update
    fact_scalar = ws.sigma / (1.0 + ws.sigma * ws.lambda_max_Q)

    w = ws.w
    w_bar = ws.w_bar
    dw = ws.dw
    ATy = ws.ATy
    ATy_bar = ws.ATy_bar
    ATdy = ws.ATdy
    last_w = ws.last_w
    last_ATy = ws.last_ATy
    fact_vec = ws.fact

    if ws.to_check
        @simd for i in eachindex(w)
            @inbounds begin
                w_i = w[i]
                w_bar_i = w_bar[i]
                ATy_i = ATy[i]
                ATy_bar_i = ATy_bar[i]
                last_w_i = last_w[i]
                last_ATy_i = last_ATy[i]

                fact = if Q_is_diag
                    fact_vec[i]
                else
                    fact_scalar
                end

                # Second part of w_bar update: add the AT*y_bar correction
                w_bar_new = w_bar_i + fact * (ATy_bar_i - ATy_i)

                # Complete w update with Halpern averaging
                w_new = Halpern_fact1 * last_w_i + Halpern_fact2 * (2.0 * w_bar_new - w_i)

                # Complete ATy update with Halpern averaging
                ATy_new = Halpern_fact1 * last_ATy_i + Halpern_fact2 * (2.0 * ATy_bar_i - ATy_i)

                w[i] = w_new
                ATy[i] = ATy_new
                w_bar[i] = w_bar_new
                dw[i] = w_bar_new - w_i
                ATdy[i] = ATy_bar_i - ATy_i
            end
        end
    else
        @simd for i in eachindex(w)
            @inbounds begin
                w_i = w[i]
                w_bar_i = w_bar[i]
                ATy_i = ATy[i]
                ATy_bar_i = ATy_bar[i]
                last_w_i = last_w[i]
                last_ATy_i = last_ATy[i]

                fact = if Q_is_diag
                    fact_vec[i]
                else
                    fact_scalar
                end

                # Second part of w_bar update: add the AT*y_bar correction
                w_bar_new = w_bar_i + fact * (ATy_bar_i - ATy_i)

                # Complete w update with Halpern averaging
                w_new = Halpern_fact1 * last_w_i + Halpern_fact2 * (2.0 * w_bar_new - w_i)

                # Complete ATy update with Halpern averaging
                ATy_new = Halpern_fact1 * last_ATy_i + Halpern_fact2 * (2.0 * ATy_bar_i - ATy_i)

                w[i] = w_new
                ATy[i] = ATy_new
            end
        end
    end
end

# CPU version of compute_Rd
function compute_Rd_cpu!(ws::HPRSOCP_workspace_cpu, sc::Scaling_info_cpu)
    # Call unified version
    compute_Rd!(ws, sc)
end

function compute_Rd_cpu!(ws::HPRSOCP_workspace_cpu, qp::QP_info_cpu, sc::Scaling_info_cpu)
    compute_Rd!(ws, qp, sc)
end

# CPU version of compute_Rp
function compute_Rp_cpu!(ws::HPRSOCP_workspace_cpu, sc::Scaling_info_cpu)
    # Call unified version
    compute_Rp!(ws, sc)
end
