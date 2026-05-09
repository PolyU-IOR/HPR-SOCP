### This file contains the main algorithm for solving quadratic programming problems using the HPR-SOCP method on GPU.

# The package is used to solve convex quadratic programming (QP) with HPR method in the paper
# HPR-SOCP: A dual Halpern Peaceman–Rachford method for solving large scale convex composite quadratic programming
# The package is developed by Kaihuang Chen · Defeng Sun · Yancheng Yuan · Guojun Zhang · Xinyuan Zhao.

# Quadratic Programming (QP) problem formulation:
#
#     minimize    (1/2) x' Q x + c' x
#     subject to  AL ≤ Ax ≤ AU
#                   l ≤ x ≤ u
#
# where:
#   - Q is a symmetric positive semidefinite matrix (n x n)
#   - c is a vector (n)
#   - A is a constraint matrix (m x n)
#   - l, u are vectors (m), lower and upper bounds for constraints
#   - x is the variable vector (n)
#

# ============================================================================
# Unified Algorithm Functions (CPU and GPU)
# ============================================================================
# These functions use multiple dispatch based on workspace type to handle
# both CPU and GPU implementations. They follow the Q operator pattern.
# ============================================================================

const SOC_DUAL_SUPPORT_CHECK_RTOL = 1e-10

function _max_soc_dual_support_violation(ws::HPRSOCP_workspace_cpu)
    max_violation = 0.0
    for i in 1:ws.number_SOC_con
        start_idx = ws.SOC_con_idx[i]
        end_idx = ws.SOC_con_idx[i+1] - 1
        t = ws.y_bar[start_idx]
        norm_s_sq = 0.0
        for j in (start_idx+1):end_idx
            norm_s_sq += ws.y_bar[j]^2
        end
        max_violation = max(max_violation, sqrt(norm_s_sq) - t)
    end
    return max(0.0, max_violation)
end

function _max_soc_dual_support_violation(ws::HPRSOCP_workspace_gpu)
    if ws.number_SOC_con == 0
        return 0.0
    end

    ws.SOC_norms_temp .= 0.0
    threads, blocks = gpu_launch_config(ws.number_SOC_con)
    if threads > 0
        @cuda threads = threads blocks = blocks compute_soc_dual_support_violation_kernel!(
            ws.SOC_norms_temp, ws.y_bar, ws.SOC_con_idx, ws.number_SOC_con
        )
        CUDA.synchronize()
    end
    return unified_absmax(ws.SOC_norms_temp)
end

function _log_soc_dual_support_check(ws::HPRSOCP_workspace, iter::Int)
    if ws.number_SOC_con == 0
        return
    end

    max_violation = _max_soc_dual_support_violation(ws)
    tol = 1e-14
    if max_violation > tol
        println(@sprintf(
            "SOC support check at iter %d: delta_{K_soc}^*(-y2) assumed zero, but max cone violation is %.6e",
            iter,
            max_violation,
        ))
    end
end

function _compute_primal_residual_components(
    ws::HPRSOCP_workspace,
    sc::HPRSOCP_scaling,
)
    if ws.m == 0
        return 0.0, 0.0, 0.0
    end

    denom = 1.0 + max(sc.norm_b_org, unified_absmax(ws.Ax))
    linear_m = ws.number_eq + ws.number_ineq

    err_total = unified_absmax(ws.Rp) / denom
    err_linear = linear_m > 0 ? unified_absmax_range(ws.Rp, 1, linear_m) / denom : 0.0
    err_soc = linear_m < ws.m ? unified_absmax_range(ws.Rp, linear_m + 1, ws.m) / denom : 0.0

    return err_total, err_linear, err_soc
end

@inline _unscale_primal_bar(v_bar::AbstractVector, col_norm::AbstractVector, b_scale::Float64) =
    b_scale .* (v_bar ./ col_norm)

@inline _unscale_dual_bar(v_bar::AbstractVector, row_norm::AbstractVector, c_scale::Float64) =
    c_scale .* (v_bar ./ row_norm)

@inline _unscale_reduced_cost_bar(z_bar::AbstractVector, col_norm::AbstractVector, c_scale::Float64) =
    c_scale .* (z_bar .* col_norm)

"""
    compute_residuals!(ws, qp, sc, res, params, iter)

Compute residuals for the HPR-SOCP algorithm. Unified function that dispatches
to appropriate implementation based on workspace type.

Uses unified operations (unified_dot, unified_norm) that dispatch based on array types.
GPU-specific operations (kernels) are handled via dispatch on workspace type.
"""
function compute_residuals!(
    ws::HPRSOCP_workspace,
    qp::HPRSOCP_QP_info,
    sc::HPRSOCP_scaling,
    res::HPRSOCP_residuals,
    params::HPRSOCP_parameters,
    iter::Int
)
    skip_q_residual = !has_quadratic_terms(qp.Q)

    ### Objective values
    # Use unified Qmap! function (already supports both operators and sparse matrices via dispatch)
    # Pass spmv_Q for GPU sparse matrices to use preprocessed CUSPARSE
    qx_dot = 0.0
    if !skip_q_residual
        if isa(qp.Q, CuSparseMatrixCSR{Float64,Int32}) && isa(ws, HPRSOCP_workspace_gpu)
            Qmap!(ws.x_bar, ws.Qx, qp.Q, ws.spmv_Q)
        else
            Qmap!(ws.x_bar, ws.Qx, qp.Q)
        end
        qx_dot = unified_dot(ws.x_bar, ws.Qx)
    end

    # Compute primal and dual objectives using unified_dot
    res.primal_obj_bar = sc.b_scale * sc.c_scale *
                         (unified_dot(ws.c, ws.x_bar) + 0.5 * qx_dot) + qp.obj_constant

    # Dual objective: always include z'x term. Constraint contributions are now
    # computed directly as y'(s + soc_rhs_full), where soc_rhs_full is aligned
    # with all model rows and zero on non-SOC rows.
    res.dual_obj_bar = sc.b_scale * sc.c_scale *
                       (-0.5 * qx_dot + unified_dot(ws.z_bar, ws.x_bar)) + qp.obj_constant
    if ws.m > 0
        res.dual_obj_bar += sc.b_scale * sc.c_scale * (unified_dot(ws.y_bar, ws.s) + unified_dot(ws.y_bar, ws.soc_rhs_full))
    end
    # _log_soc_dual_support_check(ws, iter)

    res.rel_gap_bar = abs(res.primal_obj_bar - res.dual_obj_bar) /
                      (1.0 + max(abs(res.primal_obj_bar), abs(res.dual_obj_bar)))

    ### Dual residuals
    compute_Rd!(ws, sc, skip_q_residual)
    qx_inf = skip_q_residual ? 0.0 : unified_absmax(ws.Qx)
    res.err_Rd_org_bar = unified_absmax(ws.Rd) /
                         (1.0 + max(sc.norm_c_org, unified_absmax(ws.ATdy), qx_inf))

    ### Primal residuals
    if ws.m > 0
        compute_Rp!(ws, sc)
        res.err_Rp_org_bar, res.err_Rp_linear_org_bar, res.err_Rp_soc_org_bar =
            _compute_primal_residual_components(ws, sc)
    else
        res.err_Rp_org_bar = 0.0
        res.err_Rp_linear_org_bar = 0.0
        res.err_Rp_soc_org_bar = 0.0
    end

    # Compute bounds violations at iteration 0 (device-specific via dispatch)
    if iter == 0
        compute_bounds_violation!(ws, sc, res)
    end

    res.KKTx_and_gap_org_bar = max(res.err_Rp_org_bar, res.err_Rd_org_bar, res.rel_gap_bar)

    # Track the best iterate in memory so restarts can roll back to it even
    # when HDF5 auto-save is disabled.
    if iter == 0 || res.KKTx_and_gap_org_bar < max(ws.saved_state.save_err_Rp,
        ws.saved_state.save_err_Rd,
        ws.saved_state.save_rel_gap)
        ws.saved_state.save_x .= ws.x_bar
        ws.saved_state.save_y .= ws.y_bar
        ws.saved_state.save_z .= ws.z_bar
        if !skip_q_residual
            ws.saved_state.save_w .= ws.w_bar
        end
        ws.saved_state.save_sigma = ws.sigma
        ws.saved_state.save_iter = iter
        ws.saved_state.save_err_Rp = res.err_Rp_org_bar
        ws.saved_state.save_err_Rd = res.err_Rd_org_bar
        ws.saved_state.save_primal_obj = res.primal_obj_bar
        ws.saved_state.save_dual_obj = res.dual_obj_bar
        ws.saved_state.save_rel_gap = res.rel_gap_bar
    end
end

# GPU-specific bounds violation using kernel
function compute_bounds_violation!(ws::HPRSOCP_workspace_gpu, sc::Scaling_info_gpu, res::HPRSOCP_residuals)
    threads, blocks = gpu_launch_config(ws.n)
    if threads > 0
        @cuda threads = threads blocks = blocks compute_err_lu_kernel!(ws.dx, ws.x_bar, ws.l, ws.u, sc.col_norm, sc.b_scale, ws.n)
    end
    if ws.number_SOC_var > 0
        soc_threads, soc_blocks = gpu_launch_config(ws.number_SOC_var)
        if soc_threads > 0
            @cuda threads = soc_threads blocks = soc_blocks compute_err_soc_kernel!(
                ws.dx, ws.x_bar, ws.SOC_var_idx, sc.col_norm, sc.b_scale, ws.number_SOC_var
            )
        end
    end
    res.err_Rp_org_bar = max(res.err_Rp_org_bar, unified_absmax(ws.dx))
end

# CPU-specific bounds violation using loop
function compute_bounds_violation!(ws::HPRSOCP_workspace_cpu, sc::Scaling_info_cpu, res::HPRSOCP_residuals)
    # Compute bounds violations: max(l - x, 0) for lower, max(x - u, 0) for upper
    lower_violation = max.(ws.l .- ws.x_bar, 0.0)
    upper_violation = max.(ws.x_bar .- ws.u, 0.0)
    ws.dx .= (lower_violation .+ upper_violation) .* (sc.b_scale ./ sc.col_norm)

    if ws.number_SOC_var > 0
        for i in 1:ws.number_SOC_var
            start_idx = ws.SOC_var_idx[i]
            end_idx = ws.SOC_var_idx[i+1] - 1
            t = ws.x_bar[start_idx]
            norm_s = 0.0
            ws.dx[start_idx] = t
            for j in (start_idx+1):end_idx
                ws.dx[j] = ws.x_bar[j]
                norm_s += ws.x_bar[j]^2
            end
            norm_s = sqrt(norm_s)

            if norm_s <= -t
                # leave dx as-is
            elseif norm_s <= t
                for j in start_idx:end_idx
                    ws.dx[j] = 0.0
                end
            else
                fact = (1 + t / norm_s) / 2
                ws.dx[start_idx] = (norm_s + t) / 2 - t
                for j in (start_idx+1):end_idx
                    ws.dx[j] = fact * ws.dx[j] - ws.dx[j]
                end
            end

            for j in start_idx:end_idx
                ws.dx[j] *= sc.b_scale / sc.col_norm[j]
            end
        end
    end

    res.err_Rp_org_bar = max(res.err_Rp_org_bar, unified_absmax(ws.dx))
end

# ============================================================================
# Legacy GPU-specific version (kept for backward compatibility, calls unified version)
# ============================================================================

# This function computes the residuals for the HPR-SOCP algorithm on GPU.
function compute_residuals_gpu!(ws::HPRSOCP_workspace_gpu,
    qp::QP_info_gpu,
    sc::Scaling_info_gpu,
    res::HPRSOCP_residuals,
    params::HPRSOCP_parameters,
    iter::Int,
)
    # Call unified version
    compute_residuals!(ws, qp, sc, res, params, iter)
end


# ============================================================================
# Unified save_state_to_hdf5! function
# ============================================================================

"""
    save_state_to_hdf5!(filename, ws, sc, residuals, params, iter, t_start_alg)

Unified implementation of HDF5 state saving for both GPU and CPU workspaces.

Saves the current algorithm state, best solution found so far, and algorithm
parameters to an HDF5 file. This function uses multiple dispatch via the
`to_cpu` helper to automatically transfer GPU arrays to CPU for file I/O
while being a no-op for CPU workspaces.

# Arguments
- `filename::String`: Path to HDF5 file to create/overwrite
- `ws`: Workspace (HPRSOCP_workspace_gpu or HPRSOCP_workspace_cpu)
- `sc`: Scaling information (Scaling_info_gpu or Scaling_info_cpu)
- `residuals::HPRSOCP_residuals`: Current residuals
- `params::HPRSOCP_parameters`: Algorithm parameters
- `iter::Int`: Current iteration number
- `t_start_alg::Float64`: Algorithm start time

# File Structure
The HDF5 file contains three main groups:
- `current/`: Current iteration state (solution, residuals, iteration info)
- `best/`: Best solution found so far (solution, residuals)
- `parameters/`: Algorithm parameters and settings

All solution vectors are saved in their original (unscaled) space.
"""
function save_state_to_hdf5!(
    filename::String,
    ws::Union{HPRSOCP_workspace_gpu,HPRSOCP_workspace_cpu},
    sc::Union{Scaling_info_gpu,Scaling_info_cpu},
    residuals::HPRSOCP_residuals,
    params::HPRSOCP_parameters,
    iter::Int,
    t_start_alg::Float64,
)
    skip_q_state = !has_quadratic_terms(ws.Q)
    # Transfer arrays to CPU (no-op for CPU workspace)
    x_bar = to_cpu(ws.x_bar)
    y_bar = to_cpu(ws.y_bar)
    z_bar = to_cpu(ws.z_bar)
    w_bar = skip_q_state ? x_bar : to_cpu(ws.w_bar)
    save_x = to_cpu(ws.saved_state.save_x)
    save_y = to_cpu(ws.saved_state.save_y)
    save_z = to_cpu(ws.saved_state.save_z)
    save_w = skip_q_state ? save_x : to_cpu(ws.saved_state.save_w)
    col_norm = to_cpu(sc.col_norm)
    row_norm = to_cpu(sc.row_norm)

    # Section 3.1 maps: x = s_b D_C xbar, w = s_b D_C wbar,
    # y = s_c D_R ybar, z = s_c D_C^{-1} zbar.
    x_bar_scaled = _unscale_primal_bar(x_bar, col_norm, sc.b_scale)
    w_bar_scaled = _unscale_primal_bar(w_bar, col_norm, sc.b_scale)
    save_x_scaled = _unscale_primal_bar(save_x, col_norm, sc.b_scale)
    save_w_scaled = _unscale_primal_bar(save_w, col_norm, sc.b_scale)

    if ws.m > 0
        y_bar_scaled = _unscale_dual_bar(y_bar, row_norm, sc.c_scale)
        save_y_scaled = _unscale_dual_bar(save_y, row_norm, sc.c_scale)
        z_bar_scaled = _unscale_reduced_cost_bar(z_bar, col_norm, sc.c_scale)
        save_z_scaled = _unscale_reduced_cost_bar(save_z, col_norm, sc.c_scale)
    else
        # For problems without constraints (m=0), y and dual variables are empty or not used
        y_bar_scaled = Float64[]
        save_y_scaled = Float64[]
        z_bar_scaled = _unscale_reduced_cost_bar(z_bar, col_norm, sc.c_scale)
        save_z_scaled = _unscale_reduced_cost_bar(save_z, col_norm, sc.c_scale)
    end

    # Create or open HDF5 file
    if isfile(filename)
        rm(filename, force=true)
    end
    h5open(filename, "w") do file
        # Save current iteration info
        file["current/iteration"] = iter
        file["current/time_elapsed"] = time() - t_start_alg
        file["current/timestamp"] = string(Dates.now())

        # Save current solution (scaled)
        file["current/x_org"] = x_bar_scaled
        file["current/w_org"] = w_bar_scaled
        if ws.m > 0
            file["current/y_org"] = y_bar_scaled
        end
        file["current/z_org"] = z_bar_scaled
        file["current/sigma"] = ws.sigma

        # Save current residuals
        file["current/err_Rp"] = residuals.err_Rp_org_bar
        file["current/err_Rd"] = residuals.err_Rd_org_bar
        file["current/primal_obj"] = residuals.primal_obj_bar
        file["current/dual_obj"] = residuals.dual_obj_bar
        file["current/rel_gap"] = residuals.rel_gap_bar
        file["current/KKTx_and_gap"] = residuals.KKTx_and_gap_org_bar

        # Save best solution so far (scaled)
        file["best/x_org"] = save_x_scaled
        file["best/w_org"] = save_w_scaled
        if ws.m > 0
            file["best/y_org"] = save_y_scaled
        end
        file["best/z_org"] = save_z_scaled
        file["best/sigma"] = ws.saved_state.save_sigma
        file["best/iteration"] = ws.saved_state.save_iter

        # Save best residuals
        file["best/err_Rp"] = ws.saved_state.save_err_Rp
        file["best/err_Rd"] = ws.saved_state.save_err_Rd
        file["best/primal_obj"] = ws.saved_state.save_primal_obj
        file["best/dual_obj"] = ws.saved_state.save_dual_obj
        file["best/rel_gap"] = ws.saved_state.save_rel_gap
        file["best/KKTx_and_gap"] = max(ws.saved_state.save_err_Rp, ws.saved_state.save_err_Rd, ws.saved_state.save_rel_gap)

        # Save parameters
        file["parameters/stoptol"] = params.stoptol
        file["parameters/sigma"] = params.sigma
        file["parameters/max_iter"] = params.max_iter
        file["parameters/time_limit"] = params.time_limit
        file["parameters/check_iter"] = params.check_iter
        file["parameters/warm_up"] = params.warm_up
        file["parameters/print_frequency"] = params.print_frequency
        file["parameters/device_number"] = params.device_number
        file["parameters/use_Ruiz_scaling"] = params.use_Ruiz_scaling
        file["parameters/ruiz_iterations"] = params.ruiz_iterations
        file["parameters/use_bc_scaling"] = params.use_bc_scaling
        file["parameters/bc_scaling_norm_type"] = String(params.bc_scaling_norm_type)
        file["parameters/use_l2_scaling"] = params.use_l2_scaling
        file["parameters/use_Pock_Chambolle_scaling"] = params.use_Pock_Chambolle_scaling
        file["parameters/soc_block_scaling_strategy"] = String(params.soc_block_scaling_strategy)
        file["parameters/auto_save"] = params.auto_save

        # Save initial solutions if provided
        if params.initial_x !== nothing
            file["parameters/initial_x"] = params.initial_x
        end
        if params.initial_y !== nothing
            file["parameters/initial_y"] = params.initial_y
        end
    end

    if params.verbose
        println(@sprintf("State saved to %s at iteration %d", filename, iter))
    end
end

# ============================================================================
# Unified Algorithm Functions (Block 2 Refactoring)
# ============================================================================

function sigma_gap_ratio(current_gap::Float64, best_gap::Float64, residual_indicator::Float64)
    effective_current = max(current_gap, residual_indicator)
    effective_best = max(best_gap, residual_indicator)
    return effective_current / effective_best
end

function sigma_reference_ok(relative_drift::Float64, current_progress::Float64, best_progress::Float64)
    if !isfinite(relative_drift) || !isfinite(current_progress) || !isfinite(best_progress)
        return false
    end
    return relative_drift <= 0.25 && current_progress <= max(5.0 * best_progress, 1e-12)
end

function sigma_estimation_is_stable(sigma_old::Float64, sigma_estimation::Float64)
    if !(sigma_old > 0.0) || !(sigma_estimation > 0.0)
        return false
    end
    return max(sigma_estimation / sigma_old, sigma_old / sigma_estimation) <= 10.0
end

function sigma_blend_candidate(
    sigma_old::Float64,
    sigma_estimation::Float64,
    best_sigma::Float64,
    gap_ratio::Float64,
    best_iter::Int,
    reference_ok::Bool,
)
    if !reference_ok || sigma_estimation_is_stable(sigma_old, sigma_estimation)
        return sigma_estimation
    end

    blend_weight = exp(-gap_ratio / (max(best_iter, 0) + 2))
    return exp(
        blend_weight * log(sigma_estimation) +
        (1 - blend_weight) * log(best_sigma)
    )
end

const SIGMA_REBALANCE_EXTREME_IMBALANCE = 1.0e7
const SAVED_RESTART_PRIMAL_RATIO_WORSE = 5.0
const SAVED_RESTART_PRIMAL_ERR_WORSE = 5.0
const SAVED_RESTART_DUAL_IMPROVEMENT_FLOOR = 0.5
const SAVED_RESTART_GAP_IMPROVEMENT_FLOOR = 0.8
const SAVED_RESTART_ALL_WORSE_FACTOR = 1.2
const SAVED_RESTART_GAP_WORSE_FACTOR = 1.05

function sigma_infeasibility_imbalance(err_rp::Float64, err_rd::Float64)
    lo = max(min(abs(err_rp), abs(err_rd)), 1.0e-30)
    hi = max(abs(err_rp), abs(err_rd))
    return hi / lo
end

function sigma_finish_ratio(finish_progress::Float64, stoptol::Float64)
    if !isfinite(finish_progress)
        return Inf
    end
    return max(finish_progress, 0.0) / max(stoptol, 1.0e-12)
end

function sigma_rebalance_factor(
    ratio_infeas_org::Float64,
    power::Float64,
)
    if !isfinite(ratio_infeas_org) || power <= 0.0
        return 1.0
    end
    return clamp(ratio_infeas_org^power, 1e-2, 1e2)
end

function sigma_rebalance_power(
    finish_progress::Float64,
    stoptol::Float64,
    imbalance_ratio::Float64=1.0,
)
    if imbalance_ratio >= SIGMA_REBALANCE_EXTREME_IMBALANCE
        return 1.0
    end
    finish_ratio = sigma_finish_ratio(finish_progress, stoptol)
    if !isfinite(finish_ratio) || finish_ratio > 100.0
        return 0.0
    elseif finish_ratio > 10.0
        return 0.5
    end
    return 1.0
end

function has_saved_restart_state(ws::HPRSOCP_workspace)
    return isfinite(max(
        ws.saved_state.save_err_Rp,
        ws.saved_state.save_err_Rd,
        ws.saved_state.save_rel_gap,
    ))
end

function should_backtrack_saved_best_state(
    err_rp::Float64,
    err_rd::Float64,
    rel_gap::Float64,
    save_err_rp::Float64,
    save_err_rd::Float64,
    save_rel_gap::Float64,
)
    values = (
        err_rp,
        err_rd,
        rel_gap,
        save_err_rp,
        save_err_rd,
        save_rel_gap,
    )
    all(isfinite, values) || return false

    current_kkt = max(err_rp, err_rd, rel_gap)
    saved_kkt = max(save_err_rp, save_err_rd, save_rel_gap)
    current_kkt > 1.05 * saved_kkt || return false

    current_primal_ratio = err_rp / max(err_rd, 1.0e-30)
    saved_primal_ratio = save_err_rp / max(save_err_rd, 1.0e-30)

    primal_dominance_worse =
        current_primal_ratio >= SAVED_RESTART_PRIMAL_RATIO_WORSE * max(saved_primal_ratio, 1.0) ||
        err_rp >= SAVED_RESTART_PRIMAL_ERR_WORSE * max(save_err_rp, 1.0e-30)
    dual_not_improved =
        err_rd >= max(SAVED_RESTART_DUAL_IMPROVEMENT_FLOOR * save_err_rd, 1.0e-16)
    gap_not_improved =
        rel_gap >= max(SAVED_RESTART_GAP_IMPROVEMENT_FLOOR * save_rel_gap, 1.0e-12)

    all_worse =
        err_rp >= SAVED_RESTART_ALL_WORSE_FACTOR * max(save_err_rp, 1.0e-30) &&
        err_rd >= SAVED_RESTART_ALL_WORSE_FACTOR * max(save_err_rd, 1.0e-30) &&
        rel_gap >= SAVED_RESTART_GAP_WORSE_FACTOR * max(save_rel_gap, 1.0e-12)

    return (primal_dominance_worse && dual_not_improved && gap_not_improved) || all_worse
end

function maybe_trigger_saved_best_backtrack!(
    restart_info::HPRSOCP_restart,
    ws::HPRSOCP_workspace,
    residuals::HPRSOCP_residuals,
)
    if restart_info.restart_flag <= 0 || !has_saved_restart_state(ws)
        return restart_info
    end
    if ws.saved_state.save_iter <= 0
        return restart_info
    end

    if should_backtrack_saved_best_state(
        residuals.err_Rp_org_bar,
        residuals.err_Rd_org_bar,
        residuals.rel_gap_bar,
        ws.saved_state.save_err_Rp,
        ws.saved_state.save_err_Rd,
        ws.saved_state.save_rel_gap,
    )
        restart_info.restart_flag = max(restart_info.restart_flag, 5)
    end
    return restart_info
end

function should_restore_best_restart_state(
    restart_info::HPRSOCP_restart,
    ws::HPRSOCP_workspace,
)
    return (restart_info.restart_flag == 3 &&
            isfinite(restart_info.best_kkt) &&
            restart_info.best_kkt <= 1e-6 &&
            has_saved_restart_state(ws)) ||
           (restart_info.restart_flag == 5 &&
            has_saved_restart_state(ws))
end

function restore_best_restart_state!(
    ws::HPRSOCP_workspace,
    sigma_override::Union{Nothing,Float64}=nothing,
)
    sigma_old = ws.sigma
    has_quadratic = has_quadratic_terms(ws.Q)

    ws.x .= ws.saved_state.save_x
    ws.x_bar .= ws.saved_state.save_x
    ws.last_x .= ws.saved_state.save_x

    if has_quadratic
        ws.w .= ws.saved_state.save_w
        ws.w_bar .= ws.saved_state.save_w
        ws.last_w .= ws.saved_state.save_w
    end

    ws.z_bar .= ws.saved_state.save_z
    ws.sigma = isnothing(sigma_override) ? ws.saved_state.save_sigma : sigma_override

    if ws.m > 0
        ws.y .= ws.saved_state.save_y
        ws.y_bar .= ws.saved_state.save_y
        ws.last_y .= ws.saved_state.save_y

        if has_quadratic
            unified_mul!(ws.ATy_bar, ws.AT, ws.y_bar)
            ws.ATy .= ws.ATy_bar
            ws.last_ATy .= ws.ATy_bar
        end
    end

    if has_quadratic
        Qmap!(ws.w_bar, ws.Qw_bar, ws.Q)
        ws.Qw .= ws.Qw_bar
        ws.last_Qw .= ws.Qw_bar
    end

    if ws.Q_is_diag && has_quadratic && abs(sigma_old - ws.sigma) > 1e-15
        unified_update_Q_factors!(
            ws.fact2, ws.fact, ws.fact1, ws.fact_M,
            ws.diag_Q, ws.sigma
        )
    end

    return ws
end

function reset_restart_phase!(restart_info::HPRSOCP_restart, sigma::Float64)
    restart_info.best_gap = restart_info.current_gap
    restart_info.best_kkt = Inf
    restart_info.best_sigma = sigma
    restart_info.best_iter = 0
    return restart_info
end

restart_backtrack_window(check_iter::Int) = max(400, 2 * check_iter)

function restart_long_window(iter::Int, fact::Float64, best_kkt::Float64)
    window = max(1000, round(Int, fact * iter))
    if isfinite(best_kkt)
        if best_kkt <= 1e-7
            return min(window, 2000)
        elseif best_kkt <= 1e-6
            return min(window, 5000)
        end
    end
    return window
end

function restart_backtrack_ratio(best_kkt::Float64)
    if best_kkt > 1e-2
        return 1.5
    elseif best_kkt > 1e-4
        return 1.35
    else
        return 1.2
    end
end

function should_refresh_best_sigma(
    current_gap::Float64,
    best_gap::Float64,
    current_kkt::Float64,
    best_kkt::Float64,
    best_iter::Int,
)
    improved_kkt = current_kkt < best_kkt
    compatible_gap = current_gap < best_gap && current_kkt <= 1.01 * best_kkt
    periodic_refresh = best_iter >= 10 && current_kkt <= 1.2 * best_kkt
    return improved_kkt || compatible_gap || periodic_refresh
end

function should_trigger_kkt_backtrack(
    current_gap::Float64,
    best_gap::Float64,
    current_kkt::Float64,
    best_kkt::Float64,
    inner::Int,
    check_iter::Int,
)
    if !isfinite(best_kkt) || inner < restart_backtrack_window(check_iter)
        return false
    end
    if !(current_gap > 1.2 * best_gap)
        return false
    end
    return current_kkt > restart_backtrack_ratio(best_kkt) * best_kkt
end

"""
    update_sigma!(params, restart_info, ws, qp, residuals)

Unified implementation of adaptive sigma update for both GPU and CPU.
Updates the primal penalty parameter sigma based on the progress of the algorithm.

This function uses multiple dispatch and unified operations to work with both
GPU and CPU workspaces without code duplication.

# Arguments
- `params::HPRSOCP_parameters`: Algorithm parameters
- `restart_info::HPRSOCP_restart`: Restart tracking information
- `ws::HPRSOCP_workspace`: Workspace (GPU or CPU)
- `qp::HPRSOCP_QP_info`: Problem data (GPU or CPU)
- `residuals::HPRSOCP_residuals`: Current residual values
"""

function update_sigma!(
    params::HPRSOCP_parameters,
    restart_info::HPRSOCP_restart,
    ws::HPRSOCP_workspace,
    qp::HPRSOCP_QP_info,
    residuals::HPRSOCP_residuals,
)
    # ------------------------------------------------------------------
    # Step 1. Exit conditions
    # Role:
    #   Skip sigma adaptation when sigma is fixed or restart is inactive.
    # ------------------------------------------------------------------
    if restart_info.restart_flag < 1
        return
    end

    verbose = params.verbose
    skip_q_sigma = !has_quadratic_terms(qp.Q)
    has_m = ws.m > 0
    q_is_diag = ws.Q_is_diag
    sigma_old = ws.sigma

    # if verbose
        # println("restart type is ", restart_info.restart_flag)
    # end

    use_gpu_qmap =
        !skip_q_sigma &&
        (ws isa HPRSOCP_workspace_gpu) &&
        (qp.Q isa CuSparseMatrixCSR{Float64,Int32})

    @inline function apply_Qmap!(src, dst)
        if use_gpu_qmap
            Qmap!(src, dst, qp.Q, ws.spmv_Q)
        else
            Qmap!(src, dst, qp.Q)
        end
    end

    SIGMA_MIN = 1.0e-12
    SIGMA_MAX = 1.0e12
    THETA_MAX = 1.0
    IMBALANCE_TRIGGER = 1.0e6
    SIGMA_CORRECTION_RATIO_FLOOR = 2.0
    THRESHOLD = 1.0e-8
    EPS = 1e-16
    LOG100 = log(100.0)

    sigma_lo = SIGMA_MIN
    sigma_hi = SIGMA_MAX

    # ------------------------------------------------------------------
    # Step 2. Model-based sigma proposal
    # Role:
    #   Compute sigma from a 1D surrogate model.
    #
    # Ref:
    #   The case sqrt(b/a) is the minimizer of aσ + b/σ, a standard scalar
    #   balancing model related in spirit to spectral / secant-type tuning.
    #   The general surrogate used here is customized for this solver.
    # ------------------------------------------------------------------

    if !skip_q_sigma
        @. ws.dw = ws.w_bar - ws.last_w
        apply_Qmap!(ws.dw, ws.dQw)
    end

    @. ws.dx = ws.x_bar - ws.last_x

    a = 0.0
    b = unified_dot(ws.dx, ws.dx)
    c = 0.0
    d = 0.0

    if has_m
        @. ws.dy = ws.y_bar - ws.last_y

        if !skip_q_sigma
            @. ws.ATdy = ws.ATy_bar - ws.last_ATy
            apply_Qmap!(ws.ATdy, ws.QATdy)
            a = ws.lambda_max_A * unified_dot(ws.dy, ws.dy) -
                2.0 * unified_dot(ws.dQw, ws.ATdy)
        else
            unified_mul!(ws.ATdy, ws.AT, ws.dy)
            a = ws.lambda_max_A * unified_dot(ws.dy, ws.dy)
        end
    end

    if !skip_q_sigma
        if q_is_diag
            nq = unified_norm(ws.dQw)
            a += nq * nq
        else
            a += ws.lambda_max_Q * unified_dot(ws.dw, ws.dQw)
            if has_m
                c = unified_dot(ws.ATdy, ws.QATdy)
                d = ws.lambda_max_Q
            end
        end
    end

    a = max(a, EPS)
    b = max(b, EPS)

    sigma_model =
        if skip_q_sigma
            sqrt(b / a)
        elseif q_is_diag
            if has_m
                unified_golden_Q_diag(
                    a, b, ws.diag_Q, ws.ATdy, ws.QATdy, ws.tempv;
                    lo=1e-12, hi=1e12, tol=1e-13,
                )
            else
                sqrt(b / a)
            end
        else
            if has_m
                golden(a, b, c, d; lo=1e-12, hi=1e12, tol=1e-13)
            else
                sqrt(b / a)
            end
        end

    # ------------------------------------------------------------------
    # Step 3. Historical stabilization
    # Role:
    #   Stabilize the current model-based sigma by blending it with the
    #   historically best sigma in log-scale.
    #
    # Ref:
    #   Log-scale interpolation is natural for positive scale parameters.
    #   The exact progress-ratio rule here is heuristic.
    # ------------------------------------------------------------------

    sigma_ref = restart_info.best_sigma
    current_kkt = residuals.KKTx_and_gap_org_bar
    current_gap = restart_info.current_gap
    best_kkt = restart_info.best_kkt
    best_gap = restart_info.best_gap

    kkt_ok = current_kkt <= 10.0 * best_kkt
    gap_ok = current_gap <= 10.0 * best_gap
    reference_ok = kkt_ok && gap_ok

    progress_ratio = sqrt((current_kkt / best_kkt) * (current_gap / best_gap))
    progress_ratio = clamp(progress_ratio, 1.0, 10.0)

    theta = 0.0
    sigma_base = sigma_model
    if reference_ok
        # theta = clamp(THETA_MAX * exp(1.0 - progress_ratio), 0.0, THETA_MAX)
        alpha_theta = 0.3
        theta = clamp(exp(-alpha_theta * (progress_ratio - 1.0)), 0.0, THETA_MAX)
        sigma_base = exp(theta * log(sigma_model) + (1.0 - theta) * log(sigma_ref))
        sigma_base = clamp(sigma_base, sigma_lo, sigma_hi)
    end


    # ------------------------------------------------------------------
    # Step 4. Residual-based correction
    # Role:
    #   Use primal-dual residual imbalance to further adjust sigma.
    #   If one side is much worse than the other, activate a correction state
    #   and set kappa ≈ (err_d / err_p)^beta, with hysteresis to avoid
    #   oscillatory switching.
    # ------------------------------------------------------------------

    err_p = max(residuals.err_Rp_org_bar, EPS)
    err_d = max(residuals.err_Rd_org_bar, EPS)

    err_small = min(err_p, err_d)
    err_large = max(err_p, err_d)
    imbalance_ratio = err_large / err_small

    both_finished = err_large <= THRESHOLD
    one_side_finished =
        (err_small <= 0.5 * THRESHOLD) &&
        (err_large >= 2.0 * THRESHOLD)
    extreme_imbalance = imbalance_ratio >= IMBALANCE_TRIGGER

    correction_trigger = !both_finished && (one_side_finished || extreme_imbalance)
    correction_keep =
        !both_finished &&
        ((err_small <= 0.8 * THRESHOLD) || (imbalance_ratio >= 0.1 * IMBALANCE_TRIGGER))

    current_dir = err_d >= err_p ? 1 : -1

    if both_finished
        restart_info.sigma_correction_active = false
        restart_info.sigma_correction_hold = -1
        restart_info.sigma_correction_dir = 0
    elseif correction_trigger || (restart_info.sigma_correction_active && correction_keep)
        restart_info.sigma_correction_active = true
        restart_info.sigma_correction_hold = 0
        restart_info.sigma_correction_dir = current_dir
    else
        restart_info.sigma_correction_active = false
        restart_info.sigma_correction_hold = -1
        restart_info.sigma_correction_dir = 0
    end

    beta = 0.0
    if !both_finished
        if one_side_finished
            beta = 0.5
        end
        if extreme_imbalance
            beta = 1.0
        elseif restart_info.sigma_correction_active
            beta = 0.5
        end
    end

    kappa = 1.0
    if beta > 0.0
        ratio_for_kappa = err_d / err_p
        if restart_info.sigma_correction_active
            if restart_info.sigma_correction_dir > 0
                ratio_for_kappa = max(ratio_for_kappa, SIGMA_CORRECTION_RATIO_FLOOR)
            elseif restart_info.sigma_correction_dir < 0
                ratio_for_kappa = min(ratio_for_kappa, 1.0 / SIGMA_CORRECTION_RATIO_FLOOR)
            end
        end
        kappa = exp(beta * clamp(log(ratio_for_kappa), -LOG100, LOG100))
    end

    # ------------------------------------------------------------------
    # Step 5. Final sigma update
    # Role:
    #   Combine model-based adaptation and residual correction.
    #
    # Ref:
    #   This is the synthesis step of the two ideas above; the extra
    #   directional push is a solver-specific heuristic.
    # ------------------------------------------------------------------
    sigma_new = sigma_base * kappa
    if restart_info.sigma_correction_active
        if restart_info.sigma_correction_dir > 0
            sigma_new = max(sigma_new, sigma_old * 1.5)
        elseif restart_info.sigma_correction_dir < 0
            sigma_new = min(sigma_new, sigma_old / 1.5)
        end
    end
    sigma_new = clamp(sigma_new, sigma_lo, sigma_hi)
    ws.sigma = sigma_new

    restart_info.best_iter += 1

    if q_is_diag && !skip_q_sigma && abs(sigma_old - sigma_new) > 1e-15
        unified_update_Q_factors!(
            ws.fact2, ws.fact, ws.fact1, ws.fact_M,
            ws.diag_Q, sigma_new
        )
    end

    # if verbose
    #     @printf("a=% .3e b=% .3e c=% .3e d=% .3e | sigma_mo=%.4e sigma_ref=%.4e | ref_ok=%s theta=%.2f sigma_base=%.4e \nbeta=%.2f kappa=%.2f | corr_act=%s corr_dir=%d | sigma: %.6e -> %.6e\n", a, b, c, d, sigma_model, sigma_ref, string(reference_ok), theta, sigma_base, beta, kappa, string(restart_info.sigma_correction_active), restart_info.sigma_correction_dir, sigma_old, sigma_new,)
    # end

    return
end
# ============================================================================
# Legacy GPU Wrapper (calls unified version)
# ============================================================================

# This function updates the penalty parameter (sigma) based on the current state of the algorithm.
function update_sigma_gpu!(params::HPRSOCP_parameters,
    restart_info::HPRSOCP_restart,
    ws::HPRSOCP_workspace_gpu,
    qp::QP_info_gpu,
    residuals::HPRSOCP_residuals,
)
    # Call unified implementation
    update_sigma!(params, restart_info, ws, qp, residuals)
end

# This function checks whether a restart is needed based on the current state of the algorithm.
function check_restart(restart_info::HPRSOCP_restart,
    iter::Int,
    check_iter::Int,
    sigma::Float64,
    verbose::Bool,
)
    return check_restart(restart_info, iter, check_iter, sigma, Inf, verbose)
end

function check_restart(
    restart_info::HPRSOCP_restart,
    iter::Int,
    check_iter::Int,
    sigma::Float64,
    current_kkt::Float64,
    verbose::Bool=true,
)
    restart_info.restart_flag = 0
    if restart_info.first_restart
        if iter == check_iter
            restart_info.first_restart = false
            restart_info.restart_flag = 1
            restart_info.weighted_norm = restart_info.current_gap
            restart_info.best_gap = restart_info.current_gap
            restart_info.best_kkt = current_kkt
            restart_info.best_sigma = sigma
        end
    else
        if rem(iter, check_iter) == 0
            if restart_info.current_gap < 0
                restart_info.current_gap = 1e-6
                if verbose
                    println("current_gap < 0")
                end
            end

            if restart_info.current_gap <= 0.36 * restart_info.last_gap
                restart_info.sufficient += 1
                restart_info.restart_flag = 1
            end

            if (restart_info.current_gap <= 0.8 * restart_info.last_gap) && (restart_info.current_gap > 1.00 * restart_info.save_gap)
                restart_info.necessary += 1
                restart_info.restart_flag = 2
            end

            # if restart_info.current_gap / restart_info.weighted_norm > 1e-2
            #     fact = 0.2
            # elseif restart_info.current_gap > 1e-9
            #     fact = 0.1
            # else
            # else
            fact = 0.15
            # end

            if ((restart_info.inner >= max(500, fact * iter) || (restart_info.inner >= 10000 && restart_info.current_gap <= 0.75 * restart_info.last_gap))) && (restart_info.current_gap > 0.9 * restart_info.save_gap)
                # if restart_info.inner >= max(1000, fact * iter)
                # if restart_info.inner >= restart_long_window(iter, fact, restart_info.best_kkt)
                restart_info.long += 1
                restart_info.restart_flag = 3
            end

            # if should_trigger_kkt_backtrack(
            #     restart_info.current_gap,
            #     restart_info.best_gap,
            #     current_kkt,
            #     restart_info.best_kkt,
            #     restart_info.inner,
            #     check_iter,
            # )
            #     restart_info.restart_flag = max(restart_info.restart_flag, 4)
            # end

            if should_refresh_best_sigma(
                restart_info.current_gap,
                restart_info.best_gap,
                current_kkt,
                restart_info.best_kkt,
                restart_info.best_iter,
            )
                restart_info.best_gap = restart_info.current_gap
                restart_info.best_kkt = current_kkt
                restart_info.best_sigma = sigma
                restart_info.best_iter = 0
                # if verbose
                #     println("New best gap: ", restart_info.best_gap, " at iteration ", iter, " with sigma ", sigma)
                # end
            end

            restart_info.save_gap = restart_info.current_gap
        end
    end
end

"""
    do_restart!(restart_info, ws, qp)

Unified implementation of restart operation for both GPU and CPU.

Performs a restart by resetting the algorithm state to the current averaged iterates.
This operation is triggered when the restart_flag is set (>0) by check_restart.

# What Happens During Restart
1. Set current iterates (x, w, y) to averaged iterates (x̄, w̄, ȳ)
2. Set last iterates (last_x, last_w, last_y) to averaged iterates
3. Recompute ATy_bar using the new y_bar
4. Update restart tracking information

# Arguments
- `restart_info::HPRSOCP_restart`: Restart tracking structure (modified in-place)
- `ws::HPRSOCP_workspace`: Workspace with iterate vectors (modified in-place)
- `qp::HPRSOCP_QP_info`: Problem data (used for matrix A)

# Why Restart?
Restarting helps when:
- Progress stalls (long periods without gap reduction)
- Gap reduces sufficiently (sufficient condition)
- Gap increases after recent progress (necessary condition)

# See Also
- `check_restart`: Determines when to restart based on gap progress
"""

function do_restart!(restart_info::HPRSOCP_restart,
    ws::HPRSOCP_workspace,
    qp::HPRSOCP_QP_info)
    if restart_info.restart_flag > 0
        has_quadratic = has_quadratic_terms(qp.Q)
        # Reset current iterates to main iterates
        ws.x .= ws.x_bar
        ws.last_x .= ws.x_bar
        if has_quadratic
            ws.w .= ws.w_bar
            ws.last_w .= ws.w_bar
        end

        # Handle constraint-related variables if constraints exist
        if ws.m > 0
            ws.y .= ws.y_bar
            ws.last_y .= ws.y_bar

            if has_quadratic
                # Q problems carry ATy as part of the Halpern state to save one SpMV.
                unified_mul!(ws.ATy_bar, ws.AT, ws.y_bar)
                ws.last_ATy .= ws.ATy_bar
                ws.ATy .= ws.ATy_bar
            end
        end
        if ws.noC && ws.number_SOC_con == 0 && ws.number_SOC_var == 0
            ws.Qw .= ws.Qw_bar
            ws.last_Qw .= ws.Qw_bar
        end

        # Update restart tracking information
        restart_info.last_gap = restart_info.current_gap
        restart_info.save_gap = Inf
        restart_info.times += 1
        restart_info.inner = 0
    end
end

# This function checks the stopping criteria for the HPR-SOCP algorithm on GPU.
function check_break(residuals::HPRSOCP_residuals,
    iter::Int,
    t_start_alg::Float64,
    params::HPRSOCP_parameters,
)
    if residuals.KKTx_and_gap_org_bar < params.stoptol
        return "OPTIMAL"
    end

    if iter == params.max_iter
        return "MAX_ITER"
    end

    if time() - t_start_alg > params.time_limit
        return "TIME_LIMIT"
    end

    return "CONTINUE"
end

"""
    collect_results!(ws, qp, sc, residuals, iter, t_start_alg, power_time)

Unified implementation for collecting and scaling final results from both GPU and CPU solvers.

This function:
1. Creates a new HPRSOCP_results object
2. Copies timing and iteration information
3. Scales and de-normalizes solution vectors (x, w, y, z)
4. Transfers GPU data to CPU if needed

# Arguments
- `ws::HPRSOCP_workspace`: Workspace containing solution vectors
- `qp::HPRSOCP_QP_info`: Problem data (used for dimension checking)
- `sc::HPRSOCP_scaling`: Scaling information for de-normalization
- `residuals::HPRSOCP_residuals`: Final residual values
- `iter::Int`: Final iteration count
- `t_start_alg::Float64`: Algorithm start time
- `power_time::Float64`: Total power measurement time (GPU only, default 0.0)

# Returns
- `HPRSOCP_results`: Results object with scaled solution and metadata

# Scaling Operations
- `x = b_scale * (x_bar ./ col_norm)`: Primal variable x
- `w = b_scale * (w_bar ./ col_norm)`: Slack variable w
- `y = c_scale * (y_bar ./ row_norm)`: Dual variable y
- `z = c_scale * (z_bar .* col_norm)`: Reduced cost z

# Note on GPU Transfer
For GPU workspaces, `to_cpu()` automatically transfers CuArrays to Arrays.
For CPU workspaces, it's a no-op that returns the arrays unchanged.
"""
function collect_results!(
    ws::HPRSOCP_workspace,
    qp::Union{HPRSOCP_QP_info,Nothing},
    sc::HPRSOCP_scaling,
    residuals::HPRSOCP_residuals,
    iter::Int,
    t_start_alg::Float64,
    power_time::Float64=0.0
)
    results = HPRSOCP_results()
    results.iter = iter
    results.time = time() - t_start_alg
    results.power_time = power_time
    results.residuals = residuals.KKTx_and_gap_org_bar
    results.primal_obj = residuals.primal_obj_bar
    results.gap = residuals.rel_gap_bar

    # Scale solution and transfer to CPU if needed
    # to_cpu() is a no-op for CPU arrays, transfers GPU arrays to CPU
    results.x = to_cpu(_unscale_primal_bar(ws.x_bar, sc.col_norm, sc.b_scale))
    results.w = has_quadratic_terms(ws.Q) ?
                to_cpu(_unscale_primal_bar(ws.w_bar, sc.col_norm, sc.b_scale)) :
                copy(results.x)

    # Handle dual variables (may be empty for unconstrained problems)
    if ws.m > 0
        results.y = to_cpu(_unscale_dual_bar(ws.y_bar, sc.row_norm, sc.c_scale))
    else
        results.y = Float64[]
    end

    results.z = to_cpu(_unscale_reduced_cost_bar(ws.z_bar, sc.col_norm, sc.c_scale))

    return results
end

# This function collects the results from the HPR-SOCP algorithm on GPU and prepares them for output.
function collect_results_gpu!(
    ws::HPRSOCP_workspace_gpu,
    residuals::HPRSOCP_residuals,
    sc::Scaling_info_gpu,
    iter::Int,
    t_start_alg::Float64,
    power_time::Float64,
)
    # Call unified implementation (qp not needed but kept for signature compatibility)
    return collect_results!(ws, nothing, sc, residuals, iter, t_start_alg, power_time)
end

# ============================================================================
# CPU Algorithm Functions
# ============================================================================

# CPU version of compute residuals
function compute_residuals_cpu!(ws::HPRSOCP_workspace_cpu,
    qp::QP_info_cpu,
    sc::Scaling_info_cpu,
    res::HPRSOCP_residuals,
    params::HPRSOCP_parameters,
    iter::Int)
    # Call unified version
    compute_residuals!(ws, qp, sc, res, params, iter)
end

"""
    compute_M_norm!(ws, qp)

Unified implementation of M-norm computation for both GPU and CPU.
Computes the norm of the M matrix used in convergence analysis.

The M-norm is a weighted norm that combines primal, dual, and constraint-related
terms based on the current iterate differences and problem structure.

# Arguments
- `ws::HPRSOCP_workspace`: Workspace (GPU or CPU)
- `qp::HPRSOCP_QP_info`: Problem data (GPU or CPU)

# Returns
- Float64: The computed M-norm value
"""
function compute_M_norm!(ws::HPRSOCP_workspace, qp::HPRSOCP_QP_info)
    # Initialize M terms
    M_1 = 0.0
    M_2 = (1.0 / ws.sigma) * unified_dot(ws.dx, ws.dx)
    M_3 = 0.0
    skip_q_mnorm = !has_quadratic_terms(qp.Q)

    if !skip_q_mnorm
        # Use unified Qmap! function (dispatch handles operator vs sparse matrix)
        # Pass spmv_Q for GPU sparse matrices to use preprocessed CUSPARSE
        if isa(qp.Q, CuSparseMatrixCSR{Float64,Int32}) && isa(ws, HPRSOCP_workspace_gpu)
            Qmap!(ws.dw, ws.dQw, qp.Q, ws.spmv_Q)
        else
            Qmap!(ws.dw, ws.dQw, qp.Q)
        end
        M_2 -= 2.0 * unified_dot(ws.dQw, ws.dx)
    end

    # Add constraint-related terms if constraints exist
    if ws.m > 0
        M_1 = ws.sigma * ws.lambda_max_A * unified_dot(ws.dy, ws.dy)
        unified_mul!(ws.ATdy, ws.AT, ws.dy)
        if !skip_q_mnorm
            # Pass spmv_Q for GPU sparse matrices to use preprocessed CUSPARSE
            if isa(qp.Q, CuSparseMatrixCSR{Float64,Int32}) && isa(ws, HPRSOCP_workspace_gpu)
                Qmap!(ws.ATdy, ws.QATdy, qp.Q, ws.spmv_Q)
            else
                Qmap!(ws.ATdy, ws.QATdy, qp.Q)
            end
            M_1 -= 2.0 * ws.sigma * unified_dot(ws.dQw, ws.ATdy)
        end
        M_2 += 2.0 * unified_dot(ws.ATdy, ws.dx)

        if !skip_q_mnorm
            if ws.Q_is_diag
                ws.ATdy .*= ws.fact_M
                M_3 = unified_dot(ws.ATdy, ws.QATdy) # sGS term
                M_1 += ws.sigma * unified_dot(ws.dQw, ws.dQw)
            else
                M_3 = (ws.sigma^2) / (1.0 + ws.sigma * ws.lambda_max_Q) * unified_dot(ws.ATdy, ws.QATdy)  # sGS term
                M_1 += ws.sigma * ws.lambda_max_Q * unified_dot(ws.dw, ws.dQw)
            end
        end
    elseif !ws.Q_is_diag && !skip_q_mnorm
        # No constraints case: only add Q-related term to M_1
        M_1 = ws.sigma * ws.lambda_max_Q * unified_dot(ws.dw, ws.dQw)
    end

    M_2 += max(M_1, 0.0)
    M_norm = max(M_2, 0.0) + max(M_3, 0.0)

    # Check for numerical instability
    if min(M_1, M_2, M_3) < -1e-8
        println("M_1 = $M_1, M_2 = $M_2, M_3 = $M_3, negative M norm due to numerical instability; the internal eigenvalue safety factor may be too small")
    end

    return sqrt(M_norm)
end




function compute_M_norm_1!(ws::HPRSOCP_workspace, qp::HPRSOCP_QP_info)
    # Initialize M terms
    M_1 = 0.0
    M_2 = 1.0 * unified_dot(ws.dx, ws.dx)
    M_3 = 0.0
    skip_q_mnorm = !has_quadratic_terms(qp.Q)

    if !skip_q_mnorm
        # Use unified Qmap! function (dispatch handles operator vs sparse matrix)
        # Pass spmv_Q for GPU sparse matrices to use preprocessed CUSPARSE
        if isa(qp.Q, CuSparseMatrixCSR{Float64,Int32}) && isa(ws, HPRSOCP_workspace_gpu)
            Qmap!(ws.dw, ws.dQw, qp.Q, ws.spmv_Q)
        else
            Qmap!(ws.dw, ws.dQw, qp.Q)
        end
        M_2 -= 2.0 * unified_dot(ws.dQw, ws.dx)
    end

    # Add constraint-related terms if constraints exist
    if ws.m > 0
        M_1 = ws.lambda_max_A * unified_dot(ws.dy, ws.dy)
        unified_mul!(ws.ATdy, ws.AT, ws.dy)
        if !skip_q_mnorm
            # Pass spmv_Q for GPU sparse matrices to use preprocessed CUSPARSE
            if isa(qp.Q, CuSparseMatrixCSR{Float64,Int32}) && isa(ws, HPRSOCP_workspace_gpu)
                Qmap!(ws.ATdy, ws.QATdy, qp.Q, ws.spmv_Q)
            else
                Qmap!(ws.ATdy, ws.QATdy, qp.Q)
            end
            M_1 -= 2.0 * unified_dot(ws.dQw, ws.ATdy)
        end
        M_2 += 2.0 * unified_dot(ws.ATdy, ws.dx)

        if !skip_q_mnorm
            if ws.Q_is_diag
                ws.ATdy .*= ws.fact_M
                M_3 = unified_dot(ws.ATdy, ws.QATdy) # sGS term
                M_1 += unified_dot(ws.dQw, ws.dQw)
            else
                M_3 = 1.0 / (1.0 + 1.0 * ws.lambda_max_Q) * unified_dot(ws.ATdy, ws.QATdy)  # sGS term
                M_1 += 1.0 * ws.lambda_max_Q * unified_dot(ws.dw, ws.dQw)
            end
        end
    elseif !ws.Q_is_diag && !skip_q_mnorm
        # No constraints case: only add Q-related term to M_1
        M_1 = 1.0 * ws.lambda_max_Q * unified_dot(ws.dw, ws.dQw)
    end

    M_2 += max(M_1, 0.0)
    M_norm = max(M_2, 0.0) + max(M_3, 0.0)

    # Check for numerical instability
    if min(M_1, M_2, M_3) < -1e-8
        println("M_1 = $M_1, M_2 = $M_2, M_3 = $M_3, negative M norm due to numerical instability; the internal eigenvalue safety factor may be too small")
    end

    return sqrt(M_norm)
end


# CPU version of compute M norm
function compute_M_norm_cpu!(ws::HPRSOCP_workspace_cpu, qp::QP_info_cpu)
    # Call unified implementation
    return compute_M_norm!(ws, qp)
end

# ============================================================================
# Unified Workspace Allocation
# ============================================================================

"""
    allocate_workspace(qp, params, lambda_max_A, lambda_max_Q, scaling_info, diag_Q, Q_is_diag)

Unified workspace allocation function that works for both CPU and GPU.
Dispatches to appropriate implementation based on qp type.

This replaces the separate allocate_workspace_cpu and allocate_workspace_gpu functions,
reducing code duplication and ensuring consistency.
"""
function allocate_workspace(
    qp::HPRSOCP_QP_info,
    params::HPRSOCP_parameters,
    lambda_max_A::Float64,
    lambda_max_Q::Float64,
    scaling_info::HPRSOCP_scaling,
    diag_Q::GPUOrCPUVector{Float64},
    Q_is_diag::Bool
)
    # Determine workspace type from qp type
    WS = workspace_type(qp)

    # Create workspace
    ws = WS()
    m, n = size(qp.A)
    ws.m = m
    ws.n = n
    ws.noq_soc_scratch_aty_mode = :cusparse
    ws.noC = false

    if isa(ws, HPRSOCP_workspace_gpu)
        ws.spmv_A = nothing
        ws.spmv_AT = nothing
        ws.spmv_Q = nothing
    end

    # Allocate vectors using type-dispatched helper
    ws.w = allocate_vector(WS, Float64, n)
    ws.w_hat = allocate_vector(WS, Float64, n)
    ws.w_bar = allocate_vector(WS, Float64, n)
    ws.dw = allocate_vector(WS, Float64, n)
    ws.x = allocate_vector(WS, Float64, n)
    ws.x_hat = allocate_vector(WS, Float64, n)
    ws.x_bar = allocate_vector(WS, Float64, n)
    ws.dx = allocate_vector(WS, Float64, n)
    ws.y = allocate_vector(WS, Float64, m)
    ws.y_hat = allocate_vector(WS, Float64, m)
    ws.y_bar = allocate_vector(WS, Float64, m)
    ws.dy = allocate_vector(WS, Float64, m)
    ws.z_bar = allocate_vector(WS, Float64, n)

    # Assign QP problem data
    ws.Q = qp.Q
    ws.A = qp.A
    ws.AT = qp.AT
    ws.soc_rhs = qp.soc_rhs
    ws.soc_rhs_full = qp.soc_rhs_full
    ws.AL = qp.AL
    ws.AU = qp.AU
    ws.SOC_con_idx = qp.SOC_con_idx
    ws.SOC_var_idx = qp.SOC_var_idx
    ws.number_eq = qp.number_eq
    ws.number_ineq = qp.number_ineq
    ws.number_lu_x = qp.number_lu_x
    ws.number_SOC_con = length(qp.SOC_con_idx) - 1
    ws.number_SOC_var = length(qp.SOC_var_idx) - 1
    ws.c = qp.c
    ws.l = qp.l
    ws.u = qp.u

    # Allocate work vectors
    ws.Rp = allocate_vector(WS, Float64, m)
    ws.Rd = allocate_vector(WS, Float64, n)
    ws.Ax = allocate_vector(WS, Float64, m)
    ws.ATy = allocate_vector(WS, Float64, n)
    ws.ATy_bar = allocate_vector(WS, Float64, n)
    ws.ATdy = allocate_vector(WS, Float64, n)
    ws.QATdy = allocate_vector(WS, Float64, n)
    ws.s = allocate_vector(WS, Float64, m)
    ws.Qw = allocate_vector(WS, Float64, n)
    ws.Qw_hat = allocate_vector(WS, Float64, n)
    ws.Qw_bar = allocate_vector(WS, Float64, n)
    ws.Qx = allocate_vector(WS, Float64, n)
    ws.dQw = allocate_vector(WS, Float64, n)
    ws.last_x = allocate_vector(WS, Float64, n)
    ws.last_y = allocate_vector(WS, Float64, m)
    ws.last_Qw = allocate_vector(WS, Float64, n)
    ws.last_w = allocate_vector(WS, Float64, n)
    ws.last_ATy = allocate_vector(WS, Float64, n)
    ws.tempv = allocate_vector(WS, Float64, n)

    # Set Q properties
    ws.Q_is_diag = Q_is_diag
    ws.diag_Q = convert_to_device(WS, diag_Q)

    # Allocate factorization vectors
    ws.fact1 = allocate_vector(WS, Float64, n)
    ws.fact2 = allocate_vector(WS, Float64, n)
    ws.fact = allocate_vector(WS, Float64, n)
    ws.fact_M = allocate_vector(WS, Float64, n)

    # Set eigenvalue bounds
    ws.lambda_max_A = lambda_max_A
    ws.lambda_max_Q = lambda_max_Q

    # Compute sigma
    if params.sigma == -1
        norm_b = scaling_info.norm_b
        norm_c = scaling_info.norm_c
        if norm_c > 1e-16 && norm_b > 1e-16 && norm_b < 1e16 && norm_c < 1e16
            ws.sigma = norm_b / norm_c
            # ws.sigma = 1.0
        else
            ws.sigma = 1.0
        end
    elseif params.sigma > 0
        ws.sigma = params.sigma
    else
        error("Invalid sigma value: ", params.sigma, ". It should be a positive number or -1 for automatic.")
    end

    # Compute factors for diagonal Q
    if ws.Q_is_diag
        # Use broadcasting for GPU-compatible computation
        temp = 1.0 .+ ws.sigma .* ws.diag_Q
        ws.fact1 .= 1.0 ./ temp
        ws.fact2 .= (ws.sigma .* ws.diag_Q) ./ temp
        ws.fact_M .= (ws.sigma^2 .* ws.diag_Q) ./ temp
    end

    ws.SOC_norms_temp = allocate_vector(WS, Float64, max(ws.number_SOC_con, ws.number_SOC_var))

    if isa(ws, HPRSOCP_workspace_gpu)
        configure_soc_con_projection_paths!(ws)
        configure_soc_var_projection_paths!(ws)
    end

    # Set to_check flag
    ws.to_check = true

    # Best-state tracking is used both for optional HDF5 snapshots and for
    # high-accuracy restart rollback.
    ws.saved_state = allocate_saved_state(WS)
    ws.saved_state.save_x = allocate_vector(WS, Float64, n)
    ws.saved_state.save_y = allocate_vector(WS, Float64, m)
    ws.saved_state.save_z = allocate_vector(WS, Float64, n)
    ws.saved_state.save_w = allocate_vector(WS, Float64, n)
    ws.saved_state.save_sigma = ws.sigma
    ws.saved_state.save_iter = 0
    ws.saved_state.save_err_Rp = Inf
    ws.saved_state.save_err_Rd = Inf
    ws.saved_state.save_primal_obj = Inf
    ws.saved_state.save_dual_obj = Inf
    ws.saved_state.save_rel_gap = Inf

    return ws
end

# ============================================================================
# CPU Main Update Functions
# ============================================================================

# Main update function for CPU - dispatches to appropriate update based on problem type
"""
    main_update_cpu!(ws, qp, restart_info)

CPU-specific main iteration update for the HPR-SOCP algorithm.

This function performs the core primal-dual update step using CPU-optimized loops.
It remains separate from GPU version because it calls device-specific kernels/loops.

# Algorithm Overview
The update follows the Halpern iteration scheme:
  - Compute Halpern averaging factors: α₁ = 1/(k+2), α₂ = 1 - α₁
  - Update primal variables (z, x, w) using proximal operators
  - Update dual variables (y) via gradient ascent
  - Apply Halpern averaging to produce (x̄, w̄, ȳ)

# Two Update Paths:

1. **Standard QP / custom operator with non-empty Q**:
    - Q is a sparse matrix or caller-supplied operator
   - Three-step update process:
     * Step 1: Update z, x, w (without dual correction)
     * Step 2: Update dual y variables
     * Step 3: Complete w update (add dual correction)
   - Handles diagonal Q with precomputed factor vectors
   - Handles non-diagonal Q with scalar factors

2. **Empty Q (Linear Program / SOCP without quadratic term)**:
   - No Q matrix present (LP instead of QP)
   - Simplified updates without proximal operator for Q
    - Calls: main_update_noQ_soc_cpu!

# Arguments
- `ws::HPRSOCP_workspace_cpu`: CPU workspace containing all iterate vectors
- `qp::QP_info_cpu`: Problem data (Q, A, b, c, etc.)
- `restart_info::HPRSOCP_restart`: Restart tracking (provides iteration count)

# Implementation Notes
- **Why CPU-specific**: Calls CPU loop functions instead of GPU kernels
- **Cannot be unified**: Device-specific execution paths (loops vs kernels)
- **Diagonal Q optimization**: Uses precomputed factor vectors when Q is diagonal
- **Operator dispatch**: Supports sparse matrices and caller-supplied operators
- **Structure matches GPU**: Same branching logic as main_update_gpu! for consistency

# See Also
- `main_update_gpu!`: GPU version using CUDA kernels
- `update_zxw1_cpu!`, `update_y_cpu!`, `update_w2_cpu!`: Standard QP updates
"""
function main_update_cpu!(ws::HPRSOCP_workspace_cpu, qp::QP_info_cpu, restart_info::HPRSOCP_restart)
    Halpern_fact1 = 1.0 / (restart_info.inner + 2.0)
    Halpern_fact2 = 1.0 - Halpern_fact1
    has_quadratic = has_quadratic_terms(qp.Q)

    if has_quadratic
        # Standard case with Q matrix - use unified kernels with separate Q and A modes
        update_zxw1_cpu!(ws, qp, Halpern_fact1, Halpern_fact2)
        update_y_cpu!(ws, qp, Halpern_fact1, Halpern_fact2)
        update_w2_cpu!(ws, qp, Halpern_fact1, Halpern_fact2)
    else
        # Empty Q case (linear program / SOCP) - use the dedicated no-Q path
        main_update_noQ_soc_cpu!(ws, qp, Halpern_fact1, Halpern_fact2)
    end
end

function main_update_noQ_soc_cpu!(ws::HPRSOCP_workspace_cpu,
    qp::QP_info_cpu,
    Halpern_fact1::Float64,
    Halpern_fact2::Float64)
    unified_update_zx_cpu!(ws, Halpern_fact1, Halpern_fact2)
    unified_update_y_noQ_cpu!(ws, Halpern_fact1, Halpern_fact2)
    return
end

# ============================================================================
# CPU Sigma Update
# ============================================================================

function update_sigma_cpu!(params::HPRSOCP_parameters,
    restart_info::HPRSOCP_restart,
    ws::HPRSOCP_workspace_cpu,
    qp::QP_info_cpu,
    residuals::HPRSOCP_residuals,
)
    # Call unified implementation
    update_sigma!(params, restart_info, ws, qp, residuals)
end

# ============================================================================
# CPU Collect Results
# ============================================================================

function collect_results_cpu!(ws::HPRSOCP_workspace_cpu, qp::QP_info_cpu, scaling_info::Scaling_info_cpu, results::HPRSOCP_results)
    # Call unified implementation with dummy timing values (results object already has timing)
    # This maintains backward compatibility where results object is passed in
    temp_results = collect_results!(ws, qp, scaling_info, HPRSOCP_residuals(),
        results.iter, 0.0, 0.0)

    # Copy solution vectors to the provided results object
    results.x = temp_results.x
    results.w = temp_results.w
    results.y = temp_results.y
    results.z = temp_results.z
end

# ============================================================================
# CPU Helper Functions (restart, termination, etc.)
# ============================================================================


# ============================================================================
# Unified handle_termination function
# ============================================================================

"""
    handle_termination(status, residuals, ws, scaling_info, iter, t_start_alg,
                      power_time, setup_time, iter_4, time_4, iter_6, time_6, verbose)

Unified termination handler that works for both GPU and CPU workspaces.

Collects final results, prints solution summary, and returns HPRSOCP_results.
Automatically handles GPU->CPU transfer when needed via collect_results!.

# Arguments
- `status::String`: Termination status ("OPTIMAL", "MAX_ITER", "TIME_LIMIT")
- `residuals::HPRSOCP_residuals`: Final residuals
- `ws::HPRSOCP_workspace`: Workspace (GPU or CPU)
- `scaling_info::HPRSOCP_scaling`: Scaling information (GPU or CPU)
- `iter::Int`: Final iteration number
- `t_start_alg::Float64`: Algorithm start time
- `power_time::Float64`: Time spent in power iteration
- `setup_time::Float64`: Setup time
- `iter_4, time_4`: Milestone tracking for 1e-4 accuracy
- `iter_6, time_6`: Milestone tracking for 1e-6 accuracy
- `verbose::Bool`: Whether to print output

# Returns
- `HPRSOCP_results`: Final results structure
"""
function handle_termination(
    status::String,
    residuals::HPRSOCP_residuals,
    ws::HPRSOCP_workspace,
    scaling_info::HPRSOCP_scaling,
    iter::Int,
    t_start_alg::Float64,
    power_time::Float64,
    setup_time::Float64,
    iter_4::Int,
    time_4::Float64,
    iter_6::Int,
    time_6::Float64,
    verbose::Bool
)
    # Print termination message
    if verbose
        if status == "OPTIMAL"
            println("The instance is solved, the accuracy is ", residuals.KKTx_and_gap_org_bar)
        elseif status == "MAX_ITER"
            println("The maximum number of iterations is reached, the accuracy is ",
                residuals.KKTx_and_gap_org_bar)
        elseif status == "TIME_LIMIT"
            println("The time limit is reached, the accuracy is ", residuals.KKTx_and_gap_org_bar)
        end
    end

    # Collect results using unified function (handles GPU->CPU transfer automatically)
    results = collect_results!(ws, nothing, scaling_info, residuals, iter,
        t_start_alg, power_time)

    results.status = status
    results.time_4 = time_4 == 0.0 ? results.time : time_4
    results.iter_4 = iter_4 == 0 ? iter : iter_4
    results.time_6 = time_6 == 0.0 ? results.time : time_6
    results.iter_6 = iter_6 == 0 ? iter : iter_6

    # Print solution summary
    if verbose
        println()
        println("="^80)
        println("SOLUTION SUMMARY")
        println("="^80)
        println(@sprintf("Status: %s", status))
        println(@sprintf("Iterations: %d", iter))
        println(@sprintf("Time: %.2f seconds", results.time))
        println(@sprintf("Primal Objective: %.12e", residuals.primal_obj_bar))
        println(@sprintf("Dual Objective: %.12e", residuals.dual_obj_bar))
        println(@sprintf("Primal Residual: %.6e", residuals.err_Rp_org_bar))
        println(@sprintf("Primal Residual (Linear): %.6e", residuals.err_Rp_linear_org_bar))
        println(@sprintf("Primal Residual (SOC): %.6e", residuals.err_Rp_soc_org_bar))
        println(@sprintf("Dual Residual: %.6e", residuals.err_Rd_org_bar))
        println(@sprintf("Relative Gap: %.6e", residuals.rel_gap_bar))
        println("="^80)
        println(@sprintf("Total time: %.2fs  (setup = %.2fs, solve = %.2fs)",
            setup_time + results.time, setup_time, results.time))
        println("="^80)
    end

    return results
end

# CPU version of print_problem_info
function print_problem_info(qp::HPRSOCP_QP_info, ws::HPRSOCP_workspace, params::HPRSOCP_parameters)
    if !params.verbose
        return
    end

    m, n = size(qp.A)

    println("="^80)
    println("QP PROBLEM INFORMATION")
    println("="^80)

    # Determine QP type using helper functions
    qp_type = if is_q_operator(qp.Q)
        get_operator_name(typeof(qp.Q))
    else
        # Q is a sparse matrix
        if get_Q_nnz(qp.Q) > 0
            "QP (Quadratic Program - Non-empty Q)"
        else
            "LP (Linear Program - Empty Q)"
        end
    end
    println("Problem Type: $qp_type")

    # Q matrix information
    if is_q_operator(qp.Q)
        op_name = get_operator_name(typeof(qp.Q))
        println("Q Operator: $op_name operator (implicit matrix)")
    else
        # Q is a sparse matrix
        q_size = size(qp.Q, 1)
        q_nnz = get_Q_nnz(qp.Q)
        println("Q Matrix: $(q_size)×$(q_size), nnz = $q_nnz")
        if q_nnz > 0
            println("Q is Diagonal: $(ws.Q_is_diag)")
        end
    end

    # Constraint matrix information
    if m > 0
        a_nnz = get_A_nnz(qp.A)
        println("A Matrix: $(m)×$(n), nnz = $a_nnz")
    else
        println("A Matrix: No constraints (unconstrained)")
    end

    println("Constraint Partition: linear constraints = $(qp.number_eq + qp.number_ineq), SOC blocks = $(length(qp.SOC_con_idx) - 1)")
    println("Variable Partition: boxed = $(qp.number_lu_x), SOC blocks = $(length(qp.SOC_var_idx) - 1)")

    if length(qp.SOC_con_idx) > 1
        if isa(qp, QP_info_gpu)
            println("SOC Constraint Cone Sizes: omitted in GPU mode to avoid host transfer during solve setup")
        else
            soc_con_idx = qp.SOC_con_idx
            soc_con_sizes = [soc_con_idx[i+1] - soc_con_idx[i] for i in 1:(length(soc_con_idx)-1)]
            soc_con_sizes_sorted = sort(soc_con_sizes)
            num_con_blocks = length(soc_con_sizes_sorted)
            soc_con_median = isodd(num_con_blocks) ?
                             soc_con_sizes_sorted[(num_con_blocks+1)÷2] :
                             (soc_con_sizes_sorted[num_con_blocks÷2] + soc_con_sizes_sorted[num_con_blocks÷2+1]) / 2
            println(
                "SOC Constraint Cone Sizes: min = $(first(soc_con_sizes_sorted)), median = $(soc_con_median), max = $(last(soc_con_sizes_sorted))"
            )
        end
    end

    if length(qp.SOC_var_idx) > 1
        if isa(qp, QP_info_gpu)
            println("SOC Variable Cone Sizes: omitted in GPU mode to avoid host transfer during solve setup")
        else
            soc_var_idx = qp.SOC_var_idx
            soc_var_sizes = [soc_var_idx[i+1] - soc_var_idx[i] for i in 1:(length(soc_var_idx)-1)]
            soc_var_sizes_sorted = sort(soc_var_sizes)
            num_var_blocks = length(soc_var_sizes_sorted)
            soc_var_median = isodd(num_var_blocks) ?
                             soc_var_sizes_sorted[(num_var_blocks+1)÷2] :
                             (soc_var_sizes_sorted[num_var_blocks÷2] + soc_var_sizes_sorted[num_var_blocks÷2+1]) / 2
            println(
                "SOC Variable Cone Sizes: min = $(first(soc_var_sizes_sorted)), median = $(soc_var_median), max = $(last(soc_var_sizes_sorted))"
            )
        end
    end

    println()
end

# ============================================================================
# GPU Algorithm Functions
# ============================================================================

# This function initializes the restart information for the HPR-SOCP algorithm.
function initialize_restart()
    restart_info = HPRSOCP_restart()
    restart_info.first_restart = true
    restart_info.save_gap = Inf
    restart_info.current_gap = Inf
    restart_info.last_gap = Inf
    restart_info.best_gap = Inf
    restart_info.best_kkt = Inf
    restart_info.best_sigma = 1.0
    restart_info.best_iter = 0
    restart_info.inner = 0
    restart_info.times = 0
    restart_info.sufficient = 0
    restart_info.necessary = 0
    restart_info.long = 0
    restart_info.ratio = 0
    restart_info.restart_flag = 0
    restart_info.weighted_norm = Inf
    restart_info.sigma_correction_active = false
    restart_info.sigma_correction_hold = 0
    restart_info.sigma_correction_dir = 0
    return restart_info
end

function print_step(iter::Int)
    return max(10^floor(log10(iter)) / 10, 10)
end

const SOC_VAR_SPECIALIZED_SIZES = (3, 4, 5, 8)
const SOC_VAR_LARGE_CONE_THRESHOLD = 65
const SOC_VAR_HUGE_CONE_THRESHOLD = 1024
const SOC_CON_SPECIALIZED_SIZES = (3, 4, 5)
const SOC_CON_LARGE_CONE_THRESHOLD = 65
const NOQ_SOC_SCRATCH_ATY_CUSTOM_AVG_COL_NNZ_THRESHOLD = 16.0

function build_soc_var_projection_buckets(
    soc_var_idx::AbstractVector{<:Integer};
    specialized_sizes::Tuple{Vararg{Int}}=SOC_VAR_SPECIALIZED_SIZES,
    large_cone_threshold::Int=SOC_VAR_LARGE_CONE_THRESHOLD,
    huge_cone_threshold::Int=SOC_VAR_HUGE_CONE_THRESHOLD,
)
    unsupported_sizes = setdiff(collect(specialized_sizes), collect(SOC_VAR_SPECIALIZED_SIZES))
    isempty(unsupported_sizes) || error("Unsupported SOC fast-path sizes: $(unsupported_sizes)")
    large_cone_threshold >= 2 || error("large_cone_threshold must be at least 2")
    huge_cone_threshold >= 2 || error("huge_cone_threshold must be at least 2")
    effective_huge_cone_threshold = max(huge_cone_threshold, large_cone_threshold)

    use_size3 = 3 in specialized_sizes
    use_size4 = 4 in specialized_sizes
    use_size5 = 5 in specialized_sizes
    use_size8 = 8 in specialized_sizes

    size3_starts = Int32[]
    size4_starts = Int32[]
    size5_starts = Int32[]
    size8_starts = Int32[]
    huge_starts = Int32[]
    huge_sizes = Int32[]
    large_starts = Int32[]
    large_sizes = Int32[]
    generic_starts = Int32[]
    generic_sizes = Int32[]

    for cone_idx in 1:(length(soc_var_idx)-1)
        start_idx = Int32(soc_var_idx[cone_idx])
        cone_size = Int(soc_var_idx[cone_idx+1] - soc_var_idx[cone_idx])

        if cone_size == 3 && use_size3
            push!(size3_starts, start_idx)
        elseif cone_size == 4 && use_size4
            push!(size4_starts, start_idx)
        elseif cone_size == 5 && use_size5
            push!(size5_starts, start_idx)
        elseif cone_size == 8 && use_size8
            push!(size8_starts, start_idx)
        elseif cone_size >= effective_huge_cone_threshold
            push!(huge_starts, start_idx)
            push!(huge_sizes, Int32(cone_size))
        elseif cone_size >= large_cone_threshold
            push!(large_starts, start_idx)
            push!(large_sizes, Int32(cone_size))
        else
            push!(generic_starts, start_idx)
            push!(generic_sizes, Int32(cone_size))
        end
    end

    return (
        size3_starts=size3_starts,
        size4_starts=size4_starts,
        size5_starts=size5_starts,
        size8_starts=size8_starts,
        huge_starts=huge_starts,
        huge_sizes=huge_sizes,
        large_starts=large_starts,
        large_sizes=large_sizes,
        generic_starts=generic_starts,
        generic_sizes=generic_sizes,
    )
end

function build_soc_con_projection_buckets(
    soc_con_idx::AbstractVector{<:Integer};
    specialized_sizes::Tuple{Vararg{Int}}=SOC_CON_SPECIALIZED_SIZES,
    large_cone_threshold::Int=SOC_CON_LARGE_CONE_THRESHOLD,
)
    unsupported_sizes = setdiff(collect(specialized_sizes), collect(SOC_CON_SPECIALIZED_SIZES))
    isempty(unsupported_sizes) || error("Unsupported SOC constraint fast-path sizes: $(unsupported_sizes)")
    large_cone_threshold >= 2 || error("large_cone_threshold must be at least 2")

    use_size3 = 3 in specialized_sizes
    use_size4 = 4 in specialized_sizes
    use_size5 = 5 in specialized_sizes

    size3_starts = Int32[]
    size4_starts = Int32[]
    size5_starts = Int32[]
    large_starts = Int32[]
    large_sizes = Int32[]
    generic_starts = Int32[]
    generic_sizes = Int32[]

    for cone_idx in 1:(length(soc_con_idx)-1)
        start_idx = Int32(soc_con_idx[cone_idx])
        cone_size = Int(soc_con_idx[cone_idx+1] - soc_con_idx[cone_idx])

        if cone_size == 3 && use_size3
            push!(size3_starts, start_idx)
        elseif cone_size == 4 && use_size4
            push!(size4_starts, start_idx)
        elseif cone_size == 5 && use_size5
            push!(size5_starts, start_idx)
        elseif cone_size >= large_cone_threshold
            push!(large_starts, start_idx)
            push!(large_sizes, Int32(cone_size))
        else
            push!(generic_starts, start_idx)
            push!(generic_sizes, Int32(cone_size))
        end
    end

    return (
        size3_starts=size3_starts,
        size4_starts=size4_starts,
        size5_starts=size5_starts,
        large_starts=large_starts,
        large_sizes=large_sizes,
        generic_starts=generic_starts,
        generic_sizes=generic_sizes,
    )
end

function configure_soc_con_projection_paths!(
    ws::HPRSOCP_workspace_gpu;
    specialized_sizes::Tuple{Vararg{Int}}=SOC_CON_SPECIALIZED_SIZES,
    large_cone_threshold::Int=SOC_CON_LARGE_CONE_THRESHOLD,
)
    buckets = build_soc_con_projection_buckets(
        to_cpu(ws.SOC_con_idx);
        specialized_sizes=specialized_sizes,
        large_cone_threshold=large_cone_threshold,
    )

    fast_paths = SOC_con_fast_paths_gpu()
    small_starts = Int32[]
    small_sizes = Int32[]
    append!(small_starts, buckets.size3_starts)
    append!(small_sizes, fill(Int32(3), length(buckets.size3_starts)))
    append!(small_starts, buckets.size4_starts)
    append!(small_sizes, fill(Int32(4), length(buckets.size4_starts)))
    append!(small_starts, buckets.size5_starts)
    append!(small_sizes, fill(Int32(5), length(buckets.size5_starts)))
    append!(small_starts, buckets.generic_starts)
    append!(small_sizes, buckets.generic_sizes)
    fast_paths.size3_starts = CuVector(buckets.size3_starts)
    fast_paths.size4_starts = CuVector(buckets.size4_starts)
    fast_paths.size5_starts = CuVector(buckets.size5_starts)
    fast_paths.small_starts = CuVector(small_starts)
    fast_paths.small_sizes = CuVector(small_sizes)
    fast_paths.large_starts = CuVector(buckets.large_starts)
    fast_paths.large_sizes = CuVector(buckets.large_sizes)
    fast_paths.generic_starts = CuVector(buckets.generic_starts)
    fast_paths.generic_sizes = CuVector(buckets.generic_sizes)
    fast_paths.size3_count = length(buckets.size3_starts)
    fast_paths.size4_count = length(buckets.size4_starts)
    fast_paths.size5_count = length(buckets.size5_starts)
    fast_paths.small_count = length(small_starts)
    fast_paths.large_count = length(buckets.large_starts)
    fast_paths.generic_count = length(buckets.generic_starts)

    ws.soc_con_fast_paths = fast_paths
    return ws
end

function resolve_soc_var_huge_kernel_mode(
    requested_mode::Symbol,
    cooperative_supported::Bool,
    cooperative_block_limit::Int,
    total_blocks::Int,
)
    requested_mode in (:auto, :staged, :segmented, :cooperative) ||
        error("Unsupported huge_kernel_mode=$requested_mode. Use :auto, :staged, :segmented, or :cooperative.")

    if requested_mode == :staged || total_blocks == 0
        return :staged
    end

    if requested_mode == :segmented
        return :segmented
    end

    if cooperative_supported && total_blocks <= cooperative_block_limit
        return :cooperative
    end

    return :segmented
end

function cooperative_soc_var_huge_block_limit(
    ws::HPRSOCP_workspace_gpu,
    fast_paths::SOC_var_fast_paths_gpu,
)
    fast_paths.huge_total_blocks > 0 || return 0
    CUDA.attribute(CUDA.device(), CUDA.DEVICE_ATTRIBUTE_COOPERATIVE_LAUNCH) == 1 || return 0

    kernel = @cuda launch = false unified_update_zx_noQ_SOC_huge_cooperative_kernel!(
        ws.dx, ws.z_bar, ws.x_bar, ws.x_hat, ws.last_x, ws.x, ws.ATy, ws.c, ws.sigma,
        0.5, 0.5,
        fast_paths.huge_sizes, fast_paths.huge_block_starts, fast_paths.huge_block_offsets,
        fast_paths.huge_block_cone_ids, fast_paths.huge_block_ptr,
        fast_paths.huge_partial_sums, fast_paths.huge_t_raw, fast_paths.huge_proj_t,
        fast_paths.huge_alpha, fast_paths.huge_case, fast_paths.huge_total_blocks)
    return CUDA.launch_configuration(kernel.fun).blocks
end

function configure_soc_var_projection_paths!(
    ws::HPRSOCP_workspace_gpu;
    specialized_sizes::Tuple{Vararg{Int}}=SOC_VAR_SPECIALIZED_SIZES,
    large_cone_threshold::Int=SOC_VAR_LARGE_CONE_THRESHOLD,
    huge_cone_threshold::Int=SOC_VAR_HUGE_CONE_THRESHOLD,
    huge_kernel_mode::Symbol=:auto,
)
    buckets = build_soc_var_projection_buckets(
        to_cpu(ws.SOC_var_idx);
        specialized_sizes=specialized_sizes,
        large_cone_threshold=large_cone_threshold,
        huge_cone_threshold=huge_cone_threshold,
    )

    fast_paths = SOC_var_fast_paths_gpu()
    small_starts = Int32[]
    small_sizes = Int32[]
    append!(small_starts, buckets.size3_starts)
    append!(small_sizes, fill(Int32(3), length(buckets.size3_starts)))
    append!(small_starts, buckets.size4_starts)
    append!(small_sizes, fill(Int32(4), length(buckets.size4_starts)))
    append!(small_starts, buckets.size5_starts)
    append!(small_sizes, fill(Int32(5), length(buckets.size5_starts)))
    append!(small_starts, buckets.size8_starts)
    append!(small_sizes, fill(Int32(8), length(buckets.size8_starts)))
    append!(small_starts, buckets.generic_starts)
    append!(small_sizes, buckets.generic_sizes)

    huge_block_starts = Int32[]
    huge_block_offsets = Int32[]
    huge_block_cone_ids = Int32[]
    huge_block_ptr = Int32[1]
    for (cone_id, (start_idx, cone_size_i32)) in enumerate(zip(buckets.huge_starts, buckets.huge_sizes))
        cone_size = Int(cone_size_i32)
        tail_length = cone_size - 1
        block_count = cld(tail_length, HUGE_SOC_KERNEL_THREADS)
        for block_id in 0:(block_count-1)
            push!(huge_block_starts, start_idx)
            push!(huge_block_offsets, Int32(block_id * HUGE_SOC_KERNEL_THREADS))
            push!(huge_block_cone_ids, Int32(cone_id))
        end
        push!(huge_block_ptr, Int32(length(huge_block_starts) + 1))
    end

    fast_paths.size3_starts = CuVector(buckets.size3_starts)
    fast_paths.size4_starts = CuVector(buckets.size4_starts)
    fast_paths.size5_starts = CuVector(buckets.size5_starts)
    fast_paths.size8_starts = CuVector(buckets.size8_starts)
    fast_paths.small_starts = CuVector(small_starts)
    fast_paths.small_sizes = CuVector(small_sizes)
    fast_paths.huge_starts = CuVector(buckets.huge_starts)
    fast_paths.huge_sizes = CuVector(buckets.huge_sizes)
    fast_paths.huge_block_starts = CuVector(huge_block_starts)
    fast_paths.huge_block_offsets = CuVector(huge_block_offsets)
    fast_paths.huge_block_cone_ids = CuVector(huge_block_cone_ids)
    fast_paths.huge_block_ptr = CuVector(huge_block_ptr)
    fast_paths.huge_partial_sums = CuVector(zeros(Float64, length(huge_block_starts)))
    fast_paths.huge_t_raw = CuVector(zeros(Float64, length(buckets.huge_starts)))
    fast_paths.huge_proj_t = CuVector(zeros(Float64, length(buckets.huge_starts)))
    fast_paths.huge_alpha = CuVector(zeros(Float64, length(buckets.huge_starts)))
    fast_paths.huge_case = CuVector(zeros(Int32, length(buckets.huge_starts)))
    fast_paths.large_starts = CuVector(buckets.large_starts)
    fast_paths.large_sizes = CuVector(buckets.large_sizes)
    fast_paths.generic_starts = CuVector(buckets.generic_starts)
    fast_paths.generic_sizes = CuVector(buckets.generic_sizes)
    fast_paths.size3_count = length(buckets.size3_starts)
    fast_paths.size4_count = length(buckets.size4_starts)
    fast_paths.size5_count = length(buckets.size5_starts)
    fast_paths.size8_count = length(buckets.size8_starts)
    fast_paths.small_count = length(small_starts)
    fast_paths.huge_count = length(buckets.huge_starts)
    fast_paths.huge_total_blocks = length(huge_block_starts)
    cooperative_supported = fast_paths.huge_total_blocks > 0 &&
                            CUDA.attribute(CUDA.device(), CUDA.DEVICE_ATTRIBUTE_COOPERATIVE_LAUNCH) == 1
    cooperative_block_limit =
        (huge_kernel_mode == :staged || !cooperative_supported) ? 0 :
        cooperative_soc_var_huge_block_limit(ws, fast_paths)
    fast_paths.huge_kernel_mode = resolve_soc_var_huge_kernel_mode(
        huge_kernel_mode,
        cooperative_supported,
        cooperative_block_limit,
        fast_paths.huge_total_blocks,
    )
    fast_paths.large_count = length(buckets.large_starts)
    fast_paths.large_max_size = isempty(buckets.large_sizes) ? 0 : Int(maximum(buckets.large_sizes))
    fast_paths.generic_count = length(buckets.generic_starts)

    ws.soc_var_fast_paths = fast_paths
    return ws
end

function select_noq_soc_scratch_aty_mode(ws::HPRSOCP_workspace_cpu)
    return :cusparse
end

function select_noq_soc_scratch_aty_mode(ws::HPRSOCP_workspace_gpu)
    return :cusparse
end

function configure_noq_soc_scratch_aty_mode!(ws::HPRSOCP_workspace)
    ws.noq_soc_scratch_aty_mode = select_noq_soc_scratch_aty_mode(ws)
    return ws
end

# This function updates the variables in the HPR-SOCP algorithm, when Q is diagonal, there's no proximal term on w;
# when the problem is formulated without l≤x≤u, the update is w->x->y.
"""
    main_update_gpu!(ws, qp, restart_info)

GPU-specific main iteration update for the HPR-SOCP algorithm.

This function performs the core primal-dual update step using CUDA kernels.
It remains separate from CPU version because it dispatches to GPU-optimized kernels.

# Algorithm Overview
The update follows the Halpern iteration scheme:
  - Compute Halpern averaging factors: α₁ = 1/(k+2), α₂ = 1 - α₁
  - Update primal variables (z, x, w) using proximal operators
  - Update dual variables (y) via gradient ascent
  - Apply Halpern averaging to produce (x̄, w̄, ȳ)

# Execution Modes

**Operator Mode**:
    - Q is a caller-supplied structured operator

**Sparse Matrix Mode**:
  - Q is a sparse matrix (standard QP)
  - **Empty Q** (linear program): Simplified updates without Q terms
    - **Non-empty Q**: Three-step update with solver-selected SpMV modes

# Three Update Paths:

1. **Custom Operators / Sparse Q**:
   - Calls: unified_update_zxw1_gpu!, unified_update_y_gpu!, unified_update_w2_gpu!
    - Uses operator mode for Q when needed and solver-selected sparse kernels for A

2. **Sparse Matrix Q**:
   - **Empty Q** (LP): unified_update_zx_gpu!, unified_update_y_noQ_gpu!
    - **Non-empty Q** (QP): Three-step update with solver-selected SpMV modes
   - Handles diagonal Q optimization with precomputed factors

# Arguments
- `ws::HPRSOCP_workspace_gpu`: GPU workspace containing all iterate vectors
- `qp::QP_info_gpu`: Problem data on GPU (Q, A, b, c, etc.)
- `restart_info::HPRSOCP_restart`: Restart tracking (provides iteration count)

# Implementation Notes
- **Why GPU-specific**: Launches CUDA kernels instead of CPU loops
- **Cannot be unified**: Fundamentally different execution model (parallel vs serial)
- **compute_full flag**: Derived from ws.to_check, controls full vs partial kernel execution
- **Diagonal Q optimization**: Uses precomputed factor vectors for diagonal Q
- **SpMV mode selection**: Chooses sparse multiplication strategy internally at runtime

# Performance Considerations
- Kernel launch overhead is amortized over large problem dimensions
- The internally selected SpMV modes trade off memory bandwidth vs computation
- Diagonal Q path uses vectorized operations instead of SpMV

# See Also
- `main_update_cpu!`: CPU version using loops
- `unified_update_zxw1_gpu!`, `unified_update_y_gpu!`, `unified_update_w2_gpu!`: Standard QP kernels
"""
function main_update_gpu!(ws::HPRSOCP_workspace_gpu,
    qp::QP_info_gpu,
    restart_info::HPRSOCP_restart)
    Halpern_fact1 = 1.0 / (restart_info.inner + 2.0)
    has_quadratic = has_quadratic_terms(qp.Q)
    Halpern_fact2 = 1.0 - Halpern_fact1

    if has_quadratic
        # Standard case with Q matrix - use unified kernels with separate Q and A modes
        unified_update_zxw1_gpu!(ws, qp, Halpern_fact1, Halpern_fact2)
        unified_update_y_gpu!(ws, Halpern_fact1, Halpern_fact2)
        unified_update_w2_gpu!(ws, Halpern_fact1, Halpern_fact2)
    else
        # Empty Q case (linear program) - use unified kernels with A mode only
        main_update_noQ_soc_gpu!(ws, qp, Halpern_fact1, Halpern_fact2)
    end
end

function main_update_noQ_soc_gpu!(ws::HPRSOCP_workspace_gpu,
    qp::QP_info_gpu,
    Halpern_fact1::Float64,
    Halpern_fact2::Float64)
    unified_update_zx_noQ_soc_gpu!(ws, Halpern_fact1, Halpern_fact2)
    unified_update_y_noQ_soc_gpu!(ws, Halpern_fact1, Halpern_fact2)
    return
end

# ==================== Helper Functions for Solver ====================

# Transfer model data from CPU to GPU
function transfer_to_gpu(model::QP_info_cpu, params::HPRSOCP_parameters)
    if params.verbose
        println("COPY TO GPU ...")
    end
    t_start = time()
    CUDA.synchronize()

    n = length(model.c)

    # Transfer Q to GPU using unified to_gpu interface.
    # Works for both sparse matrices and caller-supplied CPU operators.
    Q_gpu = to_gpu(model.Q)

    # Create QP_info_gpu
    qp = QP_info_gpu(
        Q_gpu,
        CuVector(model.c),
        CuSparseMatrixCSR(model.A),
        CuSparseMatrixCSR(model.AT),
        CuVector(model.soc_rhs),
        CuVector(model.soc_rhs_full),
        CuVector(model.AL),
        CuVector(model.AU),
        CuVector(model.SOC_con_idx),
        model.number_eq,
        model.number_ineq,
        CuVector(model.l),
        CuVector(model.u),
        CuVector(model.SOC_var_idx),
        model.number_lu_x,
        model.obj_constant,
    )

    CUDA.synchronize()
    transfer_time = time() - t_start
    if params.verbose
        println(@sprintf("COPY TO GPU time: %.2f seconds", transfer_time))
    end

    return qp, transfer_time
end

# Unified model preparation function
# Handles both GPU transfer and CPU copy with consistent interface
function prepare_model(model::QP_info_cpu, params::HPRSOCP_parameters)
    if params.use_gpu
        return transfer_to_gpu(model, params)
    else
        # CPU: work on a copy to avoid modifying original, no transfer time
        return deepcopy(model), 0.0
    end
end

# Print solver parameters
function print_solver_params(params::HPRSOCP_parameters, qp::Union{QP_info_gpu,QP_info_cpu})
    if !params.verbose
        return
    end

    m = size(qp.A, 1)
    n = size(qp.A, 2)

    println("="^80)
    println("SOLVER PARAMETERS:")
    println("  Problem size: m = ", m, ", n = ", n)
    println("  Device: ", params.use_gpu ? "GPU (device $(params.device_number))" : "CPU")
    println("  Stop tolerance: ", params.stoptol)
    println("  Max iterations: ", params.max_iter)
    println("  Time limit: ", params.time_limit, " seconds")
    println("  Check interval: ", params.check_iter)
    println("  Print frequency: ", params.print_frequency == -1 ? "Adaptive" : params.print_frequency)
    println("  Scaling options:")
    println("    Ruiz scaling: ", params.use_Ruiz_scaling ? "Enabled" : "Disabled")
    println("    Ruiz iterations: ", params.ruiz_iterations)
    println("    Pock-Chambolle scaling: ", params.use_Pock_Chambolle_scaling ? "Enabled" : "Disabled")
    println("    b/c scaling: ", params.use_bc_scaling ? "Enabled" : "Disabled")
    println("    b/c norm: ", params.bc_scaling_norm_type)
    println("    L2 scaling: ", params.use_l2_scaling ? "Enabled" : "Disabled")
    println("    SOC block aggregation: ", params.soc_block_scaling_strategy)

    if params.warm_up
        println("  Warm-up: Enabled (avoids JIT compilation overhead)")
    else
        println("  Warm-up: Disabled")
        println("    ⚠ WARNING: First run of each function may be slower due to JIT compilation.")
        println("    ⚠ Consider enabling warm_up for more accurate timing measurements.")
    end

    if params.initial_x !== nothing
        println("  Initial x: Provided (length ", length(params.initial_x), ")")
    end
    if params.initial_y !== nothing
        println("  Initial y: Provided (length ", length(params.initial_y), ")")
    end

    if params.auto_save
        # Calculate estimated memory for auto_save
        memory_bytes = (n + m + 2 * n) * 16  # x, y, z, w (8 bytes per Float64, 2 copies)
        memory_mb = memory_bytes / (1024 * 1024)
        memory_gb = memory_bytes / (1024 * 1024 * 1024)

        println("  Auto-save: ENABLED")
        println("    ⚠ WARNING: Auto-save will write to disk at each print iteration.")
        println("    ⚠ This may consume significant I/O bandwidth and slightly reduce speed.")
        if memory_gb >= 1.0
            println(@sprintf("    ⚠ Estimated memory for saved state: %.2f GB", memory_gb))
        elseif memory_mb >= 1.0
            println(@sprintf("    ⚠ Estimated memory for saved state: %.2f MB", memory_mb))
        elseif memory_bytes >= 1024.0
            println(@sprintf("    ⚠ Estimated memory for saved state: %.2f KB", memory_bytes / 1024))
        else
            println(@sprintf("    ⚠ Estimated memory for saved state: %.2f bytes", memory_bytes))
        end
        println("    Save file: ", params.save_filename)
    else
        println("  Auto-save: Disabled")
    end
    println("="^80)
end

# Estimate maximum eigenvalues using power iteration
function estimate_eigenvalues(qp::HPRSOCP_QP_info, params::HPRSOCP_parameters, ws::HPRSOCP_workspace)
    if params.verbose
        println("ESTIMATING MAXIMUM EIGENVALUES ...")
    end
    t_start = time()
    if isa(qp, QP_info_gpu)
        CUDA.synchronize()
    end

    m = size(qp.A, 1)

    # Estimate lambda_max_A using preprocessed SpMV structures from workspace
    if m > 0
        lambda_max_A = power_iteration_A(ws) * DEFAULT_EIG_FACTOR
    else
        lambda_max_A = 0.0
    end

    # Estimate lambda_max_Q based on Q type using unified dispatch
    lambda_max_Q = compute_lambda_max_Q(qp.Q, ws)

    if isa(qp, QP_info_gpu)
        CUDA.synchronize()
    end
    power_time = time() - t_start

    if params.verbose
        println(@sprintf("ESTIMATING MAXIMUM EIGENVALUES time = %.2f seconds", power_time))
        # println(@sprintf("estimated maximum eigenvalue of AAT = %.2e", lambda_max_A))
        # println(@sprintf("estimated maximum eigenvalue of Q = %.2e", lambda_max_Q))
    end

    return lambda_max_A, lambda_max_Q, power_time
end

# Determine SpMV mode based on problem structure
# Check if we should print at this iteration
function should_print(iter::Int, params::HPRSOCP_parameters, t_start_alg::Float64, max_iter::Int)
    if params.print_frequency == -1
        return ((rem(iter, print_step(iter)) == 0) || (iter == max_iter) ||
                (time() - t_start_alg > params.time_limit))
    elseif params.print_frequency > 0
        return ((rem(iter, params.print_frequency) == 0) || (iter == max_iter) ||
                (time() - t_start_alg > params.time_limit))
    else
        error("Invalid print_frequency: ", params.print_frequency,
            ". It should be a positive integer or -1 for automatic printing.")
    end
end

# Check and record tolerance milestones
function check_tolerance_milestones!(residuals::HPRSOCP_residuals,
    iter::Int,
    t_start_alg::Float64,
    iter_4::Ref{Int}, time_4::Ref{Float64}, first_4::Ref{Bool},
    iter_6::Ref{Int}, time_6::Ref{Float64}, first_6::Ref{Bool})
    if residuals.KKTx_and_gap_org_bar < 1e-4 && first_4[]
        time_4[] = time() - t_start_alg
        iter_4[] = iter
        first_4[] = false
        println("KKT < 1e-4 at iter = ", iter)
    end
    if residuals.KKTx_and_gap_org_bar < 1e-6 && first_6[]
        time_6[] = time() - t_start_alg
        iter_6[] = iter
        first_6[] = false
        println("KKT < 1e-6 at iter = ", iter)
    end
end

# ==================== Helper Function for CPU/GPU Dispatch ====================

# Helper function to select GPU or CPU function implementations
# ==================== Public API Functions ====================

"""
    optimize(model::QP_info_cpu, params::HPRSOCP_parameters)

Solve a QP model with optional warm-up phase.

This is the main entry point for solving QP problems. It handles:
1. Optional warm-up phase to avoid JIT compilation overhead
2. Calls solve() which does scaling, GPU transfer, and optimization

# Arguments
- `model::QP_info_cpu`: QP model built from build_from_mps(), build_from_QAbc(), etc.
- `params::HPRSOCP_parameters`: Solver parameters

# Returns
- `HPRSOCP_results`: Solution results

# Example
```julia
using HPRSOCP

model = build_from_mps("problem.mps")
params = HPRSOCP_parameters()
params.stoptol = 1e-8
params.warm_up = true
result = optimize(model, params)
```

See also: [`build_from_mps`](@ref), [`build_from_QAbc`](@ref)
"""
function optimize(model::QP_info_cpu, params::HPRSOCP_parameters)
    # Handle warmup if requested
    if params.warm_up
        if params.verbose
            println("="^80)
            println("WARM UP PHASE")
            println("  ℹ Running warmup to avoid JIT compilation overhead in main solve")
            println("="^80)
        end
        t_start_warmup = time()

        # Save original max_iter and verbose
        original_max_iter = params.max_iter
        original_verbose = params.verbose
        params.max_iter = 200
        params.verbose = false

        # Run warmup solve
        solve(model, params)

        # Restore original parameters
        params.max_iter = original_max_iter
        params.verbose = original_verbose

        warmup_time = time() - t_start_warmup
        if params.verbose
            println(@sprintf("Warmup time: %.2f seconds", warmup_time))
            println("="^80)
            println()
        end
    end

    # Main solve
    if params.verbose
        println("="^80)
        println("MAIN SOLVE")
        println("="^80)
    end

    # Run the main algorithm (scaling and GPU transfer happen inside solve)
    results = solve(model, params)

    return results
end

# Helper function: Determine if iteration should print
function should_print(iter::Int, params::HPRSOCP_parameters, t_start_alg::Float64)
    if params.print_frequency == -1
        return (rem(iter, print_step(iter)) == 0) ||
               (iter == params.max_iter) ||
               (time() - t_start_alg > params.time_limit)
    elseif params.print_frequency > 0
        return (rem(iter, params.print_frequency) == 0) ||
               (iter == params.max_iter) ||
               (time() - t_start_alg > params.time_limit)
    else
        error("Invalid print_frequency: ", params.print_frequency,
            ". It should be a positive integer or -1 for automatic printing.")
    end
end

"""
    process_initial_points!(ws, qp, params, scaling_info, m)

Process initial primal and dual points if provided in parameters.
Scales and assigns initial values to workspace variables.

# Arguments
- `ws`: Workspace containing solver state
- `qp`: QP problem information
- `params`: Solver parameters (containing initial_x and initial_y)
- `scaling_info`: Scaling information for the problem
- `m`: Number of constraints
"""
function process_initial_points!(
    ws::HPRSOCP_workspace,
    qp::HPRSOCP_QP_info,
    params::HPRSOCP_parameters,
    scaling_info::HPRSOCP_scaling,
    m::Int
)
    has_quadratic = has_quadratic_terms(qp.Q)

    # Process initial_x if provided
    if params.initial_x !== nothing
        # Convert to device array and scale
        WS = workspace_type(qp)
        initial_x_device = convert_to_device(WS, params.initial_x)
        scaled_x = initial_x_device .* scaling_info.col_norm ./ scaling_info.b_scale

        ws.x .= scaled_x
        ws.x_bar .= scaled_x
        ws.last_x .= scaled_x
        if has_quadratic
            ws.w .= scaled_x
            ws.w_bar .= scaled_x
            ws.last_w .= scaled_x
        end
    end

    # Process initial_y if provided (depends on lambda_max_A)
    if params.initial_y !== nothing
        # Warning: may have bug that quit with wrong result when we have initial points (<z,x> not equals to support function)
        # Convert to device array and scale
        WS = workspace_type(qp)
        ws.y .= convert_to_device(WS, params.initial_y)
        ws.y .= ws.y .* scaling_info.row_norm ./ scaling_info.c_scale
        ws.y_bar .= ws.y
        ws.last_y .= ws.y

        # Compute the A'*y scratch needed to initialize z_bar.
        if m > 0
            if has_quadratic
                unified_mul!(ws.ATy_bar, ws.AT, ws.y_bar)
                ws.ATy .= ws.ATy_bar
                ws.last_ATy .= ws.ATy_bar
            else
                unified_mul!(ws.ATy, ws.AT, ws.y_bar)
            end
        end

        # Compute z_bar from projection: z_bar = (x_bar - z_raw) / sigma
        # where z_raw = x_bar + sigma * (-Qx + ATy - c)
        if has_quadratic
            Qmap!(ws.x_bar, ws.Qx, qp.Q)
            tmp = .-ws.Qx .+ ws.ATy_bar .- ws.c
        else
            tmp = ws.ATy .- ws.c
        end
        z_raw = ws.x_bar .+ ws.sigma .* tmp
        ws.z_bar .= (ws.x_bar .- z_raw) ./ ws.sigma

        # Compute s for dual objective: s = proj_{[AL,AU]}(Ax - lambda_max_A * sigma * y)
        if m > 0
            unified_mul!(ws.Ax, ws.A, ws.x_bar)
            fact1 = ws.lambda_max_A * ws.sigma
            ws.s .= min.(max.(ws.Ax .- fact1 .* ws.y, ws.AL), ws.AU)
        end
    end
end

# CPU version of print_iteration_log
function print_iteration_log(iter::Int, residuals::HPRSOCP_residuals,
    ws::HPRSOCP_workspace, t_start_alg::Float64)
    println(@sprintf("%5.0f    %3.2e    %3.2e    %+7.6e    %+7.6e    %3.2e    %3.2e    %6.2f",
        iter,
        residuals.err_Rp_org_bar,
        residuals.err_Rd_org_bar,
        residuals.primal_obj_bar,
        residuals.dual_obj_bar,
        residuals.rel_gap_bar,
        ws.sigma,
        time() - t_start_alg))
end

# Helper function: Update milestone tracking for KKT thresholds
function update_milestone_tracking!(residuals::HPRSOCP_residuals, iter::Int,
    t_start_alg::Float64,
    iter_4::Int, time_4::Float64, first_4::Bool,
    iter_6::Int, time_6::Float64, first_6::Bool,
    verbose::Bool)
    if residuals.KKTx_and_gap_org_bar < 1e-4 && first_4
        time_4 = time() - t_start_alg
        iter_4 = iter
        first_4 = false
        if verbose
            println("KKT < 1e-4 at iter = ", iter)
        end
    end

    if residuals.KKTx_and_gap_org_bar < 1e-6 && first_6
        time_6 = time() - t_start_alg
        iter_6 = iter
        first_6 = false
        if verbose
            println("KKT < 1e-6 at iter = ", iter)
        end
    end

    return iter_4, time_4, first_4, iter_6, time_6, first_6
end

# Helper function: Perform main iteration step (update and norm computation)
# GPU version
function perform_iteration_step!(ws::HPRSOCP_workspace, qp::HPRSOCP_QP_info,
    params::HPRSOCP_parameters, restart_info::HPRSOCP_restart,
    iter::Int, check_iter::Int)
    # Main update - now handles both operator and sparse matrix Q within main_update_gpu!
    if isa(ws, HPRSOCP_workspace_gpu) && isa(qp, QP_info_gpu)
        main_update_gpu!(ws, qp, restart_info)
    else
        main_update_cpu!(ws, qp, restart_info)
    end
    need_last_gap = restart_info.restart_flag > 0
    need_current_gap = rem(iter + 1, check_iter) == 0
    if need_last_gap || need_current_gap
        m_norm = compute_M_norm!(ws, qp)
        if need_last_gap
            restart_info.last_gap = min(m_norm, restart_info.last_gap)
        end
        if need_current_gap
            restart_info.current_gap = m_norm
        end
    end

    restart_info.inner += 1
end

function count_empty_box_bounds(qp::QP_info_cpu)
    return sum((qp.l .== -Inf) .& (qp.u .== Inf))
end

function count_empty_box_bounds(qp::QP_info_gpu)
    empty_mask = ifelse.((qp.l .== -Inf) .& (qp.u .== Inf), Int32(1), Int32(0))
    return Int(unified_sum(empty_mask))
end

function has_mostly_empty_box_bounds(qp::HPRSOCP_QP_info)
    n = length(qp.l)
    if n == 0
        return false
    end
    return count_empty_box_bounds(qp) > 0.8 * n
end

# This function is the main solver function for the HPR-SOCP algorithm.
# It handles GPU transfer/CPU setup, scaling, and optimization.
function solve(model::QP_info_cpu, params::HPRSOCP_parameters)
    setup_start = time()

    # Validate GPU parameters before attempting GPU operations
    validate_gpu_parameters!(params)
    # Setup: GPU device (only if using GPU)
    if params.use_gpu
        CUDA.device!(params.device_number)
    end

    # Setup: GPU transfer and scaling
    diag_Q, Q_is_diag = nothing, false
    qp, transfer_time = prepare_model(model, params)

    scaling_info = scaling!(qp, params)

    diag_Q, Q_is_diag = check_Q_diagonal(qp)

    # Get problem dimensions
    m, n = size(qp.A)

    # Initialize workspace and solver state
    residuals = HPRSOCP_residuals()
    restart_info = initialize_restart()

    # Allocate workspace
    ws = allocate_workspace(qp, params, 0.0, 0.0, scaling_info, diag_Q, Q_is_diag)

    # Prepare CUSPARSE SpMV structures (GPU-only, no-op for CPU)
    prepare_workspace_spmv!(ws, qp, params.verbose)

    setup_time = time() - setup_start
    t_start_alg = time()

    # Estimate eigenvalues using power_iteration
    ws.lambda_max_A, ws.lambda_max_Q, power_time = estimate_eigenvalues(qp, params, ws)

    # Process initial points if provided
    process_initial_points!(ws, qp, params, scaling_info, m)

    configure_noq_soc_scratch_aty_mode!(ws)

    # Initialize best_sigma with the initial sigma value
    restart_info.best_sigma = ws.sigma
    print_problem_info(qp, ws, params)
    print_solver_params(params, qp)

    # Setup iteration tracking
    iter_4, time_4 = 0, 0.0
    iter_6, time_6 = 0, 0.0
    first_4, first_6 = true, true

    # Update Q factors for diagonal Q
    if ws.Q_is_diag
        unified_update_Q_factors!(ws.fact2, ws.fact, ws.fact1, ws.fact_M,
            ws.diag_Q, ws.sigma)
    end

    if params.verbose
        println("HPRSOCP SOLVER starts", params.use_gpu ? "..." : " (CPU mode)...")
        println(" iter     errRp     errRd         p_obj           d_obj          gap        sigma       time")
    end

    check_iter = params.check_iter

    # This heuristic must use the active device copy of the bounds so the
    # post-transfer solve path stays on GPU in GPU mode.
    ws.noC = has_mostly_empty_box_bounds(qp)

    # Main iteration loop
    for iter = 0:params.max_iter
        # Determine if we should print at this iteration
        print_yes = should_print(iter, params, t_start_alg, params.max_iter)

        # Compute residuals if needed
        if rem(iter, check_iter) == 0 || print_yes
            residuals.is_updated = true
            compute_residuals!(ws, qp, scaling_info, residuals, params, iter)
        else
            residuals.is_updated = false
        end

        # Check termination criteria
        status = check_break(residuals, iter, t_start_alg, params)

        # Check and perform restart if needed
        check_restart(
            restart_info,
            iter,
            check_iter,
            ws.sigma,
            residuals.KKTx_and_gap_org_bar,
            params.verbose,
        )
        # Update sigma parameter (dispatches to GPU or CPU version)
        update_sigma!(params, restart_info, ws, qp, residuals)
        # maybe_trigger_saved_best_backtrack!(restart_info, ws, residuals)

        # Perform restart
        do_restart!(restart_info, ws, qp)

        # Print iteration log
        if (print_yes || (status != "CONTINUE")) && params.verbose
            print_iteration_log(iter, residuals, ws, t_start_alg)
        end

        # Save to HDF5 if auto_save is enabled
        if (print_yes || (status != "CONTINUE")) && params.auto_save
            try
                save_state_to_hdf5!(params.save_filename, ws, scaling_info, residuals, params, iter, t_start_alg)
            catch e
                if params.verbose
                    println("Warning: Failed to save to HDF5 file: ", e)
                end
            end
        end

        # Update milestone tracking
        iter_4, time_4, first_4, iter_6, time_6, first_6 =
            update_milestone_tracking!(residuals, iter, t_start_alg,
                iter_4, time_4, first_4, iter_6, time_6, first_6, params.verbose)

        # Handle termination using unified function (dispatches based on workspace type)
        if status != "CONTINUE"
            return handle_termination(status, residuals, ws, scaling_info,
                iter, t_start_alg, power_time, setup_time,
                iter_4, time_4, iter_6, time_6, params.verbose)
        end

        next_iter = iter + 1
        ws.to_check = (rem(next_iter, check_iter) == 0) || (restart_info.restart_flag > 0)
        if params.print_frequency == -1
            ws.to_check = ws.to_check || (rem(next_iter, print_step(next_iter)) == 0)
        elseif params.print_frequency > 0
            ws.to_check = ws.to_check || (rem(next_iter, params.print_frequency) == 0)
        end

        # Perform main iteration step
        perform_iteration_step!(ws, qp, params, restart_info, iter, check_iter)
    end
end
