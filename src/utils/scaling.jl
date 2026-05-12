# This file is included by ../utils.jl.

# CPU-based scaling function for the QP problem (similar to GPU version)
# ============================================================================
# Unified Scaling Functions
# ============================================================================
#
# The scaling! function is unified to work with both CPU and GPU data.
# Device-specific operations are handled through helper functions that dispatch
# based on matrix/vector types (SparseMatrixCSC vs CuSparseMatrixCSR).
#
# Matrix Scaling Operations:
# --------------------------
# - Ruiz scaling: Equilibration using row/column max norms
# - Pock-Chambolle scaling: Equilibration using row/column sum norms
# - b/c scaling: Objective/constraint balancing (currently disabled)
#
# For sparse Q matrices, scaling is applied. For custom Q operators,
# scaling is skipped because the operator owns its normalization logic.
#
# ============================================================================

# Helper: Compute row-wise max for Ruiz scaling (CPU version)
function _compute_row_max_abs!(temp_col_norm::Vector{Float64},
    temp_norm_Q::Vector{Float64},
    A::SparseMatrixCSC, Q::SparseMatrixCSC)
    temp_col_norm .= vec(maximum(abs, A, dims=1))
    temp_norm_Q .= vec(maximum(abs, Q, dims=1))
    temp_col_norm .= max.(temp_col_norm, temp_norm_Q)
    temp_col_norm .= sqrt.(temp_col_norm)
    temp_col_norm[iszero.(temp_col_norm)] .= 1.0
    temp_col_norm .= clamp.(temp_col_norm, 1e-4, 1e4)
    return
end

# Helper: Compute row-wise max for Ruiz scaling (GPU version)
function _compute_row_max_abs!(temp_col_norm::CuVector{Float64},
    temp_norm_Q::CuVector{Float64},
    A::CuSparseMatrixCSR, Q::CuSparseMatrixCSR)
    AT_rowPtr = A.rowPtr  # For column-wise access we'd normally use AT
    AT_nzVal = A.nzVal
    Q_rowPtr = Q.rowPtr
    Q_nzVal = Q.nzVal
    n = length(temp_col_norm)

    # Note: For true GPU version, this should use AT not A
    # Assuming we have AT available through workspace
    @cuda threads = 256 blocks = ceil(Int, n / 256) compute_row_max_abs_with_Q_kernel!(
        AT_rowPtr, AT_nzVal, Q_rowPtr, Q_nzVal, temp_col_norm, n
    )
    CUDA.synchronize()
    return
end

# Helper: Compute row-wise sum for Pock-Chambolle scaling (CPU version)
function _compute_row_sum_abs!(temp_col_norm::Vector{Float64},
    temp_norm_Q::Vector{Float64},
    A::SparseMatrixCSC, Q::SparseMatrixCSC)
    temp_col_norm .= vec(sum(abs, A, dims=1))
    temp_norm_Q .= vec(sum(abs, Q, dims=1))
    temp_col_norm .= temp_col_norm .+ temp_norm_Q
    temp_col_norm .= sqrt.(temp_col_norm)
    temp_col_norm[iszero.(temp_col_norm)] .= 1.0
    temp_col_norm .= clamp.(temp_col_norm, 1e-4, 1e4)
    return
end

# Helper: Compute row-wise sum for Pock-Chambolle scaling (GPU version)
function _compute_row_sum_abs!(temp_col_norm::CuVector{Float64},
    temp_norm_Q::CuVector{Float64},
    A::CuSparseMatrixCSR, Q::CuSparseMatrixCSR)
    AT_rowPtr = A.rowPtr
    AT_nzVal = A.nzVal
    Q_rowPtr = Q.rowPtr
    Q_nzVal = Q.nzVal
    n = length(temp_col_norm)

    @cuda threads = 256 blocks = ceil(Int, n / 256) compute_col_sum_abs_with_Q_kernel!(
        AT_rowPtr, AT_nzVal, Q_rowPtr, Q_nzVal, temp_col_norm, n
    )
    CUDA.synchronize()
    return
end

# Helper: Scale matrix Q by diagonal matrices (CPU version)
function _scale_Q_matrix!(Q::SparseMatrixCSC, temp_col_norm::Vector{Float64})
    DC = spdiagm(1.0 ./ temp_col_norm)
    return DC * Q * DC
end

# Helper: Scale matrix Q by diagonal matrices (GPU version)
function _scale_Q_matrix!(Q::CuSparseMatrixCSR, temp_col_norm::CuVector{Float64})
    Q_rowPtr = Q.rowPtr
    Q_colVal = Q.colVal
    Q_nzVal = Q.nzVal
    n = length(temp_col_norm)

    threads, blocks = gpu_launch_config(n)
    if blocks > 0
        @cuda threads = threads blocks = blocks scale_csr_row_col_kernel!(
            Q_rowPtr, Q_colVal, Q_nzVal, temp_col_norm, temp_col_norm, n
        )
    end

    return Q  # Modified in-place
end

# Helper: Scale matrix A by row and column diagonal matrices (CPU version)
function _scale_A_matrix!(A::SparseMatrixCSC, temp_row_norm::Vector{Float64}, temp_col_norm::Vector{Float64})
    DR = spdiagm(1.0 ./ temp_row_norm)
    DC = spdiagm(1.0 ./ temp_col_norm)
    return DR * A * DC
end

# Helper: Scale matrix A by row and column diagonal matrices (GPU version)
function _scale_A_matrix!(A::CuSparseMatrixCSR, temp_row_norm::CuVector{Float64}, temp_col_norm::CuVector{Float64})
    A_rowPtr = A.rowPtr
    A_colVal = A.colVal
    A_nzVal = A.nzVal
    m = length(temp_row_norm)

    threads, blocks = gpu_launch_config(m)
    if blocks > 0
        @cuda threads = threads blocks = blocks scale_csr_row_col_kernel!(
            A_rowPtr, A_colVal, A_nzVal, temp_row_norm, temp_col_norm, m
        )
    end

    return A  # Modified in-place
end

const ADAPTIVE_SOC_BLOCK_THRESHOLD = 8
const POCK_RMS_SOC_BLOCK_THRESHOLD = 512
const MANY_SOC_CONSTRAINT_CONES_THRESHOLD = 100_000

uses_pre_pock_stage(strategy::Symbol) =
    strategy === :pre_pock_size_split ||
    strategy === :pre_pock_huge_var_taper ||
    strategy === :pre_pock_huge_var_taper_pock_geom ||
    strategy === :pre_pock_band_split ||
    strategy === :pre_pock_constraint_rms ||
    strategy === :pre_pock_constraint_max ||
    strategy === :pre_pock_constraint_taper ||
    strategy === :small_cone_constraint_max ||
    strategy === :count_aware_small_cone_bias

max_biased_soc_scale(block_max::Float64, rms::Float64) = sqrt(sqrt(block_max^3 * rms))
huge_var_taper_soc_scale(block_max::Float64, rms::Float64) = sqrt(sqrt(block_max * rms^3))

const SOC_SCALE_STRATEGY_MAX = Int32(1)
const SOC_SCALE_STRATEGY_RMS = Int32(2)
const SOC_SCALE_STRATEGY_GEOM = Int32(3)
const SOC_SCALE_STRATEGY_ADAPTIVE = Int32(4)
const SOC_SCALE_STRATEGY_HYBRID = Int32(5)
const SOC_SCALE_STRATEGY_RUIZ_GEOM = Int32(6)
const SOC_SCALE_STRATEGY_ROLE_SPLIT = Int32(7)
const SOC_SCALE_STRATEGY_POCK_SAFE_ROLE_SPLIT = Int32(8)
const SOC_SCALE_STRATEGY_SIZE_SPLIT_ROLE_SPLIT = Int32(9)
const SOC_SCALE_STRATEGY_PRE_POCK_SIZE_SPLIT = Int32(10)
const SOC_SCALE_STRATEGY_PRE_POCK_HUGE_VAR_TAPER = Int32(11)
const SOC_SCALE_STRATEGY_PRE_POCK_HUGE_VAR_TAPER_POCK_GEOM = Int32(12)
const SOC_SCALE_STRATEGY_PRE_POCK_BAND_SPLIT = Int32(13)
const SOC_SCALE_STRATEGY_PRE_POCK_CONSTRAINT_RMS = Int32(14)
const SOC_SCALE_STRATEGY_PRE_POCK_CONSTRAINT_MAX = Int32(15)
const SOC_SCALE_STRATEGY_PRE_POCK_CONSTRAINT_TAPER = Int32(16)
const SOC_SCALE_STRATEGY_SMALL_CONE_CONSTRAINT_MAX = Int32(17)
const SOC_SCALE_STRATEGY_COUNT_AWARE_SMALL_CONE_BIAS = Int32(18)
const SOC_SCALE_STRATEGY_PHASE_BLEND = Int32(19)
const SOC_SCALE_STRATEGY_PHASE_TAPER = Int32(20)

const SOC_SCALE_PHASE_GENERIC = Int32(0)
const SOC_SCALE_PHASE_RUIZ = Int32(1)
const SOC_SCALE_PHASE_PRE_POCK = Int32(2)
const SOC_SCALE_PHASE_POCK = Int32(3)

const SOC_SCALE_LOCATION_GENERIC = Int32(0)
const SOC_SCALE_LOCATION_CONSTRAINT = Int32(1)
const SOC_SCALE_LOCATION_VARIABLE = Int32(2)

function _throw_unsupported_soc_block_scaling_strategy(strategy::Symbol)
    error("Unsupported soc_block_scaling_strategy=$strategy. Supported modes are :max, :rms, :geom, :adaptive, :hybrid, :ruiz_geom, :role_split, :pock_safe_role_split, :size_split_role_split, :pre_pock_size_split, :pre_pock_huge_var_taper, :pre_pock_huge_var_taper_pock_geom, :pre_pock_band_split, :pre_pock_constraint_rms, :pre_pock_constraint_max, :pre_pock_constraint_taper, :small_cone_constraint_max, :count_aware_small_cone_bias, :phase_blend, and :phase_taper.")
end

@inline function soc_block_scaling_strategy_code(strategy::Symbol)
    if strategy === :max
        return SOC_SCALE_STRATEGY_MAX
    elseif strategy === :rms
        return SOC_SCALE_STRATEGY_RMS
    elseif strategy === :geom
        return SOC_SCALE_STRATEGY_GEOM
    elseif strategy === :adaptive
        return SOC_SCALE_STRATEGY_ADAPTIVE
    elseif strategy === :hybrid
        return SOC_SCALE_STRATEGY_HYBRID
    elseif strategy === :ruiz_geom
        return SOC_SCALE_STRATEGY_RUIZ_GEOM
    elseif strategy === :role_split
        return SOC_SCALE_STRATEGY_ROLE_SPLIT
    elseif strategy === :pock_safe_role_split
        return SOC_SCALE_STRATEGY_POCK_SAFE_ROLE_SPLIT
    elseif strategy === :size_split_role_split
        return SOC_SCALE_STRATEGY_SIZE_SPLIT_ROLE_SPLIT
    elseif strategy === :pre_pock_size_split
        return SOC_SCALE_STRATEGY_PRE_POCK_SIZE_SPLIT
    elseif strategy === :pre_pock_huge_var_taper
        return SOC_SCALE_STRATEGY_PRE_POCK_HUGE_VAR_TAPER
    elseif strategy === :pre_pock_huge_var_taper_pock_geom
        return SOC_SCALE_STRATEGY_PRE_POCK_HUGE_VAR_TAPER_POCK_GEOM
    elseif strategy === :pre_pock_band_split
        return SOC_SCALE_STRATEGY_PRE_POCK_BAND_SPLIT
    elseif strategy === :pre_pock_constraint_rms
        return SOC_SCALE_STRATEGY_PRE_POCK_CONSTRAINT_RMS
    elseif strategy === :pre_pock_constraint_max
        return SOC_SCALE_STRATEGY_PRE_POCK_CONSTRAINT_MAX
    elseif strategy === :pre_pock_constraint_taper
        return SOC_SCALE_STRATEGY_PRE_POCK_CONSTRAINT_TAPER
    elseif strategy === :small_cone_constraint_max
        return SOC_SCALE_STRATEGY_SMALL_CONE_CONSTRAINT_MAX
    elseif strategy === :count_aware_small_cone_bias
        return SOC_SCALE_STRATEGY_COUNT_AWARE_SMALL_CONE_BIAS
    elseif strategy === :phase_blend
        return SOC_SCALE_STRATEGY_PHASE_BLEND
    elseif strategy === :phase_taper
        return SOC_SCALE_STRATEGY_PHASE_TAPER
    end
    _throw_unsupported_soc_block_scaling_strategy(strategy)
end

@inline function soc_block_scaling_phase_code(phase::Symbol)
    if phase === :generic
        return SOC_SCALE_PHASE_GENERIC
    elseif phase === :ruiz
        return SOC_SCALE_PHASE_RUIZ
    elseif phase === :pre_pock
        return SOC_SCALE_PHASE_PRE_POCK
    elseif phase === :pock
        return SOC_SCALE_PHASE_POCK
    end
    error("Unsupported SOC block scaling phase: $phase")
end

@inline function soc_block_scaling_location_code(location::Symbol)
    if location === :generic
        return SOC_SCALE_LOCATION_GENERIC
    elseif location === :constraint
        return SOC_SCALE_LOCATION_CONSTRAINT
    elseif location === :variable
        return SOC_SCALE_LOCATION_VARIABLE
    end
    error("Unsupported SOC block scaling location: $location")
end

@inline function _soc_block_scale_value_scalar(
    block_len::Int,
    block_max::Float64,
    rms::Float64,
    strategy_code::Int32,
    phase_code::Int32,
    location_code::Int32,
    cone_count::Int32,
)
    block_len <= 0 && return 1.0
    strategy_code == SOC_SCALE_STRATEGY_MAX && return block_max

    if strategy_code == SOC_SCALE_STRATEGY_RMS
        return rms
    elseif strategy_code == SOC_SCALE_STRATEGY_GEOM
        return sqrt(block_max * rms)
    elseif strategy_code == SOC_SCALE_STRATEGY_ADAPTIVE
        return block_len <= ADAPTIVE_SOC_BLOCK_THRESHOLD ? block_max : rms
    elseif strategy_code == SOC_SCALE_STRATEGY_HYBRID
        if phase_code == SOC_SCALE_PHASE_RUIZ
            return block_len <= ADAPTIVE_SOC_BLOCK_THRESHOLD ? block_max : rms
        elseif phase_code == SOC_SCALE_PHASE_POCK
            return rms
        end
        return sqrt(block_max * rms)
    elseif strategy_code == SOC_SCALE_STRATEGY_RUIZ_GEOM
        if phase_code == SOC_SCALE_PHASE_RUIZ
            return block_len <= ADAPTIVE_SOC_BLOCK_THRESHOLD ? block_max : sqrt(block_max * rms)
        elseif phase_code == SOC_SCALE_PHASE_POCK
            return rms
        end
        return sqrt(block_max * rms)
    elseif strategy_code == SOC_SCALE_STRATEGY_ROLE_SPLIT
        if location_code == SOC_SCALE_LOCATION_CONSTRAINT
            return sqrt(block_max * rms)
        end
        if phase_code == SOC_SCALE_PHASE_RUIZ
            return block_len <= ADAPTIVE_SOC_BLOCK_THRESHOLD ? block_max : rms
        elseif phase_code == SOC_SCALE_PHASE_POCK
            return rms
        end
        return sqrt(block_max * rms)
    elseif strategy_code == SOC_SCALE_STRATEGY_POCK_SAFE_ROLE_SPLIT
        if phase_code == SOC_SCALE_PHASE_POCK
            return block_max
        end
        if location_code == SOC_SCALE_LOCATION_CONSTRAINT
            return sqrt(block_max * rms)
        end
        if phase_code == SOC_SCALE_PHASE_RUIZ
            return block_len <= ADAPTIVE_SOC_BLOCK_THRESHOLD ? block_max : rms
        end
        return sqrt(block_max * rms)
    elseif strategy_code == SOC_SCALE_STRATEGY_SIZE_SPLIT_ROLE_SPLIT
        if location_code == SOC_SCALE_LOCATION_CONSTRAINT
            return phase_code == SOC_SCALE_PHASE_POCK ? block_max : sqrt(block_max * rms)
        end
        if phase_code == SOC_SCALE_PHASE_RUIZ
            return block_len <= ADAPTIVE_SOC_BLOCK_THRESHOLD ? block_max : rms
        elseif phase_code == SOC_SCALE_PHASE_POCK
            return block_len <= POCK_RMS_SOC_BLOCK_THRESHOLD ? rms : block_max
        end
        return sqrt(block_max * rms)
    elseif strategy_code == SOC_SCALE_STRATEGY_PRE_POCK_SIZE_SPLIT
        if location_code == SOC_SCALE_LOCATION_CONSTRAINT
            return phase_code == SOC_SCALE_PHASE_POCK ? block_max : sqrt(block_max * rms)
        end
        if phase_code == SOC_SCALE_PHASE_RUIZ
            return block_len <= ADAPTIVE_SOC_BLOCK_THRESHOLD ? block_max : rms
        elseif phase_code == SOC_SCALE_PHASE_PRE_POCK
            return block_len <= POCK_RMS_SOC_BLOCK_THRESHOLD ? rms : block_max
        elseif phase_code == SOC_SCALE_PHASE_POCK
            return block_max
        end
        return sqrt(block_max * rms)
    elseif strategy_code == SOC_SCALE_STRATEGY_PRE_POCK_HUGE_VAR_TAPER
        if location_code == SOC_SCALE_LOCATION_CONSTRAINT
            return phase_code == SOC_SCALE_PHASE_POCK ? block_max : sqrt(block_max * rms)
        end
        if phase_code == SOC_SCALE_PHASE_RUIZ
            return block_len <= ADAPTIVE_SOC_BLOCK_THRESHOLD ? block_max : rms
        elseif phase_code == SOC_SCALE_PHASE_PRE_POCK
            return block_len <= POCK_RMS_SOC_BLOCK_THRESHOLD ? rms : huge_var_taper_soc_scale(block_max, rms)
        elseif phase_code == SOC_SCALE_PHASE_POCK
            return block_max
        end
        return sqrt(block_max * rms)
    elseif strategy_code == SOC_SCALE_STRATEGY_PRE_POCK_HUGE_VAR_TAPER_POCK_GEOM
        if location_code == SOC_SCALE_LOCATION_CONSTRAINT
            return phase_code == SOC_SCALE_PHASE_POCK ? block_max : sqrt(block_max * rms)
        end
        if phase_code == SOC_SCALE_PHASE_RUIZ
            return block_len <= ADAPTIVE_SOC_BLOCK_THRESHOLD ? block_max : rms
        elseif phase_code == SOC_SCALE_PHASE_PRE_POCK
            return block_len <= POCK_RMS_SOC_BLOCK_THRESHOLD ? rms : huge_var_taper_soc_scale(block_max, rms)
        elseif phase_code == SOC_SCALE_PHASE_POCK
            return block_len <= POCK_RMS_SOC_BLOCK_THRESHOLD ? block_max : sqrt(block_max * rms)
        end
        return sqrt(block_max * rms)
    elseif strategy_code == SOC_SCALE_STRATEGY_PRE_POCK_BAND_SPLIT
        if location_code == SOC_SCALE_LOCATION_CONSTRAINT
            return phase_code == SOC_SCALE_PHASE_POCK ? block_max : sqrt(block_max * rms)
        end
        if phase_code == SOC_SCALE_PHASE_RUIZ
            return block_len <= ADAPTIVE_SOC_BLOCK_THRESHOLD ? block_max : rms
        elseif phase_code == SOC_SCALE_PHASE_PRE_POCK
            return block_len <= ADAPTIVE_SOC_BLOCK_THRESHOLD ? block_max :
                   (block_len <= POCK_RMS_SOC_BLOCK_THRESHOLD ? rms : block_max)
        elseif phase_code == SOC_SCALE_PHASE_POCK
            return block_max
        end
        return sqrt(block_max * rms)
    elseif strategy_code == SOC_SCALE_STRATEGY_PRE_POCK_CONSTRAINT_RMS
        if location_code == SOC_SCALE_LOCATION_CONSTRAINT
            if phase_code == SOC_SCALE_PHASE_POCK
                return block_max
            elseif phase_code == SOC_SCALE_PHASE_PRE_POCK
                return rms
            end
            return sqrt(block_max * rms)
        end
        if phase_code == SOC_SCALE_PHASE_RUIZ
            return block_len <= ADAPTIVE_SOC_BLOCK_THRESHOLD ? block_max : rms
        elseif phase_code == SOC_SCALE_PHASE_PRE_POCK
            return block_len <= POCK_RMS_SOC_BLOCK_THRESHOLD ? rms : block_max
        elseif phase_code == SOC_SCALE_PHASE_POCK
            return block_max
        end
        return sqrt(block_max * rms)
    elseif strategy_code == SOC_SCALE_STRATEGY_PRE_POCK_CONSTRAINT_MAX
        if location_code == SOC_SCALE_LOCATION_CONSTRAINT
            if phase_code == SOC_SCALE_PHASE_POCK || phase_code == SOC_SCALE_PHASE_PRE_POCK
                return block_max
            end
            return sqrt(block_max * rms)
        end
        if phase_code == SOC_SCALE_PHASE_RUIZ
            return block_len <= ADAPTIVE_SOC_BLOCK_THRESHOLD ? block_max : rms
        elseif phase_code == SOC_SCALE_PHASE_PRE_POCK
            return block_len <= POCK_RMS_SOC_BLOCK_THRESHOLD ? rms : block_max
        elseif phase_code == SOC_SCALE_PHASE_POCK
            return block_max
        end
        return sqrt(block_max * rms)
    elseif strategy_code == SOC_SCALE_STRATEGY_PRE_POCK_CONSTRAINT_TAPER
        if location_code == SOC_SCALE_LOCATION_CONSTRAINT
            if phase_code == SOC_SCALE_PHASE_POCK
                return block_max
            elseif phase_code == SOC_SCALE_PHASE_PRE_POCK
                return huge_var_taper_soc_scale(block_max, rms)
            end
            return sqrt(block_max * rms)
        end
        if phase_code == SOC_SCALE_PHASE_RUIZ
            return block_len <= ADAPTIVE_SOC_BLOCK_THRESHOLD ? block_max : rms
        elseif phase_code == SOC_SCALE_PHASE_PRE_POCK
            return block_len <= POCK_RMS_SOC_BLOCK_THRESHOLD ? rms : block_max
        elseif phase_code == SOC_SCALE_PHASE_POCK
            return block_max
        end
        return sqrt(block_max * rms)
    elseif strategy_code == SOC_SCALE_STRATEGY_SMALL_CONE_CONSTRAINT_MAX
        if location_code == SOC_SCALE_LOCATION_CONSTRAINT
            if phase_code == SOC_SCALE_PHASE_POCK
                return block_max
            elseif phase_code == SOC_SCALE_PHASE_RUIZ || phase_code == SOC_SCALE_PHASE_PRE_POCK
                return block_len <= ADAPTIVE_SOC_BLOCK_THRESHOLD ? block_max : sqrt(block_max * rms)
            end
            return sqrt(block_max * rms)
        end
        if phase_code == SOC_SCALE_PHASE_RUIZ
            return block_len <= ADAPTIVE_SOC_BLOCK_THRESHOLD ? block_max : rms
        elseif phase_code == SOC_SCALE_PHASE_PRE_POCK
            return block_len <= POCK_RMS_SOC_BLOCK_THRESHOLD ? rms : block_max
        elseif phase_code == SOC_SCALE_PHASE_POCK
            return block_max
        end
        return sqrt(block_max * rms)
    elseif strategy_code == SOC_SCALE_STRATEGY_COUNT_AWARE_SMALL_CONE_BIAS
        if location_code == SOC_SCALE_LOCATION_CONSTRAINT
            is_many_small_cone_case =
                cone_count >= MANY_SOC_CONSTRAINT_CONES_THRESHOLD &&
                block_len <= ADAPTIVE_SOC_BLOCK_THRESHOLD
            if phase_code == SOC_SCALE_PHASE_POCK
                return block_max
            elseif is_many_small_cone_case
                if phase_code == SOC_SCALE_PHASE_RUIZ
                    return max_biased_soc_scale(block_max, rms)
                elseif phase_code == SOC_SCALE_PHASE_PRE_POCK
                    return block_max
                end
            end
            return sqrt(block_max * rms)
        end
        if phase_code == SOC_SCALE_PHASE_RUIZ
            return block_len <= ADAPTIVE_SOC_BLOCK_THRESHOLD ? block_max : rms
        elseif phase_code == SOC_SCALE_PHASE_PRE_POCK
            return block_len <= POCK_RMS_SOC_BLOCK_THRESHOLD ? rms : block_max
        elseif phase_code == SOC_SCALE_PHASE_POCK
            return block_max
        end
        return sqrt(block_max * rms)
    elseif strategy_code == SOC_SCALE_STRATEGY_PHASE_BLEND
        if phase_code == SOC_SCALE_PHASE_RUIZ
            return block_len <= ADAPTIVE_SOC_BLOCK_THRESHOLD ? block_max : rms
        elseif phase_code == SOC_SCALE_PHASE_POCK
            return block_len <= ADAPTIVE_SOC_BLOCK_THRESHOLD ? rms : sqrt(block_max * rms)
        end
        return sqrt(block_max * rms)
    elseif strategy_code == SOC_SCALE_STRATEGY_PHASE_TAPER
        if phase_code == SOC_SCALE_PHASE_RUIZ
            return block_len <= ADAPTIVE_SOC_BLOCK_THRESHOLD ? block_max : rms
        elseif phase_code == SOC_SCALE_PHASE_POCK
            return block_len <= ADAPTIVE_SOC_BLOCK_THRESHOLD ? rms :
                   sqrt(block_max * rms)
        end
        return sqrt(block_max * rms)
    end

    return block_max
end

function soc_block_scale_value(
    block_norms::AbstractVector{<:Real},
    strategy::Symbol,
    phase::Symbol=:generic,
    location::Symbol=:generic,
    cone_count::Integer=0,
)
    isempty(block_norms) && return 1.0

    block_max = Float64(maximum(block_norms))
    strategy_code = soc_block_scaling_strategy_code(strategy)
    strategy_code == SOC_SCALE_STRATEGY_MAX && return block_max

    rms = sqrt(sum(abs2, block_norms) / length(block_norms))
    return _soc_block_scale_value_scalar(
        length(block_norms),
        block_max,
        Float64(rms),
        strategy_code,
        soc_block_scaling_phase_code(phase),
        soc_block_scaling_location_code(location),
        Int32(cone_count),
    )
end

function _apply_soc_block_scaling!(
    temp_norms::AbstractVector{<:Real},
    cone_idx::AbstractVector{<:Integer},
    strategy::Symbol,
    phase::Symbol=:generic,
    location::Symbol=:generic,
)
    if length(cone_idx) <= 1
        return temp_norms
    end
    cone_count = length(cone_idx) - 1
    for i in 1:cone_count
        start_idx = cone_idx[i]
        end_idx = cone_idx[i+1] - 1
        if start_idx <= end_idx
            block_scale = soc_block_scale_value(
                @view(temp_norms[start_idx:end_idx]),
                strategy,
                phase,
                location,
                cone_count,
            )
            temp_norms[start_idx:end_idx] .= block_scale
        end
    end
    return temp_norms
end

function _apply_soc_block_scaling!(
    temp_norms::CuVector{Float64},
    cone_idx::CuVector{I},
    strategy::Symbol,
    phase::Symbol=:generic,
    location::Symbol=:generic,
) where {I<:Integer}
    if length(cone_idx) <= 1
        return temp_norms
    end

    num_blocks = length(cone_idx) - 1
    phase_code = soc_block_scaling_phase_code(phase)
    threads, blocks = gpu_launch_config(num_blocks)
    if strategy === :phase_taper
        @cuda threads = threads blocks = blocks apply_soc_block_scaling_phase_taper_kernel!(
            temp_norms,
            cone_idx,
            phase_code,
            Int32(num_blocks),
        )
        return temp_norms
    end

    strategy_code = soc_block_scaling_strategy_code(strategy)
    location_code = soc_block_scaling_location_code(location)
    @cuda threads = threads blocks = blocks apply_soc_block_scaling_kernel!(
        temp_norms,
        cone_idx,
        strategy_code,
        phase_code,
        location_code,
        Int32(num_blocks),
    )
    return temp_norms
end

function _apply_soc_block_max_scaling!(temp_norms::Vector{Float64}, cone_idx::AbstractVector{<:Integer})
    return _apply_soc_block_scaling!(temp_norms, cone_idx, :max)
end

function _apply_soc_block_max_scaling!(temp_norms::CuVector{Float64}, cone_idx::CuVector{I}) where {I<:Integer}
    return _apply_soc_block_scaling!(temp_norms, cone_idx, :max)
end

function _finalize_soc_scalar_scales!(scaling_info::Scaling_info_cpu, qp::QP_info_cpu)
    for i in 1:(length(qp.SOC_con_idx)-1)
        start_idx = qp.SOC_con_idx[i]
        end_idx = qp.SOC_con_idx[i+1] - 1
        max_norm = maximum(@view scaling_info.row_norm[start_idx:end_idx])
        scaling_info.SOC_con_scale[i] = 1.0 / (max_norm * scaling_info.b_scale)
    end
    for i in 1:(length(qp.SOC_var_idx)-1)
        start_idx = qp.SOC_var_idx[i]
        end_idx = qp.SOC_var_idx[i+1] - 1
        max_norm = maximum(@view scaling_info.col_norm[start_idx:end_idx])
        scaling_info.SOC_var_scale[i] = 1.0 / (max_norm * scaling_info.c_scale)
    end
    return scaling_info
end

function _finalize_soc_scalar_scales!(scaling_info::Scaling_info_gpu, qp::QP_info_gpu)
    num_con_blocks = length(qp.SOC_con_idx) - 1
    if num_con_blocks > 0
        threads, blocks = gpu_launch_config(num_con_blocks)
        @cuda threads = threads blocks = blocks finalize_soc_scalar_scales_kernel!(
            scaling_info.SOC_con_scale, scaling_info.row_norm, qp.SOC_con_idx, num_con_blocks, scaling_info.b_scale
        )
    end
    num_var_blocks = length(qp.SOC_var_idx) - 1
    if num_var_blocks > 0
        threads, blocks = gpu_launch_config(num_var_blocks)
        @cuda threads = threads blocks = blocks finalize_soc_scalar_scales_kernel!(
            scaling_info.SOC_var_scale, scaling_info.col_norm, qp.SOC_var_idx, num_var_blocks, scaling_info.c_scale
        )
    end
    return scaling_info
end

function _soc_rhs_norm(qp::HPRSOCP_QP_info, p::Real=Inf)
    if length(qp.SOC_con_idx) <= 1 || length(qp.soc_rhs) == 0
        return 0.0
    end
    return unified_norm(qp.soc_rhs, p)
end

clip_scaling_value(x::Real) = clamp(Float64(x), 1e-4, 1e4)

function _clip_scaling_norms!(v::AbstractVector{<:Real})
    v .= clamp.(v, 1e-4, 1e4)
    return v
end

_to_cpu_vector(v::Vector{T}) where {T} = v
_to_cpu_vector(v::CuVector{T}) where {T} = Array(v)

function _canonical_bc_norm_type(norm_type::Symbol)
    if norm_type === :l2
        return :l2
    elseif norm_type === :rms
        return :rms
    elseif norm_type === :linf || norm_type === :l_inf || norm_type === :inf
        return :linf
    end
    error("Unsupported bc scaling norm_type: $(norm_type). Use :l2, :rms, or :linf.")
end

function _finite_bound_norm(v::Vector{Float64}, norm_type::Symbol)
    canonical = _canonical_bc_norm_type(norm_type)
    if canonical === :linf
        accum = 0.0
        for x in v
            if isfinite(x)
                accum = max(accum, abs(x))
            end
        end
        return accum
    end
    accum = 0.0
    count = 0
    for x in v
        if isfinite(x)
            accum = muladd(x, x, accum)
            count += 1
        end
    end
    count == 0 && return 0.0
    if canonical === :rms
        return sqrt(accum / count)
    end
    return sqrt(accum)
end

function _finite_bound_norm(v::CuVector{Float64}, norm_type::Symbol)
    if isempty(v)
        return 0.0
    end
    canonical = _canonical_bc_norm_type(norm_type)
    finite_v = ifelse.(isfinite.(v), v, 0.0)
    if canonical === :linf
        return unified_norm(finite_v, Inf)
    end
    accum = CUDA.sum(abs2.(finite_v))
    if canonical === :rms
        count = Int(CUDA.sum(Int32.(isfinite.(v))))
        return count == 0 ? 0.0 : sqrt(accum / count)
    end
    return sqrt(accum)
end

function _vector_bc_norm(v::Vector{Float64}, norm_type::Symbol)
    isempty(v) && return 0.0
    canonical = _canonical_bc_norm_type(norm_type)
    if canonical === :linf
        return norm(v, Inf)
    elseif canonical === :rms
        return sqrt(sum(abs2, v) / length(v))
    end
    return norm(v, 2)
end

function _vector_bc_norm(v::CuVector{Float64}, norm_type::Symbol)
    isempty(v) && return 0.0
    canonical = _canonical_bc_norm_type(norm_type)
    if canonical === :linf
        return unified_norm(v, Inf)
    elseif canonical === :rms
        return sqrt(CUDA.sum(abs2.(v)) / length(v))
    end
    return unified_norm(v, 2)
end

function _soc_rhs_bc_norm(qp::HPRSOCP_QP_info, norm_type::Symbol)
    if length(qp.SOC_con_idx) <= 1 || isempty(qp.soc_rhs)
        return 0.0
    end
    return _vector_bc_norm(qp.soc_rhs, norm_type)
end

function _csr_row_abs_sums(A::CuSparseMatrixCSR)
    m, _ = size(A)
    row_sums = CUDA.zeros(Float64, m)
    if m == 0
        return row_sums
    end

    @cuda threads = 256 blocks = ceil(Int, m / 256) compute_row_abs_sum_kernel!(
        A.rowPtr, A.nzVal, row_sums, m
    )
    CUDA.synchronize()
    return row_sums
end

function compute_hat_s_b(qp::QP_info_cpu; norm_type::Symbol=:l2)
    max_bound = max(
        1.0,
        _finite_bound_norm(qp.AL, norm_type),
        _finite_bound_norm(qp.AU, norm_type),
        _soc_rhs_bc_norm(qp, norm_type),
    )
    return clip_scaling_value(max_bound)
end

function compute_hat_s_b(qp::QP_info_gpu; norm_type::Symbol=:l2)
    max_bound = max(
        1.0,
        _finite_bound_norm(qp.AL, norm_type),
        _finite_bound_norm(qp.AU, norm_type),
        _soc_rhs_bc_norm(qp, norm_type),
    )
    return clip_scaling_value(max_bound)
end

function compute_hat_s_c(qp::QP_info_cpu; norm_type::Symbol=:l2)
    c_norm = _vector_bc_norm(qp.c, norm_type)
    return clip_scaling_value(max(1.0, c_norm))
end

function compute_hat_s_c(qp::QP_info_gpu; norm_type::Symbol=:l2)
    c_norm = _vector_bc_norm(qp.c, norm_type)
    return clip_scaling_value(max(1.0, c_norm))
end

compute_bound_scale(qp::HPRSOCP_QP_info; norm_type::Symbol=:l2) = compute_hat_s_b(qp; norm_type)
compute_objective_scale(qp::HPRSOCP_QP_info; norm_type::Symbol=:l2) = compute_hat_s_c(qp; norm_type)

function compute_bc_scales(qp::HPRSOCP_QP_info; norm_type::Symbol=:l2, tau_min::Float64=1e-2, tau_max::Float64=1e2)
    hat_s_b = compute_hat_s_b(qp; norm_type)
    hat_s_c = compute_hat_s_c(qp; norm_type)
    s_b = hat_s_b
    s_c = clamp(hat_s_c, hat_s_b / tau_max, hat_s_b / tau_min)
    return s_b, s_c
end

"""
    scaling!(qp::HPRSOCP_QP_info, params::HPRSOCP_parameters)

Unified scaling function that works for both CPU and GPU QP problems.

This function applies various scaling strategies to improve numerical conditioning:
- Ruiz scaling: Row/column equilibration using max norms
- Pock-Chambolle scaling: Row/column equilibration using sum norms
- b/c scaling: Scalar normalization of the bounded linear block and the linear objective

For custom Q operators, scaling is skipped because the operator handles its own normalization.

# Arguments
- `qp::HPRSOCP_QP_info`: QP problem data (either QP_info_cpu or QP_info_gpu)
- `params::HPRSOCP_parameters`: Solver parameters controlling scaling options

# Returns
- `scaling_info`: Scaling information (Scaling_info_cpu or Scaling_info_gpu)

# Device-Specific Behavior
- CPU: Uses SparseArrays operations directly
- GPU: Uses CUDA kernels for parallel scaling operations
"""
function scaling!(qp::HPRSOCP_QP_info, params::HPRSOCP_parameters)
    device_name = isa(qp, QP_info_gpu) ? "GPU" : "CPU"

    if params.verbose
        println("SCALING QP ON $(device_name) ...")
    end
    t_start = time()

    # Perform scaling
    m, n = size(qp.A)
    soc_row_start = qp.number_linear_con + 1

    # Check if Q is an operator (not a sparse matrix)
    Q_is_operator = isa(qp.Q, Union{AbstractQOperator,AbstractQOperatorCPU})

    if Q_is_operator
        if params.verbose
            println("Q is an operator - skipping ALL scaling")
        end
        # Return minimal scaling info with no scaling applied
        row_norm = unified_ones_like(qp.AL)
        if m > 0
            row_norm = unified_ones_like(qp.AL)
        else
            row_norm = isa(qp, QP_info_gpu) ? CuVector{Float64}(undef, 0) : Vector{Float64}(undef, 0)
        end
        col_norm = unified_ones_like(qp.c)

        AL_nInf = copy(qp.AL)
        AU_nInf = copy(qp.AU)
        AL_nInf[qp.AL.==-Inf] .= 0.0
        AU_nInf[qp.AU.==Inf] .= 0.0
        norm_b_org = m > 0 ? max(unified_norm(max.(abs.(AL_nInf), abs.(AU_nInf)), Inf), _soc_rhs_norm(qp)) : 0.0
        norm_c_org = unified_norm(qp.c, Inf)

        # Create appropriate scaling info type
        if isa(qp, QP_info_gpu)
            scaling_info = Scaling_info_gpu(
                copy(qp.l), copy(qp.u),
                row_norm, col_norm,
                1.0, 1.0, 1.0, 1.0,
                norm_b_org, norm_c_org,
                CUDA.ones(Float64, max(length(qp.SOC_con_idx) - 1, 0)),
                CUDA.ones(Float64, max(length(qp.SOC_var_idx) - 1, 0))
            )
        else
            scaling_info = Scaling_info_cpu(
                copy(qp.l), copy(qp.u),
                row_norm, col_norm,
                1.0, 1.0, 1.0, 1.0,
                norm_b_org, norm_c_org,
                ones(Float64, max(length(qp.SOC_con_idx) - 1, 0)),
                ones(Float64, max(length(qp.SOC_var_idx) - 1, 0))
            )
        end

        scaling_info.norm_b = m > 0 ? max(unified_norm(max.(abs.(AL_nInf), abs.(AU_nInf))), _soc_rhs_norm(qp)) : 0.0
        scaling_info.norm_c = unified_norm(qp.c)


        return scaling_info
    end

    # For sparse Q, proceed with normal scaling
    # Initialize scaling vectors
    if isa(qp, QP_info_gpu)
        row_norm = CUDA.ones(Float64, m)
        col_norm = CUDA.ones(Float64, n)
    else
        row_norm = ones(Float64, m)
        col_norm = ones(Float64, n)
    end

    # Compute original norms for scaling info
    AL_nInf = copy(qp.AL)
    AU_nInf = copy(qp.AU)
    AL_nInf[qp.AL.==-Inf] .= 0.0
    AU_nInf[qp.AU.==Inf] .= 0.0
    norm_b_org = max(unified_norm(max.(abs.(AL_nInf), abs.(AU_nInf)), Inf), _soc_rhs_norm(qp))
    norm_c_org = unified_norm(qp.c, Inf)

    # Initialize scaling info
    if isa(qp, QP_info_gpu)
        scaling_info = Scaling_info_gpu(
            copy(qp.l), copy(qp.u),
            row_norm, col_norm,
            1.0, 1.0, 1.0, 1.0,
            norm_b_org, norm_c_org,
            CUDA.ones(Float64, max(length(qp.SOC_con_idx) - 1, 0)),
            CUDA.ones(Float64, max(length(qp.SOC_var_idx) - 1, 0))
        )
    else
        scaling_info = Scaling_info_cpu(
            copy(qp.l), copy(qp.u),
            row_norm, col_norm,
            1.0, 1.0, 1.0, 1.0,
            norm_b_org, norm_c_org,
            ones(Float64, max(length(qp.SOC_con_idx) - 1, 0)),
            ones(Float64, max(length(qp.SOC_var_idx) - 1, 0))
        )
    end

    # Temporary vectors for scaling
    if isa(qp, QP_info_gpu)
        temp_row_norm = CUDA.ones(Float64, m)
        temp_col_norm = CUDA.ones(Float64, n)
    else
        temp_row_norm = ones(Float64, m)
        temp_col_norm = ones(Float64, n)
    end

    n_threads = 0
    n_blocks = 0
    m_threads = 0
    m_blocks = 0
    q_nz_threads = 0
    q_nz_blocks = 0
    if isa(qp, QP_info_gpu)
        n_threads, n_blocks = gpu_launch_config(n)
        m_threads, m_blocks = gpu_launch_config(m)
        q_nz_threads, q_nz_blocks = gpu_launch_config(length(qp.Q.nzVal))
    end

    # Ruiz scaling
    if params.use_Ruiz_scaling
        for _ in 1:max(params.ruiz_iterations, 0)
            # Compute column-wise max of |Q| and |A| combined
            if isa(qp, QP_info_gpu)
                # GPU version: uses kernels
                AT_rowPtr = qp.AT.rowPtr
                AT_nzVal = qp.AT.nzVal
                Q_rowPtr = qp.Q.rowPtr
                Q_nzVal = qp.Q.nzVal
                @cuda threads = n_threads blocks = n_blocks compute_row_max_abs_with_Q_kernel!(
                    AT_rowPtr, AT_nzVal, Q_rowPtr, Q_nzVal, temp_col_norm, n
                )
            else
                # CPU version: uses direct operations
                temp_col_norm .= vec(maximum(abs, qp.A, dims=1))
                temp_norm_Q = vec(maximum(abs, qp.Q, dims=1))
                temp_col_norm .= sqrt.(max.(temp_col_norm, temp_norm_Q))
                temp_col_norm[iszero.(temp_col_norm)] .= 1.0
            end

            # Compute row-wise max of |A|
            if m > 0
                if isa(qp, QP_info_gpu)
                    A_rowPtr = qp.A.rowPtr
                    A_nzVal = qp.A.nzVal
                    @cuda threads = m_threads blocks = m_blocks compute_row_max_abs_kernel!(
                        A_rowPtr, A_nzVal, temp_row_norm, m
                    )
                else
                    temp_row_norm .= sqrt.(vec(maximum(abs, qp.A, dims=2)))
                    temp_row_norm[iszero.(temp_row_norm)] .= 1.0
                end
                _apply_soc_block_scaling!(temp_row_norm, qp.SOC_con_idx, params.soc_block_scaling_strategy, :ruiz, :constraint)
                _clip_scaling_norms!(temp_row_norm)
            end

            _apply_soc_block_scaling!(temp_col_norm, qp.SOC_var_idx, params.soc_block_scaling_strategy, :ruiz, :variable)
            _clip_scaling_norms!(temp_col_norm)

            # Update cumulative norms
            row_norm .*= temp_row_norm
            col_norm .*= temp_col_norm

            # Scale Q: Q = DC * Q * DC
            if isa(qp, QP_info_gpu)
                _scale_Q_matrix!(qp.Q, temp_col_norm)
            else
                DC = spdiagm(1.0 ./ temp_col_norm)
                qp.Q = DC * qp.Q * DC
            end

            # Scale A: A = DR * A * DC
            if m > 0
                if isa(qp, QP_info_gpu)
                    _scale_A_matrix!(qp.A, temp_row_norm, temp_col_norm)
                    _scale_A_matrix!(qp.AT, temp_col_norm, temp_row_norm)
                else
                    DR = spdiagm(1.0 ./ temp_row_norm)
                    DC = spdiagm(1.0 ./ temp_col_norm)
                    qp.A = DR * qp.A * DC
                end
            end

            # Scale objective and constraint bounds
            if isa(qp, QP_info_gpu)
                @cuda threads = n_threads blocks = n_blocks scale_vector_div_kernel!(
                    qp.c, temp_col_norm, n
                )

                if m > 0
                    @cuda threads = m_threads blocks = m_blocks scale_two_vectors_div_kernel!(
                        qp.AL, qp.AU, temp_row_norm, m
                    )
                    if !isempty(qp.soc_rhs)
                        qp.soc_rhs ./= @view temp_row_norm[soc_row_start:end]
                        qp.soc_rhs_full ./= temp_row_norm
                    end
                end

                @cuda threads = n_threads blocks = n_blocks scale_two_vectors_mul_kernel!(
                    qp.l, qp.u, temp_col_norm, n
                )
            else
                qp.c ./= temp_col_norm
                if m > 0
                    qp.AL ./= temp_row_norm
                    qp.AU ./= temp_row_norm
                    if !isempty(qp.soc_rhs)
                        qp.soc_rhs ./= @view temp_row_norm[soc_row_start:end]
                        qp.soc_rhs_full ./= temp_row_norm
                    end
                end
                qp.l .*= temp_col_norm
                qp.u .*= temp_col_norm
            end
        end
    end

    # Optional extra SOC stage inserted after Ruiz and before Pock.
    if uses_pre_pock_stage(params.soc_block_scaling_strategy)
        for _ in 1:2
            if isa(qp, QP_info_gpu)
                AT_rowPtr = qp.AT.rowPtr
                AT_nzVal = qp.AT.nzVal
                Q_rowPtr = qp.Q.rowPtr
                Q_nzVal = qp.Q.nzVal
                @cuda threads = n_threads blocks = n_blocks compute_row_max_abs_with_Q_kernel!(
                    AT_rowPtr, AT_nzVal, Q_rowPtr, Q_nzVal, temp_col_norm, n
                )
            else
                temp_col_norm .= vec(maximum(abs, qp.A, dims=1))
                temp_norm_Q = vec(maximum(abs, qp.Q, dims=1))
                temp_col_norm .= sqrt.(max.(temp_col_norm, temp_norm_Q))
                temp_col_norm[iszero.(temp_col_norm)] .= 1.0
            end

            if m > 0
                if isa(qp, QP_info_gpu)
                    A_rowPtr = qp.A.rowPtr
                    A_nzVal = qp.A.nzVal
                    @cuda threads = m_threads blocks = m_blocks compute_row_max_abs_kernel!(
                        A_rowPtr, A_nzVal, temp_row_norm, m
                    )
                else
                    temp_row_norm .= sqrt.(vec(maximum(abs, qp.A, dims=2)))
                    temp_row_norm[iszero.(temp_row_norm)] .= 1.0
                end
                _apply_soc_block_scaling!(temp_row_norm, qp.SOC_con_idx, params.soc_block_scaling_strategy, :pre_pock, :constraint)
                _clip_scaling_norms!(temp_row_norm)
            end

            _apply_soc_block_scaling!(temp_col_norm, qp.SOC_var_idx, params.soc_block_scaling_strategy, :pre_pock, :variable)
            _clip_scaling_norms!(temp_col_norm)

            row_norm .*= temp_row_norm
            col_norm .*= temp_col_norm

            if isa(qp, QP_info_gpu)
                _scale_Q_matrix!(qp.Q, temp_col_norm)
            else
                DC = spdiagm(1.0 ./ temp_col_norm)
                qp.Q = DC * qp.Q * DC
            end

            if m > 0
                if isa(qp, QP_info_gpu)
                    _scale_A_matrix!(qp.A, temp_row_norm, temp_col_norm)
                    _scale_A_matrix!(qp.AT, temp_col_norm, temp_row_norm)
                else
                    DR = spdiagm(1.0 ./ temp_row_norm)
                    DC = spdiagm(1.0 ./ temp_col_norm)
                    qp.A = DR * qp.A * DC
                end
            end

            if isa(qp, QP_info_gpu)
                @cuda threads = n_threads blocks = n_blocks scale_vector_div_kernel!(
                    qp.c, temp_col_norm, n
                )

                if m > 0
                    @cuda threads = m_threads blocks = m_blocks scale_two_vectors_div_kernel!(
                        qp.AL, qp.AU, temp_row_norm, m
                    )
                    if !isempty(qp.soc_rhs)
                        qp.soc_rhs ./= @view temp_row_norm[soc_row_start:end]
                        qp.soc_rhs_full ./= temp_row_norm
                    end
                end

                @cuda threads = n_threads blocks = n_blocks scale_two_vectors_mul_kernel!(
                    qp.l, qp.u, temp_col_norm, n
                )
            else
                qp.c ./= temp_col_norm
                if m > 0
                    qp.AL ./= temp_row_norm
                    qp.AU ./= temp_row_norm
                    if !isempty(qp.soc_rhs)
                        qp.soc_rhs ./= @view temp_row_norm[soc_row_start:end]
                        qp.soc_rhs_full ./= temp_row_norm
                    end
                end
                qp.l .*= temp_col_norm
                qp.u .*= temp_col_norm
            end
        end
    end

    # Pock-Chambolle scaling
    if params.use_Pock_Chambolle_scaling
        # Compute column-wise sum of |Q| and |A| combined
        if isa(qp, QP_info_gpu)
            AT_rowPtr = qp.AT.rowPtr
            AT_nzVal = qp.AT.nzVal
            Q_rowPtr = qp.Q.rowPtr
            Q_nzVal = qp.Q.nzVal
            @cuda threads = n_threads blocks = n_blocks compute_col_sum_abs_with_Q_kernel!(
                AT_rowPtr, AT_nzVal, Q_rowPtr, Q_nzVal, temp_col_norm, n
            )
        else
            temp_col_norm .= vec(sum(abs, qp.A, dims=1))
            temp_norm_Q = vec(sum(abs, qp.Q, dims=1))
            temp_col_norm .= sqrt.(temp_col_norm .+ temp_norm_Q)
            temp_col_norm[iszero.(temp_col_norm)] .= 1.0
        end

        # Compute row-wise sum of |A|
        if m > 0
            if isa(qp, QP_info_gpu)
                A_rowPtr = qp.A.rowPtr
                A_nzVal = qp.A.nzVal
                @cuda threads = m_threads blocks = m_blocks compute_row_sum_abs_kernel!(
                    A_rowPtr, A_nzVal, temp_row_norm, m
                )
            else
                temp_row_norm .= sqrt.(vec(sum(abs, qp.A, dims=2)))
                temp_row_norm[iszero.(temp_row_norm)] .= 1.0
            end
            _apply_soc_block_scaling!(temp_row_norm, qp.SOC_con_idx, params.soc_block_scaling_strategy, :pock, :constraint)
            _clip_scaling_norms!(temp_row_norm)
        end

        _apply_soc_block_scaling!(temp_col_norm, qp.SOC_var_idx, params.soc_block_scaling_strategy, :pock, :variable)
        _clip_scaling_norms!(temp_col_norm)

        # Update cumulative norms
        row_norm .*= temp_row_norm
        col_norm .*= temp_col_norm

        # Scale Q: Q = DC * Q * DC
        if isa(qp, QP_info_gpu)
            _scale_Q_matrix!(qp.Q, temp_col_norm)
        else
            DC = spdiagm(1.0 ./ temp_col_norm)
            qp.Q = DC * qp.Q * DC
        end

        # Scale A: A = DR * A * DC
        if m > 0
            if isa(qp, QP_info_gpu)
                _scale_A_matrix!(qp.A, temp_row_norm, temp_col_norm)
                _scale_A_matrix!(qp.AT, temp_col_norm, temp_row_norm)
            else
                DR = spdiagm(1.0 ./ temp_row_norm)
                DC = spdiagm(1.0 ./ temp_col_norm)
                qp.A = DR * qp.A * DC
            end
        end

        # Scale objective and bounds
        if isa(qp, QP_info_gpu)
            @cuda threads = n_threads blocks = n_blocks scale_vector_div_kernel!(
                qp.c, temp_col_norm, n
            )
            if m > 0
                @cuda threads = m_threads blocks = m_blocks scale_two_vectors_div_kernel!(
                    qp.AL, qp.AU, temp_row_norm, m
                )
                if !isempty(qp.soc_rhs)
                    qp.soc_rhs ./= @view temp_row_norm[soc_row_start:end]
                    qp.soc_rhs_full ./= temp_row_norm
                end
            end
            @cuda threads = n_threads blocks = n_blocks scale_two_vectors_mul_kernel!(
                qp.l, qp.u, temp_col_norm, n
            )
        else
            qp.c ./= temp_col_norm
            if m > 0
                qp.AL ./= temp_row_norm
                qp.AU ./= temp_row_norm
                if !isempty(qp.soc_rhs)
                    qp.soc_rhs ./= @view temp_row_norm[soc_row_start:end]
                    qp.soc_rhs_full ./= temp_row_norm
                end
            end
            qp.l .*= temp_col_norm
            qp.u .*= temp_col_norm
        end
    end

    # Scalar normalization for the bounded linear block and objective
    if params.use_bc_scaling
        b_scale, c_scale = HPRSOCP.compute_bc_scales(qp; norm_type=params.bc_scaling_norm_type)


        # b_scale = 0.1
        # if params.verbose
        #     println("b_scale: ", b_scale)
        #     println("c_scale: ", c_scale)
        # end

        # Scale Q
        if isa(qp, QP_info_gpu)
            scale_factor = b_scale / c_scale
            Q_nzVal = qp.Q.nzVal
            if q_nz_blocks > 0
                @cuda threads = q_nz_threads blocks = q_nz_blocks scale_vector_scalar_mul_kernel!(
                    Q_nzVal, scale_factor, length(Q_nzVal)
                )
            end
        else
            scale_factor = b_scale / c_scale
            qp.Q = qp.Q .* scale_factor
        end

        # Scale bounds and objective
        if isa(qp, QP_info_gpu)
            if m > 0
                @cuda threads = m_threads blocks = m_blocks scale_two_vectors_scalar_div_kernel!(
                    qp.AL, qp.AU, b_scale, m
                )
                if !isempty(qp.soc_rhs)
                    qp.soc_rhs ./= b_scale
                    qp.soc_rhs_full ./= b_scale
                end
            end
            @cuda threads = n_threads blocks = n_blocks scale_vector_scalar_div_kernel!(
                qp.c, c_scale, n
            )
            @cuda threads = n_threads blocks = n_blocks scale_two_vectors_scalar_div_kernel!(
                qp.l, qp.u, b_scale, n
            )
        else
            if m > 0
                qp.AL ./= b_scale
                qp.AU ./= b_scale
                if !isempty(qp.soc_rhs)
                    qp.soc_rhs ./= b_scale
                    qp.soc_rhs_full ./= b_scale
                end
            end
            qp.c ./= c_scale
            qp.l ./= b_scale
            qp.u ./= b_scale
        end

        scaling_info.b_scale = b_scale
        scaling_info.c_scale = c_scale
    else
        scaling_info.b_scale = 1.0
        scaling_info.c_scale = 1.0
    end

    # Update AT if CPU (GPU already modified in-place)
    if isa(qp, QP_info_cpu)
        qp.AT = qp.A'
    end

    # Symmetrize Q after scaling on CPU. The GPU path preserves symmetry through
    # diagonal scaling, and Q has already been symmetrized during model construction.
    if isa(qp, QP_info_cpu)
        # Symmetrize Q on CPU
        qp.Q = (qp.Q + transpose(qp.Q)) / 2
    end

    # Compute final norms
    AL_nInf = copy(qp.AL)
    AU_nInf = copy(qp.AU)
    AL_nInf[qp.AL.==-Inf] .= 0.0
    AU_nInf[qp.AU.==Inf] .= 0.0
    scaling_info.norm_b = max(unified_norm(max.(abs.(AL_nInf), abs.(AU_nInf))), _soc_rhs_norm(qp))
    scaling_info.norm_c = unified_norm(qp.c)

    # Store the cumulative scaling norms
    scaling_info.row_norm = row_norm
    scaling_info.col_norm = col_norm
    _finalize_soc_scalar_scales!(scaling_info, qp)

    if isa(qp, QP_info_gpu)
        CUDA.synchronize()
    end

    scaling_time = time() - t_start
    if params.verbose
        println("$(device_name) SCALING time: ", @sprintf("%.2f seconds", scaling_time))
    end

    return scaling_info
end

# ============================================================================
# Legacy Wrapper Functions
# ============================================================================

# ============================================================================
# Q Diagonal Check Function
# ============================================================================

"""
    check_Q_diagonal(qp::HPRSOCP_QP_info)

Check if Q matrix is diagonal (only needs sparsity pattern, not scaled values).
This should be called BEFORE scaling.

# Arguments
- `qp::HPRSOCP_QP_info`: QP problem data (QP_info_cpu or QP_info_gpu)

# Returns
- `diag_Q`: Diagonal elements of Q matrix (Vector on CPU, CuVector on GPU)
- `Q_is_diag::Bool`: True if Q is diagonal, false otherwise
"""
function check_Q_diagonal(qp::QP_info_cpu)
    _, n = size(qp.A)

    if isa(qp.Q, Union{AbstractQOperator,AbstractQOperatorCPU})
        return zeros(Float64, n), false
    end

    diag_Q = Vector(diag(qp.Q))
    temp_norm_Q = vec(sum(abs, qp.Q, dims=1))
    diag_Q_abs = abs.(diag_Q)
    Q_is_diag = all(temp_norm_Q .≈ diag_Q_abs)

    return diag_Q, Q_is_diag
end

function check_Q_diagonal(qp::QP_info_gpu)
    _, n = size(qp.A)

    if isa(qp.Q, Union{AbstractQOperator,AbstractQOperatorCPU})
        return CUDA.zeros(Float64, n), false
    end

    diag_Q = CUDA.zeros(Float64, n)
    if n == 0
        return diag_Q, true
    end

    is_diag = CUDA.fill(true, n)
    @cuda threads = 256 blocks = ceil(Int, n / 256) check_diagonal_kernel!(
        qp.Q.rowPtr, qp.Q.colVal, is_diag, n
    )
    @cuda threads = 256 blocks = ceil(Int, n / 256) extract_diagonal_csr_kernel!(
        qp.Q.rowPtr, qp.Q.colVal, qp.Q.nzVal, diag_Q, n
    )
    CUDA.synchronize()

    row_sums = _csr_row_abs_sums(qp.Q)
    diag_Q_abs = abs.(diag_Q)
    Q_is_diag = all(is_diag) && maximum(abs.(row_sums .- diag_Q_abs)) <= 1e-10

    return diag_Q, Q_is_diag
end

function mean(x::Vector{Float64})
    return sum(x) / length(x)
end

