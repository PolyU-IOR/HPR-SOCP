# This file is included by ../utils.jl.

# ==================== Build Functions (Public API) ====================

"""
    build_from_SOCP_data(Q, c, A, rhs, SOC_con_idx, number_eq, number_ineq, l, u, SOC_var_idx; obj_constant=0.0, verbose=true)

Build a mixed linear/SOC model in the package's canonical internal ordering:

- equality rows first
- linear inequality rows next
- SOC rows last
- linear variables first
- SOC variables last, grouped contiguously by cone

The input `rhs` is the full canonical right-hand side for all rows. Internally, the linear part
is stored in `AL`/`AU`, while the SOC part is stored separately in `soc_rhs`.

Current scope:
- supports the direct mixed linear + SOC canonical representation used by this package
- does not introduce the auxiliary-variable `A*x - b` reformulation from `soc_utils`
- expects rows/variables to already follow the canonical ordering described above
"""
function build_from_SOCP_data(
    Q::Union{SparseMatrixCSC,Matrix{Float64}},
    c::Vector{Float64},
    A::Union{SparseMatrixCSC,Matrix{Float64}},
    rhs::Vector{Float64},
    SOC_con_idx::Vector{Int},
    number_eq::Int,
    number_ineq::Int,
    l::Vector{Float64},
    u::Vector{Float64},
    SOC_var_idx::Vector{Int};
    obj_constant::Float64=0.0,
    verbose::Bool=true,
)
    Q_sparse = Q isa Matrix{Float64} ? sparse(Q) : copy(Q)
    A_sparse = A isa Matrix{Float64} ? sparse(A) : copy(A)
    c_vec = copy(c)
    rhs_vec = copy(rhs)
    l_vec = copy(l)
    u_vec = copy(u)
    soc_con_idx = copy(SOC_con_idx)
    soc_var_idx = copy(SOC_var_idx)

    m, n = size(A_sparse)
    length(c_vec) == n || error("Dimension mismatch: length(c) must equal number of columns of A.")
    length(rhs_vec) == m || error("Dimension mismatch: length(rhs) must equal number of rows of A.")
    length(l_vec) == n || error("Dimension mismatch: length(l) must equal number of variables.")
    length(u_vec) == n || error("Dimension mismatch: length(u) must equal number of variables.")
    number_eq >= 0 || error("number_eq must be nonnegative.")
    number_ineq >= 0 || error("number_ineq must be nonnegative.")
    number_eq + number_ineq <= m || error("number_eq + number_ineq cannot exceed the number of rows of A.")
    !isempty(soc_con_idx) || error("SOC_con_idx must contain at least one sentinel entry.")
    !isempty(soc_var_idx) || error("SOC_var_idx must contain at least one sentinel entry.")
    soc_con_idx[1] == number_eq + number_ineq + 1 || error("SOC_con_idx must start at number_eq + number_ineq + 1.")
    soc_con_idx[end] == m + 1 || error("SOC_con_idx must end at m + 1.")
    soc_rhs_vec = rhs_vec[(number_eq+number_ineq+1):end]
    soc_rhs_full = zeros(Float64, m)
    if !isempty(soc_rhs_vec)
        soc_rhs_full[(number_eq+number_ineq+1):end] .= soc_rhs_vec
    end

    number_lu_x = soc_var_idx[1] - 1
    soc_var_idx[end] == n + 1 || error("SOC_var_idx must end at n + 1.")
    0 <= number_lu_x <= n || error("Invalid SOC_var_idx start.")

    AL = fill(-Inf, m)
    AU = fill(Inf, m)
    if number_eq > 0
        AL[1:number_eq] .= rhs_vec[1:number_eq]
        AU[1:number_eq] .= rhs_vec[1:number_eq]
    end
    if number_ineq > 0
        lin_range = (number_eq+1):(number_eq+number_ineq)
        AL[lin_range] .= rhs_vec[lin_range]
    end

    # if verbose
    #     println("FORMULATING SOCP ...")
    #     println("  total rows = ", m, ", total cols = ", n)
    #     println("  linear constraints = ", number_eq+number_ineq)
    #     println("  SOC constraints = ", length(soc_con_idx) - 1)
    #     println("  SOC variables = ", length(soc_var_idx) - 1)
    # end

    Q_sparse = Q_sparse isa SparseMatrixCSC{Float64,Int32} ? Q_sparse : SparseMatrixCSC{Float64,Int32}(Q_sparse)
    A_sparse = A_sparse isa SparseMatrixCSC{Float64,Int32} ? A_sparse : SparseMatrixCSC{Float64,Int32}(A_sparse)

    return QP_info_cpu(
        SparseMatrixCSC{Float64,Int32}(Q_sparse),
        c_vec,
        SparseMatrixCSC{Float64,Int32}(A_sparse),
        SparseMatrixCSC{Float64,Int32}(A_sparse'),
        soc_rhs_vec,
        soc_rhs_full,
        AL,
        AU,
        soc_con_idx,
        number_eq + number_ineq,
        l_vec,
        u_vec,
        soc_var_idx,
        number_lu_x,
        obj_constant,
    )
end

"""
    build_from_cbf(filename::String; verbose::Bool=true)

Build an SOC-capable model from a CBF file using the canonical ordering used by the
reference implementation in `soc_utils/`.
"""
function build_from_cbf(filename::String; verbose::Bool=true)
    t_start = time()
    if verbose
        println("READING CBF FILE ... ", filename)
    end

    Q, c, A, rhs, SOC_con_idx, number_eq, number_ineq, l, u, SOC_var_idx, obj_constant = read_cbf(filename)

    if verbose
        println(@sprintf("READING FILE time: %.2f seconds", time() - t_start))
    end

    return build_from_SOCP_data(Q, c, A, rhs, SOC_con_idx, number_eq, number_ineq, l, u, SOC_var_idx;
        obj_constant=obj_constant, verbose=verbose)
end

# ============================================================================
