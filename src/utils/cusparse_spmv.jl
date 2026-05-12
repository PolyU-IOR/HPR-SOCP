# This file is included by ../utils.jl.

# CUSPARSE SpMV Preprocessing and Buffer Allocation
# ============================================================================

"""
    prepare_spmv_A!(A, AT, x_bar, x_hat, dx, Ax, y_bar, y, ATy)

Prepare CUSPARSE SpMV operations for A and AT matrices.
Allocates buffers and performs preprocessing (for CUDA >= 12.4).

# Arguments
- `A::CuSparseMatrixCSR`: The constraint matrix in CSR format
- `AT::CuSparseMatrixCSR`: The transpose of A in CSR format
- `x_bar, x_hat, dx, tempv::CuVector{Float64}`: Dense vectors for A operations
- `Ax::CuVector{Float64}`: Output vector for A*x
- `y_bar, y::CuVector{Float64}`: Dense vectors for AT operations
- `ATy::CuVector{Float64}`: Output vector for AT*y

# Returns
- `(spmv_A, spmv_AT)`: Tuple of CUSPARSE_spmv_A and CUSPARSE_spmv_AT structures
"""
function prepare_spmv_A!(A::CuSparseMatrixCSR{Float64,Int32},
    AT::CuSparseMatrixCSR{Float64,Int32},
    x_bar::CuVector{Float64},
    x_hat::CuVector{Float64},
    dx::CuVector{Float64},
    tempv::CuVector{Float64},
    Ax::CuVector{Float64},
    y_bar::CuVector{Float64},
    y::CuVector{Float64},
    ATy_bar::CuVector{Float64},
    ATy::CuVector{Float64})
    # Create matrix and vector descriptors
    desc_A = CUDA.CUSPARSE.CuSparseMatrixDescriptor(A, 'O')
    desc_x_bar = CUDA.CUSPARSE.CuDenseVectorDescriptor(x_bar)
    desc_x_hat = CUDA.CUSPARSE.CuDenseVectorDescriptor(x_hat)
    desc_dx = CUDA.CUSPARSE.CuDenseVectorDescriptor(dx)
    desc_tempv = CUDA.CUSPARSE.CuDenseVectorDescriptor(tempv)
    desc_Ax = CUDA.CUSPARSE.CuDenseVectorDescriptor(Ax)

    desc_AT = CUDA.CUSPARSE.CuSparseMatrixDescriptor(AT, 'O')
    desc_y_bar = CUDA.CUSPARSE.CuDenseVectorDescriptor(y_bar)
    desc_y = CUDA.CUSPARSE.CuDenseVectorDescriptor(y)
    desc_ATy_bar = CUDA.CUSPARSE.CuDenseVectorDescriptor(ATy_bar)
    desc_ATy = CUDA.CUSPARSE.CuDenseVectorDescriptor(ATy)

    CUSPARSE_handle = CUDA.CUSPARSE.handle()
    ref_one = Ref{Float64}(one(Float64))
    ref_zero = Ref{Float64}(zero(Float64))

    # Prepare A SpMV
    sz_A = Ref{Csize_t}(0)
    CUDA.CUSPARSE.cusparseSpMV_bufferSize(CUSPARSE_handle, 'N', ref_one, desc_A, desc_x_bar, ref_zero,
        desc_Ax, Float64, CUDA.CUSPARSE.CUSPARSE_SPMV_CSR_ALG2, sz_A)
    buf_A = CUDA.CuArray{UInt8}(undef, sz_A[])

    # Only call preprocess for CUDA >= 12.4
    if CUDA.CUSPARSE.version() >= v"12.4"
        CUDA.CUSPARSE.cusparseSpMV_preprocess(CUSPARSE_handle, 'N', ref_one, desc_A, desc_x_bar, ref_zero, desc_Ax,
            Float64, CUDA.CUSPARSE.CUSPARSE_SPMV_CSR_ALG2, buf_A)
    end

    spmv_A = CUSPARSE_spmv_A(CUSPARSE_handle, 'N', ref_one, desc_A, desc_x_bar, desc_x_hat, desc_dx,
        desc_tempv, ref_zero, desc_Ax, Float64, CUDA.CUSPARSE.CUSPARSE_SPMV_CSR_ALG2, buf_A)

    # Prepare AT SpMV
    sz_AT = Ref{Csize_t}(0)
    CUDA.CUSPARSE.cusparseSpMV_bufferSize(CUSPARSE_handle, 'N', ref_one, desc_AT, desc_y_bar, ref_zero,
        desc_ATy, Float64, CUDA.CUSPARSE.CUSPARSE_SPMV_CSR_ALG2, sz_AT)
    buf_AT = CUDA.CuArray{UInt8}(undef, sz_AT[])

    # Only call preprocess for CUDA >= 12.4
    if CUDA.CUSPARSE.version() >= v"12.4"
        CUDA.CUSPARSE.cusparseSpMV_preprocess(CUSPARSE_handle, 'N', ref_one, desc_AT, desc_y_bar, ref_zero, desc_ATy,
            Float64, CUDA.CUSPARSE.CUSPARSE_SPMV_CSR_ALG2, buf_AT)
    end

    spmv_AT = CUSPARSE_spmv_AT(CUSPARSE_handle, 'N', ref_one, desc_AT, desc_y_bar, desc_y,
        ref_zero, desc_ATy_bar, desc_ATy, Float64, CUDA.CUSPARSE.CUSPARSE_SPMV_CSR_ALG2, buf_AT)

    return spmv_A, spmv_AT
end

# Note: prepare_spmv_Q! is now in Q_operators/sparse_matrix_operator.jl

# ============================================================================
# Helper Functions for CUSPARSE SpMV Operations
# ============================================================================

"""
    spmv_A_operation!(ws, vec_in, vec_out)

Perform A * vec_in -> vec_out using preprocessed CUSPARSE if available.
Falls back to standard CUSPARSE.mv! if preprocessing not available.

# Note
This uses ws.spmv_A which contains the preprocessed buffer and descriptors.
The descriptor for vec_in is determined by which descriptor matches the input vector.
"""
function spmv_A_operation!(ws::HPRSOCP_workspace_gpu, vec_in::CuVector{Float64}, vec_out::CuVector{Float64})
    if ws.spmv_A !== nothing
        # Use preprocessed CUSPARSE spmv - need to determine which descriptor to use
        # The spmv_A struct has desc_x_bar, desc_x_hat, desc_dx
        # We'll use cusparseSpMV with the appropriate descriptor
        # For simplicity, create a temporary descriptor for the input vector
        desc_in = CUDA.CUSPARSE.CuDenseVectorDescriptor(vec_in)
        desc_out = CUDA.CUSPARSE.CuDenseVectorDescriptor(vec_out)
        CUDA.CUSPARSE.cusparseSpMV(ws.spmv_A.handle, ws.spmv_A.operator, ws.spmv_A.alpha,
            ws.spmv_A.desc_A, desc_in, ws.spmv_A.beta, desc_out,
            ws.spmv_A.compute_type, ws.spmv_A.alg, ws.spmv_A.buf)
    else
        CUDA.CUSPARSE.mv!('N', 1, ws.A, vec_in, 0, vec_out, 'O', CUDA.CUSPARSE.CUSPARSE_SPMV_CSR_ALG2)
    end
end

"""
    spmv_AT_operation!(ws, vec_in, vec_out)

Perform AT * vec_in -> vec_out using preprocessed CUSPARSE if available.
Falls back to standard CUSPARSE.mv! if preprocessing not available.
"""
function spmv_AT_operation!(ws::HPRSOCP_workspace_gpu, vec_in::CuVector{Float64}, vec_out::CuVector{Float64})
    if ws.spmv_AT !== nothing
        desc_in = CUDA.CUSPARSE.CuDenseVectorDescriptor(vec_in)
        desc_out = CUDA.CUSPARSE.CuDenseVectorDescriptor(vec_out)
        CUDA.CUSPARSE.cusparseSpMV(ws.spmv_AT.handle, ws.spmv_AT.operator, ws.spmv_AT.alpha,
            ws.spmv_AT.desc_AT, desc_in, ws.spmv_AT.beta, desc_out,
            ws.spmv_AT.compute_type, ws.spmv_AT.alg, ws.spmv_AT.buf)
    else
        CUDA.CUSPARSE.mv!('N', 1, ws.AT, vec_in, 0, vec_out, 'O', CUDA.CUSPARSE.CUSPARSE_SPMV_CSR_ALG2)
    end
end
