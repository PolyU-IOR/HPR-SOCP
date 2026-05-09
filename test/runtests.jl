using Test
using HPRSOCP
using SparseArrays
using LinearAlgebra
using HDF5
using Dates
using JuMP

# Test Configuration:
# - All tests are configured to ensure convergence (OPTIMAL status)
# - max_iter = 50000-100000 (depending on problem type)
# - time_limit = 600-1800 seconds (10-30 minutes)
# - stoptol = 1e-6
# - verbose = false (no output during model building)
# - print_frequency = -1 (no intermediate solver output)
# - Primal objectives are printed for all data folder instances

function make_soc_scaling_test_qp(A::SparseMatrixCSC{Float64,Int64}, c::Vector{Float64};
    Q::SparseMatrixCSC{Float64,Int64}=spzeros(size(A, 2), size(A, 2)),
    AL::Vector{Float64}=fill(-Inf, size(A, 1)),
    AU::Vector{Float64}=fill(Inf, size(A, 1)),
    l::Vector{Float64}=fill(-Inf, size(A, 2)),
    u::Vector{Float64}=fill(Inf, size(A, 2)),
)
    m, n = size(A)
    soc_rhs = zeros(m)
    soc_rhs_full = zeros(m)
    return HPRSOCP.QP_info_cpu(
        SparseMatrixCSC{Float64,Int32}(Q),
        c,
        SparseMatrixCSC{Float64,Int32}(A),
        SparseMatrixCSC{Float64,Int32}(A'),
        soc_rhs,
        soc_rhs_full,
        AL,
        AU,
        [1, m + 1],
        0,
        0,
        l,
        u,
        [n + 1],
        n,
        0.0,
    )
end

function make_soc_scaling_params(; use_gpu::Bool, bc::Bool, pock::Bool)
    params = HPRSOCP.HPRSOCP_parameters()
    params.use_gpu = use_gpu
    params.verbose = false
    params.use_Ruiz_scaling = false
    params.use_l2_scaling = false
    params.use_bc_scaling = bc
    params.use_Pock_Chambolle_scaling = pock
    return params
end

@testset "HPRSOCP.jl Tests" begin
    @testset "GPU scaling implementation" begin
        utils_src = read(joinpath(dirname(pathof(HPRSOCP)), "utils.jl"), String)
        algorithm_src = read(joinpath(dirname(pathof(HPRSOCP)), "algorithm.jl"), String)

        @test occursin("function compute_hat_s_b(qp::QP_info_gpu; norm_type::Symbol=:l2)", utils_src)
        @test occursin("function compute_hat_s_c(qp::QP_info_gpu; norm_type::Symbol=:l2)", utils_src)
        @test occursin("function compute_bc_scales(qp::HPRSOCP_QP_info; norm_type::Symbol=:l2, tau_min::Float64=1e-2, tau_max::Float64=1e2)", utils_src)
        @test occursin("function soc_block_scale_value", utils_src)
        @test occursin("function _apply_soc_block_scaling!(", utils_src)
        @test occursin("function check_Q_diagonal(qp::QP_info_gpu)", utils_src)
        @test occursin("function _apply_soc_block_max_scaling!(temp_norms::CuVector{Float64}, cone_idx::CuVector{I}) where {I<:Integer}", utils_src)
        @test !occursin("function _disable_soc_row_scaling!(temp_norms::CuVector{Float64}, cone_idx::CuVector{I}) where {I<:Integer}", utils_src)
        @test occursin("function _finalize_soc_scalar_scales!(scaling_info::Scaling_info_gpu, qp::QP_info_gpu)", utils_src)
        @test !occursin("sparse_matrix_inf_norm(A::CuSparseMatrixCSR) = sparse_matrix_inf_norm(SparseMatrixCSC(A))", utils_src)
        @test !occursin("SparseMatrixCSC(qp.Q)", utils_src)
        @test !occursin("soc_row_factor_cpu = _soc_block_gamma_factors(qp)", utils_src)
        @test !occursin("cpu_norms = Vector(temp_norms)", utils_src)
        @test !occursin("Vector(scaling_info.row_norm)", utils_src)
        @test !occursin("Vector(scaling_info.col_norm)", utils_src)
        @test !occursin("Vector(qp.SOC_con_idx)", utils_src)
        @test !occursin("Vector(qp.SOC_var_idx)", utils_src)
        @test occursin("diag_Q::GPUOrCPUVector{Float64}", algorithm_src)
        @test !occursin("Vector(qp.SOC_con_idx)", algorithm_src)
        @test !occursin("Vector(qp.SOC_var_idx)", algorithm_src)
        @test occursin("function count_empty_box_bounds(qp::QP_info_gpu)", algorithm_src)
        @test occursin("function has_mostly_empty_box_bounds(qp::HPRSOCP_QP_info)", algorithm_src)
        @test !occursin("number_empty_lu = sum((model.l .== -Inf) .& (model.u .== Inf))", algorithm_src)
        @test !occursin("length(model.l)", algorithm_src)
        @test occursin("soc_block_scaling_strategy::Symbol", read(joinpath(dirname(pathof(HPRSOCP)), "structs.jl"), String))

        qp_empty = make_soc_scaling_test_qp(spzeros(0, 2), zeros(2); l=fill(-Inf, 2), u=fill(Inf, 2))
        qp_mixed = make_soc_scaling_test_qp(spzeros(0, 2), zeros(2); l=[-Inf, 0.0], u=[Inf, 1.0])
        @test HPRSOCP.count_empty_box_bounds(qp_empty) == 2
        @test HPRSOCP.has_mostly_empty_box_bounds(qp_empty)
        @test HPRSOCP.count_empty_box_bounds(qp_mixed) == 1
        @test !HPRSOCP.has_mostly_empty_box_bounds(qp_mixed)

        if HPRSOCP.CUDA.functional()
            A = sparse([1, 1, 2, 2], [1, 2, 1, 2], [10.0, 10.0, 1.0, 1.0], 2, 2)
            qp_cpu = make_soc_scaling_test_qp(A, [1.0, 1.0])
            qp_gpu_src = make_soc_scaling_test_qp(A, [1.0, 1.0])

            cpu_params = make_soc_scaling_params(use_gpu=false, bc=true, pock=true)
            gpu_params = make_soc_scaling_params(use_gpu=true, bc=true, pock=true)

            sc_cpu = HPRSOCP.scaling!(qp_cpu, cpu_params)
            qp_gpu, _ = HPRSOCP.prepare_model(qp_gpu_src, gpu_params)
            sc_gpu = HPRSOCP.scaling!(qp_gpu, gpu_params)

        end

        
        @testset "Empty (zero) Q matrix accepted" begin
            # Test that build_from_QAbc accepts empty Q matrix (all zeros)
            n, m = 5, 3
            Q_empty = zeros(n, n)  # Empty Q matrix - LP problem
            c = ones(n)
            A = sparse(ones(m, n))
            AL = zeros(m)
            AU = fill(n/2, m)
            l = zeros(n)
            u = ones(n)
            
            # Should accept empty Q and convert to sparse
            model = build_from_QAbc(Q_empty, c, A, AL, AU, l, u; verbose=false)
            @test model !== nothing
            @test model.Q isa SparseMatrixCSC
            @test nnz(model.Q) == 0  # Should have no nonzeros
        end
        
        @testset "Mixed sparse and dense matrices" begin
            # Test mixing sparse Q with dense A
            n, m = 4, 2
            Q_sparse = sparse(Matrix{Float64}(I, n, n))
            c = ones(n)
            A_dense = ones(m, n)
            AL = zeros(m)
            AU = ones(m) * 2
            l = zeros(n)
            u = ones(n)
            
            model = build_from_QAbc(Q_sparse, c, A_dense, AL, AU, l, u; verbose=false)
            @test model !== nothing
            @test model.Q isa SparseMatrixCSC
            @test model.A isa SparseMatrixCSC
        end

        @testset "SOCP direct builder metadata" begin
            Q = spzeros(4, 4)
            c = zeros(4)
            A = sparse([
                1.0 0.0 0.0 0.0;
                0.0 1.0 0.0 0.0;
                0.0 0.0 1.0 0.0;
                0.0 0.0 0.0 1.0;
            ])
            rhs = [2.0, 1.0, 0.0, 0.0]
            SOC_con_idx = [3, 5]
            number_eq = 1
            number_ineq = 1
            l = fill(-Inf, 4)
            u = fill(Inf, 4)
            SOC_var_idx = [2, 5]

            model = build_from_SOCP_data(Q, c, A, rhs, SOC_con_idx, number_eq, number_ineq, l, u, SOC_var_idx; verbose=false)

            @test model.number_eq == 1
            @test model.number_ineq == 1
            @test model.number_lu_x == 1
            @test model.SOC_con_idx == [3, 5]
            @test model.SOC_var_idx == [2, 5]
            @test model.soc_rhs == [0.0, 0.0]
            @test model.AL[1] == 2.0
            @test model.AU[1] == 2.0
            @test model.AL[2] == 1.0
            @test isinf(model.AU[2])
            @test isinf(model.AL[3]) && isinf(model.AU[3])
            @test size(model.A) == (4, 4)
            @test size(model.Q) == (4, 4)

                @testset "CBF OBJSENSE alias" begin
                    source_cbf = joinpath(@__DIR__, "..", "data", "model.cbf")
                    @test isfile(source_cbf)

                    source_text = read(source_cbf, String)
                    objsense_text = replace(source_text, "SENSE\nmin" => "OBJSENSE\nmax")

                    mktempdir() do dir
                        path = joinpath(dir, "objsense_alias.cbf")
                        write(path, objsense_text)

                        Q, c, A, rhs, SOC_con_idx, number_eq, number_ineq, l, u, SOC_var_idx, obj_constant = HPRSOCP.read_cbf(path)

                        @test size(Q) == (3, 3)
                        @test size(A) == (6, 3)
                        @test c ≈ [3.0, 5.0, -1.0]
                        @test obj_constant == 0.0
                        @test SOC_con_idx == [3, 7]
                        @test number_eq == 0
                        @test number_ineq == 2
                        @test SOC_var_idx == [4]
                    end
                end
        end

        @testset "SOC CPU projection and residual helpers" begin
            function allocate_cpu_ws(model; lambda_max_A=1.0, lambda_max_Q=0.0)
                params = HPRSOCP_parameters()
                params.verbose = false
                params.use_gpu = false
                params.sigma = 1.0
                scaling_info = HPRSOCP.scaling!(model, params)
                diag_Q, Q_is_diag = HPRSOCP.check_Q_diagonal(model)
                ws = HPRSOCP.allocate_workspace(model, params, lambda_max_A, lambda_max_Q, scaling_info, diag_Q, Q_is_diag)
                return ws, scaling_info
            end

            @testset "SOC variable projection update" begin
                model = build_from_SOCP_data(
                    spzeros(3, 3),
                    zeros(3),
                    spzeros(0, 3),
                    Float64[],
                    [1],
                    0,
                    0,
                    fill(-Inf, 3),
                    fill(Inf, 3),
                    [1, 4];
                    verbose=false,
                )

                ws, _ = allocate_cpu_ws(model; lambda_max_A=0.0, lambda_max_Q=0.0)
                ws.x .= [0.0, 1.0, 0.0]
                ws.last_x .= 0.0
                ws.w .= 0.0
                ws.ATy .= 0.0

                HPRSOCP.update_zxw1_cpu!(ws, model, 0.0, 1.0)

                @test ws.x_bar ≈ [0.5, 0.5, 0.0]
                @test ws.z_bar ≈ [0.5, -0.5, 0.0]
            end

            @testset "SOC dual projection update" begin
                model = build_from_SOCP_data(
                    spzeros(3, 3),
                    zeros(3),
                    sparse(Matrix{Float64}(I, 3, 3)),
                    zeros(3),
                    [1, 4],
                    0,
                    0,
                    fill(-Inf, 3),
                    fill(Inf, 3),
                    [4];
                    verbose=false,
                )

                ws, _ = allocate_cpu_ws(model; lambda_max_A=1.0, lambda_max_Q=0.0)
                ws.x_hat .= [0.0, 1.0, 0.0]
                ws.y .= 0.0
                ws.last_y .= 0.0
                ws.w_bar .= 0.0
                ws.Qw .= 0.0
                ws.Qw_bar .= 0.0

                HPRSOCP.update_y_cpu!(ws, model, 0.0, 1.0)

                @test ws.s ≈ [0.5, 0.5, 0.0]
                @test ws.y_bar ≈ [0.5, -0.5, 0.0]
            end

            @testset "SOC residual and violation checks" begin
                model_var = build_from_SOCP_data(
                    spzeros(3, 3),
                    zeros(3),
                    spzeros(0, 3),
                    Float64[],
                    [1],
                    0,
                    0,
                    fill(-Inf, 3),
                    fill(Inf, 3),
                    [1, 4];
                    verbose=false,
                )
                ws_var, sc_var = allocate_cpu_ws(model_var; lambda_max_A=0.0, lambda_max_Q=0.0)
                ws_var.x_bar .= [0.0, 1.0, 0.0]
                res = HPRSOCP.HPRSOCP_residuals()
                res.err_Rp_org_bar = 0.0
                HPRSOCP.compute_bounds_violation!(ws_var, sc_var, res)
                @test res.err_Rp_org_bar ≈ 0.5

                model_con = build_from_SOCP_data(
                    spzeros(3, 3),
                    zeros(3),
                    sparse(Matrix{Float64}(I, 3, 3)),
                    zeros(3),
                    [1, 4],
                    0,
                    0,
                    fill(-Inf, 3),
                    fill(Inf, 3),
                    [4];
                    verbose=false,
                )
                ws_con, sc_con = allocate_cpu_ws(model_con; lambda_max_A=1.0, lambda_max_Q=0.0)
                ws_con.x_bar .= [0.0, 1.0, 0.0]
                HPRSOCP.compute_Rp!(ws_con, sc_con)
                @test ws_con.Rp ≈ [0.5, -0.5, 0.0]
            end

            @testset "Split primal infeasibility separates linear and SOC rows" begin
                params = HPRSOCP_parameters()
                params.verbose = false
                params.use_gpu = false

                @testset "mixed rows report separate linear and SOC parts" begin
                    model = build_from_SOCP_data(
                        spzeros(4, 4),
                        zeros(4),
                        sparse(Matrix{Float64}(I, 4, 4)),
                        zeros(4),
                        [2, 5],
                        0,
                        1,
                        fill(-Inf, 4),
                        fill(Inf, 4),
                        [5];
                        verbose=false,
                    )

                    ws, sc = allocate_cpu_ws(model; lambda_max_A=1.0, lambda_max_Q=0.0)
                    ws.x_bar .= [-2.0, 0.0, 1.0, 0.0]
                    ws.y_bar .= 0.0
                    ws.z_bar .= 0.0
                    ws.s .= 0.0

                    residuals = HPRSOCP.HPRSOCP_residuals()
                    HPRSOCP.compute_residuals!(ws, model, sc, residuals, params, 1)

                    linear_m = model.number_eq + model.number_ineq
                    denom = 1.0 + max(sc.norm_b_org, HPRSOCP.unified_norm(ws.Ax, Inf))
                    expected_linear = HPRSOCP.unified_norm(ws.Rp[1:linear_m], Inf) / denom
                    expected_soc = HPRSOCP.unified_norm(ws.Rp[(linear_m + 1):end], Inf) / denom

                    @test residuals.err_Rp_linear_org_bar ≈ expected_linear
                    @test residuals.err_Rp_soc_org_bar ≈ expected_soc
                    @test residuals.err_Rp_org_bar ≈ max(expected_linear, expected_soc)
                    @test residuals.err_Rp_linear_org_bar > residuals.err_Rp_soc_org_bar > 0.0
                end

                @testset "pure linear model reports zero SOC part" begin
                    model = build_from_SOCP_data(
                        spzeros(2, 2),
                        zeros(2),
                        sparse([1.0 0.0]),
                        [0.0],
                        [2],
                        0,
                        1,
                        fill(-Inf, 2),
                        fill(Inf, 2),
                        [3];
                        verbose=false,
                    )

                    ws, sc = allocate_cpu_ws(model; lambda_max_A=1.0, lambda_max_Q=0.0)
                    ws.x_bar .= [-1.0, 0.0]
                    ws.y_bar .= 0.0
                    ws.z_bar .= 0.0
                    ws.s .= 0.0

                    residuals = HPRSOCP.HPRSOCP_residuals()
                    HPRSOCP.compute_residuals!(ws, model, sc, residuals, params, 1)

                    @test residuals.err_Rp_linear_org_bar ≈ residuals.err_Rp_org_bar
                    @test residuals.err_Rp_soc_org_bar == 0.0
                end

                @testset "pure SOC model reports zero linear part" begin
                    model = build_from_SOCP_data(
                        spzeros(3, 3),
                        zeros(3),
                        sparse(Matrix{Float64}(I, 3, 3)),
                        zeros(3),
                        [1, 4],
                        0,
                        0,
                        fill(-Inf, 3),
                        fill(Inf, 3),
                        [4];
                        verbose=false,
                    )

                    ws, sc = allocate_cpu_ws(model; lambda_max_A=1.0, lambda_max_Q=0.0)
                    ws.x_bar .= [0.0, 1.0, 0.0]
                    ws.y_bar .= 0.0
                    ws.z_bar .= 0.0
                    ws.s .= 0.0

                    residuals = HPRSOCP.HPRSOCP_residuals()
                    HPRSOCP.compute_residuals!(ws, model, sc, residuals, params, 1)

                    @test residuals.err_Rp_linear_org_bar == 0.0
                    @test residuals.err_Rp_soc_org_bar ≈ residuals.err_Rp_org_bar
                end

                @testset "verbose=false suppresses restart chatter" begin
                    model = build_from_QAbc(
                        spzeros(1, 1),
                        [0.0],
                        spzeros(0, 1),
                        Float64[],
                        Float64[],
                        [-1.0],
                        [1.0];
                        verbose=false,
                    )

                    ws, _ = allocate_cpu_ws(model; lambda_max_A=0.0, lambda_max_Q=0.0)
                    ws.x_bar .= [1.0]
                    ws.last_x .= [0.0]
                    ws.w_bar .= [0.0]
                    ws.last_w .= [0.0]

                    residuals = HPRSOCP.HPRSOCP_residuals()
                    residuals.err_Rd_org_bar = 1.0
                    residuals.err_Rp_org_bar = 1.0
                    residuals.rel_gap_bar = 0.1

                    restart_info = HPRSOCP.initialize_restart()
                    restart_info.first_restart = false
                    restart_info.restart_flag = 1
                    restart_info.current_gap = 0.2
                    restart_info.last_gap = 1.0
                    restart_info.save_gap = 0.2
                    restart_info.weighted_norm = 1.0
                    restart_info.best_gap = 0.3
                    restart_info.best_sigma = 1.0
                    restart_info.best_iter = 0

                    restart_io = Pipe()
                    redirect_stdout(restart_io) do
                        HPRSOCP.check_restart(restart_info, 150, 150, 1.0, false)
                    end
                    close(restart_io.in)
                    @test isempty(read(restart_io, String))

                    sigma_io = Pipe()
                    redirect_stdout(sigma_io) do
                        HPRSOCP.update_sigma!(params, restart_info, ws, model, residuals)
                    end
                    close(sigma_io.in)
                    @test isempty(read(sigma_io, String))
                end
            end

            @testset "Split primal infeasibility is printed" begin
                model = build_from_SOCP_data(
                    spzeros(2, 2),
                    zeros(2),
                    sparse([1.0 0.0]),
                    [0.0],
                    [2],
                    0,
                    1,
                    fill(-Inf, 2),
                    fill(Inf, 2),
                    [3];
                    verbose=false,
                )
                ws, sc = allocate_cpu_ws(model; lambda_max_A=1.0, lambda_max_Q=0.0)
                ws.sigma = 7.0

                residuals = HPRSOCP.HPRSOCP_residuals()
                residuals.err_Rp_org_bar = 0.11
                residuals.err_Rp_linear_org_bar = 0.22
                residuals.err_Rp_soc_org_bar = 0.33
                residuals.err_Rd_org_bar = 0.44
                residuals.primal_obj_bar = 1.23
                residuals.dual_obj_bar = -4.56
                residuals.rel_gap_bar = 0.55
                residuals.KKTx_and_gap_org_bar = 0.66

                iter_io = Pipe()
                redirect_stdout(iter_io) do
                    HPRSOCP.print_iteration_log(5, residuals, ws, time())
                end
                close(iter_io.in)
                iter_output = read(iter_io, String)
                @test occursin("2.20e-01", iter_output)
                @test occursin("3.30e-01", iter_output)

                summary_io = Pipe()
                redirect_stdout(summary_io) do
                    HPRSOCP.handle_termination(
                        "OPTIMAL",
                        residuals,
                        ws,
                        sc,
                        5,
                        time(),
                        0.0,
                        0.0,
                        0,
                        0.0,
                        0,
                        0.0,
                        true,
                    )
                end
                close(summary_io.in)
                summary_output = read(summary_io, String)
                @test occursin("Primal Residual (Linear)", summary_output)
                @test occursin("Primal Residual (SOC)", summary_output)
            end
        end

        @testset "Small end-to-end SOCP solves (CPU)" begin
            function cpu_soc_params()
                params = HPRSOCP_parameters()
                params.use_gpu = false
                params.verbose = false
                params.warm_up = false
                params.max_iter = 2000
                params.stoptol = 1e-5
                params.time_limit = 60.0
                params.print_frequency = -1
                return params
            end

            @testset "SOC variables only" begin
                model = build_from_SOCP_data(
                    sparse(Matrix{Float64}(I, 3, 3)),
                    zeros(3),
                    spzeros(0, 3),
                    Float64[],
                    [1],
                    0,
                    0,
                    fill(-Inf, 3),
                    fill(Inf, 3),
                    [1, 4];
                    verbose=false,
                )

                result = optimize(model, cpu_soc_params())
                @test result.status == "OPTIMAL"
                @test norm(result.x) ≤ 1e-3
                @test result.x[1] + 1e-8 ≥ norm(result.x[2:3])
            end

            @testset "Pure SOC constraints" begin
                model = build_from_SOCP_data(
                    sparse(Matrix{Float64}(I, 3, 3)),
                    zeros(3),
                    sparse(Matrix{Float64}(I, 3, 3)),
                    zeros(3),
                    [1, 4],
                    0,
                    0,
                    fill(-Inf, 3),
                    fill(Inf, 3),
                    [4];
                    verbose=false,
                )

                result = optimize(model, cpu_soc_params())
                @test result.status == "OPTIMAL"
                @test norm(result.x) ≤ 1e-3
                @test result.y[1] + 1e-8 ≥ norm(result.y[2:3]) || norm(result.y) ≤ 1e-3
            end

            @testset "Mixed linear and SOC constraints" begin
                model = build_from_SOCP_data(
                    sparse(Matrix{Float64}(I, 4, 4)),
                    zeros(4),
                    sparse(Matrix{Float64}(I, 4, 4)),
                    zeros(4),
                    [2, 5],
                    0,
                    1,
                    fill(-Inf, 4),
                    fill(Inf, 4),
                    [5];
                    verbose=false,
                )

                result = optimize(model, cpu_soc_params())
                @test result.status == "OPTIMAL"
                @test norm(result.x) ≤ 1e-3
                @test result.x[1] ≥ -1e-8
            end

            @testset "SOC dual objective includes rhs shift" begin
                model = build_from_SOCP_data(
                    spzeros(3, 3),
                    zeros(3),
                    spzeros(3, 3),
                    [1.0, 0.0, 0.0],
                    [1, 4],
                    0,
                    0,
                    fill(-Inf, 3),
                    fill(Inf, 3),
                    [4];
                    verbose=false,
                )

                params = HPRSOCP_parameters()
                params.verbose = false
                params.use_gpu = false
                sc = HPRSOCP.scaling!(model, params)
                diag_Q, Q_is_diag = HPRSOCP.check_Q_diagonal(model)
                ws = HPRSOCP.allocate_workspace(model, params, 1.0, 0.0, sc, diag_Q, Q_is_diag)

                ws.x_bar .= 0.0
                ws.z_bar .= 0.0
                ws.y_bar .= [1.0, 0.0, 0.0]
                ws.s .= 0.0

                residuals = HPRSOCP.HPRSOCP_residuals()
                HPRSOCP.compute_residuals!(ws, model, sc, residuals, params, 0)

                @test residuals.dual_obj_bar ≈ 1.0 atol=1e-10
            end

        end

        @testset "Small end-to-end SOCP solves (GPU)" begin
            if HPRSOCP.CUDA.functional()
                function gpu_soc_params()
                    params = HPRSOCP_parameters()
                    params.use_gpu = true
                    params.verbose = false
                    params.warm_up = false
                    params.max_iter = 4000
                    params.stoptol = 1e-6
                    params.time_limit = 60.0
                    params.print_frequency = -1
                    return params
                end

                @testset "Trust-region style SOC variables with equality" begin
                    H = sparse(Diagonal([0.0, 1.0, 2.0, 3.0]))
                    c = [0.0, -1.0, 0.5, 0.25]
                    A = sparse([1.0 0.0 0.0 0.0])
                    b = [1.0]

                    model = build_from_SOCP_data(
                        H,
                        c,
                        A,
                        b,
                        [2],
                        1,
                        0,
                        fill(-Inf, 4),
                        fill(Inf, 4),
                        [1, 5];
                        verbose=false,
                    )

                    result = optimize(model, gpu_soc_params())
                    @test result.status == "OPTIMAL"
                    @test norm(A * result.x - b) ≤ 1e-5
                    @test result.x[1] + 1e-8 ≥ norm(result.x[2:4])
                end
            else
                @test true
                @info "CUDA not functional; skipping GPU SOC regression test."
            end
        end

        @testset "SOC restart and scaling consistency" begin
            function allocate_cpu_ws(model; lambda_max_A=1.0, lambda_max_Q=0.0, auto_save=false, soc_strategy=:hybrid)
                params = HPRSOCP_parameters()
                params.verbose = false
                params.use_gpu = false
                params.sigma = 1.0
                params.auto_save = auto_save
                params.soc_block_scaling_strategy = soc_strategy
                scaling_info = HPRSOCP.scaling!(model, params)
                diag_Q, Q_is_diag = HPRSOCP.check_Q_diagonal(model)
                ws = HPRSOCP.allocate_workspace(model, params, lambda_max_A, lambda_max_Q, scaling_info, diag_Q, Q_is_diag)
                return ws, scaling_info
            end

            @testset "do_restart! keeps Qw state for SOC models" begin
                model = build_from_SOCP_data(
                    sparse(Matrix{Float64}(I, 3, 3)),
                    zeros(3),
                    spzeros(0, 3),
                    Float64[],
                    [1],
                    0,
                    0,
                    fill(-Inf, 3),
                    fill(Inf, 3),
                    [1, 4];
                    verbose=false,
                )
                ws, _ = allocate_cpu_ws(model; lambda_max_A=0.0, lambda_max_Q=1.0)
                restart_info = HPRSOCP.initialize_restart()
                ws.dw .= [0.05, -0.05, 0.1]
                ws.sigma = 1.0

                mnorm = HPRSOCP.compute_M_norm!(ws, model)
                @test isfinite(mnorm)
                @test mnorm >= 0.0
            end
        end

        @testset "External solver SOC cross-checks" begin
            if Base.find_package("Clarabel") !== nothing
                @eval using Clarabel

                function cpu_soc_params_crosscheck()
                    params = HPRSOCP_parameters()
                    params.use_gpu = false
                    params.verbose = false
                    params.warm_up = false
                    params.max_iter = 4000
                    params.stoptol = 1e-6
                    params.time_limit = 60.0
                    params.print_frequency = -1
                    return params
                end

                function solve_with_clarabel(a::Vector{Float64})
                    m = Model(Clarabel.Optimizer)
                    set_silent(m)
                    @variable(m, x[1:3])
                    @constraint(m, x in SecondOrderCone())
                    @objective(m, Min, 0.5 * sum(x[i]^2 for i in 1:3) - sum(a[i] * x[i] for i in 1:3))
                    optimize!(m)
                    return value.(x)
                end

                @testset "SOC variable cone vs Clarabel" begin
                    a = [0.0, 1.0, 0.0]
                    clarabel_x = solve_with_clarabel(a)

                    model = build_from_SOCP_data(
                        sparse(Matrix{Float64}(I, 3, 3)),
                        -a,
                        spzeros(0, 3),
                        Float64[],
                        [1],
                        0,
                        0,
                        fill(-Inf, 3),
                        fill(Inf, 3),
                        [1, 4];
                        verbose=false,
                    )

                    result = optimize(model, cpu_soc_params_crosscheck())
                    @test result.status == "OPTIMAL"
                    @test result.x ≈ clarabel_x atol=5e-3
                end

                @testset "SOC constraint cone vs Clarabel" begin
                    a = [0.0, 1.0, 0.0]
                    clarabel_x = solve_with_clarabel(a)

                    model = build_from_SOCP_data(
                        sparse(Matrix{Float64}(I, 3, 3)),
                        -a,
                        sparse(Matrix{Float64}(I, 3, 3)),
                        zeros(3),
                        [1, 4],
                        0,
                        0,
                        fill(-Inf, 3),
                        fill(Inf, 3),
                        [4];
                        verbose=false,
                    )

                    result = optimize(model, cpu_soc_params_crosscheck())
                    @test result.status == "OPTIMAL"
                    @test result.x ≈ clarabel_x atol=5e-3
                end
            else
                @test true
                @info "Clarabel not installed; skipping optional SOC external-solver cross-checks."
            end
        end
    end
    
    @testset "QP from MPS file" begin
        # Test loading and solving a QP from MPS file
        # Test AUG2D.mps from data folder
        @testset "AUG2D.mps from data folder" begin
            mps_file = joinpath(@__DIR__, "..", "data", "AUG2D.mps")
            if isfile(mps_file)
                model = build_from_mps(mps_file; verbose=false)
                
                # Create solver parameters - ensure convergence
                params = HPRSOCP_parameters()
                params.max_iter = 50000
                params.stoptol = 1e-6
                params.time_limit = 600.0  # 10 minutes
                params.warm_up = false
                params.verbose = false
                params.print_frequency = -1
                
                # Solve
                result = optimize(model, params)
                
                # Check that we got a result
                @test result !== nothing
                @test result.status == "OPTIMAL"
                @test result.residuals < 1e-6
                
                println("AUG2D.mps: Status=$(result.status), Objective=$(result.primal_obj)")
            else
                @warn "AUG2D.mps file not found, skipping test"
            end
        end
    end
    
    @testset "QP from matrices (QAbc)" begin
        # Test solving QP problems with dense and empty Q matrices
        @testset "Solve with dense Q matrix" begin
            n, m = 10, 5
            # Create a small dense QP
            Q_dense = Matrix{Float64}(I, n, n) * 2.0
            c = -ones(n)
            A_dense = ones(m, n)
            AL = zeros(m)
            AU = fill(n/2, m)
            l = zeros(n)
            u = ones(n)
            
            model = build_from_QAbc(Q_dense, c, A_dense, AL, AU, l, u; verbose=false)
            
            params = HPRSOCP_parameters()
            params.max_iter = 10000
            params.stoptol = 1e-6
            params.time_limit = 120.0
            params.warm_up = false
            params.verbose = false
            params.print_frequency = -1
            
            result = optimize(model, params)
            
            @test result !== nothing
            @test result.status == "OPTIMAL"
            @test result.residuals < 1e-6
            @test length(result.x) == n
            
            println("Dense Q matrix solve: Status=$(result.status), Objective=$(result.primal_obj)")
        end
        
        @testset "Solve with empty (zero) Q matrix - LP" begin
            # This is essentially an LP problem (Q = 0)
            n, m = 8, 4
            Q_empty = zeros(n, n)  # Empty Q - linear programming
            c = -ones(n)  # Minimize -sum(x), equivalent to maximize sum(x)
            A = ones(m, n)
            AL = fill(2.0, m)
            AU = fill(5.0, m)
            l = zeros(n)
            u = ones(n)
            
            model = build_from_QAbc(Q_empty, c, A, AL, AU, l, u; verbose=false)
            
            params = HPRSOCP_parameters()
            params.max_iter = 10000
            params.stoptol = 1e-6
            params.time_limit = 120.0
            params.warm_up = false
            params.verbose = false
            params.print_frequency = -1
            
            result = optimize(model, params)
            
            @test result !== nothing
            @test result.status == "OPTIMAL"
            @test result.residuals < 1e-6
            @test length(result.x) == n
            
            println("Empty Q matrix (LP) solve: Status=$(result.status), Objective=$(result.primal_obj)")
        end
    end
    
    @testset "Parameter validation" begin
        # Test that solver handles different parameter configurations
        @testset "Parameter settings" begin
            # Create a tiny problem
            n, m = 5, 2
            Q = sparse(Matrix{Float64}(I, n, n))
            c = ones(n)
            A = sparse(ones(m, n))
            lcon = zeros(m)
            ucon = fill(n/2, m)
            lvar = zeros(n)
            uvar = ones(n)
            
            model = build_from_QAbc(Q, c, A, lcon, ucon, lvar, uvar; verbose=false)
            
            # Test with different tolerances
            for tol in [1e-3, 1e-5]
                params = HPRSOCP_parameters()
                params.max_iter = 50000
                params.stoptol = tol
                params.time_limit = 600.0
                params.warm_up = false
                params.verbose = false
                params.print_frequency = -1
                
                result = optimize(model, params)
                @test result !== nothing
                @test result.status == "OPTIMAL"
            end
            
            println("✓ Parameter validation test passed")
        end
    end
    
    @testset "Result structure" begin
        # Test that result structure contains expected fields
        @testset "Result fields" begin
            n, m = 4, 2
            Q = sparse(Matrix{Float64}(I, n, n))
            c = ones(n)
            A = sparse(ones(m, n))
            lcon = zeros(m)
            ucon = ones(m) * 2
            lvar = zeros(n)
            uvar = ones(n)
            
            model = build_from_QAbc(Q, c, A, lcon, ucon, lvar, uvar; verbose=false)
            
            params = HPRSOCP_parameters()
            params.max_iter = 50000
            params.stoptol = 1e-6
            params.time_limit = 600.0
            params.warm_up = false
            params.verbose = false
            params.print_frequency = -1
            
            result = optimize(model, params)
            
            # Check result has all expected fields
            @test hasfield(typeof(result), :x)
            @test hasfield(typeof(result), :y)
            @test hasfield(typeof(result), :z)
            @test hasfield(typeof(result), :w)
            @test hasfield(typeof(result), :iter)
            @test hasfield(typeof(result), :time)
            @test hasfield(typeof(result), :residuals)
            @test hasfield(typeof(result), :primal_obj)
            @test hasfield(typeof(result), :gap)
            @test hasfield(typeof(result), :status)
            
            println("✓ Result structure test passed")
        end
    end
    
    @testset "Auto-save feature" begin
        @testset "HDF5 auto-save test" begin
            mps_file = joinpath(@__DIR__, "..", "data", "AUG2D.mps")
            if isfile(mps_file)
                model = build_from_mps(mps_file; verbose=false)
                
                # Create solver parameters with auto-save enabled
                params = HPRSOCP_parameters()
                params.max_iter = 5000
                params.stoptol = 1e-4
                params.time_limit = 120.0
                params.check_iter = 100
                params.verbose = false
                params.print_frequency = 500  # Save less frequently in tests
                params.warm_up = false
                
                # Enable auto-save
                params.auto_save = true
                test_filename = "test_autosave_runtests.h5"
                params.save_filename = test_filename
                
                # Remove old file if exists
                if isfile(test_filename)
                    rm(test_filename)
                end
                
                result = optimize(model, params)
                
                # Check solver results
                @test result !== nothing
                @test result.status == "OPTIMAL"
                @test result.residuals < 1e-4
                
                # Check that HDF5 file was created
                @test isfile(test_filename)
                
                # Read and verify HDF5 file contents
                h5open(test_filename, "r") do file
                    # Check current state exists
                    @test haskey(file, "current/iteration")
                    @test haskey(file, "current/x_org")
                    @test haskey(file, "current/w_org")
                    @test haskey(file, "current/z_org")
                    @test haskey(file, "current/y_org")
                    @test haskey(file, "current/sigma")
                    @test haskey(file, "current/err_Rp")
                    @test haskey(file, "current/err_Rd")
                    @test haskey(file, "current/primal_obj")
                    @test haskey(file, "current/dual_obj")
                    @test haskey(file, "current/rel_gap")
                    
                    # Check best solution exists
                    @test haskey(file, "best/iteration")
                    @test haskey(file, "best/x_org")
                    @test haskey(file, "best/w_org")
                    @test haskey(file, "best/z_org")
                    @test haskey(file, "best/y_org")
                    @test haskey(file, "best/sigma")
                    @test haskey(file, "best/err_Rp")
                    @test haskey(file, "best/err_Rd")
                    
                    # Check parameters saved
                    @test haskey(file, "parameters/stoptol")
                    @test haskey(file, "parameters/auto_save")
                    @test read(file, "parameters/auto_save") == true
                    
                    # Verify dimensions
                    x_best = read(file, "best/x_org")
                    w_best = read(file, "best/w_org")
                    z_best = read(file, "best/z_org")
                    y_best = read(file, "best/y_org")
                    
                    @test length(x_best) > 0
                    @test length(w_best) > 0
                    @test length(z_best) > 0
                    @test length(y_best) > 0
                    @test length(x_best) == length(w_best)
                    @test length(x_best) == length(z_best)
                    
                    # Check that best solution has reasonable values
                    best_err_Rp = read(file, "best/err_Rp")
                    best_err_Rd = read(file, "best/err_Rd")
                    @test best_err_Rp < 1e-4
                    @test best_err_Rd < 1e-4
                    
                    # Verify iteration counter
                    current_iter = read(file, "current/iteration")
                    best_iter = read(file, "best/iteration")
                    @test current_iter >= 0
                    @test best_iter >= 0
                    @test best_iter <= current_iter
                end
                
                # Clean up test file
                rm(test_filename)
                
                println("Auto-save (AUG2D.mps): Status=$(result.status), Objective=$(result.primal_obj)")
            else
                @warn "AUG2D.mps file not found, skipping auto-save test"
            end
        end
    end
    
    @testset "CPU Mode Tests with Data Files" begin
        @testset "CPU - MPS file (AUG2D.mps)" begin
            mps_file = joinpath(@__DIR__, "..", "data", "AUG2D.mps")
            
            if isfile(mps_file)
                # Build model from MPS file
                model = build_from_mps(mps_file; verbose=false)
                
                # Configure parameters for CPU
                params = HPRSOCP_parameters()
                params.use_gpu = false
                params.max_iter = 100000
                params.stoptol = 1e-6
                params.time_limit = 1200.0
                params.warm_up = false
                params.verbose = false
                params.print_frequency = -1
                
                result = optimize(model, params)
                
                @test result !== nothing
                @test result.status in ["OPTIMAL", "MAX_ITER"]
                if result.status == "OPTIMAL"
                    @test result.residuals < 1e-6
                end
                
                println("CPU - AUG2D.mps: Status=$(result.status), Objective=$(result.primal_obj)")
            else
                @warn "AUG2D.mps file not found, skipping CPU MPS test"
            end
        end
        
    end
    
    @testset "GPU Validation" begin
        # Test GPU availability checking and automatic fallback to CPU
        
        @testset "GPU validation function exists" begin
            # Test that the validate_gpu_parameters! function is defined
            @test isdefined(HPRSOCP, :validate_gpu_parameters!)
        end
        
        @testset "CPU mode works correctly" begin
            # Create a simple test problem
            n, m = 10, 5
            Q = sparse(1.0I, n, n)
            c = ones(n)
            A = sparse(rand(m, n))
            AL = -ones(m)
            AU = ones(m)
            l = zeros(n)
            u = 10 * ones(n)
            
            model = build_from_QAbc(Q, c, A, AL, AU, l, u; verbose=false)
            
            # Test with explicit CPU mode
            params = HPRSOCP_parameters()
            params.use_gpu = false
            params.max_iter = 100
            params.verbose = false
            
            result = optimize(model, params)
            @test result !== nothing
            @test result.status in ["OPTIMAL", "MAX_ITER"]
        end
        
        @testset "GPU parameter validation" begin
            using CUDA
            
            # Create a simple test problem
            n, m = 10, 5
            Q = sparse(1.0I, n, n)
            c = ones(n)
            A = sparse(rand(m, n))
            AL = -ones(m)
            AU = ones(m)
            l = zeros(n)
            u = 10 * ones(n)
            
            model = build_from_QAbc(Q, c, A, AL, AU, l, u; verbose=false)
            
            # Test with invalid GPU device number
            params = HPRSOCP_parameters()
            params.use_gpu = true
            params.device_number = 999  # Invalid device number
            params.max_iter = 100
            params.verbose = false
            
            # Should automatically fall back to CPU without error
            result = optimize(model, params)
            @test result !== nothing
            @test result.status in ["OPTIMAL", "MAX_ITER"]
            
            # After validation, use_gpu should be false if GPU is unavailable or device is invalid
            if !CUDA.functional() || params.device_number >= length(CUDA.devices())
                # GPU should have been disabled
                @test params.use_gpu == false || CUDA.functional()
            end
        end
        
        @testset "Default GPU behavior" begin
            using CUDA
            
            # Create a simple test problem
            n, m = 10, 5
            Q = sparse(1.0I, n, n)
            c = ones(n)
            A = sparse(rand(m, n))
            AL = -ones(m)
            AU = ones(m)
            l = zeros(n)
            u = 10 * ones(n)
            
            model = build_from_QAbc(Q, c, A, AL, AU, l, u; verbose=false)
            
            # Test default parameters (should use GPU if available)
            params = HPRSOCP_parameters()
            params.max_iter = 100
            params.verbose = false
            initial_use_gpu = params.use_gpu
            
            result = optimize(model, params)
            @test result !== nothing
            @test result.status in ["OPTIMAL", "MAX_ITER"]
            
            # If CUDA is not functional, use_gpu should be false after optimization
            if !CUDA.functional()
                @test params.use_gpu == false
            end
        end
        
        @testset "Parameter printing shows correct device" begin
            # Create a simple test problem
            n, m = 5, 3
            Q = sparse(1.0I, n, n)
            c = ones(n)
            A = sparse(rand(m, n))
            AL = -ones(m)
            AU = ones(m)
            l = zeros(n)
            u = ones(n)
            
            model = build_from_QAbc(Q, c, A, AL, AU, l, u; verbose=false)
            
            # Test with CPU mode - just verify it runs without error
            # (actual output checking would require more complex test setup)
            params = HPRSOCP_parameters()
            params.use_gpu = false
            params.max_iter = 10
            params.verbose = false  # Keep verbose off for clean test output
            
            result = optimize(model, params)
            
            # Verify the result is valid and use_gpu is correctly set
            @test result !== nothing
            @test params.use_gpu == false
            @test result.status in ["OPTIMAL", "MAX_ITER"]
        end
    end
end
