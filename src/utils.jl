"""
    validate_gpu_parameters!(params::HPRSOCP_parameters)

Validates GPU-related parameters and adjusts settings if GPU is requested but not available.

# Arguments
- `params::HPRSOCP_parameters`: The solver parameters to validate

# Behavior
- If `use_gpu=true` but CUDA is not functional, sets `use_gpu=false` and warns user
- If `use_gpu=true` but device_number is invalid, sets `use_gpu=false` and warns user
- Validates that device_number is within valid range [0, num_devices-1]
"""
function validate_gpu_parameters!(params::HPRSOCP_parameters)
    if params.use_gpu
        # Check if CUDA is functional
        if !CUDA.functional()
            @warn "GPU requested but CUDA is not functional. Falling back to CPU execution."
            params.use_gpu = false
            return
        end

        # Check if device_number is valid
        num_devices = length(CUDA.devices())
        if params.device_number < 0 || params.device_number >= num_devices
            @warn "Invalid GPU device number $(params.device_number). Valid range is [0, $(num_devices-1)]. Falling back to CPU execution."
            params.use_gpu = false
            return
        end

        if params.auto_save
            @warn "GPU auto_save copies iterates to CPU for HDF5 output during the solve. Disable auto_save to keep the iteration path GPU-resident until final result collection."
        end
    end
end

# Read data from a mps file
function read_mps(file::String)
    if file[end-3:end] == ".mps" || file[end-4:end] == ".MPS"
        io = open(file)
        qp = Logging.with_logger(Logging.NullLogger()) do
            QPSReader.readqps(io, mpsformat=:free)
        end
        close(io)
    else
        error("Unsupported file format. Please provide a .mps file.")
    end
    # constraint matrix
    A = sparse(qp.arows, qp.acols, qp.avals, qp.ncon, qp.nvar)
    lcon = qp.lcon
    ucon = qp.ucon

    # quadratic part
    Q = sparse(qp.qrows, qp.qcols, qp.qvals, qp.nvar, qp.nvar)
    # the Q matrix is not symmetric, so we need to symmetrize it
    diag_Q = diag(Q)
    Q = Q + Q' - Diagonal(diag_Q)

    # linear part
    c = qp.c
    c0 = qp.c0

    # bounds
    lvar = qp.lvar
    uvar = qp.uvar

    return Q, c, A, lcon, ucon, lvar, uvar, c0
end

mutable struct CBFProblemData
    acoord
    bcoord
    con
    dcoord
    fcoord
    hcoord
    intlist
    nconstr
    nvar
    var
    objacoord
    objfcoord
    objoffset
    psdcon
    psdvar
    sense
end

function _read_cbf_raw(filename::String)
    file = open(filename, "r")
    lines = readlines(file)
    close(file)

    acoord = []
    bcoord = []
    nconstr = 0
    con = []
    dcoord = []
    fcoord = []
    hcoord = []
    intlist = []
    nvar = 0
    var = []
    objacoord = []
    objfcoord = []
    objoffset = 0.0
    psdcon = []
    psdvar = []
    sense = "min"

    i = 1
    while i <= length(lines)
        if startswith(lines[i], "#")
            i += 1
            continue
        end
        line = lines[i]
        if line == "ACOORD"
            nnz = parse(Int, lines[i+1])
            acoord = Vector{Tuple{Int,Int,Float64}}(undef, nnz)
            i += 1
            for j in 1:nnz
                i += 1
                parts = split(lines[i])
                acoord[j] = (parse(Int, parts[1]) + 1, parse(Int, parts[2]) + 1, parse(Float64, parts[3]))
            end
        elseif line == "BCOORD"
            nnz = parse(Int, lines[i+1])
            bcoord = Vector{Tuple{Int,Float64}}(undef, nnz)
            i += 1
            for j in 1:nnz
                i += 1
                parts = split(lines[i])
                bcoord[j] = (parse(Int, parts[1]) + 1, parse(Float64, parts[2]))
            end
        elseif line == "CON"
            parts = split(lines[i+1])
            nconstr = parse(Int, parts[1])
            number_con_groups = parse(Int, parts[2])
            con = Vector{Tuple{String,Int}}(undef, number_con_groups)
            i += 1
            for j in 1:number_con_groups
                i += 1
                parts = split(lines[i])
                con[j] = (parts[1], parse(Int, parts[2]))
            end
        elseif line == "DCOORD"
            nnz = parse(Int, lines[i+1])
            dcoord = Vector{Tuple{Int,Int,Int,Float64}}(undef, nnz)
            i += 1
            for j in 1:nnz
                i += 1
                parts = split(lines[i])
                dcoord[j] = (parse(Int, parts[1]) + 1, parse(Int, parts[2]) + 1, parse(Int, parts[3]) + 1, parse(Float64, parts[4]))
            end
        elseif line == "FCOORD"
            nnz = parse(Int, lines[i+1])
            fcoord = Vector{Tuple{Int,Int,Int,Int,Float64}}(undef, nnz)
            i += 1
            for j in 1:nnz
                i += 1
                parts = split(lines[i])
                fcoord[j] = (parse(Int, parts[1]) + 1, parse(Int, parts[2]) + 1, parse(Int, parts[3]) + 1, parse(Int, parts[4]) + 1, parse(Float64, parts[5]))
            end
        elseif line == "HCOORD" || line == "hCOORD"
            nnz = parse(Int, lines[i+1])
            hcoord = Vector{Tuple{Int,Int,Int,Int,Float64}}(undef, nnz)
            i += 1
            for j in 1:nnz
                i += 1
                parts = split(lines[i])
                hcoord[j] = (parse(Int, parts[1]) + 1, parse(Int, parts[2]) + 1, parse(Int, parts[3]) + 1, parse(Int, parts[4]) + 1, parse(Float64, parts[5]))
            end
        elseif line == "INT"
            nnz = parse(Int, lines[i+1])
            intlist = Vector{Int}(undef, nnz)
            i += 1
            for j in 1:nnz
                i += 1
                intlist[j] = parse(Int, split(lines[i])[1]) + 1
            end
        elseif line == "VAR"
            parts = split(lines[i+1])
            nvar = parse(Int, parts[1])
            number_var_groups = parse(Int, parts[2])
            var = Vector{Tuple{String,Int}}(undef, number_var_groups)
            i += 1
            for j in 1:number_var_groups
                i += 1
                parts = split(lines[i])
                var[j] = (parts[1], parse(Int, parts[2]))
            end
        elseif line == "OBJACOORD"
            nnz = parse(Int, lines[i+1])
            objacoord = Vector{Tuple{Int,Float64}}(undef, nnz)
            i += 1
            for j in 1:nnz
                i += 1
                parts = split(lines[i])
                objacoord[j] = (parse(Int, parts[1]) + 1, parse(Float64, parts[2]))
            end
        elseif line == "OBJFCOORD"
            nnz = parse(Int, lines[i+1])
            objfcoord = Vector{Tuple{Int,Int,Int,Float64}}(undef, nnz)
            i += 1
            for j in 1:nnz
                i += 1
                parts = split(lines[i])
                objfcoord[j] = (parse(Int, parts[1]) + 1, parse(Int, parts[2]) + 1, parse(Int, parts[3]) + 1, parse(Float64, parts[4]))
            end
        elseif line == "OBJBCOORD"
            objoffset = parse(Float64, lines[i+1])
            i += 1
        elseif line == "PSDCON"
            nnz = parse(Int, lines[i+1])
            psdcon = Vector{Int}(undef, nnz)
            i += 1
            for j in 1:nnz
                i += 1
                psdcon[j] = parse(Int, split(lines[i])[1]) + 1
            end
        elseif line == "PSDVAR"
            nnz = parse(Int, lines[i+1])
            psdvar = Vector{Int}(undef, nnz)
            i += 1
            for j in 1:nnz
                i += 1
                psdvar[j] = parse(Int, split(lines[i])[1]) + 1
            end
        elseif line == "SENSE" || line == "OBJSENSE"
            sense = _cbf_normalize_obj_sense(lines[i+1])
            i += 1
        end
        i += 1
    end

    return CBFProblemData(acoord, bcoord, con, dcoord, fcoord, hcoord, intlist, nconstr, nvar, var, objacoord, objfcoord, objoffset, psdcon, psdvar, sense)
end


function _cbf_next_data_line(io::IO)
    while !eof(io)
        line = readline(io)
        n = ncodeunits(line)
        start = 1
        while start <= n
            b = codeunit(line, start)
            if b == 0x20 || b == 0x09 || b == 0x0d
                start += 1
            else
                break
            end
        end
        start > n && continue
        codeunit(line, start) == UInt8('#') && continue

        stop = n
        while stop >= start
            b = codeunit(line, stop)
            if b == 0x20 || b == 0x09 || b == 0x0d
                stop -= 1
            else
                break
            end
        end
        return start == 1 && stop == n ? line : SubString(line, start, stop)
    end
    return nothing
end

@inline function _cbf_parse_int_pair(line::AbstractString)
    tokens = eachsplit(line)
    state = iterate(tokens)
    state === nothing && error("Invalid CBF line: expected two integers.")
    first_token, token_state = state
    state = iterate(tokens, token_state)
    state === nothing && error("Invalid CBF line: expected two integers.")
    second_token, _ = state
    return parse(Int, first_token), parse(Int, second_token)
end

@inline function _cbf_parse_group_entry(line::AbstractString)
    tokens = eachsplit(line)
    state = iterate(tokens)
    state === nothing && error("Invalid CBF line: expected a cone tag and length.")
    cone_token, token_state = state
    state = iterate(tokens, token_state)
    state === nothing && error("Invalid CBF line: expected a cone tag and length.")
    len_token, _ = state
    return String(cone_token), parse(Int, len_token)
end

@inline function _cbf_skip_ascii_spaces(data::Vector{UInt8}, idx::Int, stop::Int)
    while idx <= stop
        b = data[idx]
        if b == 0x20 || b == 0x09 || b == 0x0d
            idx += 1
        else
            break
        end
    end
    return idx
end

@inline function _cbf_parse_ascii_int(data::Vector{UInt8}, idx::Int, stop::Int)
    idx = _cbf_skip_ascii_spaces(data, idx, stop)
    idx <= stop || error("Invalid CBF line: expected integer token.")

    sign = 1
    b = data[idx]
    if b == UInt8('-')
        sign = -1
        idx += 1
    elseif b == UInt8('+')
        idx += 1
    end

    value = 0
    saw_digit = false
    while idx <= stop
        b = data[idx]
        if UInt8('0') <= b <= UInt8('9')
            value = 10 * value + Int(b - UInt8('0'))
            saw_digit = true
            idx += 1
        else
            break
        end
    end
    saw_digit || error("Invalid CBF line: expected integer token.")
    return sign * value, idx
end

@inline function _cbf_parse_ascii_float(data::Vector{UInt8}, idx::Int, stop::Int)
    idx = _cbf_skip_ascii_spaces(data, idx, stop)
    idx <= stop || error("Invalid CBF line: expected floating-point token.")

    endptr = Ref{Ptr{Cchar}}()
    p = Ptr{Cchar}(pointer(data, idx))
    value = ccall(:strtod, Cdouble, (Ptr{Cchar}, Ref{Ptr{Cchar}}), p, endptr)
    consumed = Int(endptr[] - p)
    consumed > 0 || error("Invalid CBF line: expected floating-point token.")
    idx + consumed <= stop + 1 || error("Invalid CBF line: floating-point token overran line.")
    return Float64(value), idx + consumed
end

@inline function _cbf_parse_int_pair(data::Vector{UInt8}, start::Int, stop::Int)
    first_val, idx = _cbf_parse_ascii_int(data, start, stop)
    second_val, _ = _cbf_parse_ascii_int(data, idx, stop)
    return first_val, second_val
end

@inline function _cbf_parse_group_entry(data::Vector{UInt8}, start::Int, stop::Int)
    idx = _cbf_skip_ascii_spaces(data, start, stop)
    idx <= stop || error("Invalid CBF line: expected a cone tag and length.")
    cone_start = idx
    while idx <= stop
        b = data[idx]
        if b == 0x20 || b == 0x09
            break
        end
        idx += 1
    end
    cone_stop = idx - 1
    cone_stop >= cone_start || error("Invalid CBF line: expected a cone tag and length.")

    cone_type = if cone_stop == cone_start
        b = data[cone_start]
        b == UInt8('Q') ? "Q" : b == UInt8('F') ? "F" : String(copy(@view data[cone_start:cone_stop]))
    elseif cone_stop == cone_start + 1 && data[cone_start] == UInt8('L')
        b = data[cone_start+1]
        b == UInt8('+') ? "L+" : b == UInt8('-') ? "L-" : b == UInt8('=') ? "L=" : String(copy(@view data[cone_start:cone_stop]))
    else
        String(copy(@view data[cone_start:cone_stop]))
    end
    len_val, _ = _cbf_parse_ascii_int(data, idx, stop)
    return cone_type, len_val
end

@inline function _cbf_normalize_obj_sense(line::AbstractString)
    sense = lowercase(strip(line))
    if sense == "min" || sense == "minimize" || sense == "minimise"
        return "min"
    elseif sense == "max" || sense == "maximize" || sense == "maximise"
        return "max"
    end
    error("Unsupported CBF objective sense: $(repr(line)).")
end

@inline function _cbf_parse_triplet_entry_ascii(data::Vector{UInt8}, start::Int, stop::Int)
    row_idx, idx = _cbf_parse_ascii_int(data, start, stop)
    col_idx, idx = _cbf_parse_ascii_int(data, idx, stop)
    value, _ = _cbf_parse_ascii_float(data, idx, stop)
    return row_idx + 1, col_idx + 1, value
end

@inline function _cbf_parse_value_entry_ascii(data::Vector{UInt8}, start::Int, stop::Int)
    row_idx, idx = _cbf_parse_ascii_int(data, start, stop)
    value, _ = _cbf_parse_ascii_float(data, idx, stop)
    return row_idx + 1, value
end

@inline function _cbf_next_line_bounds(data::Vector{UInt8}, pos::Int)
    n = length(data)
    pos <= n || return nothing

    line_start = pos
    while pos <= n
        b = data[pos]
        if b == 0x0a || b == 0x0d
            break
        end
        pos += 1
    end
    line_stop = pos - 1

    if pos <= n && data[pos] == 0x0d
        pos += 1
        if pos <= n && data[pos] == 0x0a
            pos += 1
        end
    elseif pos <= n && data[pos] == 0x0a
        pos += 1
    end

    return line_start, line_stop, pos
end

@inline function _cbf_next_data_line(data::Vector{UInt8}, pos::Int)
    while true
        span = _cbf_next_line_bounds(data, pos)
        isnothing(span) && return nothing
        line_start, line_stop, pos = span

        while line_start <= line_stop
            b = data[line_start]
            if b == 0x20 || b == 0x09 || b == 0x0d
                line_start += 1
            else
                break
            end
        end
        line_start > line_stop && continue
        data[line_start] == UInt8('#') && continue

        while line_stop >= line_start
            b = data[line_stop]
            if b == 0x20 || b == 0x09 || b == 0x0d
                line_stop -= 1
            else
                break
            end
        end

        return line_start, line_stop, pos
    end
end

@inline function _cbf_line_equals(data::Vector{UInt8}, start::Int, stop::Int, literal::AbstractString)
    n = ncodeunits(literal)
    stop - start + 1 == n || return false
    @inbounds for k in 1:n
        data[start+k-1] == codeunit(literal, k) || return false
    end
    return true
end

@inline function _cbf_skip_ascii_spaces(line::AbstractString, idx::Int)
    n = ncodeunits(line)
    while idx <= n
        b = codeunit(line, idx)
        if b == 0x20 || b == 0x09
            idx += 1
        else
            break
        end
    end
    return idx
end

@inline function _cbf_parse_ascii_int(line::AbstractString, idx::Int)
    idx = _cbf_skip_ascii_spaces(line, idx)
    n = ncodeunits(line)
    idx <= n || error("Invalid CBF line: expected integer token.")

    sign = 1
    b = codeunit(line, idx)
    if b == UInt8('-')
        sign = -1
        idx += 1
    elseif b == UInt8('+')
        idx += 1
    end

    value = 0
    saw_digit = false
    while idx <= n
        b = codeunit(line, idx)
        if 0x30 <= b <= 0x39
            value = 10 * value + (b - 0x30)
            saw_digit = true
            idx += 1
        else
            break
        end
    end
    saw_digit || error("Invalid CBF line: expected integer token.")
    return sign * value, idx
end

@inline function _cbf_parse_ascii_float(line::AbstractString, idx::Int)
    idx = _cbf_skip_ascii_spaces(line, idx)
    n = ncodeunits(line)
    idx <= n || error("Invalid CBF line: expected floating-point token.")

    endptr = Ref{Ptr{Cchar}}()
    p = Ptr{Cchar}(Ptr{UInt8}(pointer(line)) + (idx - 1))
    value = ccall(:strtod, Cdouble, (Ptr{Cchar}, Ref{Ptr{Cchar}}), p, endptr)
    consumed = Int(endptr[] - p)
    consumed > 0 || error("Invalid CBF line: expected floating-point token.")
    return Float64(value), idx + consumed
end

@inline function _cbf_parse_triplet_entry_ascii(line::AbstractString)
    row_idx, idx = _cbf_parse_ascii_int(line, 1)
    col_idx, idx = _cbf_parse_ascii_int(line, idx)
    value, _ = _cbf_parse_ascii_float(line, idx)
    return row_idx + 1, col_idx + 1, value
end

@inline function _cbf_parse_value_entry_ascii(line::AbstractString)
    row_idx, idx = _cbf_parse_ascii_int(line, 1)
    value, _ = _cbf_parse_ascii_float(line, idx)
    return row_idx + 1, value
end

@inline _cbf_parse_triplet_entry(line::AbstractString) = _cbf_parse_triplet_entry_ascii(line)

@inline _cbf_parse_value_entry(line::AbstractString) = _cbf_parse_value_entry_ascii(line)

@inline function _cbf_parse_line_triplet!(out::NTuple{3,Base.RefValue}, line::AbstractString)
    row_idx, idx = _cbf_parse_ascii_int(line, 1)
    col_idx, idx = _cbf_parse_ascii_int(line, idx)
    value, _ = _cbf_parse_ascii_float(line, idx)
    out[1][] = row_idx + 1
    out[2][] = col_idx + 1
    out[3][] = value
    return nothing
end

@inline function _cbf_skip_data_lines(io::IO, count::Int)
    for _ in 1:count
        readline(io)
    end
end

@inline _cbf_is_supported_cone(cone_type::AbstractString) =
    cone_type == "L+" || cone_type == "L-" || cone_type == "L=" || cone_type == "Q" || cone_type == "F"

function _cbf_prepare_layout(con::Vector{Tuple{String,Int}}, var::Vector{Tuple{String,Int}}; nconstr::Int=-1, nvar::Int=-1)
    isempty(con) && error("Invalid CBF file: missing CON section.")
    isempty(var) && error("Invalid CBF file: missing VAR section.")

    for cgrp in con
        _cbf_is_supported_cone(cgrp[1]) || error("Unsupported constraint cone type $(cgrp[1]).")
    end
    for vgrp in var
        _cbf_is_supported_cone(vgrp[1]) || error("Unsupported variable cone type $(vgrp[1]).")
    end

    m = 0
    for (_, len) in con
        m += len
    end
    n = 0
    for (_, len) in var
        n += len
    end
    (nconstr == -1 || nconstr == m) || error("CBF CON size mismatch: declared nconstr=$nconstr but group sum is $m.")
    (nvar == -1 || nvar == n) || error("CBF VAR size mismatch: declared nvar=$nvar but group sum is $n.")

    number_eq = 0
    number_ineq = 0
    number_soc_rows = 0
    soc_con_lens = Int[]
    for (cone_type, len) in con
        if cone_type == "L="
            number_eq += len
        elseif cone_type == "L+" || cone_type == "L-"
            number_ineq += len
        elseif cone_type == "Q"
            number_soc_rows += len
            push!(soc_con_lens, len)
        end
    end

    total_rows = number_eq + number_ineq + number_soc_rows
    row_new = zeros(Int, m)
    row_a_sign = ones(Int8, m)
    row_b_sign = fill(Int8(-1), m)

    eq_ptr = 1
    ineq_ptr = number_eq + 1
    soc_ptr = number_eq + number_ineq + 1
    r0 = 1
    for (cone_type, len) in con
        r1 = r0 + len - 1
        if cone_type == "L="
            @inbounds for r in r0:r1
                row_new[r] = eq_ptr
                eq_ptr += 1
            end
        elseif cone_type == "L+"
            @inbounds for r in r0:r1
                row_new[r] = ineq_ptr
                ineq_ptr += 1
            end
        elseif cone_type == "L-"
            @inbounds for r in r0:r1
                row_new[r] = ineq_ptr
                row_a_sign[r] = -1
                row_b_sign[r] = 1
                ineq_ptr += 1
            end
        elseif cone_type == "Q"
            @inbounds for r in r0:r1
                row_new[r] = soc_ptr
                soc_ptr += 1
            end
        end
        r0 = r1 + 1
    end
    @assert eq_ptr == number_eq + 1
    @assert ineq_ptr == number_eq + number_ineq + 1
    @assert soc_ptr == total_rows + 1

    number_soc_vars = 0
    for (cone_type, len) in var
        if cone_type == "Q"
            number_soc_vars += len
        end
    end
    number_lu_x = n - number_soc_vars

    lb = Vector{Float64}(undef, n)
    ub = Vector{Float64}(undef, n)
    col_new = zeros(Int, n)
    SOC_var_idx = Int[number_lu_x+1]

    linear_ptr = 1
    soc_var_ptr = number_lu_x + 1
    old_col = 1
    for (cone_type, len) in var
        old_last = old_col + len - 1
        if cone_type == "Q"
            @inbounds for old_idx in old_col:old_last
                col_new[old_idx] = soc_var_ptr
                lb[soc_var_ptr] = -Inf
                ub[soc_var_ptr] = Inf
                soc_var_ptr += 1
            end
            push!(SOC_var_idx, soc_var_ptr)
        else
            lower = cone_type == "L+" || cone_type == "L=" ? 0.0 : -Inf
            upper = cone_type == "L-" || cone_type == "L=" ? 0.0 : Inf
            @inbounds for old_idx in old_col:old_last
                col_new[old_idx] = linear_ptr
                lb[linear_ptr] = lower
                ub[linear_ptr] = upper
                linear_ptr += 1
            end
        end
        old_col = old_last + 1
    end
    @assert linear_ptr == number_lu_x + 1
    @assert soc_var_ptr == n + 1

    SOC_con_idx = Int[number_eq+number_ineq+1]
    for len in soc_con_lens
        push!(SOC_con_idx, SOC_con_idx[end] + len)
    end
    @assert SOC_con_idx[end] - 1 == total_rows
    @assert SOC_var_idx[end] - 1 == n

    return (
        m=m,
        n=n,
        total_rows=total_rows,
        number_eq=number_eq,
        number_ineq=number_ineq,
        row_new=row_new,
        row_a_sign=row_a_sign,
        row_b_sign=row_b_sign,
        col_new=col_new,
        lb=lb,
        ub=ub,
        SOC_con_idx=SOC_con_idx,
        SOC_var_idx=SOC_var_idx,
        number_lu_x=number_lu_x,
    )
end

@inline function _cbf_store_mapped_triplet!(
    I::AbstractVector{Ti},
    J::AbstractVector{Ti},
    V::Vector{Float64},
    nnz_kept::Int,
    line::AbstractString,
    row_new::Vector{Int},
    row_a_sign::Vector{Int8},
    col_new::Vector{Int},
) where {Ti<:Integer}
    r_old, c_old_idx, val = _cbf_parse_triplet_entry_ascii(line)
    r_new = row_new[r_old]
    r_new == 0 && return nnz_kept

    a_val = Float64(row_a_sign[r_old]) * val
    a_val == 0.0 && return nnz_kept

    nnz_kept += 1
    I[nnz_kept] = convert(Ti, r_new)
    J[nnz_kept] = convert(Ti, col_new[c_old_idx])
    V[nnz_kept] = a_val
    return nnz_kept
end

@inline function _cbf_init_mapped_csc_state(n::Int)
    return (
        col_head=fill(Int32(0), n),
        col_tail=fill(Int32(0), n),
        col_count=fill(Int32(0), n),
        last_row=fill(Int32(0), n),
        needs_sort=falses(n),
    )
end

@inline function _cbf_store_mapped_csc_entry!(
    bucket_rows::Vector{Int32},
    bucket_vals::Vector{Float64},
    bucket_next::Vector{Int32},
    col_head::Vector{Int32},
    col_tail::Vector{Int32},
    col_count::Vector{Int32},
    last_row::Vector{Int32},
    needs_sort::BitVector,
    nnz_kept::Int,
    r_old::Int,
    c_old_idx::Int,
    val::Float64,
    row_new::Vector{Int},
    row_a_sign::Vector{Int8},
    col_new::Vector{Int},
)
    r_new = row_new[r_old]
    r_new == 0 && return nnz_kept

    a_val = Float64(row_a_sign[r_old]) * val
    a_val == 0.0 && return nnz_kept

    c_new = col_new[c_old_idx]
    nnz_kept += 1
    bucket_rows[nnz_kept] = Int32(r_new)
    bucket_vals[nnz_kept] = a_val
    bucket_next[nnz_kept] = Int32(0)

    prev_idx = col_tail[c_new]
    if prev_idx == 0
        col_head[c_new] = Int32(nnz_kept)
    else
        bucket_next[prev_idx] = Int32(nnz_kept)
    end
    col_tail[c_new] = Int32(nnz_kept)
    col_count[c_new] = col_count[c_new] + Int32(1)

    r_new_i32 = Int32(r_new)
    if r_new_i32 < last_row[c_new]
        needs_sort[c_new] = true
    end
    last_row[c_new] = r_new_i32
    return nnz_kept
end

@inline function _cbf_store_mapped_csc_line!(
    bucket_rows::Vector{Int32},
    bucket_vals::Vector{Float64},
    bucket_next::Vector{Int32},
    col_head::Vector{Int32},
    col_tail::Vector{Int32},
    col_count::Vector{Int32},
    last_row::Vector{Int32},
    needs_sort::BitVector,
    nnz_kept::Int,
    line::AbstractString,
    row_new::Vector{Int},
    row_a_sign::Vector{Int8},
    col_new::Vector{Int},
)
    r_old, c_old_idx, val = _cbf_parse_triplet_entry_ascii(line)
    return _cbf_store_mapped_csc_entry!(
        bucket_rows,
        bucket_vals,
        bucket_next,
        col_head,
        col_tail,
        col_count,
        last_row,
        needs_sort,
        nnz_kept,
        r_old,
        c_old_idx,
        val,
        row_new,
        row_a_sign,
        col_new,
    )
end

function _cbf_build_mapped_csc_from_buckets(
    bucket_rows::Vector{Int32},
    bucket_vals::Vector{Float64},
    bucket_next::Vector{Int32},
    col_head::Vector{Int32},
    col_count::Vector{Int32},
    needs_sort::BitVector,
    total_rows::Int,
    n::Int,
    nnz_kept::Int,
)
    if total_rows == 0 || n == 0 || nnz_kept == 0
        return SparseMatrixCSC{Float64,Int32}(spzeros(Float64, total_rows, n))
    end

    colptr = Vector{Int32}(undef, n + 1)
    rowval = Vector{Int32}(undef, nnz_kept)
    nzval = Vector{Float64}(undef, nnz_kept)
    colptr[1] = Int32(1)

    scratch_rows = Int32[]
    scratch_vals = Float64[]
    write_ptr = 1
    @inbounds for col in 1:n
        head = col_head[col]
        if head == 0
            colptr[col+1] = Int32(write_ptr)
            continue
        end

        if needs_sort[col]
            count = Int(col_count[col])
            resize!(scratch_rows, count)
            resize!(scratch_vals, count)
            idx = head
            k = 1
            while idx != 0
                scratch_rows[k] = bucket_rows[idx]
                scratch_vals[k] = bucket_vals[idx]
                idx = bucket_next[idx]
                k += 1
            end
            perm = sortperm(view(scratch_rows, 1:count))

            prev_row = Int32(0)
            accum = 0.0
            have_prev = false
            for p in perm
                row = scratch_rows[p]
                val = scratch_vals[p]
                if have_prev && row == prev_row
                    accum += val
                else
                    if have_prev && accum != 0.0
                        rowval[write_ptr] = prev_row
                        nzval[write_ptr] = accum
                        write_ptr += 1
                    end
                    prev_row = row
                    accum = val
                    have_prev = true
                end
            end
            if have_prev && accum != 0.0
                rowval[write_ptr] = prev_row
                nzval[write_ptr] = accum
                write_ptr += 1
            end
        else
            idx = head
            prev_row = Int32(0)
            accum = 0.0
            have_prev = false
            while idx != 0
                row = bucket_rows[idx]
                val = bucket_vals[idx]
                if have_prev && row == prev_row
                    accum += val
                else
                    if have_prev && accum != 0.0
                        rowval[write_ptr] = prev_row
                        nzval[write_ptr] = accum
                        write_ptr += 1
                    end
                    prev_row = row
                    accum = val
                    have_prev = true
                end
                idx = bucket_next[idx]
            end
            if have_prev && accum != 0.0
                rowval[write_ptr] = prev_row
                nzval[write_ptr] = accum
                write_ptr += 1
            end
        end

        colptr[col+1] = Int32(write_ptr)
    end

    resize!(rowval, write_ptr - 1)
    resize!(nzval, write_ptr - 1)
    return SparseMatrixCSC{Float64,Int32}(total_rows, n, colptr, rowval, nzval)
end

function _cbf_build_mapped_csc_from_lines(
    lines,
    row_new::Vector{Int},
    row_a_sign::Vector{Int8},
    col_new::Vector{Int},
    total_rows::Int,
    n::Int,
)
    state = _cbf_init_mapped_csc_state(n)
    bucket_rows = Vector{Int32}(undef, length(lines))
    bucket_vals = Vector{Float64}(undef, length(lines))
    bucket_next = Vector{Int32}(undef, length(lines))
    nnz_kept = 0
    for line in lines
        nnz_kept = _cbf_store_mapped_csc_line!(
            bucket_rows,
            bucket_vals,
            bucket_next,
            state.col_head,
            state.col_tail,
            state.col_count,
            state.last_row,
            state.needs_sort,
            nnz_kept,
            line,
            row_new,
            row_a_sign,
            col_new,
        )
    end
    return _cbf_build_mapped_csc_from_buckets(
        bucket_rows,
        bucket_vals,
        bucket_next,
        state.col_head,
        state.col_count,
        state.needs_sort,
        total_rows,
        n,
        nnz_kept,
    )
end

function _read_cbf_io(filename::String)
    if !(endswith(filename, ".cbf") || endswith(filename, ".CBF"))
        error("Unsupported file format. Please provide a .cbf file.")
    end

    con = Tuple{String,Int}[]
    var = Tuple{String,Int}[]
    nconstr = -1
    nvar = -1
    sense = "min"
    obj_constant = 0.0
    layout_ready = false
    n = 0
    total_rows = 0
    number_eq = 0
    number_ineq = 0
    number_lu_x = 0
    row_new = Int[]
    row_a_sign = Int8[]
    row_b_sign = Int8[]
    col_new = Int[]
    lb = Float64[]
    ub = Float64[]
    SOC_con_idx = Int[]
    SOC_var_idx = Int[]
    c = Float64[]
    b_new = Float64[]
    I = Int32[]
    J = Int32[]
    V = Float64[]
    nnz_kept = 0
    raw_a_rows = Int[]
    raw_a_cols = Int[]
    raw_a_vals = Float64[]
    raw_b_idx = Int[]
    raw_b_vals = Float64[]
    raw_c_idx = Int[]
    raw_c_vals = Float64[]

    open(filename, "r") do io
        while true
            line = _cbf_next_data_line(io)
            isnothing(line) && break

            if line == "CON"
                nconstr, ngrp = _cbf_parse_int_pair(readline(io))
                resize!(con, ngrp)
                for j in 1:ngrp
                    con[j] = _cbf_parse_group_entry(readline(io))
                end
                if !layout_ready && !isempty(var)
                    layout = _cbf_prepare_layout(con, var; nconstr=nconstr, nvar=nvar)
                    n = layout.n
                    total_rows = layout.total_rows
                    number_eq = layout.number_eq
                    number_ineq = layout.number_ineq
                    number_lu_x = layout.number_lu_x
                    row_new = layout.row_new
                    row_a_sign = layout.row_a_sign
                    row_b_sign = layout.row_b_sign
                    col_new = layout.col_new
                    lb = layout.lb
                    ub = layout.ub
                    SOC_con_idx = layout.SOC_con_idx
                    SOC_var_idx = layout.SOC_var_idx
                    c = zeros(Float64, n)
                    b_new = zeros(Float64, total_rows)
                    layout_ready = true
                end
            elseif line == "VAR"
                nvar, ngrp = _cbf_parse_int_pair(readline(io))
                resize!(var, ngrp)
                for j in 1:ngrp
                    var[j] = _cbf_parse_group_entry(readline(io))
                end
                if !layout_ready && !isempty(con)
                    layout = _cbf_prepare_layout(con, var; nconstr=nconstr, nvar=nvar)
                    n = layout.n
                    total_rows = layout.total_rows
                    number_eq = layout.number_eq
                    number_ineq = layout.number_ineq
                    number_lu_x = layout.number_lu_x
                    row_new = layout.row_new
                    row_a_sign = layout.row_a_sign
                    row_b_sign = layout.row_b_sign
                    col_new = layout.col_new
                    lb = layout.lb
                    ub = layout.ub
                    SOC_con_idx = layout.SOC_con_idx
                    SOC_var_idx = layout.SOC_var_idx
                    c = zeros(Float64, n)
                    b_new = zeros(Float64, total_rows)
                    layout_ready = true
                end
            elseif line == "ACOORD"
                nnz = parse(Int, readline(io))
                if !layout_ready
                    sizehint!(raw_a_rows, length(raw_a_rows) + nnz)
                    sizehint!(raw_a_cols, length(raw_a_cols) + nnz)
                    sizehint!(raw_a_vals, length(raw_a_vals) + nnz)
                    for _ in 1:nnz
                        r_old, c_old_idx, val = _cbf_parse_triplet_entry_ascii(readline(io))
                        push!(raw_a_rows, r_old)
                        push!(raw_a_cols, c_old_idx)
                        push!(raw_a_vals, val)
                    end
                else
                    resize!(I, nnz_kept + nnz)
                    resize!(J, nnz_kept + nnz)
                    resize!(V, nnz_kept + nnz)
                    for _ in 1:nnz
                        nnz_kept = _cbf_store_mapped_triplet!(I, J, V, nnz_kept, readline(io), row_new, row_a_sign, col_new)
                    end
                end
            elseif line == "BCOORD"
                nnz = parse(Int, readline(io))
                if !layout_ready
                    sizehint!(raw_b_idx, length(raw_b_idx) + nnz)
                    sizehint!(raw_b_vals, length(raw_b_vals) + nnz)
                    for _ in 1:nnz
                        idx, val = _cbf_parse_value_entry_ascii(readline(io))
                        push!(raw_b_idx, idx)
                        push!(raw_b_vals, val)
                    end
                else
                    for _ in 1:nnz
                        idx, val = _cbf_parse_value_entry_ascii(readline(io))
                        r_new = row_new[idx]
                        if r_new != 0
                            b_new[r_new] = Float64(row_b_sign[idx]) * val
                        end
                    end
                end
            elseif line == "OBJACOORD"
                nnz = parse(Int, readline(io))
                if !layout_ready
                    sizehint!(raw_c_idx, length(raw_c_idx) + nnz)
                    sizehint!(raw_c_vals, length(raw_c_vals) + nnz)
                    for _ in 1:nnz
                        idx, val = _cbf_parse_value_entry_ascii(readline(io))
                        push!(raw_c_idx, idx)
                        push!(raw_c_vals, val)
                    end
                else
                    for _ in 1:nnz
                        idx, val = _cbf_parse_value_entry_ascii(readline(io))
                        c[col_new[idx]] = val
                    end
                end
            elseif line == "OBJBCOORD"
                obj_constant = parse(Float64, readline(io))
            elseif line == "SENSE" || line == "OBJSENSE"
                sense = _cbf_normalize_obj_sense(readline(io))
            elseif line in ("DCOORD", "HCOORD", "hCOORD", "PSDCON", "PSDVAR", "INT", "FCOORD", "OBJFCOORD")
                nnz = parse(Int, readline(io))
                nnz == 0 || error(
                    line in ("DCOORD", "HCOORD", "hCOORD", "PSDCON") ? "PSD constraints are not supported." :
                    line in ("FCOORD", "OBJFCOORD", "PSDVAR") ? "Matrix variables are not supported." :
                    "Integer variables are not supported."
                )
                _cbf_skip_data_lines(io, nnz)
            end
        end
    end

    if !layout_ready
        layout = _cbf_prepare_layout(con, var; nconstr=nconstr, nvar=nvar)
        n = layout.n
        total_rows = layout.total_rows
        number_eq = layout.number_eq
        number_ineq = layout.number_ineq
        number_lu_x = layout.number_lu_x
        row_new = layout.row_new
        row_a_sign = layout.row_a_sign
        row_b_sign = layout.row_b_sign
        col_new = layout.col_new
        lb = layout.lb
        ub = layout.ub
        SOC_con_idx = layout.SOC_con_idx
        SOC_var_idx = layout.SOC_var_idx
        c = zeros(Float64, n)
        b_new = zeros(Float64, total_rows)
        layout_ready = true
    end

    if !isempty(raw_c_idx)
        @inbounds for k in eachindex(raw_c_idx)
            c[col_new[raw_c_idx[k]]] = raw_c_vals[k]
        end
    end

    if !isempty(raw_b_idx)
        @inbounds for k in eachindex(raw_b_idx)
            r_new = row_new[raw_b_idx[k]]
            if r_new != 0
                b_new[r_new] = Float64(row_b_sign[raw_b_idx[k]]) * raw_b_vals[k]
            end
        end
    end

    if !isempty(raw_a_vals)
        resize!(I, nnz_kept + length(raw_a_vals))
        resize!(J, nnz_kept + length(raw_a_vals))
        resize!(V, nnz_kept + length(raw_a_vals))
        @inbounds for k in eachindex(raw_a_vals)
            r_old = raw_a_rows[k]
            r_new = row_new[r_old]
            r_new == 0 && continue

            a_val = Float64(row_a_sign[r_old]) * raw_a_vals[k]
            a_val == 0.0 && continue

            nnz_kept += 1
            I[nnz_kept] = Int32(r_new)
            J[nnz_kept] = Int32(col_new[raw_a_cols[k]])
            V[nnz_kept] = a_val
        end
    end

    resize!(I, nnz_kept)
    resize!(J, nnz_kept)
    resize!(V, nnz_kept)

    A_new = total_rows == 0 ? spzeros(Float64, 0, n) : SparseArrays.sparse!(I, J, V, total_rows, n)
    dropzeros!(A_new)

    if sense == "max"
        c .*= -1
        obj_constant *= -1
    end

    return spzeros(Float64, n, n), c, SparseMatrixCSC{Float64,Int32}(A_new), b_new, SOC_con_idx, number_eq, number_ineq, lb, ub, SOC_var_idx, obj_constant
end

function _read_cbf_mmap(data::Vector{UInt8})
    con = Tuple{String,Int}[]
    var = Tuple{String,Int}[]
    nconstr = -1
    nvar = -1
    sense = "min"
    obj_constant = 0.0
    layout_ready = false
    n = 0
    total_rows = 0
    number_eq = 0
    number_ineq = 0
    number_lu_x = 0
    row_new = Int[]
    row_a_sign = Int8[]
    row_b_sign = Int8[]
    col_new = Int[]
    lb = Float64[]
    ub = Float64[]
    SOC_con_idx = Int[]
    SOC_var_idx = Int[]
    c = Float64[]
    b_new = Float64[]
    I = Int32[]
    J = Int32[]
    V = Float64[]
    nnz_kept = 0
    raw_a_rows = Int[]
    raw_a_cols = Int[]
    raw_a_vals = Float64[]
    raw_b_idx = Int[]
    raw_b_vals = Float64[]
    raw_c_idx = Int[]
    raw_c_vals = Float64[]

    pos = 1
    while true
        span = _cbf_next_data_line(data, pos)
        isnothing(span) && break
        line_start, line_stop, pos = span

        if _cbf_line_equals(data, line_start, line_stop, "CON")
            span = _cbf_next_line_bounds(data, pos)
            isnothing(span) && error("Invalid CBF file: unexpected EOF after CON.")
            line_start, line_stop, pos = span
            nconstr, ngrp = _cbf_parse_int_pair(data, line_start, line_stop)
            resize!(con, ngrp)
            for j in 1:ngrp
                span = _cbf_next_line_bounds(data, pos)
                isnothing(span) && error("Invalid CBF file: unexpected EOF inside CON.")
                line_start, line_stop, pos = span
                con[j] = _cbf_parse_group_entry(data, line_start, line_stop)
            end
            if !layout_ready && !isempty(var)
                layout = _cbf_prepare_layout(con, var; nconstr=nconstr, nvar=nvar)
                n = layout.n
                total_rows = layout.total_rows
                number_eq = layout.number_eq
                number_ineq = layout.number_ineq
                number_lu_x = layout.number_lu_x
                row_new = layout.row_new
                row_a_sign = layout.row_a_sign
                row_b_sign = layout.row_b_sign
                col_new = layout.col_new
                lb = layout.lb
                ub = layout.ub
                SOC_con_idx = layout.SOC_con_idx
                SOC_var_idx = layout.SOC_var_idx
                c = zeros(Float64, n)
                b_new = zeros(Float64, total_rows)
                layout_ready = true
            end
        elseif _cbf_line_equals(data, line_start, line_stop, "VAR")
            span = _cbf_next_line_bounds(data, pos)
            isnothing(span) && error("Invalid CBF file: unexpected EOF after VAR.")
            line_start, line_stop, pos = span
            nvar, ngrp = _cbf_parse_int_pair(data, line_start, line_stop)
            resize!(var, ngrp)
            for j in 1:ngrp
                span = _cbf_next_line_bounds(data, pos)
                isnothing(span) && error("Invalid CBF file: unexpected EOF inside VAR.")
                line_start, line_stop, pos = span
                var[j] = _cbf_parse_group_entry(data, line_start, line_stop)
            end
            if !layout_ready && !isempty(con)
                layout = _cbf_prepare_layout(con, var; nconstr=nconstr, nvar=nvar)
                n = layout.n
                total_rows = layout.total_rows
                number_eq = layout.number_eq
                number_ineq = layout.number_ineq
                number_lu_x = layout.number_lu_x
                row_new = layout.row_new
                row_a_sign = layout.row_a_sign
                row_b_sign = layout.row_b_sign
                col_new = layout.col_new
                lb = layout.lb
                ub = layout.ub
                SOC_con_idx = layout.SOC_con_idx
                SOC_var_idx = layout.SOC_var_idx
                c = zeros(Float64, n)
                b_new = zeros(Float64, total_rows)
                layout_ready = true
            end
        elseif _cbf_line_equals(data, line_start, line_stop, "ACOORD")
            span = _cbf_next_line_bounds(data, pos)
            isnothing(span) && error("Invalid CBF file: unexpected EOF after ACOORD.")
            line_start, line_stop, pos = span
            nnz, _ = _cbf_parse_ascii_int(data, line_start, line_stop)
            if !layout_ready
                sizehint!(raw_a_rows, length(raw_a_rows) + nnz)
                sizehint!(raw_a_cols, length(raw_a_cols) + nnz)
                sizehint!(raw_a_vals, length(raw_a_vals) + nnz)
                for _ in 1:nnz
                    span = _cbf_next_line_bounds(data, pos)
                    isnothing(span) && error("Invalid CBF file: unexpected EOF inside ACOORD.")
                    line_start, line_stop, pos = span
                    r_old, c_old_idx, val = _cbf_parse_triplet_entry_ascii(data, line_start, line_stop)
                    push!(raw_a_rows, r_old)
                    push!(raw_a_cols, c_old_idx)
                    push!(raw_a_vals, val)
                end
            else
                resize!(I, nnz_kept + nnz)
                resize!(J, nnz_kept + nnz)
                resize!(V, nnz_kept + nnz)
                for _ in 1:nnz
                    span = _cbf_next_line_bounds(data, pos)
                    isnothing(span) && error("Invalid CBF file: unexpected EOF inside ACOORD.")
                    line_start, line_stop, pos = span
                    r_old, c_old_idx, val = _cbf_parse_triplet_entry_ascii(data, line_start, line_stop)
                    r_new = row_new[r_old]
                    r_new == 0 && continue

                    a_val = Float64(row_a_sign[r_old]) * val
                    a_val == 0.0 && continue

                    nnz_kept += 1
                    I[nnz_kept] = Int32(r_new)
                    J[nnz_kept] = Int32(col_new[c_old_idx])
                    V[nnz_kept] = a_val
                end
            end
        elseif _cbf_line_equals(data, line_start, line_stop, "BCOORD")
            span = _cbf_next_line_bounds(data, pos)
            isnothing(span) && error("Invalid CBF file: unexpected EOF after BCOORD.")
            line_start, line_stop, pos = span
            nnz, _ = _cbf_parse_ascii_int(data, line_start, line_stop)
            if !layout_ready
                sizehint!(raw_b_idx, length(raw_b_idx) + nnz)
                sizehint!(raw_b_vals, length(raw_b_vals) + nnz)
                for _ in 1:nnz
                    span = _cbf_next_line_bounds(data, pos)
                    isnothing(span) && error("Invalid CBF file: unexpected EOF inside BCOORD.")
                    line_start, line_stop, pos = span
                    idx, val = _cbf_parse_value_entry_ascii(data, line_start, line_stop)
                    push!(raw_b_idx, idx)
                    push!(raw_b_vals, val)
                end
            else
                for _ in 1:nnz
                    span = _cbf_next_line_bounds(data, pos)
                    isnothing(span) && error("Invalid CBF file: unexpected EOF inside BCOORD.")
                    line_start, line_stop, pos = span
                    idx, val = _cbf_parse_value_entry_ascii(data, line_start, line_stop)
                    r_new = row_new[idx]
                    if r_new != 0
                        b_new[r_new] = Float64(row_b_sign[idx]) * val
                    end
                end
            end
        elseif _cbf_line_equals(data, line_start, line_stop, "OBJACOORD")
            span = _cbf_next_line_bounds(data, pos)
            isnothing(span) && error("Invalid CBF file: unexpected EOF after OBJACOORD.")
            line_start, line_stop, pos = span
            nnz, _ = _cbf_parse_ascii_int(data, line_start, line_stop)
            if !layout_ready
                sizehint!(raw_c_idx, length(raw_c_idx) + nnz)
                sizehint!(raw_c_vals, length(raw_c_vals) + nnz)
                for _ in 1:nnz
                    span = _cbf_next_line_bounds(data, pos)
                    isnothing(span) && error("Invalid CBF file: unexpected EOF inside OBJACOORD.")
                    line_start, line_stop, pos = span
                    idx, val = _cbf_parse_value_entry_ascii(data, line_start, line_stop)
                    push!(raw_c_idx, idx)
                    push!(raw_c_vals, val)
                end
            else
                for _ in 1:nnz
                    span = _cbf_next_line_bounds(data, pos)
                    isnothing(span) && error("Invalid CBF file: unexpected EOF inside OBJACOORD.")
                    line_start, line_stop, pos = span
                    idx, val = _cbf_parse_value_entry_ascii(data, line_start, line_stop)
                    c[col_new[idx]] = val
                end
            end
        elseif _cbf_line_equals(data, line_start, line_stop, "OBJBCOORD")
            span = _cbf_next_line_bounds(data, pos)
            isnothing(span) && error("Invalid CBF file: unexpected EOF after OBJBCOORD.")
            line_start, line_stop, pos = span
            obj_constant, _ = _cbf_parse_ascii_float(data, line_start, line_stop)
        elseif _cbf_line_equals(data, line_start, line_stop, "SENSE") || _cbf_line_equals(data, line_start, line_stop, "OBJSENSE")
            span = _cbf_next_line_bounds(data, pos)
            isnothing(span) && error("Invalid CBF file: unexpected EOF after objective sense section.")
            line_start, line_stop, pos = span
            sense = _cbf_normalize_obj_sense(String(copy(@view data[line_start:line_stop])))
        else
            is_dcoord = _cbf_line_equals(data, line_start, line_stop, "DCOORD")
            is_hcoord = _cbf_line_equals(data, line_start, line_stop, "HCOORD")
            is_hcoord_lower = _cbf_line_equals(data, line_start, line_stop, "hCOORD")
            is_psdcon = _cbf_line_equals(data, line_start, line_stop, "PSDCON")
            is_psdvar = _cbf_line_equals(data, line_start, line_stop, "PSDVAR")
            is_int = _cbf_line_equals(data, line_start, line_stop, "INT")
            is_fcoord = _cbf_line_equals(data, line_start, line_stop, "FCOORD")
            is_objfcoord = _cbf_line_equals(data, line_start, line_stop, "OBJFCOORD")

            if !(is_dcoord || is_hcoord || is_hcoord_lower || is_psdcon || is_psdvar || is_int || is_fcoord || is_objfcoord)
                continue
            end

            span = _cbf_next_line_bounds(data, pos)
            isnothing(span) && error("Invalid CBF file: unexpected EOF after section header.")
            line_start, line_stop, pos = span
            nnz, _ = _cbf_parse_ascii_int(data, line_start, line_stop)
            nnz == 0 || error(
                is_dcoord || is_hcoord || is_hcoord_lower || is_psdcon ? "PSD constraints are not supported." :
                is_fcoord || is_objfcoord || is_psdvar ? "Matrix variables are not supported." :
                "Integer variables are not supported."
            )
            for _ in 1:nnz
                span = _cbf_next_line_bounds(data, pos)
                isnothing(span) && error("Invalid CBF file: unexpected EOF inside unsupported section.")
                _, _, pos = span
            end
        end
    end

    if !layout_ready
        layout = _cbf_prepare_layout(con, var; nconstr=nconstr, nvar=nvar)
        n = layout.n
        total_rows = layout.total_rows
        number_eq = layout.number_eq
        number_ineq = layout.number_ineq
        number_lu_x = layout.number_lu_x
        row_new = layout.row_new
        row_a_sign = layout.row_a_sign
        row_b_sign = layout.row_b_sign
        col_new = layout.col_new
        lb = layout.lb
        ub = layout.ub
        SOC_con_idx = layout.SOC_con_idx
        SOC_var_idx = layout.SOC_var_idx
        c = zeros(Float64, n)
        b_new = zeros(Float64, total_rows)
        layout_ready = true
    end

    if !isempty(raw_c_idx)
        @inbounds for k in eachindex(raw_c_idx)
            c[col_new[raw_c_idx[k]]] = raw_c_vals[k]
        end
    end

    if !isempty(raw_b_idx)
        @inbounds for k in eachindex(raw_b_idx)
            r_new = row_new[raw_b_idx[k]]
            if r_new != 0
                b_new[r_new] = Float64(row_b_sign[raw_b_idx[k]]) * raw_b_vals[k]
            end
        end
    end

    if !isempty(raw_a_vals)
        resize!(I, nnz_kept + length(raw_a_vals))
        resize!(J, nnz_kept + length(raw_a_vals))
        resize!(V, nnz_kept + length(raw_a_vals))
        @inbounds for k in eachindex(raw_a_vals)
            r_old = raw_a_rows[k]
            r_new = row_new[r_old]
            r_new == 0 && continue

            a_val = Float64(row_a_sign[r_old]) * raw_a_vals[k]
            a_val == 0.0 && continue

            nnz_kept += 1
            I[nnz_kept] = Int32(r_new)
            J[nnz_kept] = Int32(col_new[raw_a_cols[k]])
            V[nnz_kept] = a_val
        end
    end

    resize!(I, nnz_kept)
    resize!(J, nnz_kept)
    resize!(V, nnz_kept)

    A_new = total_rows == 0 ? spzeros(Float64, 0, n) : SparseArrays.sparse!(I, J, V, total_rows, n)
    dropzeros!(A_new)

    if sense == "max"
        c .*= -1
        obj_constant *= -1
    end

    return spzeros(Float64, n, n), c, SparseMatrixCSC{Float64,Int32}(A_new), b_new, SOC_con_idx, number_eq, number_ineq, lb, ub, SOC_var_idx, obj_constant
end

function read_cbf(filename::String)
    if !(endswith(filename, ".cbf") || endswith(filename, ".CBF"))
        error("Unsupported file format. Please provide a .cbf file.")
    end

    open(filename, "r") do io
        data = try
            Mmap.mmap(io)
        catch
            return _read_cbf_io(filename)
        end

        if isempty(data) || (data[end] != 0x0a && data[end] != 0x0d)
            return _read_cbf_io(filename)
        end
        return _read_cbf_mmap(data)
    end
end


function read_cbf_old(filename::String)
    if !(endswith(filename, ".cbf") || endswith(filename, ".CBF"))
        error("Unsupported file format. Please provide a .cbf file.")
    end

    problem = _read_cbf_raw(filename)

    if !isempty(problem.dcoord) || !isempty(problem.hcoord) || !isempty(problem.psdcon)
        error("PSD constraints are not supported.")
    end
    if !isempty(problem.fcoord) || !isempty(problem.psdvar)
        error("Matrix variables are not supported.")
    end
    if !isempty(problem.intlist)
        error("Integer variables are not supported.")
    end

    supported_cones = Set(["L+", "L-", "L=", "Q", "F"])
    for con in problem.con
        con[1] in supported_cones || error("Unsupported constraint cone type $(con[1]).")
    end
    for var in problem.var
        var[1] in supported_cones || error("Unsupported variable cone type $(var[1]).")
    end

    m = sum(getindex.(problem.con, 2))
    n = sum(getindex.(problem.var, 2))
    @assert problem.nconstr == m
    @assert problem.nvar == n

    arows = getindex.(problem.acoord, 1)
    acols = getindex.(problem.acoord, 2)
    avals = getindex.(problem.acoord, 3)
    A = sparse(arows, acols, avals, m, n)

    b = zeros(Float64, m)
    for (idx, val) in problem.bcoord
        b[idx] = val
    end

    A_eq = spzeros(m == 0 ? 0 : 0, n)
    A_ineq = spzeros(m == 0 ? 0 : 0, n)
    A_soc = spzeros(m == 0 ? 0 : 0, n)
    b_eq = Float64[]
    b_ineq = Float64[]
    b_soc = Float64[]
    soc_con_lens = Int[]
    current_row = 1

    for (cone_type, len) in problem.con
        rows = current_row:(current_row+len-1)
        if cone_type == "L="
            A_eq = vcat(A_eq, A[rows, :])
            append!(b_eq, -b[rows])
        elseif cone_type == "L+"
            A_ineq = vcat(A_ineq, A[rows, :])
            append!(b_ineq, -b[rows])
        elseif cone_type == "L-"
            A_ineq = vcat(A_ineq, -A[rows, :])
            append!(b_ineq, b[rows])
        elseif cone_type == "Q"
            A_soc = vcat(A_soc, A[rows, :])
            append!(b_soc, -b[rows])
            push!(soc_con_lens, len)
        elseif cone_type == "F"
            # Free constraint block contributes nothing.
        end
        current_row += len
    end

    number_eq = length(b_eq)
    number_ineq = length(b_ineq)
    A_new = vcat(A_eq, A_ineq, A_soc)
    b_new = vcat(b_eq, b_ineq, b_soc)

    SOC_con_idx = [number_eq + number_ineq + 1]
    for len in soc_con_lens
        push!(SOC_con_idx, SOC_con_idx[end] + len)
    end
    @assert SOC_con_idx[end] - 1 == size(A_new, 1)

    c = zeros(Float64, n)
    for (idx, val) in problem.objacoord
        c[idx] = val
    end
    obj_constant = problem.objoffset
    if problem.sense == "max"
        c .*= -1
        obj_constant *= -1
    end

    lb = Float64[]
    ub = Float64[]
    linear_idx = Int[]
    next_col = 1
    soc_var_lens = Int[]
    soc_cols = Int[]
    for (cone_type, len) in problem.var
        cols = next_col:(next_col+len-1)
        if cone_type == "L+"
            append!(lb, zeros(Float64, len))
            append!(ub, fill(Inf, len))
            append!(linear_idx, cols)
        elseif cone_type == "L-"
            append!(lb, fill(-Inf, len))
            append!(ub, zeros(Float64, len))
            append!(linear_idx, cols)
        elseif cone_type == "L="
            append!(lb, zeros(Float64, len))
            append!(ub, zeros(Float64, len))
            append!(linear_idx, cols)
        elseif cone_type == "F"
            append!(lb, fill(-Inf, len))
            append!(ub, fill(Inf, len))
            append!(linear_idx, cols)
        elseif cone_type == "Q"
            push!(soc_var_lens, len)
            append!(soc_cols, cols)
        end
        next_col += len
    end

    number_lu_x = length(lb)
    perm = vcat(linear_idx, soc_cols)
    A_new = A_new[:, perm]
    c = c[perm]
    append!(lb, fill(-Inf, length(soc_cols)))
    append!(ub, fill(Inf, length(soc_cols)))

    SOC_var_idx = [number_lu_x + 1]
    for len in soc_var_lens
        push!(SOC_var_idx, SOC_var_idx[end] + len)
    end
    @assert SOC_var_idx[end] - 1 == n

    return sparse(zeros(n, n)), c, sparse(A_new), b_new, SOC_con_idx, number_eq, number_ineq, lb, ub, SOC_var_idx, obj_constant
end

# Formulate the QP problem with the C constraints (l ≤ x ≤ u)
function qp_formulation(Q::SparseMatrixCSC,
    c::Vector{Float64},
    A::SparseMatrixCSC,
    AL::Vector{Float64},
    AU::Vector{Float64},
    l::Vector{Float64},
    u::Vector{Float64},
    c0::Float64=0.0)

    # ====================================================================
    # Input Validation
    # ====================================================================

    # Check Q matrix properties
    m_Q, n_Q = size(Q)
    if m_Q != n_Q
        error("Q matrix must be square. Got size ($m_Q, $n_Q).")
    end
    n = n_Q

    # Check Q is symmetric (within tolerance)
    if nnz(Q) > 0
        Q_diff = Q - Q'
        if norm(Q_diff, Inf) > 1e-10
            @warn "Q matrix is not symmetric (max deviation: $(norm(Q_diff, Inf))). Symmetrizing Q = 0.5*(Q + Q')."
            Q = 0.5 * (Q + Q')
            dropzeros!(Q)
        end
    end

    # Check vector dimensions
    if length(c) != n
        error("Dimension mismatch: Q is $n×$n but c has length $(length(c)).")
    end
    if length(l) != n
        error("Dimension mismatch: Q is $n×$n but l has length $(length(l)).")
    end
    if length(u) != n
        error("Dimension mismatch: Q is $n×$n but u has length $(length(u)).")
    end

    # Check A matrix dimensions
    m_A, n_A = size(A)
    if n_A != n
        error("Dimension mismatch: Q is $n×$n but A has $n_A columns.")
    end
    if length(AL) != m_A
        error("Dimension mismatch: A has $m_A rows but AL has length $(length(AL)).")
    end
    if length(AU) != m_A
        error("Dimension mismatch: A has $m_A rows but AU has length $(length(AU)).")
    end

    # Check bound consistency
    infeasible_bounds = findall(l .> u)
    if !isempty(infeasible_bounds)
        error("Infeasible variable bounds: l > u at indices: $(infeasible_bounds[1:min(5, length(infeasible_bounds))]) $(length(infeasible_bounds) > 5 ? "..." : "")")
    end

    infeasible_constraints = findall(AL .> AU)
    if !isempty(infeasible_constraints)
        error("Infeasible constraint bounds: AL > AU at rows: $(infeasible_constraints[1:min(5, length(infeasible_constraints))]) $(length(infeasible_constraints) > 5 ? "..." : "")")
    end

    # Check for NaN or Inf in problem data (except bounds which can be ±Inf)
    if any(isnan, Q.nzval) || any(isinf, Q.nzval)
        error("Q matrix contains NaN or Inf values.")
    end
    if any(isnan, c) || any(isinf, c)
        error("c vector contains NaN or Inf values.")
    end
    if any(isnan, A.nzval) || any(isinf, A.nzval)
        error("A matrix contains NaN or Inf values.")
    end
    if any(isnan.(AL) .& isfinite.(AL)) || any(isnan.(AU) .& isfinite.(AU))
        error("Constraint bounds AL or AU contain NaN values.")
    end
    if any(isnan.(l) .& isfinite.(l)) || any(isnan.(u) .& isfinite.(u))
        error("Variable bounds l or u contain NaN values.")
    end

    # ====================================================================
    # Problem Preprocessing
    # ====================================================================

    # Remove the rows of A that are all zeros
    abs_A = abs.(A)
    del_row = findall(sum(abs_A, dims=2)[:, 1] .== 0)    # rows that AL and AU are -Inf and Inf
    del_row = union(del_row, findall((AL .== -Inf) .& (AU .== Inf)))

    if length(del_row) > 0
        keep_rows = setdiff(1:size(A, 1), del_row)
        A = A[keep_rows, :]
        AL = AL[keep_rows]
        AU = AU[keep_rows]
        println("Deleted ", length(del_row), " rows of A that are all zeros.")
    end

    idxE = findall(AL .== AU)
    idxG = findall((AL .> -Inf) .& (AU .== Inf))
    idxL = findall((AL .== -Inf) .& (AU .< Inf))
    idxB = findall((AL .> -Inf) .& (AU .< Inf))
    idxB = setdiff(idxB, idxE)

    # check dimension of Q, c, A, l, u, AL, AU
    # println("problem information: nRow = ", size(A, 1), ", nCol = ", size(A, 2), ", nnz Q = ", nnz(Q), ", nnz A = ", nnz(A))
    # println("                     number of equalities = ", length(idxE))
    # println("                     number of inequalities = ", length(idxG) + length(idxL) + length(idxB))
    @assert size(Q, 1) == size(Q, 2)
    @assert size(Q, 1) == length(c)
    @assert size(A, 2) == length(c)
    @assert length(l) == length(u)
    @assert length(l) == size(Q, 1)
    @assert length(AL) == length(AU)
    @assert length(AL) == size(A, 1)


    number_eq = length(idxE)
    number_ineq = size(A, 1) - number_eq
    number_lu_x = length(l)
    SOC_con_idx = [size(A, 1) + 1]
    SOC_var_idx = [length(l) + 1]

    standard_qp = QP_info_cpu(Q, c, A, A', Float64[], zeros(Float64, size(A, 1)), AL, AU, SOC_con_idx, number_eq, number_ineq, l, u, SOC_var_idx, number_lu_x, c0)

    # Return the modified qp
    return standard_qp
end

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
    soc_row_start = qp.number_eq + qp.number_ineq + 1

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

# ==================== Build Functions (Public API) ====================

"""
    build_from_mps(filename::String; verbose::Bool=true)

Build a QP model from an MPS file.

This function reads a QP problem from an MPS file and returns a CPU-based model
that can be solved with `optimize()` or `solve()`.

# Arguments
- `filename::String`: Path to the MPS file
- `verbose::Bool`: Whether to print progress information (default: true)

# Returns
- `QP_info_cpu`: QP model ready to be solved

# Example
```julia
using HPRSOCP

model = build_from_mps("problem.mps")
params = HPRSOCP_parameters()
result = optimize(model, params)
```

See also: [`build_from_QAbc`](@ref), [`build_from_cbf`](@ref), [`optimize`](@ref)
"""
function build_from_mps(filename::String; verbose::Bool=true)
    t_start = time()
    if verbose
        println("READING FILE ... ", filename)
    end
    Q, c, A, lcon, ucon, lvar, uvar, c0 = read_mps(filename)
    read_time = time() - t_start
    if verbose
        println(@sprintf("READING FILE time: %.2f seconds", read_time))
    end

    t_start = time()
    if verbose
        println("FORMULATING QP ...")
    end
    standard_qp = qp_formulation(Q, c, A, lcon, ucon, lvar, uvar, c0)
    if verbose
        println(@sprintf("FORMULATING QP time: %.2f seconds", time() - t_start))
    end

    return standard_qp
end

"""
    build_from_QAbc(Q, c, A, AL, AU, l, u, obj_constant=0.0)

Build a QP model from matrix form.

This function creates a QP problem from the standard form:
    min  0.5 <x,Qx> + <c,x> + obj_constant
    s.t. AL <= Ax <= AU
         l <= x <= u

Accepts both sparse and dense matrices for Q and A. Dense matrices will be automatically
converted to sparse format for efficient computation.

# Arguments
- `Q::Union{SparseMatrixCSC, Matrix{Float64}}`: Quadratic objective matrix (n × n). Can be sparse or dense.
- `c::Vector{Float64}`: Linear objective coefficients (length n)
- `A::Union{SparseMatrixCSC, Matrix{Float64}}`: Constraint matrix (m × n). Can be sparse or dense.
- `AL::Vector{Float64}`: Lower bounds for constraints Ax (length m)
- `AU::Vector{Float64}`: Upper bounds for constraints Ax (length m)
- `l::Vector{Float64}`: Lower bounds for variables x (length n)
- `u::Vector{Float64}`: Upper bounds for variables x (length n)
- `obj_constant::Float64`: Constant term in objective function (default: 0.0)

# Returns
- `QP_info_cpu`: QP model ready to be solved

# Example
```julia
using SparseArrays, HPRSOCP

# Example 1: Sparse matrices
Q = sparse([2.0 0.0; 0.0 2.0])
c = [-3.0, -5.0]
A = sparse([-1.0 -2.0; -3.0 -1.0])
AL = [-10.0, -12.0]
AU = [Inf, Inf]
l = [0.0, 0.0]
u = [Inf, Inf]

model = build_from_QAbc(Q, c, A, AL, AU, l, u)
params = HPRSOCP_parameters()
result = optimize(model, params)

# Example 2: Dense matrices (automatically converted)
n = 10
Q = zeros(n, n)  # Empty or dense Q matrix
Q[1,1] = 2.0
c = ones(n)
A = ones(5, n)  # Dense constraint matrix
AL = -Inf * ones(5)
AU = ones(5)
l = zeros(n)
u = ones(n)
model = build_from_QAbc(Q, c, A, AL, AU, l, u)
```

See also: [`build_from_mps`](@ref), [`build_from_cbf`](@ref), [`optimize`](@ref)
"""
function build_from_QAbc(Q::Union{SparseMatrixCSC,Matrix{Float64}},
    c::Vector{Float64},
    A::Union{SparseMatrixCSC,Matrix{Float64}},
    AL::Vector{Float64},
    AU::Vector{Float64},
    l::Vector{Float64},
    u::Vector{Float64},
    obj_constant::Float64=0.0;
    verbose::Bool=true)

    # Convert dense matrices to sparse if needed
    if Q isa Matrix{Float64}
        if verbose
            println("Converting dense Q matrix to sparse format...")
        end
        Q = sparse(Q)
    end

    if A isa Matrix{Float64}
        if verbose
            println("Converting dense A matrix to sparse format...")
        end
        A = sparse(A)
    end

    # Create copies to avoid modifying the input
    Q = copy(Q)
    c = copy(c)
    A = copy(A)
    lcon = copy(AL)
    ucon = copy(AU)
    lvar = copy(l)
    uvar = copy(u)

    t_start = time()
    if verbose
        println("FORMULATING QP ...")
    end
    standard_qp = qp_formulation(Q, c, A, lcon, ucon, lvar, uvar, obj_constant)
    if verbose
        println(@sprintf("FORMULATING QP time: %.2f seconds", time() - t_start))
    end

    return standard_qp
end

function _is_effectively_diagonal(P::SparseMatrixCSC{Float64,Int32}; atol::Float64=1e-10)
    I, J, V = findnz(P)
    @inbounds for k in eachindex(V)
        if I[k] != J[k] && abs(V[k]) > atol
            return false
        end
    end
    return true
end

function _append_sparse_block_to_triplets!(
    I::Vector{Int},
    J::Vector{Int},
    V::Vector{Float64},
    row_offset::Int,
    A_block::SparseMatrixCSC{Float64,Int32},
)
    block_I, block_J, block_V = findnz(A_block)
    @inbounds for k in eachindex(block_V)
        push!(I, block_I[k] + row_offset)
        push!(J, block_J[k])
        push!(V, block_V[k])
    end
    return nothing
end

@inline function _find_local_index(idxs::Vector{Int}, idx::Int)
    @inbounds for k in eachindex(idxs)
        if idxs[k] == idx
            return k
        end
    end
    return 0
end

@inline function _accumulate_local_pair!(idxs::Vector{Int}, vals::Vector{Float64}, idx::Int, val::Float64)
    pos = _find_local_index(idxs, idx)
    if pos == 0
        push!(idxs, idx)
        push!(vals, val)
    else
        vals[pos] += val
    end
    return nothing
end

function _append_dense_row_triplets!(
    soc_I::Vector{Int},
    soc_J::Vector{Int},
    soc_V::Vector{Float64},
    row::Int,
    cols::Vector{Int},
    vals::Vector{Float64},
)
    @inbounds for k in eachindex(cols)
        val = vals[k]
        if val != 0.0
            push!(soc_I, row)
            push!(soc_J, cols[k])
            push!(soc_V, val)
        end
    end
    return nothing
end

function _normalize_soc_rhs_block!(rhs_block::Vector{Float64}; floor::Float64=1.0)
    isempty(rhs_block) && return 1.0
    block_scale = maximum(abs, rhs_block)
    alpha = 1.0 / max(block_scale, floor)
    if alpha != 1.0
        rhs_block .*= alpha
    end
    return alpha
end

function _normalize_soc_block!(
    A_block::SparseMatrixCSC{Float64,Int32},
    rhs_block::Vector{Float64};
    mode::Symbol=:rhs,
    floor::Float64=1.0,
)
    isempty(rhs_block) && return A_block, rhs_block
    rhs_scale = maximum(abs, rhs_block)

    if mode == :rhs
        block_scale = rhs_scale
    elseif mode == :rhs_row
        row_sq = zeros(Float64, size(A_block, 1))
        I, _, V = findnz(A_block)
        @inbounds for k in eachindex(V)
            row_sq[I[k]] += V[k]^2
        end
        row_scale = isempty(row_sq) ? 0.0 : sqrt(maximum(row_sq))
        block_scale = max(rhs_scale, row_scale)
    else
        error("Unsupported soc_block_scale_mode=$mode. Supported modes are :rhs and :rhs_row.")
    end

    alpha = 1.0 / max(block_scale, floor)
    if alpha == 1.0
        return A_block, rhs_block
    end
    rhs_scaled = copy(rhs_block)
    rhs_scaled .*= alpha
    A_scaled = SparseMatrixCSC{Float64,Int32}(alpha .* A_block)
    return A_scaled, rhs_scaled
end

function _normalize_new_soc_block_segment!(
    I::Vector{Int},
    V::Vector{Float64},
    rhs_soc::Vector{Float64},
    rhs_lo::Int,
    rhs_hi::Int,
    nz_lo::Int,
    nz_hi::Int,
    row_offset::Int;
    mode::Symbol=:rhs,
    floor::Float64=1.0,
)
    if rhs_lo > rhs_hi
        return 1.0
    end

    rhs_scale = maximum(abs, @view rhs_soc[rhs_lo:rhs_hi])

    if mode == :rhs
        block_scale = rhs_scale
    elseif mode == :rhs_row
        nrows = rhs_hi - rhs_lo + 1
        row_sq = zeros(Float64, nrows)
        if nz_lo <= nz_hi
            @inbounds for k in nz_lo:nz_hi
                local_row = I[k] - row_offset
                if 1 <= local_row <= nrows
                    row_sq[local_row] += V[k]^2
                end
            end
        end
        row_scale = isempty(row_sq) ? 0.0 : sqrt(maximum(row_sq))
        block_scale = max(rhs_scale, row_scale)
    else
        error("Unsupported soc_block_scale_mode=$mode. Supported modes are :rhs and :rhs_row.")
    end

    alpha = 1.0 / max(block_scale, floor)
    if alpha == 1.0
        return alpha
    end

    rhs_soc[rhs_lo:rhs_hi] .*= alpha
    if nz_lo <= nz_hi
        @inbounds for k in nz_lo:nz_hi
            V[k] *= alpha
        end
    end
    return alpha
end

function _normalize_soc_triplets_by_rhs!(
    I::Vector{Int},
    V::Vector{Float64},
    rhs_soc::Vector{Float64},
    soc_ptr_local::Vector{Int};
    row_shift::Int=0,
    floor::Float64=1.0,
)
    if length(soc_ptr_local) <= 1 || isempty(rhs_soc)
        return nothing
    end

    soc_rows = length(rhs_soc)
    row_scale = ones(Float64, soc_rows)

    for i in 1:(length(soc_ptr_local)-1)
        lo = soc_ptr_local[i]
        hi = soc_ptr_local[i+1] - 1
        if lo > hi
            continue
        end
        block_scale = maximum(abs, @view rhs_soc[lo:hi])
        alpha = 1.0 / max(block_scale, floor)
        if alpha != 1.0
            rhs_soc[lo:hi] .*= alpha
            row_scale[lo:hi] .= alpha
        end
    end

    @inbounds for k in eachindex(V)
        local_row = I[k] - row_shift
        if 1 <= local_row <= soc_rows
            V[k] *= row_scale[local_row]
        end
    end

    return nothing
end

function _bounded_linear_rows_to_canonical_triplets(
    A::SparseMatrixCSC{Float64,Int32},
    lcon::Vector{Float64},
    ucon::Vector{Float64},
)
    size(A, 1) == length(lcon) || error("Constraint lower-bound vector length must match the number of rows of A.")
    size(A, 1) == length(ucon) || error("Constraint upper-bound vector length must match the number of rows of A.")

    m, n = size(A)
    eq_dest = zeros(Int, m)
    lower_dest = zeros(Int, m)
    upper_dest = zeros(Int, m)
    beq = Float64[]
    bineq = Float64[]
    sizehint!(beq, m)
    sizehint!(bineq, m)

    number_eq = 0
    number_ineq = 0

    for i in 1:m
        li = lcon[i]
        ui = ucon[i]

        if !isfinite(li) && !isfinite(ui)
            continue
        elseif isfinite(li) && isfinite(ui) && li == ui
            number_eq += 1
            eq_dest[i] = number_eq
            push!(beq, li)
        else
            if isfinite(li)
                number_ineq += 1
                lower_dest[i] = number_ineq
                push!(bineq, li)
            end
            if isfinite(ui)
                number_ineq += 1
                upper_dest[i] = number_ineq
                push!(bineq, -ui)
            end
        end
    end

    total_linear_rows = number_eq + number_ineq
    I = Int[]
    J = Int[]
    V = Float64[]
    sizehint!(I, 2 * nnz(A))
    sizehint!(J, 2 * nnz(A))
    sizehint!(V, 2 * nnz(A))

    rowvals_A = rowvals(A)
    nzvals_A = nonzeros(A)
    for col in 1:n
        for p in nzrange(A, col)
            row = rowvals_A[p]
            val = nzvals_A[p]

            dest_eq = eq_dest[row]
            if dest_eq != 0
                push!(I, dest_eq)
                push!(J, col)
                push!(V, val)
            end

            dest_lower = lower_dest[row]
            if dest_lower != 0
                push!(I, number_eq + dest_lower)
                push!(J, col)
                push!(V, val)
            end

            dest_upper = upper_dest[row]
            if dest_upper != 0
                push!(I, number_eq + dest_upper)
                push!(J, col)
                push!(V, -val)
            end
        end
    end

    rhs_linear = Vector{Float64}(undef, total_linear_rows)
    if number_eq > 0
        rhs_linear[1:number_eq] .= beq
    end
    if number_ineq > 0
        rhs_linear[(number_eq+1):end] .= bineq
    end

    return I, J, V, rhs_linear, number_eq, number_ineq
end

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
        number_eq,
        number_ineq,
        l_vec,
        u_vec,
        soc_var_idx,
        number_lu_x,
        obj_constant,
    )
end

function _build_from_SOCP_data_owned(
    Q::SparseMatrixCSC,
    c::Vector{Float64},
    A::SparseMatrixCSC,
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
    m, n = size(A)
    length(c) == n || error("Dimension mismatch: length(c) must equal number of columns of A.")
    length(rhs) == m || error("Dimension mismatch: length(rhs) must equal number of rows of A.")
    length(l) == n || error("Dimension mismatch: length(l) must equal number of variables.")
    length(u) == n || error("Dimension mismatch: length(u) must equal number of variables.")
    number_eq >= 0 || error("number_eq must be nonnegative.")
    number_ineq >= 0 || error("number_ineq must be nonnegative.")
    number_eq + number_ineq <= m || error("number_eq + number_ineq cannot exceed the number of rows of A.")
    !isempty(SOC_con_idx) || error("SOC_con_idx must contain at least one sentinel entry.")
    !isempty(SOC_var_idx) || error("SOC_var_idx must contain at least one sentinel entry.")
    SOC_con_idx[1] == number_eq + number_ineq + 1 || error("SOC_con_idx must start at number_eq + number_ineq + 1.")
    SOC_con_idx[end] == m + 1 || error("SOC_con_idx must end at m + 1.")

    soc_rhs_vec = rhs[(number_eq+number_ineq+1):end]
    soc_rhs_full = zeros(Float64, m)
    if !isempty(soc_rhs_vec)
        soc_rhs_full[(number_eq+number_ineq+1):end] .= soc_rhs_vec
    end

    number_lu_x = SOC_var_idx[1] - 1
    SOC_var_idx[end] == n + 1 || error("SOC_var_idx must end at n + 1.")
    0 <= number_lu_x <= n || error("Invalid SOC_var_idx start.")

    AL = fill(-Inf, m)
    AU = fill(Inf, m)
    if number_eq > 0
        AL[1:number_eq] .= rhs[1:number_eq]
        AU[1:number_eq] .= rhs[1:number_eq]
    end
    if number_ineq > 0
        lin_range = (number_eq+1):(number_eq+number_ineq)
        AL[lin_range] .= rhs[lin_range]
    end

    # if verbose
    #     println("FORMULATING SOCP ...")
    #     println("  total rows = ", m, ", total cols = ", n)
    #     println("  equalities = ", number_eq)
    #     println("  linear inequalities = ", number_ineq)
    #     println("  SOC constraints = ", length(SOC_con_idx) - 1)
    #     println("  SOC variables = ", length(SOC_var_idx) - 1)
    # end

    Q_owned = Q isa SparseMatrixCSC{Float64,Int32} ? Q : SparseMatrixCSC{Float64,Int32}(Q)
    A_owned = A isa SparseMatrixCSC{Float64,Int32} ? A : SparseMatrixCSC{Float64,Int32}(A)

    return QP_info_cpu(
        Q_owned,
        c,
        A_owned,
        SparseMatrixCSC{Float64,Int32}(A_owned'),
        soc_rhs_vec,
        soc_rhs_full,
        AL,
        AU,
        SOC_con_idx,
        number_eq,
        number_ineq,
        l,
        u,
        SOC_var_idx,
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

    return _build_from_SOCP_data_owned(Q, c, A, rhs, SOC_con_idx, number_eq, number_ineq, l, u, SOC_var_idx;
        obj_constant=obj_constant, verbose=verbose)
end

# ============================================================================
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

# Run the dataset of QP problems from a specified directory and save the results to a CSV file
function run_dataset(data_path::String, result_path::String, params::HPRSOCP_parameters)

    files = readdir(data_path)

    # Specify the path and filename for the CSV file
    csv_file = joinpath(result_path, "HPRSOCP_result.csv")

    # redirect the output to a file
    log_path = joinpath(result_path, "HPRSOCP_log.txt")

    if !isdir(result_path)
        mkdir(result_path)
    end

    io = open(log_path, "a")

    # if csv file exists, read the existing results, where each column is an any array
    if isfile(csv_file)
        result_table = CSV.read(csv_file, DataFrame)
        namelist = Vector{Any}(result_table.name[1:end-2])
        iterlist = Vector{Any}(result_table.iter[1:end-2])
        timelist = Vector{Any}(result_table.alg_time[1:end-2])
        reslist = Vector{Any}(result_table.res[1:end-2])
        objlist = Vector{Any}(result_table.primal_obj[1:end-2])
        statuslist = Vector{Any}(result_table.status[1:end-2])
        iter4list = Vector{Any}(result_table.iter_4[1:end-2])
        time3list = Vector{Any}(result_table.time_4[1:end-2])
        iter6list = Vector{Any}(result_table.iter_6[1:end-2])
        time6list = Vector{Any}(result_table.time_6[1:end-2])
        powerlist = Vector{Any}(result_table.power_time[1:end-2])
    else
        namelist = []
        iterlist = []
        timelist = []
        reslist = []
        objlist = []
        statuslist = []
        iter4list = []
        time3list = []
        iter6list = []
        time6list = []
        powerlist = []
    end

    for i = 1:length(files)
        file = files[i]
        file_lc = lowercase(file)
        if (endswith(file_lc, ".mps") || endswith(file_lc, ".cbf")) && !(file in namelist)
            FILE_NAME = joinpath(data_path, file)
            println(@sprintf("solving the problem %d", i), @sprintf(": %s", file))

            redirect_stdout(io) do
                println(@sprintf("solving the problem %d", i), @sprintf(": %s", file))
                println("main run starts: ----------------------------------------------------------------------------------------------------------")
                t_start_all = time()
                model = nothing
                if endswith(file_lc, ".cbf")
                    model = build_from_cbf(FILE_NAME, verbose=params.verbose)
                else
                    model = build_from_mps(FILE_NAME, verbose=params.verbose)
                end
                results = optimize(model, params)
                params.warm_up = false  # disable warm-up for next runs
                all_time = time() - t_start_all
                println("main run ends----------------------------------------------------------------------------------------------------------")


                println("iter = ", results.iter,
                    @sprintf("  time = %3.2e", results.time),
                    @sprintf("  residual = %3.2e", results.residuals),
                    @sprintf("  primal_obj = %3.15e", results.primal_obj),
                )

                push!(namelist, file)
                push!(iterlist, results.iter)
                push!(timelist, min(results.time, params.time_limit))
                push!(reslist, results.residuals)
                push!(objlist, results.primal_obj)
                push!(statuslist, results.status)
                push!(iter4list, results.iter_4)
                push!(time3list, min(results.time_4, params.time_limit))
                push!(iter6list, results.iter_6)
                push!(time6list, min(results.time_6, params.time_limit))
                push!(powerlist, results.power_time)

            end

            result_table = DataFrame(name=namelist,
                iter=iterlist,
                alg_time=timelist,
                res=reslist,
                primal_obj=objlist,
                status=statuslist,
                iter_4=iter4list,
                time_4=time3list,
                iter_6=iter6list,
                time_6=time6list,
                power_time=powerlist,
            )

            # compute the shifted geometric mean of the algorithm_time, put it in the last row
            geomean_time = exp(mean(log.(timelist .+ 10.0))) - 10.0
            geomean_time_4 = exp(mean(log.(time3list .+ 10.0))) - 10.0
            geomean_time_6 = exp(mean(log.(time6list .+ 10.0))) - 10.0
            geomean_iter = exp(mean(log.(iterlist .+ 10.0))) - 10.0
            geomean_iter_4 = exp(mean(log.(iter4list .+ 10.0))) - 10.0
            geomean_iter_6 = exp(mean(log.(iter6list .+ 10.0))) - 10.0
            push!(result_table, ["SGM10", geomean_iter, geomean_time, "", "", "", geomean_iter_4, geomean_time_4, geomean_iter_6, geomean_time_6, ""])

            # count the number of solved instances, termlist = "OPTIMAL" means solved
            solved = count(x -> x < params.time_limit, timelist)
            solved_3 = count(x -> x < params.time_limit, time3list)
            solved_6 = count(x -> x < params.time_limit, time6list)
            push!(result_table, ["solved", "", solved, "", "", "", "", solved_3, "", solved_6, ""])

            CSV.write(csv_file, result_table)
        end
    end

    close(io)
end

# ============================================================================
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
