# This file is included by ../utils.jl.

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

