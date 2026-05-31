#!/usr/bin/env julia
#
# Single-species disordered Bose-Hubbard chain.
# Sparse Hamiltonian construction plus POLFED selected diagonalization.
#
# Hamiltonian:
# H = -t sum_<i,j> (b_i^dag b_j + h.c.)
#     + U sum_i n_i (n_i - 1)
#     + sum_i eps_i n_i
#
# The Hilbert space is restricted to a fixed total boson number N and a maximum
# on-site occupation nmax.

using LinearAlgebra
using SparseArrays
using Random
using Printf
using DelimitedFiles

function parse_commandline()
    args = Dict{String,Any}(
        "L" => 10,
        "N" => -1,
        "nmax" => 2,
        "t" => 1.0,
        "U" => 4.0,
        "W" => 6.0,
        "nreal" => 20,
        "seed" => 1234,
        "howmany" => 400,
        "target" => "middle",
        "block" => 4,
        "boundary" => "periodic",
        "full" => false,
        "outdir" => "polfed_bose_hubbard_results",
        "save-fields" => false,
    )
    int_args = Set(["L", "N", "nmax", "nreal", "seed", "howmany", "block"])
    float_args = Set(["t", "U", "W"])
    string_args = Set(["target", "boundary", "outdir"])
    flag_args = Set(["full", "save-fields"])

    i = 1
    while i <= length(ARGS)
        token = ARGS[i]
        if !startswith(token, "--")
            error("Unexpected positional argument: $(token)")
        end
        key = token[3:end]
        if key in flag_args
            args[key] = true
            i += 1
            continue
        end
        if i == length(ARGS)
            error("Missing value after --$(key)")
        end
        value = ARGS[i + 1]
        if key in int_args
            args[key] = parse(Int, value)
        elseif key in float_args
            args[key] = parse(Float64, value)
        elseif key in string_args
            args[key] = value
        else
            error("Unknown option --$(key)")
        end
        i += 2
    end

    return args
end

function generate_fixedN_occupations(L::Int, N::Int, nmax::Int)
    if L < 1
        error("L must be positive. Got L=$(L).")
    end
    if N < 0
        error("Particle number must be nonnegative. Got N=$(N).")
    end
    if nmax < 1
        error("nmax must be positive. Got nmax=$(nmax).")
    end
    if N > L * nmax
        error("No states in this sector: N=$(N) exceeds L*nmax=$(L*nmax).")
    end

    states = Vector{Int}[]
    current = zeros(Int, L)

    function fill_site(site::Int, remaining::Int)
        if site == L
            if remaining <= nmax
                current[site] = remaining
                push!(states, copy(current))
            end
            return
        end

        max_here = min(nmax, remaining)
        for n in 0:max_here
            current[site] = n
            fill_site(site + 1, remaining - n)
        end
    end

    fill_site(1, N)
    return states
end

occupation_key(occ::Vector{Int}) = Tuple(occ)

function diagonal_energy(occ::Vector{Int}, L::Int, U::Float64,
                         eps::Vector{Float64})
    e = 0.0
    for i in 1:L
        e += U * occ[i] * (occ[i] - 1)
        e += eps[i] * occ[i]
    end
    return e
end

function hopping_pairs(L::Int, boundary::String)
    pairs = Tuple{Int,Int}[]
    for i in 1:(L - 1)
        push!(pairs, (i, i + 1))
    end

    if boundary == "periodic" && L > 2
        push!(pairs, (L, 1))
    elseif boundary != "open"
        error("Unknown boundary=$(boundary). Use open or periodic.")
    end

    return pairs
end

function hop_state(occ::Vector{Int}, from::Int, to::Int, nmax::Int)
    if occ[from] == 0 || occ[to] >= nmax
        return nothing, 0.0
    end

    newocc = copy(occ)
    amp = sqrt(Float64(occ[from] * (occ[to] + 1)))
    newocc[from] -= 1
    newocc[to] += 1
    return newocc, amp
end

function build_bose_hubbard_sparse(L::Int, N::Int, nmax::Int, t::Float64,
                                   U::Float64, eps::Vector{Float64};
                                   boundary::String = "periodic")
    basis = generate_fixedN_occupations(L, N, nmax)
    dim = length(basis)

    index = Dict{Tuple{Vararg{Int}},Int}(occupation_key(s) => i for (i, s) in enumerate(basis))
    bonds = hopping_pairs(L, boundary)

    rows = Int[]
    cols = Int[]
    vals = Float64[]
    sizehint!(rows, dim * (1 + 2 * length(bonds)))
    sizehint!(cols, dim * (1 + 2 * length(bonds)))
    sizehint!(vals, dim * (1 + 2 * length(bonds)))

    for (col, occ) in enumerate(basis)
        push!(rows, col)
        push!(cols, col)
        push!(vals, diagonal_energy(occ, L, U, eps))

        for (i, j) in bonds
            newocc, amp = hop_state(occ, i, j, nmax)
            if newocc !== nothing
                push!(rows, index[occupation_key(newocc)])
                push!(cols, col)
                push!(vals, -t * amp)
            end

            newocc, amp = hop_state(occ, j, i, nmax)
            if newocc !== nothing
                push!(rows, index[occupation_key(newocc)])
                push!(cols, col)
                push!(vals, -t * amp)
            end
        end
    end

    H = sparse(rows, cols, vals, dim, dim)
    return 0.5 * (H + H')
end

function parse_target(target_string::String)
    t = lowercase(strip(target_string))
    if t == "middle"
        return :middle
    elseif t == "maxdos"
        return :maxdos
    else
        return parse(Float64, target_string)
    end
end

function normalize_eigenvalue_result(result)
    vals = result isa Tuple ? result[1] : result
    if hasproperty(vals, :values)
        vals = getproperty(vals, :values)
    elseif hasproperty(vals, :eigenvalues)
        vals = getproperty(vals, :eigenvalues)
    end
    return vals
end

function polfed_energies(rng::AbstractRNG, H::SparseMatrixCSC{Float64,Int},
                         howmany::Int, target, block::Int)
    try
        @eval using Polfed
    catch err
        msg = sprint(showerror, err)
        error("Package Polfed is required only for POLFED runs, but it is not available in this Julia environment. " *
              "Use --full for small tests or install/copy the Julia project with Polfed. Original error: $(msg)")
    end

    dim = size(H, 1)

    if block <= 1
        x0 = randn(rng, dim)
        x0 ./= norm(x0)
    else
        x0 = randn(rng, dim, block)
        x0 = Matrix(qr(x0).Q)
    end

    result = try
        Base.invokelatest(Polfed.polfed, H, x0, howmany, target)
    catch err
        msg = sprint(showerror, err)
        error("POLFED call failed for polfed(H, x0, howmany, target). " *
              "Check the installed Polfed.jl API on the cluster. Original error: $(msg)")
    end

    vals = sort(collect(real(normalize_eigenvalue_result(result))))
    if length(vals) > howmany
        return vals[1:howmany]
    end
    return vals
end

function full_energies(H::SparseMatrixCSC{Float64,Int})
    vals = eigvals(Matrix(Hermitian(Matrix(H))))
    return sort(collect(real(vals)))
end

function write_outputs(outfile::String, E::Matrix{Float64}, args, L::Int, N::Int,
                       nmax::Int, dim_expected::Integer, t::Float64, U::Float64,
                       W::Float64, naccepted::Int, nrequested::Int,
                       nattempted::Int, nskipped::Int, boundary::String,
                       mode_label::String)
    tmpfile = outfile * ".tmp"
    open(tmpfile, "w") do io
        println(io, "# Single-species disordered Bose-Hubbard chain")
        println(io, "# H = -t sum_<i,j> (b_i^dag b_j + h.c.) + U sum_i n_i*(n_i-1) + sum_i eps_i*n_i")
        @printf(io, "# L=%d N=%d nmax=%d dim=%d t=%.16g U=%.16g W=%.16g naccepted=%d nrequested=%d nattempted=%d nskipped=%d boundary=%s mode=%s\n",
                L, N, nmax, dim_expected, t, U, W, naccepted, nrequested,
                nattempted, nskipped, boundary, mode_label)
        println(io, "# Each column is one accepted disorder realization.")
        println(io, "# Incomplete POLFED realizations are skipped, not padded with NaN.")
        println(io, "# POLFED target=$(args["target"]), requested howmany=$(args["howmany"]).")
        writedlm(io, E[:, 1:naccepted])
    end
    mv(tmpfile, outfile; force = true)
end

function write_fields(fieldfile::String, fields::Matrix{Float64}, nreal_done::Int)
    tmpfile = fieldfile * ".tmp"
    open(tmpfile, "w") do io
        println(io, "# Disorder fields eps_i. Each column is one accepted realization.")
        writedlm(io, fields[:, 1:nreal_done])
    end
    mv(tmpfile, fieldfile; force = true)
end

function append_skipped(skipfile::String, attempt::Int, returned::Int, finite_count::Int,
                        requested::Int, reason::String)
    new_file = !isfile(skipfile)
    open(skipfile, "a") do io
        if new_file
            println(io, "# attempt returned finite requested reason")
        end
        @printf(io, "%d %d %d %d %s\n", attempt, returned, finite_count, requested, reason)
    end
end

function main()
    args = parse_commandline()

    L = args["L"]
    N = args["N"] < 0 ? L : args["N"]
    nmax = args["nmax"]
    t = args["t"]
    U = args["U"]
    W = args["W"]
    nreal = args["nreal"]
    seed = args["seed"]
    howmany = args["howmany"]
    target = parse_target(args["target"])
    block = args["block"]
    boundary = lowercase(args["boundary"])
    use_full = args["full"]
    outdir = args["outdir"]
    save_fields = args["save-fields"]

    if L < 1
        error("L must be positive. Got L=$(L).")
    end
    if N < 0
        error("N must be nonnegative after defaulting. Got N=$(N).")
    end
    if nmax < 1
        error("nmax must be positive. Got nmax=$(nmax).")
    end
    if N > L * nmax
        error("N must satisfy N <= L*nmax. Got N=$(N), L=$(L), nmax=$(nmax).")
    end
    if nreal < 1
        error("nreal must be positive. Got nreal=$(nreal).")
    end
    if howmany < 1
        error("howmany must be positive. Got howmany=$(howmany).")
    end
    if block < 1
        error("block must be positive. Got block=$(block).")
    end
    if W < 0
        error("W must be nonnegative. Got W=$(W).")
    end

    mkpath(outdir)
    rng = MersenneTwister(seed)

    dim_expected = length(generate_fixedN_occupations(L, N, nmax))
    mode_label = use_full ? "full" : "polfed"

    filename = @sprintf("energies_%s_U%.6g_L%d_N%d_nmax%d_W%.6g_boundary%s_nreal%d.txt",
                        mode_label, U, L, N, nmax, W, boundary, nreal)
    outfile = joinpath(outdir, filename)
    partial_outfile = replace(outfile, ".txt" => "_partial.txt")
    fieldfile = joinpath(outdir, @sprintf("fields_U%.6g_L%d_N%d_nmax%d_W%.6g_boundary%s_nreal%d.txt",
                                          U, L, N, nmax, W, boundary, nreal))
    skipfile = joinpath(outdir, @sprintf("skipped_U%.6g_L%d_N%d_nmax%d_W%.6g_boundary%s_nreal%d.txt",
                                         U, L, N, nmax, W, boundary, nreal))

    @printf("Bose-Hubbard: L=%d, N=%d, nmax=%d, dim=%d\n", L, N, nmax, dim_expected)
    @printf("t=%.8g, U=%.8g, W=%.8g, nreal=%d, boundary=%s, mode=%s\n",
            t, U, W, nreal, boundary, mode_label)
    @printf("Writing partial results to %s\n", partial_outfile)
    flush(stdout)

    nrows = use_full ? dim_expected : howmany
    E = fill(NaN, nrows, nreal)
    all_fields = zeros(Float64, L, nreal)
    naccepted = 0
    nskipped = 0

    for r in 1:nreal
        eps = [2.0 * W * rand(rng) - W for _ in 1:L]

        @printf("realization %d / %d ... ", r, nreal)
        flush(stdout)

        H = build_bose_hubbard_sparse(L, N, nmax, t, U, eps; boundary = boundary)

        vals = use_full ? full_energies(H) : polfed_energies(rng, H, howmany, target, block)
        finite_count = count(isfinite, vals)

        if length(vals) < nrows || finite_count < nrows
            nskipped += 1
            reason = length(vals) < nrows ? "too_few_eigenvalues" : "nonfinite_eigenvalues"
            append_skipped(skipfile, r, length(vals), finite_count, nrows, reason)
            @printf("skipped. returned=%d/%d finite=%d reason=%s\n",
                    length(vals), nrows, finite_count, reason)
            flush(stdout)
            continue
        end

        naccepted += 1
        E[1:nrows, naccepted] .= vals[1:nrows]
        all_fields[:, naccepted] .= eps

        @printf("accepted %d. Emin=%.8f, Emax=%.8f\n",
                naccepted, minimum(vals), maximum(vals))
        write_outputs(partial_outfile, E, args, L, N, nmax, dim_expected, t, U,
                      W, naccepted, nreal, r, nskipped, boundary, mode_label)
        if save_fields
            write_fields(fieldfile, all_fields, naccepted)
        end
        flush(stdout)
    end

    write_outputs(outfile, E, args, L, N, nmax, dim_expected, t, U, W,
                  naccepted, nreal, nreal, nskipped, boundary, mode_label)
    @printf("Saved final energies: %s\n", outfile)
    @printf("Accepted realizations: %d / %d\n", naccepted, nreal)
    @printf("Skipped realizations: %d\n", nskipped)
    if nskipped > 0
        @printf("Skipped-realization log: %s\n", skipfile)
    end
    if save_fields
        @printf("Saved fields: %s\n", fieldfile)
    end
end

main()
