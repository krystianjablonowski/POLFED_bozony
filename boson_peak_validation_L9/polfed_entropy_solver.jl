#!/usr/bin/env julia
"""Run one Bose-Hubbard realization and one U point with Polfed.jl.

Only scalar observables and the mean Q-sector distribution are written.  The
potentially multi-gigabyte eigenvector matrix never leaves this process.
"""

using DelimitedFiles
using LinearAlgebra
using Printf
using Random
using SparseArrays
using Statistics
using Polfed


function parse_args()
    values = Dict{String,String}()
    index = 1
    while index <= length(ARGS)
        key = ARGS[index]
        startswith(key, "--") || error("Unexpected argument: $(key)")
        index == length(ARGS) && error("Missing value after $(key)")
        values[key[3:end]] = ARGS[index + 1]
        index += 2
    end
    required = ["L", "N", "nmax", "t", "U", "epsilon-file", "output",
                "howmany", "selected", "target", "block", "seed",
                "overestimate-iters", "eigentol",
                "gap-edge-discard"]
    missing = [key for key in required if !haskey(values, key)]
    isempty(missing) || error("Missing arguments: " * join(missing, ", "))
    return values
end


function basis_states(L::Int, N::Int, nmax::Int)
    states = Vector{Vector{Int}}()
    current = zeros(Int, L)
    function recurse(site::Int, remaining::Int)
        if site == L
            if 0 <= remaining <= nmax
                current[site] = remaining
                push!(states, copy(current))
            end
            return
        end
        lower = max(0, remaining - nmax * (L - site))
        upper = min(nmax, remaining)
        for occupation in lower:upper
            current[site] = occupation
            recurse(site + 1, remaining - occupation)
        end
    end
    recurse(1, N)
    return states
end


function build_hamiltonian(states, L::Int, nmax::Int, t::Float64,
                           U::Float64, epsilon::Vector{Float64})
    dim = length(states)
    lookup = Dict{Tuple{Vararg{Int}},Int}(Tuple(state) => i for (i, state) in enumerate(states))
    rows = Int[]
    cols = Int[]
    data = Float64[]
    sizehint!(rows, dim * (1 + 2 * (L - 1)))
    sizehint!(cols, dim * (1 + 2 * (L - 1)))
    sizehint!(data, dim * (1 + 2 * (L - 1)))
    q = zeros(Int, dim)

    for (column, state) in enumerate(states)
        q[column] = sum(n * (n - 1) for n in state)
        diagonal = U * q[column] + sum(epsilon[i] * state[i] for i in 1:L)
        push!(rows, column); push!(cols, column); push!(data, diagonal)
        for site in 1:(L - 1)
            for (origin, destination_site) in ((site, site + 1), (site + 1, site))
                a, b = state[origin], state[destination_site]
                (a == 0 || b >= nmax) && continue
                target = copy(state)
                target[origin] -= 1
                target[destination_site] += 1
                push!(rows, lookup[Tuple(target)])
                push!(cols, column)
                push!(data, -t * sqrt(Float64(a * (b + 1))))
            end
        end
    end
    H = sparse(rows, cols, data, dim, dim)
    return 0.5 * (H + H'), q
end


function parse_target(value::String)
    lowered = lowercase(strip(value))
    lowered == "middle" && return :middle
    lowered == "maxdos" && return :maxdos
    return parse(Float64, value)
end


function gap_ratio(values::Vector{Float64}, edge_discard::Int)
    length(values) < 3 && return NaN, 0
    spacings = diff(sort(values))
    ratios = min.(spacings[1:end-1], spacings[2:end]) ./
             max.(spacings[1:end-1], spacings[2:end])
    ratios = ratios[isfinite.(ratios)]
    if edge_discard > 0 && length(ratios) > 2 * edge_discard
        ratios = ratios[(edge_discard + 1):(end - edge_discard)]
    end
    isempty(ratios) && return NaN, 0
    return mean(ratios), length(ratios)
end


function observables(vectors::Matrix{Float64}, q::Vector{Int})
    dim, count = size(vectors)
    qmax = maximum(q; init=0)
    entropy = zeros(count)
    entropy2 = zeros(count)
    ipr = zeros(count)
    qmean = zeros(count)
    qvar = zeros(count)
    hq = zeros(count)
    sintra = zeros(count)
    p_q_mean = zeros(qmax + 1)

    for column in 1:count
        probabilities = abs2.(view(vectors, :, column))
        entropy[column] = -sum(p > 0 ? p * log(p) : 0.0 for p in probabilities)
        ipr[column] = sum(abs2, probabilities)
        entropy2[column] = -log(ipr[column])
        p_q = zeros(qmax + 1)
        for row in eachindex(probabilities)
            p_q[q[row] + 1] += probabilities[row]
        end
        p_q_mean .+= p_q
        qmean[column] = sum((index - 1) * p_q[index] for index in eachindex(p_q))
        qvar[column] = sum(((index - 1) - qmean[column])^2 * p_q[index]
                           for index in eachindex(p_q))
        hq[column] = -sum(p > 0 ? p * log(p) : 0.0 for p in p_q)
        sintra[column] = entropy[column] - hq[column]
    end
    p_q_mean ./= count
    identity_error = maximum(abs.(entropy .- (hq .+ sintra)); init=0.0)
    logdim = log(dim)
    return Dict(
        "entropy" => mean(entropy), "entropy_norm" => mean(entropy) / logdim,
        "entropy2" => mean(entropy2), "entropy2_norm" => mean(entropy2) / logdim,
        "IPR" => mean(ipr), "Q_mean" => mean(qmean), "Q_variance" => mean(qvar),
        "H_Q" => mean(hq), "S_intra_Q" => mean(sintra),
        "entropy_identity_error" => identity_error, "P_Q" => p_q_mean,
    )
end


function sampled_diagnostics(H, values, vectors)
    count = size(vectors, 2)
    sample_count = min(count, 24)
    indices = unique(round.(Int, range(1, count; length=sample_count)))
    sampled_vectors = vectors[:, indices]
    sampled_values = values[indices]
    residual = H * sampled_vectors - sampled_vectors .* reshape(sampled_values, 1, :)
    denominators = max.(sqrt.(sum(abs2, H * sampled_vectors; dims=1)),
                        abs.(reshape(sampled_values, 1, :)), 1.0)
    residual_max = maximum(sqrt.(sum(abs2, residual; dims=1)) ./ denominators)
    gram = sampled_vectors' * sampled_vectors
    orthogonality = maximum(abs.(gram - I))
    return residual_max, orthogonality
end


function main()
    args = parse_args()
    L = parse(Int, args["L"]); N = parse(Int, args["N"])
    nmax = parse(Int, args["nmax"]); t = parse(Float64, args["t"])
    U = parse(Float64, args["U"]); howmany = parse(Int, args["howmany"])
    selected = parse(Int, args["selected"]); block = parse(Int, args["block"])
    seed = parse(Int, args["seed"]); edge = parse(Int, args["gap-edge-discard"])
    epsilon = vec(readdlm(args["epsilon-file"], Float64))
    length(epsilon) == L || error("Expected $(L) disorder values, got $(length(epsilon))")
    states = basis_states(L, N, nmax)
    H, q = build_hamiltonian(states, L, nmax, t, U, epsilon)
    dim = size(H, 1)
    1 <= selected <= howmany < dim || error("Require 1 <= selected <= howmany < D")

    rng = MersenneTwister(seed)
    if block <= 1
        x0 = normalize(randn(rng, dim))
    else
        factorization = qr(randn(rng, dim, block))
        x0 = Matrix(factorization.Q[:, 1:block])
    end
    started = time()
    fact = Polfed.PolfedCore.FactorizationConfig(
        overestimate_iters=parse(Float64, args["overestimate-iters"]),
        eigentol=parse(Float64, args["eigentol"]),
    )
    values, vectors, report = Polfed.polfed(
        H, x0, howmany, parse_target(args["target"]);
        produce_report=true, fact=fact
    )
    elapsed = time() - started
    order = sortperm(real.(values))
    values = collect(real.(values[order]))
    vectors = Matrix(real.(vectors[:, order]))
    if length(values) < selected
        println(stderr, "===== POLFED REPORT =====")
        try
            Polfed.PolfedCore.display_report(report)
        catch report_error
            show(stderr, MIME("text/plain"), report)
            println(stderr)
            println(stderr, "display_report fallback: ", report_error)
        end
        error(
            "POLFED returned $(length(values)) eigenpairs, fewer than the $(selected) required for observables"
        )
    end

    first_selected = fld(length(values) - selected, 2) + 1
    selected_range = first_selected:(first_selected + selected - 1)
    selected_values = values[selected_range]
    selected_vectors = vectors[:, selected_range]
    obs = observables(selected_vectors, q)
    rmean, rcount = gap_ratio(selected_values, edge)
    residual_max, orthogonality = sampled_diagnostics(H, selected_values, selected_vectors)
    sigma = 0.5 * (minimum(values) + maximum(values))

    header = ["entropy", "entropy_norm", "entropy2", "entropy2_norm", "IPR",
              "gap_ratio", "gap_ratio_count", "Q_mean", "Q_variance", "H_Q",
              "S_intra_Q", "entropy_identity_error", "P_Q", "E_min_over_t",
              "E_max_over_t", "sigma_over_t", "solver_time_s", "residual_max",
              "orthogonality_error", "selected_states"]
    row = [obs["entropy"], obs["entropy_norm"], obs["entropy2"], obs["entropy2_norm"],
           obs["IPR"], rmean, rcount, obs["Q_mean"], obs["Q_variance"], obs["H_Q"],
           obs["S_intra_Q"], obs["entropy_identity_error"],
           join((@sprintf("%.17g", value) for value in obs["P_Q"]), ";"),
           minimum(values) / t, maximum(values) / t, sigma / t, elapsed,
           residual_max, orthogonality, selected]
    open(args["output"], "w") do io
        println(io, join(header, '\t'))
        println(io, join((value isa AbstractFloat ? @sprintf("%.17g", value) : string(value)
                          for value in row), '\t'))
    end
    @printf("POLFED D=%d requested=%d selected=%d time=%.3fs report=%s\n",
            dim, howmany, selected, elapsed, string(typeof(report)))
end


main()
