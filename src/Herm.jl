module Herm
# `Hermitian matrices for clustering directed graphs: insights and applications`

using
    LinearAlgebra,
    SparseArrays,
    Statistics,
    KrylovKit,
    Graphs,
    Clustering,
    Random,
    Distributions

export herm

function herm(dg::SimpleDiGraph, K::Int; ϵ::T=0.0, normalize::Bool=false) where {T<:Number}
    n = nv(dg)
    Util._check_positive(K)
    if ϵ == 0.0
        p = mean(degree(dg)) / n
        ϵ = 10.0 * sqrt(p * (n / K) * log(p * (n / K)))
    else
        Util._check_positive(ϵ)
    end

    l = isodd(K) ? K - 1 : K

    A = Util.herm_adjacency_matrix(dg; α=1.0im)
    if normalize
        A = Util.normalize_rw(A)
    end
    λ, G, _ = eigsolve(A, l, :LM)
    G = hcat(G...)
    if any(abs.(λ) .> ϵ)
        indices = λ .> ϵ
        λ = λ[indices]
        G = G[:, indices]
    end

    inp = hcat(real.(G), imag.(G))
    ret = kmeans(inp', K)

    assignments(ret), inp
end

function sampleDSBM(k::Int, n::Int, p::Float64, q::Float64, F::AbstractMatrix; seed::Int=42)
    Util._check_positive(k)
    Util._check_positive(n)
    Util._check_probability(p)
    Util._check_probability(q)
    (size(F, 1) != k || size(F, 2) != k) && throw(ErrorException("The size of F must be k x k"))
    any(F .< 0) && throw(DomainError("The element of F must be non-negative"))
    !all(isapprox.(F + F', 1.0; atol=1e-8)) && throw(DomainError("The sum of symmetric elements must be 1"))
    Random.seed!(seed)

    N = k * n
    clst = [i % k + 1 for i in 0:(N-1)] |> shuffle
    idxs = Int[]
    idys = Int[]
    edge_probs = rand(Uniform(), fld(N * (N - 1), 2))
    dir_probs = rand(Uniform(), fld(N * (N - 1), 2))
    cnt = 1
    for i in 1:N
        ci = clst[i]
        for j in i+1:N
            cj = clst[j]
            edge_prob = edge_probs[cnt]
            dir_prob = dir_probs[cnt]
            st = dir_prob < F[ci, cj] ? i : j
            ed = st == j ? i : j

            if (ci == cj && edge_prob < p) ||
               (ci != cj && edge_prob < q)
                push!(idxs, st)
                push!(idys, ed)
            end
            cnt += 1
        end
    end
    dg = DiGraph(N)
    for i in eachindex(idxs)
        add_edge!(dg, idxs[i], idys[i])
    end

    return Dict(
        :clst => clst,
        :dg => dg,
    )
end

include("Util.jl")
end
