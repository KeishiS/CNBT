module SimpleHerm

using
    Graphs,
    LinearAlgebra,
    SparseArrays,
    KrylovKit,
    Clustering

export
    simpleHerm

function herm_adjacency_matrix(dg::SimpleDiGraph; α=1.0im)
    ᾱ = conj(α)
    n = nv(dg)
    m = ne(dg)
    for i in 1:n
        has_edge(dg, i, i) && throw(ErrorException("self-loop should not exist"))
    end

    idxs = sizehint!(Vector{Int}(), 2m)
    idys = sizehint!(Vector{Int}(), 2m)
    vals = sizehint!(Vector{ComplexF64}(), 2m)
    for i in 1:n
        for j in (i+1):n
            chk₁ = has_edge(dg, i, j)
            chk₂ = has_edge(dg, j, i)

            if chk₁ || chk₂
                elem = chk₁ && chk₂ ? 1.0 :
                       chk₁ ? α : ᾱ
                push!(idxs, i)
                push!(idys, j)
                push!(vals, elem)

                push!(idxs, j)
                push!(idys, i)
                push!(vals, conj(elem))
            end
        end
    end

    sparse(idxs, idys, vals, n, n)
end

function herm_normalized_laplacian(dg::SimpleDiGraph; α=1.0im)
    H = herm_adjacency_matrix(dg; α=α)
    d = indegree(dg) + outdegree(dg) # TODO
    invd = Util.revdeg(d; p=-0.5)

    I - (H .* invd) .* invd'
end

function simpleHerm(dg::SimpleDiGraph, K; α=nothing, ϵ::Float64=1e-8)
    if isnothing(α)
        α = Util.root_of_unity(K)
    end

    L = herm_normalized_laplacian(dg; α=α)
    _, ϕ, _ = eigsolve(L, 1, :SR)
    ϕ = ϕ[1]
    d = indegree(dg) + outdegree(dg)
    F = ϕ ./ sqrt.(d .+ ϵ)
    F = [real.(F) imag.(F)]

    ret = kmeans(F', K)
    assignments(ret), F
end


include("Util.jl")

end
