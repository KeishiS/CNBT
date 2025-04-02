module BISYM

include("Util.jl")
using
    LinearAlgebra,
    SparseArrays,
    Statistics,
    Graphs,
    KrylovKit,
    Clustering

export bisym

function bisym(dg::SimpleDiGraph, K::Int; discounted::Bool=true, ϵ::Float64=1e-8)
    A = adjacency_matrix(dg)
    din = 1 ./ sqrt.(indegree(dg) .+ ϵ)
    dout = 1 ./ sqrt.(outdegree(dg) .+ ϵ)

    B =
        if discounted
            Symmetric(((A .* dout) .* din') * (A' .* dout'))
        else
            A * A'
        end
    C =
        if discounted
            Symmetric(((A' .* din) .* dout') * (A .* din'))
        else
            A' * A
        end
    U = B + C
    λ, X, _ = eigsolve(U, K)
    X = hcat(view(X, 1:K)...)
    if discounted
        X = X ./ (norm.(eachrow(X)) .+ ϵ)
    end
    ret = kmeans(X', K)

    assignments(ret), X
end

end
