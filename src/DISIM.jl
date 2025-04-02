module DISIM
# `Co-clustering for directed graphs: the Stochastic co-Blockmodel and spectral algorithm Di-Sim`

include("Util.jl")
using
    LinearAlgebra,
    SparseArrays,
    Statistics,
    Graphs,
    KrylovKit,
    Clustering

export disim

function disim(dg::SimpleDiGraph, K::Int; τ::T=0.0, ϵ::Float64=1e-8, type::Symbol=:Both) where {T<:Number}

    md::Float64 = τ != 0.0 ? Float64(τ) : mean(outdegree(dg))
    Util._check_positive(md)
    A = adjacency_matrix(dg)
    dr = 1 ./ sqrt.(outdegree(dg) .+ md)
    dc = 1 ./ sqrt.(indegree(dg) .+ md)

    L = (A .* dr) .* dc'
    d, u, v, _ = svdsolve(L, K)
    Xl = hcat(view(u, 1:K)...)
    Xr = hcat(view(v, 1:K)...)

    XL = Xl ./ (norm.(eachrow(Xl)) .+ ϵ)
    XR = Xr ./ (norm.(eachrow(Xr)) .+ ϵ)

    inp =
        type == :Both ? hcat(XL, XR) :
        type == :Left ? XL : XR
    ret = kmeans(inp', K)

    assignments(ret), inp
end

end
