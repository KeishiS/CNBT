module CNBT

using
    Graphs,
    LinearAlgebra,
    SparseArrays,
    Statistics,
    KrylovKit,
    ArnoldiMethod,
    Random,
    Distributions,
    Clustering

export
    cnbt_matrix,
    outvec,
    invec,
    cnbtSC,
    Util,
    SimpleHerm,
    Herm,
    DISIM,
    BISYM

function cnbt_matrix(dg::SimpleDiGraph; α=1.0im, normalize=false)
    ug = Graph(dg)
    egs = ug |> edges |> collect
    edgeidmap = Dict{Edge,Int}()
    m = 0

    for e in egs
        i, j = src(e), dst(e)
        i == j && continue
        if !haskey(edgeidmap, e)
            m += 1
            edgeidmap[e] = m
        end
    end
    for e in reverse.(egs)
        i, j = src(e), dst(e)
        i == j && continue
        if !haskey(edgeidmap, e)
            m += 1
            edgeidmap[e] = m
        end
    end


    d_both = [ # bidirectional degree
        length(outneighbors(dg, v) ∩ inneighbors(dg, v))
        for v in vertices(dg)
    ]
    d_out = [ # out only degree
        length(setdiff(outneighbors(dg, v), inneighbors(dg, v)))
        for v in vertices(dg)
    ]
    d_in = [ # in only degree
        length(setdiff(inneighbors(dg, v), outneighbors(dg, v)))
        for v in vertices(dg)
    ]
    d = d_both + d_out + d_in

    idxs = sizehint!(Vector{Int}(), m)
    idys = sizehint!(Vector{Int}(), m)
    vals = sizehint!(Vector{ComplexF64}(), m)
    all_edges = edgeidmap |> keys |> collect

    for id in eachindex(all_edges)
        e₁ = all_edges[id]
        id₁ = edgeidmap[e₁]
        u, v = src(e₁), dst(e₁)
        for t in neighbors(ug, v)
            u == t && continue
            id₂ = edgeidmap[Edge(v, t)]

            chk₁ = has_edge(dg, Edge(v, t))
            chk₂ = has_edge(dg, Edge(t, v))
            val = chk₁ && chk₂ ? 1.0 :
                  chk₁ ? α :
                  chk₂ ? conj(α) : 0.0
            if normalize
                val /= d[v] - 1
            end
            if chk₁ || chk₂
                push!(idxs, id₁)
                push!(idys, id₂)
                push!(vals, val)
            end
        end
    end

    sparse(idxs, idys, vals, m, m), edgeidmap
end

function outvec(g::AbstractVecOrMat, dg::SimpleDiGraph, emap::Dict{Edge,Int}; α::ComplexF64=1.0im)
    gᵒᵘᵗ = zeros(eltype(g), (nv(dg), size(g, 2)))
    for (e, id) in emap
        st, ed = src(e), dst(e)
        f = Edge(ed, st)
        if has_edge(dg, e) && has_edge(dg, f)
            gᵒᵘᵗ[st, :] .+= g[id, :]
        elseif has_edge(dg, e)
            gᵒᵘᵗ[st, :] .+= g[id, :] .* α
        elseif has_edge(dg, f)
            gᵒᵘᵗ[st, :] .+= g[id, :] .* conj(α)
        end
    end

    size(g, 2) == 1 ? vec(gᵒᵘᵗ) : gᵒᵘᵗ
end

function invec(g::AbstractVecOrMat, dg::SimpleDiGraph, emap::Dict{Edge,Int}; α::ComplexF64=1.0im)
    gⁱⁿ = zeros(eltype(g), (nv(dg), size(g, 2)))
    for (e, id) in emap
        ed = dst(e)
        gⁱⁿ[ed, :] .+= g[id, :]
    end

    size(g, 2) == 1 ? vec(gⁱⁿ) : gⁱⁿ
end

function cnbtSC(dg::SimpleDiGraph, K;
    α=nothing, num=nothing,
    type::Symbol=:Out, normalize::Bool=true)
    if isnothing(α)
        α = Util.root_of_unity(K)
    end
    if isnothing(num)
        num = fld(K, 2)
    end
    edge2node =
        if type == :Out
            outvec
        else
            invec
        end

    N = nv(dg)
    Bα, emap = cnbt_matrix(dg; α=α)
    λ1, ϕ1, _ = eigsolve(Bα, num, :SR)
    λ2, ϕ2, _ = eigsolve(Bα, num, :LR)
    conds = collect(zip(vcat(λ1, λ2), vcat(ϕ1, ϕ2)))
    sort!(conds, by=x -> x[1] |> real |> abs, rev=true)
    ev = edge2node(hcat(getindex.(conds, 2)[1:num]...), dg, emap; α=α)
    inp = hcat(real.(ev), imag.(ev))
    if normalize
        inp = inp ./ norm.(eachrow(inp))
    end
    ret = kmeans(inp', K)
    return assignments(ret), inp
end

include("Util.jl")
include("SimpleHerm.jl")
include("Herm.jl")
include("DISIM.jl")
include("BISYM.jl")

end # module CNBT
