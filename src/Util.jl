module Util

using
    LinearAlgebra,
    SparseArrays,
    Distributions,
    Graphs,
    Random,
    Distributions,
    Clustering

export
    root_of_unity,
    revdeg,
    ari,
    sample_3dsbm,
    sample_3dcsbm,
    constrained_kmeans

function herm_adjacency_matrix(dg::SimpleDiGraph; α::ComplexF64=1.0im)
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

function normalize_rw(A::AbstractMatrix)
    d = sum(abs.(A); dims=2) |> vec
    any(d .== 0) && throw(DomainError("something wrong"))
    invd = 1 ./ d

    A .* invd
end

function ari(pred::AbstractVector, test::AbstractVector)
    length(pred) != length(test) && throw(ErrorException("Two arguments must be the same length vector."))

    randindex(pred, test)[1]
end

function root_of_unity(k)
    cos(2π / k) + sin(2π / k) * im
end

function revdeg(deg::AbstractVector{T} where {T<:Real}; p::Real=-1.0)
    if any(deg .< 0)
        throw(DomainError(deg, "each value must be non-negative"))
    end

    ret = zeros(length(deg))
    ret[deg.>0] .= deg[deg.>0] .^ p

    ret
end

function sample_3dsbm(;
    seed::Int=42, n::Int=1000, c::Real=5, ϵ::Real=5, η::Real=1)
    Random.seed!(seed)

    K = 3
    N = K * n
    γin = 2c / (1 + η * (1 + ϵ))
    γforw = (2c * ϵ * η) / (1 + η * (1 + ϵ))
    γrev = (2c * η) / (1 + η * (1 + ϵ))
    clst = [i % K + 1 for i in 0:(N-1)] |> shuffle
    idx = Int[]
    idy = Int[]
    for i in 1:N
        cᵢ = clst[i]
        for j in 1:N
            i == j && continue
            cⱼ = clst[j]
            rnd = rand(Uniform())
            if cᵢ == cⱼ
                if rnd <= (γin / N) / 2
                    push!(idx, i)
                    push!(idy, j)
                elseif rnd <= γin / N
                    push!(idx, j)
                    push!(idy, i)
                end
            elseif (cᵢ % K) + 1 == cⱼ
                if rnd <= γforw / N
                    push!(idx, i)
                    push!(idy, j)
                end
            elseif (cⱼ % K) + 1 == cᵢ
                if rnd <= γrev / N
                    push!(idx, i)
                    push!(idy, j)
                end
            end
        end
    end
    dg = DiGraph(N)
    for i in eachindex(idx)
        add_edge!(dg, idx[i], idy[i])
    end

    return Dict(
        :clst => clst,
        :dg => dg,
    )
end

function sample_3dcsbm(;
    seed::Int=42, n::Int=1000, c::Real=0.5, ϵ::Real=5, η::Real=1, λ::Real=2.5)
    Random.seed!(seed)

    K = 3
    N = K * n
    θs = rand(Pareto(λ - 1), N)
    γin = 2c / (1 + η * (1 + ϵ))
    γforw = (2c * ϵ * η) / (1 + η * (1 + ϵ))
    γrev = (2c * η) / (1 + η * (1 + ϵ))
    clst = [i % K + 1 for i in 0:(N-1)] |> shuffle
    idx = Int[]
    idy = Int[]
    for i in 1:N
        cᵢ = clst[i]
        θᵢ = θs[i]
        for j in 1:N
            i == j && continue
            cⱼ = clst[j]
            θⱼ = θs[j]
            rnd = rand(Uniform())
            if cᵢ == cⱼ
                if rnd <= min(1, θᵢ * θⱼ * γin / N) / 2
                    push!(idx, i)
                    push!(idy, j)
                elseif rnd <= min(1, θᵢ * θⱼ * γin / N)
                    push!(idx, j)
                    push!(idy, i)
                end
            elseif (cᵢ % K) + 1 == cⱼ
                if rnd <= min(1, θᵢ * θⱼ * γforw / N)
                    push!(idx, i)
                    push!(idy, j)
                end
            elseif (cⱼ % K) + 1 == cᵢ
                if rnd <= min(1, θᵢ * θⱼ * γrev / N)
                    push!(idx, i)
                    push!(idy, j)
                end
            end
        end
    end
    dg = DiGraph(N)
    for i in eachindex(idx)
        add_edge!(dg, idx[i], idy[i])
    end

    return Dict(
        :clst => clst,
        :dg => dg,
    )
end

function _check_positive(x::T) where {T<:Number}
    if x <= 0
        throw(DomainError(x, "The value must be positive: $(x)"))
    end
end

function _check_probability(x::T) where {T<:Number}
    (x < 0 || 1 < x) && throw(DomainError("The value must be included in [0,1]"))
end

function constrained_kmeans(inp::AbstractMatrix, K::Int; n_loop::Int=500, ϵ::Float64=1e-6)
    D, N = size(inp)
    xs = zeros(Float64, D, K)
    σ2s = ones(Float64, K)

    # select initial centers
    X = inp ./ norm.(eachcol(inp))'
    dist = ones(Float64, N) ./ N
    selected = Int[]
    for k in 1:K
        while true
            s = rand(Categorical(dist))
            if !(s in selected)
                push!(selected, s)
                break
            end
        end
        xs[:, k] .= X[:, selected[end]]

        dist .= Inf
        for l in 1:k
            dist .= min.(
                dist,
                norm.(eachcol((X .- xs[:, l]))) .^ 2
            )
        end
        dist ./= sum(dist)
    end

    # Loop
    rs = zeros(Float64, N, K)
    rs_nxt = zeros(Float64, N, K)
    qs = ones(Float64, K) ./ K
    for _ in 1:n_loop
        # update rs
        for i in 1:N
            for k in 1:K
                rs_nxt[i, k] = qs[k] * pdf(MvNormal(xs[:, k], I .* σ2s[k]), inp[:, i])
            end
            rs_nxt[i, :] .= rs_nxt[i, :] ./ sum(rs_nxt[i, :])
        end
        Ns = vec(sum(rs_nxt; dims=1))

        # update qs
        qs .= vec(sum(rs_nxt; dims=1))
        qs ./= sum(qs)

        # update xs
        for k in 1:K
            xs[:, k] .= inp * rs_nxt[:, k]
            xs[:, k] ./= norm(xs[:, k])
        end

        # update σs
        for k in 1:K
            σ2s[k] = sum(rs_nxt[:, k] .* norm.(eachcol(inp .- xs[:, k])) .^ 2) / (Ns[k] * D)
        end

        if norm(rs - rs_nxt) / N < ϵ
            break
        else
            rs .= rs_nxt
        end
    end

    assign = argmax.(eachrow(rs))
    KmeansResult(
        xs,
        assign,
        maximum.(eachrow(rs)),
        Int[count(assign .== k) for k in 1:K],
        zeros(K),
        0.0,
        0,
        true
    )
end

end
