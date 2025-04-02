using Test, ProgressMeter, Logging
using Plots, LaTeXStrings
using LinearAlgebra, Statistics
using DataFrames, DataFramesMeta, CSV, Printf, Clustering
using CNBT

@everywhere begin
    using CNBT
    function trial_checkAlg(jobs, results)
        @info "[START `trial_checkAlg`] workerID: $(myid()), Hostname: $(gethostname())"

        n = 1000
        K = 5
        indices = [(1, 2), (2, 3), (3, 4), (4, 5), (5, 1)]
        while isopen(jobs)
            if !isready(jobs)
                sleep(1)
                continue
            end

            try
                p, η, trial, seed, method = take!(jobs)
                F = ones(K, K) ./ 2.0
                for (idx, idy) in indices
                    F[idx, idy] = 1 - η
                    F[idy, idx] = η
                end
                data = Herm.sampleDSBM(K, n, p, p, F; seed=seed)

                if method == :cnbtSC || method == :cnbtSC_out
                    pred, _ = cnbtSC(data[:dg], K; type=:Out)
                elseif method == :cnbtSC_in
                    pred, _ = cnbtSC(data[:dg], K; type=:In)
                elseif method == :Herm
                    pred, _ = Herm.herm(data[:dg], K)
                elseif method == :SimpleHerm
                    pred, _ = SimpleHerm.simpleHerm(data[:dg], K)
                elseif method == :DISIM_LR
                    pred, _ = DISIM.disim(data[:dg], K; type=:Both)
                elseif method == :DISIM_L
                    pred, _ = DISIM.disim(data[:dg], K; type=:Left)
                elseif method == :DISIM_R
                    pred, _ = DISIM.disim(data[:dg], K; type=:Right)
                elseif method == :BISYM
                    pred, _ = BISYM.bisym(data[:dg], K; discounted=false)
                elseif method == :DDSYM
                    pred, _ = BISYM.bisym(data[:dg], K; discounted=true)
                end

                put!(results, (
                    p, η, trial, seed, method, Util.ari(pred, data[:clst])
                ))

            catch e
                @warn "[SOMETHING WRONG: $(e)] workerID: $(myid()), Hostname: $(gethostname())"
                break
            end
        end

        @info "[END `trial_checkAlg`] workerID: $(myid()), Hostname: $(gethostname())"
    end
end



@testset "experiment1" begin
    if (length(ARGS) != 0) && !("experiment1" in ARGS)
        @info "Skip testset: Experiment 1"
        return
    end
    outDir = "out"
    outfile = joinpath(outDir, "output_check-alg.csv")
    summaryfile = joinpath(outDir, "summary_check-alg.csv")
    if !isdir(outDir)
        mkdir(outDir)
    end

    methods = [:DISIM_LR, :BISYM, :DDSYM, :Herm, :SimpleHerm, :cnbtSC_out, :cnbtSC_in]
    n_trials = 10
    ps = [0.45, 0.5, 0.6, 0.8] / 100.0
    ηs = range(0, 0.3, length=11)
    n = 1000
    K = 5

    df =
        if isfile(outfile)
            CSV.read(outfile, DataFrame)
        else
            DataFrame(
                p=Float64[],
                η=Float64[],
                trial=Int[],
                seed=Int[],
                method=String[],
                ari=Float64[]
            )
        end

    # p, η, trial, seed, method
    jobs = RemoteChannel(() -> Channel{Tuple{Float64,Float64,Int,Int,Symbol}}(
        length(ps) * length(ηs) * n_trials * length(methods)
    ))
    # p, η, trial, seed, method, ari
    results = RemoteChannel(() -> Channel{Tuple{Float64,Float64,Int,Int,Symbol,Float64}}(
        length(ps) * length(ηs) * n_trials * length(methods)
    ))

    # Wakeup workers
    for p in workers()
        remote_do(trial_checkAlg, p, jobs, results)
    end

    # Insert jobs
    cntJobs = 0
    for (a, p) in enumerate(ps)
        for (b, η) in enumerate(ηs)
            for trial in 1:n_trials
                seed = (trial - 1) + (b - 1) * n_trials + (a - 1) * n_trials * length(ηs)

                for method in methods
                    if any(@with df (:p .== p .&& :η .== η .&& :trial .== trial .&& :method .== String(method)))
                        continue
                    end

                    put!(jobs, (
                        p, η, trial, seed, method
                    ))
                    cntJobs += 1
                end
            end
        end
    end

    @info "total jobs: $(cntJobs)"
    # Collect results
    pbar = Progress(cntJobs, desc="Experiment 1")
    for _ in 1:cntJobs
        p, η, trial, seed, method, val = take!(results)
        push!(df, (
            p, η, trial, seed, String(method), val
        ))
        CSV.write(outfile, df)
        next!(pbar)
    end
    close(jobs)
    close(results)

    gb = @groupby(df, [:p, :η, :method])
    summary = @combine(gb, :meanARI = mean(:ari), :stdARI = std(:ari))
    CSV.write(summaryfile, summary)

    linesty = [:solid, :dash, :dot, :dashdot, :dashdotdot]
    marksty = [:circle, :rect, :diamond]
    for subdf in @groupby(summary, :p)
        plt = plot(
            ylims=(-0.01, 1.0),
            xticks=range(0, 0.3, step=0.05),
            yticks=range(0, 1, step=0.1),
            xlabel=L"\eta", ylabel=isapprox(subdf[1, :p], 0.0045; atol=1e-6) ? "ARI" : "",
            guidefontsize=18, tickfontsize=14,
            legendfontsize=16, legend_columns=2,
            legend=isapprox(subdf[1, :p], 0.0045; atol=1e-6)
        )
        for (id, result) in enumerate(@groupby(subdf, :method))
            plot!(
                result[!, :η], result[!, :meanARI], yerror=result[!, :stdARI],
                marker=marksty[cld(id, length(linesty))],
                linestyle=linesty[(id-1)%length(linesty)+1],
                label=String(result[1, :method])
            )
        end
        savefig(plt,
            joinpath(
                outDir,
                @sprintf("dsbm_%.4f.pdf", subdf[1, :p])
            )
        )
    end

end
