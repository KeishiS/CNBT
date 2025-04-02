using DataFrames: Sort
using Test, ProgressMeter, Logging, JLD2
using Plots, LaTeXStrings, StatsPlots
using LinearAlgebra, Statistics, Clustering
using DataFrames, DataFramesMeta, CSV, Printf
using CNBT

@everywhere begin
    using CNBT, Graphs
    using Logging

    sampling_func = Dict{Symbol,Function}(
        :dsbm => Util.sample_3dsbm,
        :dcsbm => Util.sample_3dcsbm,
    )

    clst_func = Dict{Symbol,Function}(
        :cnbtSC => (x, y) -> cnbtSC(x, y; type=:Out),
        :cnbtSC_out => (x, y) -> cnbtSC(x, y; type=:Out),
        :cnbtSC_in => (x, y) -> cnbtSC(x, y; type=:In),
        :Herm => Herm.herm,
        :SimpleHerm => SimpleHerm.simpleHerm,
        :DISIM_LR => (x, y) -> DISIM.disim(x, y; type=:Both),
        :DISIM_L => (x, y) -> DISIM.disim(x, y; type=:Left),
        :DISIM_R => (x, y) -> DISIM.disim(x, y; type=:Right),
        :BISYM => (x, y) -> BISYM.bisym(x, y; discounted=false),
        :DDSYM => (x, y) -> BISYM.bisym(x, y; discounted=true),
    )

    function trial_synthetics(jobs, results)
        @info "[START `trial_synthetics`] workerID: $(myid()), Hostname: $(gethostname())"

        K = 3
        while isopen(jobs)
            if !isready(jobs)
                sleep(1)
                continue
            end

            try
                sampling, ϵ, η, trial, seed, method = take!(jobs)
                data = sampling_func[sampling](; seed=seed, ϵ=ϵ, η=η)
                dg, _ = induced_subgraph(data[:dg], findall(degree(data[:dg]) .> 0))
                clst = data[:clst][findall(degree(data[:dg]) .> 0)]

                pred, _ = clst_func[method](dg, K)
                put!(results, (
                    sampling, ϵ, η, trial, seed, method, Util.ari(pred, clst)
                ))
            catch e
                @warn "[SOMETHING WRONG: $(e)] workerID: $(myid()), Hostname: $(gethostname())"
                break
            end
        end

        @info "[END `trial_synthetics`] workerID: $(myid()), Hostname: $(gethostname())"
    end
end

@testset "synthetics" begin
    testname = "synthetics"
    if (length(ARGS) != 0) && !(testname in ARGS)
        @info "Skip testset: $(testname)"
        return
    end

    samplings = [:dsbm, :dcsbm]
    methods = [:DISIM_LR, :BISYM, :DDSYM, :Herm, :SimpleHerm, :cnbtSC_out, :cnbtSC_in]
    stepwidth = 0.10
    n_trials = 30
    ϵs = range(1, 5, step=stepwidth)
    ηs = range(0.5, 3, step=stepwidth)

    outDir = "out"
    outfile = joinpath(outDir, "output_$(testname).csv")
    if !isdir(outDir)
        mkdir(outDir)
    end

    df =
        if isfile(outfile)
            CSV.read(outfile, DataFrame)
        else
            DataFrame(
                sampling=String[],
                ϵ=Float64[],
                η=Float64[],
                trial=Int[],
                seed=Int[],
                method=String[],
                ari=Float64[]
            )
        end

    # sampling, ϵ, η, trial, seed, method
    jobs =
        RemoteChannel(() -> Channel{Tuple{Symbol,Float64,Float64,Int,Int,Symbol}}(
            length(samplings) * length(ϵs) * length(ηs) * n_trials * length(methods)
        ))

    # sampling, ϵ, η, trial, seed, method, ari
    results =
        RemoteChannel(() -> Channel{Tuple{Symbol,Float64,Float64,Int,Int,Symbol,Float64}}(
            length(samplings) * length(ϵs) * length(ηs) * n_trials * length(methods)
        ))

    # Wakeup workers
    for p in workers()
        remote_do(trial_synthetics, p, jobs, results)
    end

    # Insert jobs
    cntJobs = 0
    for (a, ϵ) in enumerate(ϵs)
        for (b, η) in enumerate(ηs)
            for sampling in samplings
                for method in methods
                    for trial in 1:n_trials
                        seed = (a - 1) * n_trials * length(ηs) + (b - 1) * n_trials + trial
                        if any(@with df (:ϵ .== ϵ .&& :η .== η .&& :trial .== trial .&& :method .== String(method)))
                            continue
                        end

                        put!(jobs, (
                            sampling, ϵ, η, trial, seed, method
                        ))
                        cntJobs += 1
                    end
                end
            end
        end
    end

    # Collect results
    pbar = Progress(cntJobs, desc="Experiment 2")
    for cnt in 1:cntJobs
        sampling, ϵ, η, trial, seed, method, val = take!(results)
        push!(df, (
            String(sampling), ϵ, η, trial, seed, String(method), val
        ))
        next!(pbar)
        if cnt % 10 == 0
            CSV.write(outfile, df)
        end
    end
    close(jobs)
    close(results)
    CSV.write(outfile, df)

    # contour over (ϵ, η)
    for sampling in samplings
        for method in methods
            subdf = @subset df (:sampling .== String(sampling) .&& :method .== String(method))
            gb = @groupby subdf [:ϵ, :η]
            summary = @combine gb :meanARI = mean(:ari)
            summary = sort(summary, [:ϵ, :η])

            vals = unstack(summary, :ϵ, :η, :meanARI)[:, 2:end] |> Matrix
            ϵs = summary[:, :ϵ] |> unique |> sort
            ηs = summary[:, :η] |> unique |> sort

            plt = contour(
                ηs, ϵs, vals;
                xlabel=L"\eta", ylabel=L"\epsilon",
                levels=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8],
                clabels=true, color=:grays, fill=true,
                guidefontsize=18, tickfontsize=14, clabelsize=12,
                right_margin=Plots.Measures.Length(:mm, 5),
                colorbar=false
            )
            savefig(joinpath(outDir, "$(String(sampling))_$(String(method)).pdf"))
        end
    end

    # Grouped Boxplot with ϵ=4
    for sampling in samplings
        subdf = @chain df begin
            @subset :ϵ .== 4.0 .&& :sampling .== String(sampling)
            @subset in.(:η, Ref([1.0, 1.5, 2.0, 2.5, 3.0]))
            @select :η :method :ari
        end
        plt = @df subdf groupedboxplot(
            :η, :ari; group=:method,
            xlabel=L"\eta", ylabel="ARI",
            left_margin=Plots.Measures.Length(:mm, 10),
            bottom_margin=Plots.Measures.Length(:mm, 5),
            guidefontsize=18, tickfontsize=14,
            legendfontsize=16, legend_columns=1, legend=:outerright,
            outliers=false, size=(1200, 500)
        )
        savefig(joinpath(outDir, "eps_$(String(sampling)).pdf"))
    end
end
