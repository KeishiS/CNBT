using Test
using Distributed

projpath = joinpath(@__DIR__, "../")
addprocs(
    [("localhost", 1)];
    topology=:master_worker,
    enable_threaded_blas=true,
    exeflags="--project=$(projpath)",
    dir=projpath
)

# include("experiment1.jl")
include("experiment2.jl")
