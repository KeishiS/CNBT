# README

This repository contains source codes for "Complex non-backtracking matrix for directed graphs".

Our proposed algorithm is included in `src/CNBT.jl`. You can use it as follows.

```
> julia --project=.
(CNBT) pkg> instantiate
julia> using CNBT, Clustering
julia> data = Util.sample_3dsbm()
julia> dg, _ = induced_subgraph(data[:dg], findall(degree(data[:dg]) .> 0))
julia> clst = data[:clst][findall(degree(data[:dg]) .> 0)]
julia> q, _ = cnbtSC(dg, 3)
julia> randindex(clst, q)[1]
```

To reproduce our plots, please execute test codes as follows:

```
> julia --project=.
> ]test
```

Our program `test/runtests.jl` uses `Distributed` to perform parallel processing.
Therefore, please enable `sshd` on your local PC. Additionally, since executing the test code takes a long time, you can speed up the process by adding `addprocs` into `runtests.jl` to distribute the processing across remote servers.
