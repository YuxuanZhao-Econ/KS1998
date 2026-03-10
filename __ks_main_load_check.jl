include("scripts/run_ks.jl")
include("scripts/make_figures.jl")
println(isdefined(Main, :run_ks_pipeline))
println(isdefined(Main, :make_figures_pipeline))
