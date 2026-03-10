include(joinpath(@__DIR__, "scripts", "run_ks.jl"))
include(joinpath(@__DIR__, "scripts", "make_figures.jl"))

payload = run_ks_pipeline()
make_figures_pipeline()
