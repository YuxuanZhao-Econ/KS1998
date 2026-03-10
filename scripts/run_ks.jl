if !isdefined(Main, :KSFunctions)
    include(joinpath(@__DIR__, "..", "src", "KSFunctions.jl"))
end
using .KSFunctions
using Printf
using Dates
using Serialization

if !isdefined(Main, :baseline_config)
    include(joinpath(@__DIR__, "config.jl"))
end

function run_ks_pipeline(; cfg = baseline_config(),
    result_path = joinpath(@__DIR__, "..", "output", "results", "ks_result.jls"))

    res = solve_ks(cfg.par;
        coeff_init = cfg.coeff_init,
        max_outer = cfg.max_outer,
        damping = cfg.damping,
        r2_tol = cfg.r2_tol,
        base_seed = cfg.base_seed,
        vary_seed = cfg.vary_seed)

    kgrid, Kgrid = make_grids(cfg.par)
    V_star, pol_star = solve_household_pfi(cfg.par, kgrid, Kgrid, res.coeff;
        max_policy_iter = cfg.hh_max_policy_iter,
        eval_iter = cfg.hh_eval_iter,
        tol = cfg.hh_tol)

    payload = (
        config = cfg,
        solve_result = res,
        final_value = V_star,
        final_policy = pol_star,
        saved_at = Dates.now(),
    )

    mkpath(dirname(result_path))
    serialize(result_path, payload)

    println("\nFinal perceived law of motion (rows: bad, good):")
    println("[a_z  b_z] =")
    display(res.coeff)
    println("R2 by state = ", res.r2)
    println()
    println("Bad-state law:  log(K') = $(round(res.coeff[1,1], digits=4)) + $(round(res.coeff[1,2], digits=4))*log(K)")
    println("Good-state law: log(K') = $(round(res.coeff[2,1], digits=4)) + $(round(res.coeff[2,2], digits=4))*log(K)")
    println("Saved binary result to $(result_path)")

    mkpath(joinpath(@__DIR__, "..", "output", "logs"))
    open(joinpath(@__DIR__, "..", "output", "logs", "solve_ks_summary.txt"), "w") do io
        println(io, "Run timestamp: ", Dates.now())
        println(io, "Final perceived law of motion (rows: bad, good):")
        println(io, res.coeff)
        println(io, "R2 by state = ", res.r2)
        println(io, "Saved result file = ", result_path)
    end

    return payload
end

if abspath(PROGRAM_FILE) == @__FILE__
    run_ks_pipeline()
end
