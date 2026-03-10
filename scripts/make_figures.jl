if !isdefined(Main, :KSFunctions)
    include(joinpath(@__DIR__, "..", "src", "KSFunctions.jl"))
end
using .KSFunctions
using Plots, Statistics, Printf
using Serialization

if !isdefined(Main, :baseline_config)
    include(joinpath(@__DIR__, "config.jl"))
end

function make_figures_pipeline(; result_path = joinpath(@__DIR__, "..", "output", "results", "ks_result.jls"),
    figure_dir = joinpath(@__DIR__, "..", "output", "figures"))

    if !isfile(result_path)
        error("Missing $(result_path). Run scripts/run_ks.jl first.")
    end

    saved = deserialize(result_path)
    cfg = saved.config
    res = saved.solve_result
    par = cfg.par
    kgrid, Kgrid = make_grids(par)
    coeff_star = res.coeff
    pol_star = saved.final_policy

    iK_mid = nearest_index(Kgrid, median(Kgrid))
    k = kgrid

    mkpath(figure_dir)

    kp_BU = policy_line(pol_star, kgrid, iK_mid, 1, 1)
    kp_BE = policy_line(pol_star, kgrid, iK_mid, 2, 1)
    kp_GU = policy_line(pol_star, kgrid, iK_mid, 1, 2)
    kp_GE = policy_line(pol_star, kgrid, iK_mid, 2, 2)

    p1 = plot(k, kp_BU, lw=2.5, label="k:BU", xlabel="k", ylabel="k'")
    plot!(p1, k, kp_BE, lw=2.5, label="k:BE")
    plot!(p1, k, kp_GU, lw=2.5, label="k:GU")
    plot!(p1, k, kp_GE, lw=2.5, label="k:GE")
    plot!(p1, k, k, lw=1.5, ls=:dash, color=:black, label="45 deg")
    title!(p1, "Policy Functions (medium K)")
    savefig(p1, joinpath(figure_dir, "figure_policy_functions.png"))

    Kmid = Kgrid[iK_mid]
    r_b, w_b = prices(par, Kmid, 1)
    r_g, w_g = prices(par, Kmid, 2)

    k_BU_f, mpc_BU = mpc_from_policy_loess(k, kp_BU, r_b, 0.0; span = 0.25)
    k_BE_f, mpc_BE = mpc_from_policy_loess(k, kp_BE, r_b, w_b * par.l_tilde; span = 0.25)
    k_GU_f, mpc_GU = mpc_from_policy_loess(k, kp_GU, r_g, 0.0; span = 0.25)
    k_GE_f, mpc_GE = mpc_from_policy_loess(k, kp_GE, r_g, w_g * par.l_tilde; span = 0.25)

    p2 = plot(k_BU_f, mpc_BU, lw=2.5, label="c:BU", xlabel="k", ylabel="MPC")
    plot!(p2, k_BE_f, mpc_BE, lw=2.5, label="c:BE")
    plot!(p2, k_GU_f, mpc_GU, lw=2.5, label="c:GU")
    plot!(p2, k_GE_f, mpc_GE, lw=2.5, label="c:GE")
    title!(p2, "MPCs (medium K)")
    savefig(p2, joinpath(figure_dir, "figure_mpcs.png"))

    K_series, z_path = simulate_economy(par, kgrid, Kgrid, pol_star, coeff_star; seed = cfg.base_seed)
    T = length(K_series) - 1
    K_forecast = similar(K_series)
    K_forecast[1] = K_series[1]
    for t in 1:T
        iz = z_path[t]
        K_forecast[t + 1] = exp(coeff_star[iz, 1] + coeff_star[iz, 2] * log(K_series[t]))
    end

    t_plot = (par.burn_in + 1):min(par.burn_in + 1000, length(K_series))
    p3 = plot(t_plot .- par.burn_in, K_series[t_plot], lw=2.2, label="Cross-sectional", xlabel="t", ylabel="K")
    plot!(p3, t_plot .- par.burn_in, K_forecast[t_plot], lw=2.2, label="Forecasted")
    title!(p3, "Den Haan Diagnostic")
    savefig(p3, joinpath(figure_dir, "figure_denhaan.png"))

    println("Loaded result from $(result_path)")
    println("Saved figures to $(figure_dir)")

    return (; p1, p2, p3)
end

if abspath(PROGRAM_FILE) == @__FILE__
    make_figures_pipeline()
end
