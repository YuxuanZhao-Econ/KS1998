module KSFunctions

using Random, Statistics, LinearAlgebra, Printf
using Base.Threads

export KSParams, state_index, util, make_grids, prices, nearest_index, linear_interp_weights,
       cubic_spline_interp_K, continuation_value, bellman_rhs, continuous_argmax,
       solve_household_vfi, solve_household_pfi, draw_next_z, p_emp_next,
       simulate_economy, fit_law_of_motion, solve_ks,
       policy_line, tricube, interp_1d, mpc_from_policy_loess,
       simulate_economy_with_unemp, mc_r2_diagnostics


# ====================================
# 1) Initialization and parameters
# ====================================
Base.@kwdef mutable struct KSParams
    beta::Float64
    sigma::Float64
    alpha::Float64
    delta::Float64
    l_tilde::Float64
    z_vals::Vector{Float64}
    u_rate::Vector{Float64}
    N_vals::Vector{Float64}
    Pz::Matrix{Float64}
    Pze::Matrix{Float64}
    k_min::Float64
    k_max::Float64
    nk::Int
    K_min::Float64
    K_max::Float64
    nK::Int
    N_agents::Int
    T_sim::Int
    burn_in::Int
end

@inline state_index(ie::Int, iz::Int) = (iz - 1) * 2 + ie
util(c, sigma) = c <= 0 ? -1.0e12 : (sigma == 1.0 ? log(c) : (c^(1.0 - sigma) - 1.0) / (1.0 - sigma))

# ====================================
# 2) Basic Helper Functions: Grids, interpolation, Bellman RHS, and optimization.
# ===================================

function make_grids(par::KSParams)
    kgrid = collect(range(par.k_min, par.k_max, length = par.nk))
    Kgrid = collect(range(par.K_min, par.K_max, length = par.nK))
    return kgrid, Kgrid
end

function prices(par::KSParams, K::Float64, iz::Int)
    z = par.z_vals[iz]
    N = par.N_vals[iz]
    Kn = K / max(N, 1.0e-8)
    r = par.alpha * z * Kn^(par.alpha - 1.0) - par.delta
    w = (1.0 - par.alpha) * z * Kn^(par.alpha)
    return r, w
end

function nearest_index(grid::Vector{Float64}, x::Float64)
    j = searchsortedfirst(grid, x)
    if j <= 1
        return 1
    elseif j > length(grid)
        return length(grid)
    else
        return abs(grid[j] - x) < abs(grid[j - 1] - x) ? j : j - 1
    end
end

function linear_interp_weights(grid::Vector{Float64}, x::Float64)
    j = searchsortedfirst(grid, x)
    if j <= 1
        return 1, 1, 0.0
    elseif j > length(grid)
        n = length(grid)
        return n, n, 0.0
    else
        lo = j - 1
        hi = j
        w = (x - grid[lo]) / (grid[hi] - grid[lo])
        return lo, hi, clamp(w, 0.0, 1.0)
    end
end

@inline function cubic_spline_interp_K(V, ikp::Int, iep::Int, izp::Int, iKlo::Int, iKhi::Int, wK::Float64)
    nK = size(V, 2)
    if iKlo == iKhi || iKlo <= 1 || iKhi >= nK
        return (1.0 - wK) * V[ikp, iKlo, iep, izp] + wK * V[ikp, iKhi, iep, izp]
    end

    i0 = iKlo - 1
    i1 = iKlo
    i2 = iKhi
    i3 = iKhi + 1

    y0 = V[ikp, i0, iep, izp]
    y1 = V[ikp, i1, iep, izp]
    y2 = V[ikp, i2, iep, izp]
    y3 = V[ikp, i3, iep, izp]

    t = wK
    a0 = -0.5 * y0 + 1.5 * y1 - 1.5 * y2 + 0.5 * y3
    a1 = y0 - 2.5 * y1 + 2.0 * y2 - 0.5 * y3
    a2 = -0.5 * y0 + 0.5 * y2
    a3 = y1

    v = ((a0 * t + a1) * t + a2) * t + a3
    lo = min(y1, y2)
    hi = max(y1, y2)
    return clamp(v, lo, hi)
end

@inline function continuation_value(V, kgrid, kp::Float64, iep::Int, izp::Int, iKlo::Int, iKhi::Int, wK::Float64)
    iklo, ikhi, wk = linear_interp_weights(kgrid, kp)
    vlo = cubic_spline_interp_K(V, iklo, iep, izp, iKlo, iKhi, wK)
    vhi = cubic_spline_interp_K(V, ikhi, iep, izp, iKlo, iKhi, wK)
    return (1.0 - wk) * vlo + wk * vhi
end

@inline function bellman_rhs(par::KSParams, V, kgrid, cash::Float64, kp::Float64, cur_state::Int, iKlo::Int, iKhi::Int, wK::Float64)
    c = cash - kp
    u = util(c, par.sigma)
    if u <= -1.0e11
        return -1.0e18
    end

    ev = 0.0
    for izp in 1:2
        for iep in 1:2
            nxt_state = state_index(iep, izp)
            p = par.Pze[cur_state, nxt_state]
            ev += p * continuation_value(V, kgrid, kp, iep, izp, iKlo, iKhi, wK)
        end
    end

    return u + par.beta * ev
end

function continuous_argmax(par::KSParams, V, kgrid, cash::Float64, cur_state::Int, iKlo::Int, iKhi::Int, wK::Float64;
    n_coarse::Int = 25, tol::Float64 = 1e-4, max_iter::Int = 80)

    lower = par.k_min
    upper = min(par.k_max, cash - 1.0e-10)
    if upper <= lower + 1.0e-10
        kp = lower
        return kp, bellman_rhs(par, V, kgrid, cash, kp, cur_state, iKlo, iKhi, wK)
    end

    xs = collect(range(lower, upper, length = max(n_coarse, 3)))
    vals = similar(xs)
    for j in eachindex(xs)
        vals[j] = bellman_rhs(par, V, kgrid, cash, xs[j], cur_state, iKlo, iKhi, wK)
    end

    best_j = argmax(vals)
    best_x = xs[best_j]
    best_v = vals[best_j]

    left = best_j == 1 ? xs[1] : xs[best_j - 1]
    right = best_j == length(xs) ? xs[end] : xs[best_j + 1]
    if right <= left + tol
        return best_x, best_v
    end

    phi = (sqrt(5.0) - 1.0) / 2.0
    c = right - phi * (right - left)
    d = left + phi * (right - left)
    fc = bellman_rhs(par, V, kgrid, cash, c, cur_state, iKlo, iKhi, wK)
    fd = bellman_rhs(par, V, kgrid, cash, d, cur_state, iKlo, iKhi, wK)

    for _ in 1:max_iter
        if (right - left) <= tol * max(1.0, abs(c) + abs(d))
            break
        end
        if fc > fd
            right = d
            d = c
            fd = fc
            c = right - phi * (right - left)
            fc = bellman_rhs(par, V, kgrid, cash, c, cur_state, iKlo, iKhi, wK)
        else
            left = c
            c = d
            fc = fd
            d = left + phi * (right - left)
            fd = bellman_rhs(par, V, kgrid, cash, d, cur_state, iKlo, iKhi, wK)
        end
    end

    if fc > fd
        cand_x, cand_v = c, fc
    else
        cand_x, cand_v = d, fd
    end

    return cand_v > best_v ? (cand_x, cand_v) : (best_x, best_v)
end

# ====================================
# 3) Main Solver Functions: VFI and PFI
# ===================================

function solve_household_vfi(par::KSParams, kgrid, Kgrid, coeff; max_iter = 2000, tol = 1e-6, opt_tol = 1e-4)
    nk, nK = length(kgrid), length(Kgrid)
    V = zeros(nk, nK, 2, 2)
    Vnew = similar(V)
    pol = zeros(nk, nK, 2, 2)

    for it in 1:max_iter
        diff = 0.0
        for iz in 1:2
            for iK in 1:nK
                K = Kgrid[iK]
                r, w = prices(par, K, iz)
                Knext = exp(coeff[iz, 1] + coeff[iz, 2] * log(K))
                iKlo, iKhi, wK = linear_interp_weights(Kgrid, Knext)

                for ie in 1:2
                    labor_income = ie == 2 ? w * par.l_tilde : 0.0
                    cur_state = state_index(ie, iz)

                    for ik in 1:nk
                        cash = (1.0 + r) * kgrid[ik] + labor_income
                        best_kp, best_v = continuous_argmax(par, V, kgrid, cash, cur_state, iKlo, iKhi, wK; tol = opt_tol)
                        Vnew[ik, iK, ie, iz] = best_v
                        pol[ik, iK, ie, iz] = best_kp
                        diff = max(diff, abs(best_v - V[ik, iK, ie, iz]))
                    end
                end
            end
        end

        V, Vnew = Vnew, V
        if it % 50 == 0
            @printf("  VFI iter %d, diff = %.3e\n", it, diff)
        end
        if diff < tol
            @printf("  VFI converged at iter %d, diff = %.3e\n", it, diff)
            return V, pol
        end
    end

    @warn "VFI reached max_iter without full convergence"
    return V, pol
end

function solve_household_pfi(par::KSParams, kgrid, Kgrid, coeff;
    max_policy_iter = 200, eval_iter = 25, tol = 1e-6, final_eval_iter = 200,
    opt_tol = 1e-4, policy_tol = 1e-4, improve_tol = 1e-8)

    nk, nK = length(kgrid), length(Kgrid)
    V = zeros(nk, nK, 2, 2)
    Vnew = similar(V)

    pol = Array{Float64}(undef, nk, nK, 2, 2)
    pol_new = similar(pol)

    for iz in 1:2, iK in 1:nK, ie in 1:2, ik in 1:nk
        pol[ik, iK, ie, iz] = kgrid[ik]
    end

    for pit in 1:max_policy_iter
        V_prev_pit = copy(V)
        diff_eval = Inf

        for _ in 1:eval_iter
            diff_eval = 0.0
            for iz in 1:2
                for iK in 1:nK
                    K = Kgrid[iK]
                    r, w = prices(par, K, iz)
                    Knext = exp(coeff[iz, 1] + coeff[iz, 2] * log(K))
                    iKlo, iKhi, wK = linear_interp_weights(Kgrid, Knext)

                    for ie in 1:2
                        labor_income = ie == 2 ? w * par.l_tilde : 0.0
                        cur_state = state_index(ie, iz)

                        for ik in 1:nk
                            kp = pol[ik, iK, ie, iz]
                            Vnew[ik, iK, ie, iz] = bellman_rhs(
                                par, V, kgrid,
                                (1.0 + r) * kgrid[ik] + labor_income,
                                kp, cur_state, iKlo, iKhi, wK
                            )
                            diff_eval = max(diff_eval, abs(Vnew[ik, iK, ie, iz] - V[ik, iK, ie, iz]))
                        end
                    end
                end
            end

            V, Vnew = Vnew, V
            if diff_eval < tol
                break
            end
        end

        n_change = 0
        for iz in 1:2
            for iK in 1:nK
                K = Kgrid[iK]
                r, w = prices(par, K, iz)
                Knext = exp(coeff[iz, 1] + coeff[iz, 2] * log(K))
                iKlo, iKhi, wK = linear_interp_weights(Kgrid, Knext)

                for ie in 1:2
                    labor_income = ie == 2 ? w * par.l_tilde : 0.0
                    cur_state = state_index(ie, iz)

                    for ik in 1:nk
                        cash = (1.0 + r) * kgrid[ik] + labor_income
                        old_kp = pol[ik, iK, ie, iz]
                        old_v = bellman_rhs(par, V, kgrid, cash, old_kp, cur_state, iKlo, iKhi, wK)

                        new_kp, new_v = continuous_argmax(
                            par, V, kgrid, cash, cur_state, iKlo, iKhi, wK;
                            tol = opt_tol
                        )

                        if (new_v - old_v > improve_tol) && (abs(new_kp - old_kp) > policy_tol)
                            pol_new[ik, iK, ie, iz] = new_kp
                            n_change += 1
                        else
                            pol_new[ik, iK, ie, iz] = old_kp
                        end
                    end
                end
            end
        end

        pol, pol_new = pol_new, pol
        diff_V = maximum(abs.(V .- V_prev_pit))

        if pit % 10 == 0
            @printf("  PFI iter %d, eval diff = %.3e, dV = %.3e, n_change = %d\n",
                    pit, diff_eval, diff_V, n_change)
        end

        if n_change == 0
            for fe in 1:final_eval_iter
                diff_eval = 0.0
                for iz in 1:2
                    for iK in 1:nK
                        K = Kgrid[iK]
                        r, w = prices(par, K, iz)
                        Knext = exp(coeff[iz, 1] + coeff[iz, 2] * log(K))
                        iKlo, iKhi, wK = linear_interp_weights(Kgrid, Knext)

                        for ie in 1:2
                            labor_income = ie == 2 ? w * par.l_tilde : 0.0
                            cur_state = state_index(ie, iz)

                            for ik in 1:nk
                                kp = pol[ik, iK, ie, iz]
                                Vnew[ik, iK, ie, iz] = bellman_rhs(
                                    par, V, kgrid,
                                    (1.0 + r) * kgrid[ik] + labor_income,
                                    kp, cur_state, iKlo, iKhi, wK
                                )
                                diff_eval = max(diff_eval, abs(Vnew[ik, iK, ie, iz] - V[ik, iK, ie, iz]))
                            end
                        end
                    end
                end

                V, Vnew = Vnew, V

                if diff_eval < tol
                    @printf("  PFI converged at iter %d, eval diff = %.3e, dV = %.3e\n",
                            pit, diff_eval, diff_V)
                    return V, pol
                end
            end

            @printf("  PFI policy converged at iter %d (n_change = 0). Final eval diff = %.3e\n",
                    pit, diff_eval)
            return V, pol
        end
    end

    @warn "PFI reached max_policy_iter without full policy convergence"
    return V, pol
end


# ====================================
# 4) Simulation and regression for law of motion
# ===================================

function draw_next_z(iz::Int, Pz::Matrix{Float64}, rng)
    udraw = rand(rng)
    return udraw <= Pz[iz, 1] ? 1 : 2
end

function p_emp_next(par::KSParams, ie::Int, iz::Int, izp::Int)
    cur_state = state_index(ie, iz)
    pz = par.Pz[iz, izp]
    if pz <= 1.0e-12
        return ie == 2 ? 1.0 : 0.0
    end
    emp_state = state_index(2, izp)
    p_emp = par.Pze[cur_state, emp_state] / pz
    return clamp(p_emp, 0.0, 1.0)
end

function simulate_economy(par::KSParams, kgrid, Kgrid, pol, coeff; seed = 1234)
    rng = MersenneTwister(seed)
    T, N = par.T_sim, par.N_agents

    k_val = fill(kgrid[cld(length(kgrid), 2)], N)
    e_state = [rand(rng) < (1.0 - par.u_rate[2]) ? 2 : 1 for _ in 1:N]
    z_path = ones(Int, T + 1)
    z_path[1] = 2

    K_series = zeros(T + 1)
    K_series[1] = mean(k_val)

    for t in 1:T
        iz = z_path[t]
        Kt = mean(k_val)
        K_series[t] = Kt

        iKlo, iKhi, wK = linear_interp_weights(Kgrid, Kt)

        izp = draw_next_z(iz, par.Pz, rng)
        z_path[t + 1] = izp

        k_next_val = similar(k_val)
        e_next_state = similar(e_state)
        u_e = rand(rng, N)

        Base.Threads.@threads :static for i in 1:N
            ie = e_state[i]

            iklo, ikhi, wk = linear_interp_weights(kgrid, k_val[i])

            kp_11 = pol[iklo, iKlo, ie, iz]
            kp_12 = pol[iklo, iKhi, ie, iz]
            kp_21 = pol[ikhi, iKlo, ie, iz]
            kp_22 = pol[ikhi, iKhi, ie, iz]

            kp_lo = (1.0 - wK) * kp_11 + wK * kp_12
            kp_hi = (1.0 - wK) * kp_21 + wK * kp_22
            kp_star = (1.0 - wk) * kp_lo + wk * kp_hi

            k_next_val[i] = clamp(kp_star, par.k_min, par.k_max)

            p_emp = p_emp_next(par, ie, iz, izp)
            e_next_state[i] = u_e[i] < p_emp ? 2 : 1
        end

        k_val = k_next_val
        e_state = e_next_state
        K_series[t + 1] = mean(k_val)
    end

    return K_series, z_path
end

function fit_law_of_motion(K_series, z_path; burn_in = 1000)
    coeff = zeros(2, 2)
    r2 = zeros(2)

    for z in 1:2
        idx = [t for t in burn_in:(length(K_series) - 1) if z_path[t] == z]
        x = log.(K_series[idx])
        y = log.(K_series[idx .+ 1])

        X = hcat(ones(length(x)), x)
        b = X \ y
        coeff[z, :] .= b

        yhat = X * b
        ssr = sum((y .- yhat) .^ 2)
        sst = sum((y .- mean(y)) .^ 2)
        r2[z] = 1.0 - ssr / max(sst, 1.0e-12)
    end

    return coeff, r2
end

# ====================================
# 5) Main outer loop to solve for equilibrium coefficients
# ====================================

function solve_ks(par::KSParams;
    coeff_init = [0.05 0.95; 0.05 0.95],
    max_outer = 30, damping = 0.9, r2_tol = 0.9999,
    base_seed = 1234, vary_seed = true)

    kgrid, Kgrid = make_grids(par)
    coeff = copy(coeff_init)

    V = zeros(length(kgrid), length(Kgrid), 2, 2)
    pol = zeros(length(kgrid), length(Kgrid), 2, 2)
    K_series = zeros(par.T_sim + 1)
    z_path = ones(Int, par.T_sim + 1)
    r2 = zeros(2)

    for out in 1:max_outer
        @printf("\n=== Outer iteration %d ===\n", out)
        @printf("  current coeff used in HH/simulation:\n")
        @printf("    bad state : a = %.4f, b = %.4f\n", coeff[1,1], coeff[1,2])
        @printf("    good state: a = %.4f, b = %.4f\n", coeff[2,1], coeff[2,2])

        V, pol = solve_household_pfi(
            par, kgrid, Kgrid, coeff;
            tol = 1e-6, max_policy_iter = 2000, eval_iter = 25
        )

        seed_used = vary_seed ? (base_seed + out) : base_seed
        K_series, z_path = simulate_economy(par, kgrid, Kgrid, pol, coeff; seed = seed_used)
        new_coeff, r2 = fit_law_of_motion(K_series, z_path; burn_in = par.burn_in)

        gap = maximum(abs.(new_coeff .- coeff))

        @printf("  regression-implied new coeff:\n")
        @printf("    bad state : a = %.4f, b = %.4f, R2 = %.6f\n", new_coeff[1,1], new_coeff[1,2], r2[1])
        @printf("    good state: a = %.4f, b = %.4f, R2 = %.6f\n", new_coeff[2,1], new_coeff[2,2], r2[2])
        @printf("  max coefficient gap = %.4e\n", gap)

        coeff_next = damping .* coeff .+ (1.0 - damping) .* new_coeff

        @printf("  updated coeff after damping:\n")
        @printf("    bad state : a = %.4f, b = %.4f\n", coeff_next[1,1], coeff_next[1,2])
        @printf("    good state: a = %.4f, b = %.4f\n", coeff_next[2,1], coeff_next[2,2])

        coeff .= coeff_next

        if (r2[1] > r2_tol) && (r2[2] > r2_tol)
            @printf("Converged: outer loop stopped at iter %d (R2 criterion met).\n", out)
            return (; coeff, r2, V, pol, kgrid, Kgrid, K_series, z_path)
        end
    end

    @warn "Outer loop reached max_outer without meeting R2 threshold."
    return (; coeff, r2, V, pol, kgrid, Kgrid, K_series, z_path)
end

# ====================================
# 6) Visualizations: extracting policy lines, MPC estimation, etc.
# ====================================

policy_line(pol, kgrid, iK, ie, iz) = vec(pol[:, iK, ie, iz])

function tricube(u)
    a = abs(u)
    return a < 1.0 ? (1.0 - a^3)^3 : 0.0
end

function interp_1d(xgrid, ygrid, x)
    ilo, ihi, w = linear_interp_weights(xgrid, x)
    return (1.0 - w) * ygrid[ilo] + w * ygrid[ihi]
end

function mpc_from_policy_loess(kgrid, kp, r, labor_income; nfine = 1200, span = 0.25)
    kfine = collect(range(first(kgrid), last(kgrid), length = nfine))
    kp_fine = [interp_1d(kgrid, kp, x) for x in kfine]

    m = (1.0 .+ r) .* kfine .+ labor_income
    c = m .- kp_fine

    n = length(m)
    q = max(10, ceil(Int, span * n))
    mpc = similar(m)

    for i in 1:n
        x0 = m[i]
        d = abs.(m .- x0)
        h = partialsort!(copy(d), q)
        h = max(h, 1.0e-8)

        w = [tricube(di / h) for di in d]
        x1 = m .- x0
        x2 = x1 .^ 2

        s00 = sum(w)
        s01 = sum(w .* x1)
        s02 = sum(w .* x2)
        s11 = sum(w .* x1 .* x1)
        s12 = sum(w .* x1 .* x2)
        s22 = sum(w .* x2 .* x2)

        t0 = sum(w .* c)
        t1 = sum(w .* x1 .* c)
        t2 = sum(w .* x2 .* c)

        XWX = [s00 s01 s02;
               s01 s11 s12;
               s02 s12 s22]
        XWy = [t0, t1, t2]

        beta = XWX \ XWy
        mpc[i] = beta[2]
    end

    return kfine, mpc
end

function simulate_economy_with_unemp(par::KSParams, kgrid, Kgrid, pol, coeff; seed = 1234)
    rng = MersenneTwister(seed)
    T, N = par.T_sim, par.N_agents

    k_val = fill(kgrid[cld(length(kgrid), 2)], N)
    e_state = [rand(rng) < (1.0 - par.u_rate[2]) ? 2 : 1 for _ in 1:N]
    z_path = ones(Int, T + 1)
    z_path[1] = 2

    K_series = zeros(T + 1)
    u_series = zeros(T + 1)
    u_target = zeros(T + 1)

    K_series[1] = mean(k_val)
    u_series[1] = mean(e_state .== 1)
    u_target[1] = par.u_rate[z_path[1]]

    for t in 1:T
        iz = z_path[t]
        Kt = mean(k_val)
        K_series[t] = Kt
        u_series[t] = mean(e_state .== 1)
        u_target[t] = par.u_rate[iz]

        iKlo, iKhi, wK = linear_interp_weights(Kgrid, Kt)
        izp = draw_next_z(iz, par.Pz, rng)
        z_path[t + 1] = izp

        k_next_val = similar(k_val)
        e_next_state = similar(e_state)
        u_e = rand(rng, N)

        for i in 1:N
            ie = e_state[i]
            iklo, ikhi, wk = linear_interp_weights(kgrid, k_val[i])

            kp_11 = pol[iklo, iKlo, ie, iz]
            kp_12 = pol[iklo, iKhi, ie, iz]
            kp_21 = pol[ikhi, iKlo, ie, iz]
            kp_22 = pol[ikhi, iKhi, ie, iz]

            kp_lo = (1.0 - wK) * kp_11 + wK * kp_12
            kp_hi = (1.0 - wK) * kp_21 + wK * kp_22
            kp_star = (1.0 - wk) * kp_lo + wk * kp_hi
            k_next_val[i] = clamp(kp_star, par.k_min, par.k_max)

            p_emp = p_emp_next(par, ie, iz, izp)
            e_next_state[i] = u_e[i] < p_emp ? 2 : 1
        end

        k_val = k_next_val
        e_state = e_next_state

        K_series[t + 1] = mean(k_val)
        u_series[t + 1] = mean(e_state .== 1)
        u_target[t + 1] = par.u_rate[z_path[t + 1]]
    end

    return (; K_series, z_path, u_series, u_target)
end

function mc_r2_diagnostics(par, kgrid, Kgrid, pol, coeff; nrep = 30, base_seed = 1000)
    r2_bad = zeros(nrep)
    r2_good = zeros(nrep)
    a_bad = zeros(nrep)
    b_bad = zeros(nrep)
    a_good = zeros(nrep)
    b_good = zeros(nrep)

    for rep in 1:nrep
        seed = base_seed + rep
        K_series, z_path = simulate_economy(par, kgrid, Kgrid, pol, coeff; seed = seed)
        coeff_rep, r2_rep = fit_law_of_motion(K_series, z_path; burn_in = par.burn_in)

        a_bad[rep] = coeff_rep[1,1]
        b_bad[rep] = coeff_rep[1,2]
        a_good[rep] = coeff_rep[2,1]
        b_good[rep] = coeff_rep[2,2]
        r2_bad[rep] = r2_rep[1]
        r2_good[rep] = r2_rep[2]
    end

    return (; r2_bad, r2_good, a_bad, b_bad, a_good, b_good)
end

end


