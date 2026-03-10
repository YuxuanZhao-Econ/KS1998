function baseline_config()
    par = KSFunctions.KSParams(
        beta = 0.99,
        sigma = 1.0,
        alpha = 0.36,
        delta = 0.025,
        l_tilde = 0.3271,
        z_vals = [0.99, 1.01],
        u_rate = [0.10, 0.04],
        N_vals = [0.2944, 0.3142],
        Pz = [0.875 0.125; 0.125 0.875],
        Pze = [0.600 0.275 0.036 0.089;
               0.031 0.844 0.002 0.123;
               0.107 0.018 0.333 0.542;
               0.009 0.116 0.023 0.852],
        k_min = 0.0,
        k_max = 30.0,
        nk = 300,
        K_min = 1.0,
        K_max = 30.0,
        nK = 60,
        N_agents = 3000,
        T_sim = 6000,
        burn_in = 1000,
    )

    coeff_init = [0.05 0.95; 0.05 0.95]

    return (
        par = par,
        coeff_init = coeff_init,
        max_outer = 15,
        damping = 0.9,
        r2_tol = 0.9999,
        base_seed = 1234,
        vary_seed = false,
        hh_tol = 1e-6,
        hh_max_policy_iter = 2000,
        hh_eval_iter = 25,
    )
end

function fast_config()
    par = KSFunctions.KSParams(
        beta = 0.99,
        sigma = 1.0,
        alpha = 0.36,
        delta = 0.025,
        l_tilde = 0.3271,
        z_vals = [0.99, 1.01],
        u_rate = [0.10, 0.04],
        N_vals = [0.2944, 0.3142],
        Pz = [0.875 0.125; 0.125 0.875],
        Pze = [0.600 0.275 0.036 0.089;
               0.031 0.844 0.002 0.123;
               0.107 0.018 0.333 0.542;
               0.009 0.116 0.023 0.852],
        k_min = 0.0,
        k_max = 30.0,
        nk = 150,
        K_min = 1.0,
        K_max = 30.0,
        nK = 35,
        N_agents = 1000,
        T_sim = 2000,
        burn_in = 300,
    )

    coeff_init = [0.05 0.95; 0.05 0.95]

    return (
        par = par,
        coeff_init = coeff_init,
        max_outer = 10,
        damping = 0.9,
        r2_tol = 0.999,
        base_seed = 1234,
        vary_seed = false,
        hh_tol = 1e-6,
        hh_max_policy_iter = 1000,
        hh_eval_iter = 20,
    )
end

