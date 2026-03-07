function solve_household_pfi(par::KSParams, kgrid, Kgrid, coeff; max_policy_iter = 200, eval_iter = 25, tol = 1e-5)
    nk, nK = length(kgrid), length(Kgrid)
    V = zeros(nk, nK, 2, 2)
    Vnew = similar(V)

    pol = Array{Int}(undef, nk, nK, 2, 2)
    pol_new = similar(pol)

    for iz in 1:2, iK in 1:nK, ie in 1:2, ik in 1:nk
        pol[ik, iK, ie, iz] = ik
    end

    for pit in 1:max_policy_iter
        diff_eval = Inf

        # Policy evaluation (for fixed policy)
        for pe in 1:eval_iter
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
                            ikp = pol[ik, iK, ie, iz]
                            kp = kgrid[ikp]
                            c = (1.0 + r) * kgrid[ik] + labor_income - kp
                            u = util(c, par.sigma)

                            if u <= -1.0e11
                                Vnew[ik, iK, ie, iz] = -1.0e18
                            else
                                ev = 0.0
                                for izp in 1:2
                                    for iep in 1:2
                                        nxt_state = state_index(iep, izp)
                                        p = par.Pze[cur_state, nxt_state)
                                        Vcont = (1.0 - wK) * V[ikp, iKlo, iep, izp] + wK * V[ikp, iKhi, iep, izp]
                                        ev += p * Vcont
                                    end
                                end
                                Vnew[ik, iK, ie, iz] = u + par.beta * ev
                            end

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

        if pit % 50 == 0
            @printf("  PFI iter %d, eval diff = %.3e\n", pit, diff_eval)
        end

        # Policy improvement
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
                        best_v = -1.0e18
                        best_ikp = pol[ik, iK, ie, iz]

                        for ikp in 1:nk
                            kp = kgrid[ikp]
                            c = cash - kp
                            u = util(c, par.sigma)
                            if u <= -1.0e11
                                continue
                            end

                            ev = 0.0
                            for izp in 1:2
                                for iep in 1:2
                                    nxt_state = state_index(iep, izp)
                                    p = par.Pze[cur_state, nxt_state]
                                    Vcont = (1.0 - wK) * V[ikp, iKlo, iep, izp] + wK * V[ikp, iKhi, iep, izp]
                                    ev += p * Vcont
                                end
                            end

                            val = u + par.beta * ev
                            if val > best_v
                                best_v = val
                                best_ikp = ikp
                            end
                        end

                        pol_new[ik, iK, ie, iz] = best_ikp
                        n_change += (best_ikp != pol[ik, iK, ie, iz]) ? 1 : 0
                    end
                end
            end
        end

        pol, pol_new = pol_new, pol

        if n_change == 0
            @printf("  PFI policy converged at iter %d, eval diff = %.3e\n", pit, diff_eval)
            return V, pol
        end
    end

    @warn "PFI reached max_policy_iter without full policy convergence"
    return V, pol
end

