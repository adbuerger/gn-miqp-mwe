#
# This file is part of gn-miqp-mwe.
#
# Copyright (c) 2022 Adrian Bürger, Angelika Altmann-Dieses, Moritz Diehl.
# Developed at HS Karlsruhe and IMTEK, University of Freiburg.
# All rights reserved.
#
# The BSD 3-Clause License
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
#    list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
#    contributors may be used to endorse or promote products derived from
#    this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#

import time
import warnings
import numpy as np
import matplotlib.pyplot as plt
import casadi as ca

try:
    import gurobipy as gp
except ImportError:
    warnings.warn("gurobipy not found; this is required for solving the GN-MIQP.")

try:
    import pycombina as pyc
except ImportError:
    warnings.warn("pycombina not found; this is required for solving the CIA problem.")


class MIOCPDecomposition:
    def _set_timing(self):

        t0 = 0
        tf = 1.5
        self.N = 30

        self.dt = (tf - t0) / self.N

        self.t = np.linspace(t0, tf, self.N + 1)

    def _set_system_and_objective(self):

        x = ca.SX.sym("x", 1)
        b = ca.SX.sym("u", 1)

        xdot = x**3 - b

        self.x_ref = 0.7
        r = x - self.x_ref

        self.ode = ca.Function("ode", [x, b], [xdot])
        self.r = ca.Function("r", [x], [r])

        self.nx = x.numel()
        self.nb = b.numel()

        self.min_up_time_b = None

    def set_min_up_time(self, min_up_time):

        # Prevent over-fulfillment of dwell time constraints,
        # cf. https://github.com/adbuerger/pycombina/issues/7

        self.min_up_time_b = min_up_time - 1e-6

    def _setup_rk4_integrator(self):

        x0 = ca.SX.sym("x0", self.nx)
        p = ca.SX.sym("p", self.nb)

        K1 = self.ode(x0, p)
        K2 = self.ode(x0 + (self.dt / 2) * K1, p)
        K3 = self.ode(x0 + (self.dt / 2) * K2, p)
        K4 = self.ode(x0 + self.dt * K3, p)

        xf = x0 + (self.dt / 6) * (K1 + 2 * K2 + 2 * K3 + K4)

        self.rk4 = ca.Function("rk4", [x0, p], [xf])

    def _set_bounds_and_initials(self):

        self.x0 = 0.8

        self.x_min = -np.inf
        self.x_max = np.inf
        self.x_init = 0.8

        self.b_min = 0
        self.b_max = 1
        self.b_init = 1

    def _setup_discretization(self):

        self.V = []
        self.R = []
        self.g = []
        self.b = []

        self.idx_b = []
        idx = 0

        x_k = self.x0

        for k in range(self.N):

            b_k = ca.SX.sym(f"b_{k}", self.nb)
            for j in range(self.nb):
                self.idx_b += [idx]
                idx += 1
            self.V.append(b_k)
            self.b.append(b_k)

            x_k_next = ca.SX.sym(f"x_{k+1}", self.nx)
            for j in range(self.nx):
                idx += 1
            self.V.append(x_k_next)

            self.g.append(x_k_next - self.rk4(x_k, b_k))

            self.R += [self.r(x_k)]

            x_k = x_k_next

        self.R += [self.r(x_k)]

    def _set_min_up_time_constraints(self) -> None:

        self._g_mut = []

        if self.min_up_time_b is not None:

            for k in range(0, self.N, 1):

                it = 0
                uptime = self.dt

                while uptime < self.min_up_time_b:

                    idx_1 = k - 1
                    idx_2 = k - (it + 2)

                    if idx_1 >= 0:
                        b_idx_1 = self.b[idx_1]
                    else:
                        b_idx_1 = 0

                    if idx_2 >= 0:
                        b_idx_2 = self.b[idx_2]
                    else:
                        b_idx_2 = 0

                    self._g_mut.append(-self.b[k] + b_idx_1 - b_idx_2)

                    it += 1
                    uptime += self.dt

        self._g_mut = ca.veccat(*self._g_mut)
        self._lbg_mut = -np.inf * np.ones(self._g_mut.numel())
        self._ubg_mut = np.zeros(self._g_mut.numel())

    def _setup_nlp(self):

        self.V = ca.veccat(*self.V)
        self.R = ca.veccat(*self.R)
        self.f = 0.5 * ca.mtimes(self.R.T, self.R)
        self.g = ca.veccat(*self.g, self._g_mut)

        self.nlpsolver = ca.nlpsol(
            "nlpsolver", "ipopt", {"x": self.V, "f": self.f, "g": self.g}
        )

    def __init__(self):

        self._set_timing()
        self._set_system_and_objective()
        self._setup_rk4_integrator()

    def _setup_bounds_and_initials(self):

        V_lb = []
        V_ub = []
        V_0 = []

        for k in range(self.N):

            V_lb += [self.b_min]
            V_ub += [self.b_max]
            V_0 += [self.b_init]

            V_lb += [self.x_min]
            V_ub += [self.x_max]
            V_0 += [self.x_init]

        self.V_lb = ca.veccat(*V_lb)
        self.V_ub = ca.veccat(*V_ub)
        self.V_0 = ca.veccat(*V_0)

        self.lbg = np.zeros(self.g.numel())
        self.ubg = np.zeros(self.g.numel())
        self.lbg[-self._g_mut.numel() :] = -np.inf

    def _solve_nlp(self):

        self.sol_nlp = self.nlpsolver(
            lbx=self.V_lb, ubx=self.V_ub, x0=self.V_0, lbg=self.lbg, ubg=self.ubg
        )

    def _select_results(self):

        x_opt = [self.x0]
        b_opt = []

        offset = 0

        for k in range(self.N):

            b_opt += [self.sol_nlp["x"][offset : offset + self.nb]]
            offset += self.nb

            x_opt += [self.sol_nlp["x"][offset : offset + self.nx]]
            offset += self.nx

        self.b_opt = np.asarray(ca.veccat(*b_opt))
        self.x_opt = np.asarray(ca.veccat(*x_opt))
        self.V_opt = self.sol_nlp["x"]

    def _save_relaxed_solution(self):

        self.x_rel = self.x_opt
        self.b_rel = self.b_opt

    def solve_nlp(self):

        self._set_bounds_and_initials()
        self._setup_discretization()
        self._set_min_up_time_constraints()
        self._setup_nlp()
        self._setup_bounds_and_initials()
        self._solve_nlp()
        self._select_results()
        self._save_relaxed_solution()

    def solve_cia(self, use_pycombina_bnb=True):

        b_rel = np.hstack([self.b_opt, 1 - self.b_opt])

        ba = pyc.BinApprox(t=self.t, b_rel=b_rel)

        if self.min_up_time_b:
            ba.set_min_up_times([self.min_up_time_b, 0])

        if use_pycombina_bnb:
            cia = pyc.CombinaBnB(ba)
        else:
            cia = pyc.CombinaMILP(ba)

        t_start = time.time()
        cia.solve()
        self.runtime_cia = time.time() - t_start

        self.b_opt = ba.b_bin[0, :]
        self.b_opt_cia = self.b_opt
        self.obj_cia = ba.eta

    def simulate(self):

        self.x_opt = [self.x0]

        for k in range(self.N):

            self.x_opt.append(float(self.rk4(self.x_opt[-1], self.b_opt[k])))

        self.x_opt = np.asarray(self.x_opt)

    def _setup_gnmiqp(self):

        dR = ca.jacobian(self.R, self.V)

        self.B = ca.mtimes([dR.T, dR])
        dg = ca.jacobian(self.g, self.V)

        P = ca.Function("P", [self.V], [self.B])
        q = ca.Function(
            "q", [self.V], [ca.mtimes(dR.T, self.R) - ca.mtimes(self.B, self.V)]
        )

        A = ca.Function("A", [self.V], [dg])

        g_b = ca.SX.sym("g_b", self.g.shape)
        b = ca.Function("b", [g_b, self.V], [g_b - self.g + ca.mtimes(dg, self.V)])

        self.q_k = np.asarray(q(self.V_opt))
        self.A_k = np.asarray(A(self.V_opt))

        self.lb_k = np.squeeze(b(self.lbg, self.V_opt))
        self.ub_k = np.squeeze(b(self.ubg, self.V_opt))

        self.P_k = np.asarray(P(self.V_opt))

    def _solve_gnmiqp(self):

        vtype = np.zeros(self.V.shape, dtype=object)
        vtype[:] = gp.GRB.CONTINUOUS

        lbx = -np.inf * np.ones(self.A_k.shape[1])
        ubx = np.inf * np.ones(self.A_k.shape[1])

        vtype[self.idx_b] = gp.GRB.BINARY
        vtype = np.squeeze(vtype).tolist()

        m = gp.Model()

        x = m.addMVar(self.A_k.shape[1], vtype=vtype, lb=lbx, ub=ubx)

        m.setObjective(0.5 * (x @ self.P_k @ x) + self.q_k.T @ x)

        m.addConstr(self.A_k @ x >= self.lb_k)
        m.addConstr(self.A_k @ x <= self.ub_k)

        t_start = time.time()
        m.optimize()
        self.runtime_gn_miqp = time.time() - t_start

        self.res_miqp = x.X
        self.b_opt = self.res_miqp[::2]

    def solve_gnmiqp(self):

        self._setup_gnmiqp()
        self._solve_gnmiqp()

    def get_objective_value(self):

        f = 0

        for k in range(self.N + 1):

            f += 0.5 * float(self.x_opt[k] - self.x_ref) ** 2

        print(f"Least squares objective value: {f}")

        return f

    def plot_results(self, plotname="results"):

        pgf_with_rc_fonts = {
            "font.size": 16,
            "legend.handlelength": 1.0,
            "legend.columnspacing": 1.2,
            "legend.edgecolor": "None",
        }

        plt.rcParams.update(pgf_with_rc_fonts)

        plt.close("all")

        _, ax = plt.subplots(2, 1, figsize=(8, 4), sharex=True)

        ax[0].step(
            self.t,
            self.x_ref * np.ones(self.t.shape),
            label=r"$x_\mathrm{ref}$",
            linestyle="dashed",
            color="C7",
            where="post",
            linewidth=0.7,
        )
        ax[0].plot(
            self.t, self.x_rel, label=r"$x^{\ast}$", linestyle="dotted", color="C0"
        )
        ax[0].plot(self.t, self.x_opt, label=r"$x^{\ast\ast\ast}$", color="C0")
        ax[0].set_ylim([0.6, 1.0])
        ax[0].set_ylabel("$x$")
        ax[0].legend(loc="upper left", ncol=3, bbox_to_anchor=(0.0, 1.1))

        ax[1].step(
            self.t[:-1],
            self.b_rel,
            label=r"$b^{\ast}$",
            linestyle="dotted",
            color="C1",
            where="post",
        )
        ax[1].step(
            self.t[:-1], self.b_opt, label=r"$b^{\ast\ast}$", color="C1", where="post"
        )
        ax[1].set_ylim([-0.1, 1.1])
        ax[1].set_ylabel("$b$")
        ax[1].legend(loc="center right", ncol=2)

        ax[-1].set_xlim([self.t[0], self.t[-1]])
        ax[-1].set_xlabel("$t$")

        for ax_k in ax:

            ax_k.spines["top"].set_visible(False)
            ax_k.spines["right"].set_visible(False)

        plt.savefig(f"{plotname}.pdf", format="pdf", bbox_inches="tight")


if __name__ == "__main__":

    miocp_cia = MIOCPDecomposition()
    miocp_cia.set_min_up_time(min_up_time=3 * miocp_cia.dt)
    miocp_cia.solve_nlp()
    miocp_cia.solve_cia()
    miocp_cia.simulate()
    miocp_cia.get_objective_value()
    miocp_cia.plot_results(plotname="results_cia")

    miocp_gnmiqp = MIOCPDecomposition()
    miocp_gnmiqp.set_min_up_time(min_up_time=3 * miocp_gnmiqp.dt)
    miocp_gnmiqp.solve_nlp()
    miocp_gnmiqp.solve_gnmiqp()
    miocp_gnmiqp.simulate()
    miocp_gnmiqp.get_objective_value()
    miocp_gnmiqp.plot_results(plotname="results_gnmiqp")
