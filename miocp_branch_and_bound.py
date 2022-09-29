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

import heapq
import numpy as np
import matplotlib.pyplot as plt
import casadi as ca


class Node:
    def __init__(self, parent, obj, depth, b, xf):

        self.parent = parent
        self.obj = obj
        self.depth = depth
        self.b = b
        self.xf = xf

    def get_b(self):

        if self.parent is None:
            return [self.b]
        else:
            return self.parent.get_b() + [self.b]

    def get_depth(self):

        if self.parent is None:
            return [self.depth]
        else:
            return self.parent.get_depth() + [self.depth]


class MIOCPBranchAndBound:
    def _set_timing(self):

        t0 = 0
        tf = 1.5
        self.N = 30

        self.dt = (tf - t0) / self.N

        self.t = np.linspace(t0, tf, self.N + 1)

    def _set_system_and_objective(self):

        x = ca.SX.sym("x")
        b = ca.SX.sym("b", 1)

        xdot = x**3 - b

        self.x_ref = 0.7
        obj = 0.5 * (x - self.x_ref) ** 2

        self.ode = ca.Function("ode", [x, b], [xdot])
        self.obj = ca.Function("obj", [x], [obj])

        self.nx = x.numel()
        self.nb = b.numel()

    def _set_bounds_and_initials(self):

        self.x0 = 0.8

        self.x_min = 0.0
        self.x_max = 1.0

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

    def __init__(self):

        self._set_timing()
        self._set_system_and_objective()
        self._set_bounds_and_initials()
        self._setup_rk4_integrator()

    def _create_node(self, parent, b):

        if parent is not None:
            xf = parent.xf
            obj = parent.obj
            depth = parent.depth
            b_prev = parent.b
        else:
            xf = self.x0
            obj = self.obj(self.x0)
            depth = 0
            b_prev = 0

        if (b != b_prev) and (b_prev == 0):
            uptime = 0
        else:
            uptime = self.min_up_time_b - self.dt

        while (uptime < self.min_up_time_b) and (depth < self.N):
            xf = self.rk4(xf, b)
            depth += 1
            obj += self.obj(xf)
            uptime += self.dt

        if depth > self.N:
            print("here")

        if xf > self.x_max:
            return None

        if obj > self.ub:
            return None

        return depth, obj, Node(parent, obj, depth, b, xf)

    def _setup_queue(self):

        self.q = []

    def _setup_ub(self, ub):

        self.ub = ub

    def _create_initial_nodes(self):

        for b in [0, 1]:
            node = self._create_node(None, b)
            if node is not None:
                heapq.heappush(self.q, node)

    def _run_bnb(self):

        self.i = 0
        self.best_node = None

        while not len(self.q) == 0:

            self.i += 1
            _, _, parent = heapq.heappop(self.q)

            if parent.depth == self.N:
                if parent.obj < self.ub:

                    self.best_node = parent
                    self.ub = parent.obj

                    print(f"New best node with obj = {parent.obj}")

            else:
                for b in [0, 1]:
                    node = self._create_node(parent, b)
                    if node is not None:
                        heapq.heappush(self.q, node)

            if not (self.i % 1000):
                print(f"Iteration {self.i}, current best obj = {self.ub}")

        print(f"Best solution after {self.i} iterations: obj = {self.best_node.obj}")

    def _collect_solution(self):

        self.b_opt = np.empty(self.N)
        self.b_opt[:] = np.nan

        self.b_opt[[0] + self.best_node.get_depth()[:-1]] = self.best_node.get_b()

        for k, b_k in enumerate(self.b_opt):
            if np.isnan(b_k):
                self.b_opt[k] = self.b_opt[k - 1]

    def _simulate(self):

        self.x_opt = [self.x0]

        for k in range(self.N):

            self.x_opt.append(float(self.rk4(self.x_opt[-1], self.b_opt[k])))

        self.x_opt = np.asarray(self.x_opt)

    def run(self, ub):

        self._setup_queue()
        self._setup_ub(ub=ub)
        self._create_initial_nodes()
        self._run_bnb()
        self._collect_solution()
        self._simulate()

    def get_objective_value(self):

        f = 0

        for k in range(self.N + 1):

            f += 0.5 * float(self.x_opt[k] - self.x_ref) ** 2

        print(f"Least squares objective value: {f}")

        return f

    def plot_results(self, plotname="results"):

        plt.rc("text", usetex=True)

        pgf_with_rc_fonts = {
            "font.size": 19,
            "legend.handlelength": 1.0,
            "legend.columnspacing": 1.2,
            "legend.edgecolor": "None",
        }

        plt.rcParams.update(pgf_with_rc_fonts)

        plt.close("all")

        fig, ax = plt.subplots(2, 1, figsize=(8, 4), sharex=True)

        ax[0].step(
            self.t,
            self.x_ref * np.ones(self.t.shape),
            label=r"$x_\mathrm{ref}$",
            linestyle="dashed",
            color="C7",
            where="post",
            linewidth=0.7,
        )
        ax[0].plot(self.t, self.x_opt, label=r"$x^{\ast\ast\ast}$", color="k")
        ax[0].set_ylim([0.6, 1.0])
        ax[0].set_ylabel("$x$")
        ax[0].legend(loc="upper left", ncol=3)

        ax[1].step(
            self.t[:-1], self.b_opt, label=r"$b^{\ast\ast}$", color="k", where="post"
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

    miocp_bnb = MIOCPBranchAndBound()
    miocp_bnb.set_min_up_time(min_up_time=3 * miocp_bnb.dt)
    miocp_bnb.run(ub=0.05)
    miocp_bnb.get_objective_value()
    miocp_bnb.plot_results(plotname="results_bnb")
