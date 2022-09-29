# gn-miqp-mwe - A minimum working example for the Gauss-Newton-based decomposition algorithm for nonlinear mixed-integer optimal control problems

This repository provides a minimum working example to illustrate potential advantages of the Gauss-Newton-based decomposition approach presented in [[1]](#1) for a numerical case study of Mixed-Integer Optimal Control (MIOC) of a simple nonlinear and unstable system, cf. [[2]](#2), Example 8.17, pp. 577-579.

We describe the problem setup and implementation and compare the results of the proposed approach to results obtained using the Combinatorial Integral Approximation (CIA) problem [[3]](#3).

## Requirements

Python version >= 3.6 is required. The Python packages required for running the algorithms are listed in the file `requirements.txt`, such as `casadi` [[4]](#4) and `gurobipy` [[5]](#5). In addition to these packages, [pycombina](https://github.com/adbuerger/pycombina) [[6]](#6) is required.

While the GN-MIQP in this example is sufficiently small so that it can be solved using the restricted trial license of Gurobi included in the pip-version of `gurobipy`, note that a different license might be required for larger problems or production use, cf. [[5]](#5).

## Licenses

The source code for the examples provided in this repository is released under the BSD 3-clause license. The provided PDF documentation is licensed under [CC-BY 4.0](https://creativecommons.org/licenses/by/4.0/).

## References
<a id="1">[1]</a> 
A Bürger, C Zeile, A Altmann-Dieses, S Sager, and M Diehl. A Gauss-Newton-based decomposition algorithm for nonlinear mixed-integer optimal control problems. Submitted, 2022. Available at https://optimization-online.org/2022/04/8890/.

<a id="2">[2]</a> 
J B Rawlings, D Q Mayne, and M M Diehl. Model Predictive Control: Theory, Computation, and Design. Nob Hill, 2nd edition, 2020. 3rd printing. Available at https://sites.engineering.ucsb.edu/~jbraw/mpc/.

<a id="3">[3]</a> 
S Sager, M Jung, and C Kirches. Combinatorial integral approximation. Mathematical Methods of Operations Research, 73(3):363, 2011.

<a id="4">[4]</a> 
J A E Andersson, J Gillis, G Horn, J B Rawlings, and M Diehl. CasADi – A software framework for nonlinear optimization and optimal control. Mathematical Programming Computation, 11(1):1-36, 2019.

<a id="5">[5]</a> 
Gurobi Optimization, LLC. Gurobi optimizer reference manual. https://www.gurobi.com, 2022. Last accessed September 18, 2022.

<a id="6">[6]</a> 
A Bürger, C Zeile, M Hahn, A Altmann-Dieses, S Sager, and M Diehl. pycombina: An open-source tool for solving combinatorial approximation problems arising in mixed-integer optimal control. In IFAC-PapersOnLine, volume 53, pages 6502-6508, 2020. 21st IFAC World Congress.
