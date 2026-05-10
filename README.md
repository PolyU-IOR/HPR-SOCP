# HPR-SOCP: A GPU Solver for Second-Order Cone Programming in Julia

> **HPR-SOCP: A dual Halpern Peaceman–Rachford (HPR) method for solving large-scale second-order cone programming (SOCP) problems.**

---

## SOCP Problem Formulation
HPR-SOCP solves convex quadratic second-order cone programs of the following form:

$$
\begin{aligned}
\min_{x=(x_1,x_2)}\quad
& \frac{1}{2}\langle x,Qx\rangle+\langle c,x\rangle \\
\mathrm{s.t.}\quad
& A_1x\in \mathcal K_A,\quad
  A_2x-b_2\in \mathcal K_{\mathrm{soc}},\\
& x_1\in\mathcal C,\quad x_2\in\mathcal K_v,
\end{aligned}
$$

with $Q\succeq 0$, $x_1\in\mathbb R^{n_1}$, $x_2\in\mathbb R^{n_2}$, and $n_1+n_2=n$, where

$$
\begin{aligned}
\mathcal K_A
&:= \\{ y\in\mathbb R^{m_A}\mid l_c\le y\le u_c \\},&
\mathcal C
&:= \\{ x_1\in\mathbb R^{n_1}\mid l_x\le x_1\le u_x \\},\\
\mathcal K_{\mathrm{soc}}
&:=\mathcal Q^{m_1}\times\cdots\times\mathcal Q^{m_p},&
\mathcal K_v
&:=\mathcal Q^{n_{2,1}}\times\cdots\times\mathcal Q^{n_{2,q}},\\
\mathcal Q^m
&:= \\{(t,\xi)\in\mathbb R\times\mathbb R^{m-1}\mid \|\xi\|_2\le t\\}.&
\end{aligned}
$$


# Getting Started

## Prerequisites

Before using HPR-SOCP, make sure the following dependencies are installed:

- **Julia** 
- **CUDA** (Required for GPU acceleration; install the appropriate version for your GPU and Julia)
- Required Julia packages

> To install the required Julia packages and build the HPR-SOCP environment, run:
```bash
julia --project -e 'using Pkg; Pkg.instantiate()'
```

> To verify that CUDA is properly installed and working with Julia, run:
```julia
using CUDA
CUDA.versioninfo()
```

---

## Usage 1: Test Instances in CBF Format

### Setting Data and Result Paths

> Before running the scripts, please modify **`run_single_file.jl`** or **`run_dataset.jl`** in the demo directory to specify the data path and result path according to your setup.

### Running a Single Instance

To test the script on a single instance:

```bash
julia --project demo/run_single_file.jl
```

### Running All Instances in a Directory

To process all files in a directory:

```bash
julia --project demo/run_dataset.jl
```
---

## Note on First-Time Execution Performance

You may notice that solving a single instance — or the first instance in a dataset — appears slow. This is due to Julia’s Just-In-Time (JIT) compilation, which compiles code on first execution.

> **💡 Tip for Better Performance:**  
> To reduce repeated compilation overhead, it’s recommended to run scripts from an **IDE like VS Code** or the **Julia REPL** in the terminal.

#### Start Julia REPL with the project environment:

```bash
julia --project
```

Then, at the Julia REPL, run demo/run_single_file.jl (or other scripts):

```julia
include("demo/run_single_file.jl")
```

> **CAUTION:**  
> If you encounter the error message:  
> `Error: Error during loading of extension AtomixCUDAExt of Atomix, use Base.retry_load_extensions() to retry`.
>
> Don’t panic — this is usually a transient issue. Simply wait a few moments; the extension typically loads successfully on its own.

---

## Parameters

Below is a list of the parameters in HPR-SOCP along with their default values and usage:

<table>
  <thead>
    <tr>
      <th>Parameter</th>
      <th>Default Value</th>
      <th>Description</th>
    </tr>
  </thead>
  <tbody>
    <tr><td><code>stoptol</code></td><td><code>1e-6</code></td><td>Stopping tolerance for convergence checks.</td></tr>
    <tr><td><code>sigma</code></td><td><code>-1 (auto)</code></td><td>Initial value of the σ parameter used in the algorithm.</td></tr>
    <tr><td><code>max_iter</code></td><td><code>typemax(Int32)</code></td><td>Maximum number of iterations allowed.</td></tr>
    <tr><td><code>time_limit</code></td><td><code>3600.0</code></td><td>Maximum allowed runtime (seconds) for the algorithm.</td></tr>
    <tr><td><code>check_iter</code></td><td><code>150</code></td><td>Frequency (in iterations) to check for convergence or perform other checks.</td></tr>
    <tr><td><code>warm_up</code></td><td><code>false</code></td><td>Determines if a warm-up phase is performed before main execution.</td></tr>
    <tr><td><code>print_frequency</code></td><td><code>-1 (auto)</code></td><td>Frequency (in iterations) for printing progress or logging information.</td></tr>
    <tr><td><code>device_number</code></td><td><code>0</code></td><td>GPU device number (e.g., 0, 1, 2, 3).</td></tr>
    <tr><td><code>use_Ruiz_scaling</code></td><td><code>true</code></td><td>Whether to apply Ruiz scaling to the problem data.</td></tr>
    <tr><td><code>ruiz_iterations</code></td><td><code>10</code></td><td>Number of Ruiz scaling passes.</td></tr>
    <tr><td><code>use_bc_scaling</code></td><td><code>true</code></td><td>Whether to apply b/c scaling.</td></tr>
    <tr><td><code>bc_scaling_norm_type</code></td><td><code>:l2</code></td><td>Scalar summary used for b/c scaling (<code>:l2</code>, <code>:rms</code>, or <code>:linf</code>).</td></tr>
    <tr><td><code>use_l2_scaling</code></td><td><code>false</code></td><td>Whether to apply L2-norm based scaling.</td></tr>
    <tr><td><code>use_Pock_Chambolle_scaling</code></td><td><code>true</code></td><td>Whether to apply Pock-Chambolle scaling to the problem data.</td></tr>
    <tr><td><code>soc_block_scaling_strategy</code></td><td><code>:phase_taper</code></td><td>Shared SOC block aggregation rule used during scaling.</td></tr>
    <tr><td><code>initial_x</code></td><td><code>nothing</code></td><td>Initial primal solution for warm-start.</td></tr>
    <tr><td><code>initial_y</code></td><td><code>nothing</code></td><td>Initial dual solution for warm-start.</td></tr>
    <tr><td><code>auto_save</code></td><td><code>false</code></td><td>Automatically save best x, y, z, w, and sigma during optimization.</td></tr>
    <tr><td><code>save_filename</code></td><td><code>"HPRSOCP_autosave.h5"</code></td><td>Filename for auto-save HDF5 file.</td></tr>
    <tr><td><code>verbose</code></td><td><code>true</code></td><td>Enable verbose output during optimization.</td></tr>
    <tr><td><code>use_gpu</code></td><td><code>true</code></td><td>Whether to use GPU acceleration (requires CUDA).</td></tr>
  </tbody>
</table>

---

# Result Explanation

After solving an instance, you can access the result variables as shown below:

```julia
# Example from /demo/run_single_file.jl
println("Objective value: ", result.primal_obj)
println("x1 = ", result.x[1])
println("x2 = ", result.x[2])
```

<table>
  <thead>
    <tr>
      <th>Category</th>
      <th>Variable</th>
      <th>Description</th>
    </tr>
  </thead>
  <tbody>
    <tr><td><b>Iteration Counts</b></td><td><code>iter</code></td><td>Total number of iterations performed by the algorithm.</td></tr>
    <tr><td></td><td><code>iter_4</code></td><td>Number of iterations required to achieve an accuracy of 1e-4.</td></tr>
    <tr><td></td><td><code>iter_6</code></td><td>Number of iterations required to achieve an accuracy of 1e-6.</td></tr>
    <tr><td><b>Time Metrics</b></td><td><code>time</code></td><td>Total time in seconds taken by the algorithm.</td></tr>
    <tr><td></td><td><code>time_4</code></td><td>Time in seconds taken to achieve an accuracy of 1e-4.</td></tr>
    <tr><td></td><td><code>time_6</code></td><td>Time in seconds taken to achieve an accuracy of 1e-6.</td></tr>
    <tr><td></td><td><code>power_time</code></td><td>Time in seconds spent estimating the spectral bounds used during setup.</td></tr>
    <tr><td><b>Objective Values</b></td><td><code>primal_obj</code></td><td>The primal objective value obtained.</td></tr>
    <tr><td></td><td><code>gap</code></td><td>The final relative duality gap.</td></tr>
    <tr><td><b>Residuals</b></td><td><code>residuals</code></td><td>The overall termination metric <code>max(primal residual, dual residual, relative gap)</code>.</td></tr>
    <tr><td><b>Algorithm Status</b></td><td><code>status</code></td><td>The final status of the algorithm:<br/>- <code>OPTIMAL</code>: Found optimal solution<br/>- <code>MAX_ITER</code>: Max iterations reached<br/>- <code>TIME_LIMIT</code>: Time limit reached</td></tr>
    <tr><td><b>Solution Vectors</b></td><td><code>x</code></td><td>The final primal variable vector in the original scaling.</td></tr>
    <tr><td></td><td><code>y</code></td><td>The final dual vector for linear and SOC constraints in the original scaling.</td></tr>
    <tr><td></td><td><code>z</code></td><td>The final reduced-cost / bound-dual vector in the original scaling.</td></tr>
    <tr><td></td><td><code>w</code></td><td>The auxiliary primal vector.</td></tr>
  </tbody>
</table>

---


## Citation

Kaihuang Chen, Defeng Sun, and Guojun Zhang, “HPR-SOCP: A GPU-Accelerated Halpern–Peaceman–Rachford Solver for Second-Order Cone Programming,” May 2026.

If you use HPR-SOCP in your research, please cite this repository for now.

# HPR-SOCP
