# Parallel Implementation of the Standard Gradient Descent Algorithm to Optimize a Linear Regression Model Using C Language

# Project Overview
This project implements and optimizes a linear regression model using gradient descent, parallelized with OpenMP (OMP) for improved performance. The primary goal is to minimize the mean squared error (MSE) cost function while evaluating the performance of parallel execution on different core configurations. We tested the parallel implementation on Anvil HPC with up to 512 cores to assess scalability and speedup.

# Key Features
- Serial and Parallel Implementations: Standard and parallelized versions of gradient descent in C.
- Performance Evaluation: Strong and weak scaling tests on Anvil HPC.
- OpenMP-based Parallelization: Multi-threading for optimized execution.
- Automated Job Scripts: SLURM scripts for HPC execution and benchmarking.

# Tech Stack
- Programming Language: C
- Parallel Computing: OpenMP (OMP)
- HPC Execution: Anvil HPC (SLURM job scripts)
- Visualization: Python (Matplotlib for performance graphs)

# Project Files
- serial.c – Serial implementation of gradient descent for linear regression.
- parallel.c – OpenMP-parallelized version of gradient descent.
- parallel_anvil.c – Optimized parallel implementation for Anvil HPC.
- anvilstrong128.job – SLURM job script for strong scaling test (128 cores).
- anvilweak128.job – SLURM job script for weak scaling test (128 cores).
- anvilstrong512.job – SLURM job script for strong scaling test (512 cores).
- anvilweak512.job – SLURM job script for weak scaling test (512 cores).
- StrongScaleTest128.dat / WeakScaleTest128.dat – Data from scaling tests (128 cores).
- StrongScaleTest512.dat / WeakScaleTest512.dat – Data from scaling tests (512 cores).
- plot_scaling.py – Python script for visualizing strong/weak scaling results.

# How to Run the Project
1. Compile and Run Serial Version in Linux Environment:
```bash
gcc -o serial serial.c -lm
./serial <num_data_points> <num_features> <iterations>
```

2. Compile and Run Parallel Version (OpenMP) in Linux Environtment:
```bash
gcc -o parallel parallel.c -fopenmp -lm
./parallel <num_data_points> <num_features> <iterations> <num_threads>
```

3. Running on Anvil HPC
Submit job scripts to Anvil using SLURM:
```bash
sbatch anvilstrong128.job
sbatch anvilweak128.job
sbatch anvilstrong512.job
sbatch anvilweak512.job
```

# Results & Insights
- *Speedup Gains:* Significant improvements up to 64 cores, followed by diminishing returns beyond 128 cores.
- *Parallel Efficiency:* Performance gains were notable, but MPI-based parallelization may be needed for better scaling beyond 128 cores.
- *Strong vs. Weak Scaling:* Strong scaling showed consistent speedup, while weak scaling maintained efficiency at increasing problem sizes.

# Conclusion
This project successfully demonstrates parallel gradient descent optimization using OpenMP and evaluates its performance on a high-performance computing cluster. Future improvements could explore MPI-based parallelization for scaling beyond a single compute node.

# References
- Linear Regression in Machine Learning
- Gradient Descent Optimization
- Weighted Mean Square Error Implementation
