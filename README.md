# 🍎 [BadApple-LAMMPS](https://youtu.be/iAOHLZOBsHA)

[**Bad Apple!!**](https://youtu.be/FtutLA63Cp8) but it's on real-time Molecular Dynamics (MD) simulations in [**LAMMPS**](https://github.com/lammps/lammps) via a custom steering fix, [OpenCV](https://github.com/opencv/opencv) distance fields, and [Kokkos](https://github.com/kokkos/kokkos) GPU acceleration! ⚡

## Preview

![Bad_Apple!!_LAMMPS](bad_apple_lammps.gif)

---

## 🔬 How It Works

Instead of simply setting atom positions frame-by-frame, **BadApple-LAMMPS** simulates interacting particles inside a 2D Yukawa fluid driven by a dynamically evolving potential well:

1. **Video Ingestion & Silhouette Processing**:
   For each frame from `bad_apple.mp4`, the custom fix (`fix_bad_apple` / `fix_bad_apple_kokkos`) converts the image to grayscale, resizes it to match the box aspect ratio ($e.g.$ $480 \times 360$), and applies binary thresholding.
2. **Euclidean Distance Transform (EDT)**:
   Using `cv::distanceTransform`, a 2D scalar field $d(x, y)$ is computed representing the Euclidean distance from any background point to the nearest silhouette boundary.
3. **Host-to-Device Memory Transfer (Kokkos)**:
   The computed binary mask and distance field matrices are uploaded from Host mirror views to GPU Device views (`Kokkos::View`) via optimized direct DMA copies (`Kokkos::deep_copy`).
4. **Dynamic Steering Force & GPU Parallel Kernel**:
   At every MD timestep, atoms outside the white silhouette experience a restoring force evaluated in parallel on GPU execution space along the negative gradient of the distance field:
   $$\mathbf{F}_{\text{steering}} = -k_{\text{spring}} \nabla d(x, y)$$
   Particles are smoothly pulled into the silhouette shape with zero CPU-GPU transfer bottlenecks during force integration.
5. **Interatomic Repulsion & Fluid Packing**:
   Inside the silhouette, particles interact via a screened Coulomb (Yukawa) repulsive pair potential (`pair_style yukawa` or `pair_style yukawa/kk`), preventing atom overlap and creating uniform fluid packing.
6. **Damped Langevin Dynamics**:
   A strong Langevin thermostat dissipates kinetic energy quickly so atoms conform cleanly to moving contours without runaway velocities.

---

## 🛠️ Prerequisites

- **LAMMPS** (Recent stable C++ release with Kokkos support)
- **OpenCV 4** C++ development libraries (`libopencv-dev` on Debian/Ubuntu, `opencv` on macOS/Arch/Fedora)
- **CUDA Toolkit** (Optional, for NVIDIA GPU acceleration) or **ROCm / OpenMP** (for AMD/Multi-core Kokkos backends)
- **C++17 Compatible Compiler** (`g++`, `clang++`, or `nvcc` via Kokkos wrapper `nvcc_wrapper`)
- **Video file**: `bad_apple.mp4` (standard Bad Apple music video)
- *(Optional)* **OVITO** or **VMD** for trajectory rendering and visualization---

## 📦 Installation & Build

### 1. Copy Custom Fix Files

Copy the CPU and Kokkos fix files into your LAMMPS `src/` directory:

```bash
cp fix_bad_apple.cpp fix_bad_apple.h \
   fix_bad_apple_kokkos.cpp fix_bad_apple_kokkos.h \
   lammps/src/
```
### 2. Build via CMake (Recommended)
#### Option A: GPU Build (CUDA + Kokkos)

```bash
mkdir -p lammps/build && cd lammps/build

cmake ../cmake -D CMAKE_BUILD_TYPE=Release \
               -D PKG_KOKKOS=ON \
               -D Kokkos_ENABLE_CUDA=ON \
               -D Kokkos_ARCH_AMPERE80=ON \     # Adjust to your GPU arch (e.g. TURING75, AMPERE80, ADA89, HOPPER90)
               -D CMAKE_CXX_COMPILER=$(pwd)/../lib/kokkos/bin/nvcc_wrapper \
               -D CMAKE_CXX_FLAGS="$(pkg-config --cflags opencv4)" \
               -D CMAKE_EXE_LINKER_FLAGS="$(pkg-config --libs opencv4)"

cmake --build . -j$(nproc)
```

#### Option B: CPU Build (Kokkos OpenMP / Serial)

```bash
mkdir -p lammps/build && cd lammps/build

cmake ../cmake -D CMAKE_BUILD_TYPE=Release \
               -D PKG_KOKKOS=ON \
               -D Kokkos_ENABLE_OPENMP=ON \
               -D CMAKE_CXX_FLAGS="$(pkg-config --cflags opencv4)" \
               -D CMAKE_EXE_LINKER_FLAGS="$(pkg-config --libs opencv4)"

cmake --build . -j$(nproc)
```
#### Option C: Standard CPU Build (Without Kokkos)

```bash
mkdir -p lammps/build && cd lammps/build

cmake ../cmake -D CMAKE_BUILD_TYPE=Release \
               -D CMAKE_CXX_FLAGS="$(pkg-config --cflags opencv4)" \
               -D CMAKE_EXE_LINKER_FLAGS="$(pkg-config --libs opencv4)"

cmake --build . -j$(nproc)
```

### 3. Build via Traditional Make (Alternative)

1. Enable the Kokkos package (if building with Kokkos):
```bash
cd lammps/src
make yes-kokkos
```

2. Edit `lammps/src/MAKE/Makefile.mpi` (or your target Makefile) to append OpenCV flags:
```Makefile
EXTRA_INC += $(shell pkg-config --cflags opencv4)
EXTRA_LIB += $(shell pkg-config --libs opencv4)
```

3. Compile with Kokkos CUDA or OpenMP:
```bash
# For CUDA:
make kokkos_cuda_mpi -j$(nproc)

# For OpenMP / CPU:
make kokkos_omp -j$(nproc)
```

---

## 🚀 Running the Simulation

Plae `bad_apple.mp4` and `bad_apple.in` in your working directory.

### Running with Kokkos Accerlation (GPU)
```bash
mpirun -np 1 lmp -k on g 1 -sf kk -pk kokkos gpu/aware on -in bad_apple.in
```
### Running with Kokkos OpenMP (CPU Multi-threading)
```bash
mpirun -np 1 lmp -k on t 16 -sf kk -in bad_apple.in
```

### Running Standard CPU (MPI)
```bash
mpirun -np 8 lmp -in bad_apple.in
```

---

## ⚙️ Simulation Details & Parameters

From [`bad_apple.in`](file:///home/bgkang/Projects/BadApple-LAMMPS/bad_apple.in):

| Parameter | Value | Description |
| :--- | :--- | :--- |
| **Dimensions** | $480 \times 360$ Å (2D) | Matches $480 \times 360$ pixel grid aspect ratio |
| **Number of Atoms** | 75,000 | Randomly seeded carbon atoms ($m = 12.011$ g/mol) |
| **Pair Potential** | `yukawa 1.0 4.0` | Screened Coulomb potential ($\kappa = 1.0\text{ Å}^{-1}, r_c = 4.0\text{ Å}$) |
| **Pair Coeff** | `1 1 5.0 4.0` | Repulsive energy prefactor $A = 5.0\text{ eV}\cdot\text{Å}$ |
| **Fix Bad Apple** | `bad_apple.mp4 100 5.0 480 360` | Video path, update every 100 steps, $k_{\text{spring}} = 5.0$, $480\times 360$ res |
| **Thermostat** | `langevin 2000 2000 0.1 424242` | Damping friction with $\tau_{\text{damp}} = 0.1\text{ ps}$ at $T = 2000\text{ K}$ |
| **Integrator** | `nve/limit 0.3` | NVE with maximum displacement capped at $0.3\text{ Å/step}$ |
| **Timestep** | $0.001\text{ ps}$ ($1\text{ fs}$) | Integration timestep in `metal` units |
| **Output** | `dump 1 all xyz 100 bad_apple_sim.xyz` | Writes trajectory every 100 steps |

### `fix bad_apple` Syntax

```lammps
fix ID group-ID bad_apple <video_file> <N_every> <k_spring> <width> <height>
```
- `<video_file>`: Path to input MP4 video file.
- `<N_every>`: Advance to the next video frame every `N` timesteps.
- `<k_spring>`: Force constant scaling the distance-transform gradient.
- `<width>`, `<height>`: Image grid resolution mapped to simulation box dimensions.

---

## 🎨 Visualization & Rendering

To visualize and render the output trajectory [`bad_apple_sim.xyz`]:

1. Open **[OVITO](https://www.ovito.org/)**.
2. Load `bad_apple_sim.xyz`.
3. Adjust particle radius (e.g. $0.6 - 0.9\text{ Å}$) and color scheme (e.g., white particles on a black background).
4. Export the animation as a video or GIF.

---

## 📜 License

Distributed under the MIT License. See `LICENSE` for more information.
