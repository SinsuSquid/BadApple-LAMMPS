# BadApple-LAMMPS 🍎

Render the iconic **Bad Apple!!** shadow art video using real-time Molecular Dynamics (MD) simulations in [**LAMMPS**](https://github.com/lammps/lammps) via a custom steering fix and OpenCV distance fields.

## Preview

![Bad_Apple!!_LAMMPS](bad_apple_lammps.gif)

---

## 🔬 How It Works

Instead of simply setting atom positions frame-by-frame, **BadApple-LAMMPS** simulates 100,000 interacting particles inside a 2D Yukawa fluid driven by a dynamically evolving potential well:

1. **Video Ingestion & Silhouette Processing**:
   For each frame from `bad_apple.mp4`, the custom fix (`fix_bad_apple`) converts the image to grayscale, resizes it to match the box aspect ratio ($480 \times 360$), and applies binary thresholding.
2. **Euclidean Distance Transform (EDT)**:
   Using `cv::distanceTransform`, a 2D scalar field $d(x, y)$ is computed representing the Euclidean distance from any background point to the nearest silhouette boundary.
3. **Dynamic Steering Force**:
   At every MD timestep, atoms outside the white silhouette experience a restoring force along the negative gradient of the distance field:
   $$\mathbf{F}_{\text{steering}} = -k_{\text{spring}} \nabla d(x, y)$$
   This pulls particles smoothly into the silhouette shape.
4. **Interatomic Repulsion & Fluid Packing**:
   Inside the silhouette, particles interact via a screened Coulomb (Yukawa) repulsive pair potential (`pair_style yukawa`), preventing atom overlap and creating uniform fluid-like packing.
5. **Damped Langevin Dynamics**:
   A strong Langevin thermostat dissipates kinetic energy quickly so atoms conform cleanly to moving contours without excessive oscillation or runaway velocities.

---

## 🛠️ Prerequisites

- **LAMMPS** (Recent stable C++ release)
- **OpenCV 4** C++ development libraries (`libopencv-dev` on Debian/Ubuntu, `opencv` on macOS/Arch/Fedora)
- **C++14/17 Compiler** (`g++` or `clang++`)
- **Video file**: `bad_apple.mp4` (standard Bad Apple music video)
- *(Optional)* **OVITO** or **VMD** for trajectory rendering and visualization

---

## 📦 Installation & Build

### Method 1: CMake (Recommended)

1. Clone or download LAMMPS source code:
   ```bash
   git clone -b stable https://github.com/lammps/lammps.git lammps
   ```

2. Copy [`fix_bad_apple.cpp`](file:///home/bgkang/Projects/BadApple-LAMMPS/fix_bad_apple.cpp) and [`fix_bad_apple.h`](file:///home/bgkang/Projects/BadApple-LAMMPS/fix_bad_apple.h) into the LAMMPS source directory:
   ```bash
   cp fix_bad_apple.cpp fix_bad_apple.h lammps/src/
   ```

3. Build LAMMPS with CMake and OpenCV linked:
   ```bash
   mkdir -p lammps/build && cd lammps/build
   cmake ../cmake -D CMAKE_BUILD_TYPE=Release \
                  -D CMAKE_CXX_FLAGS="$(pkg-config --cflags opencv4)" \
                  -D CMAKE_EXE_LINKER_FLAGS="$(pkg-config --libs opencv4)"
   cmake --build . -j$(nproc)
   ```
   *(The compiled executable will be located at `lammps/build/lmp`)*

### Method 2: Traditional Make

1. Copy [`fix_bad_apple.cpp`](file:///home/bgkang/Projects/BadApple-LAMMPS/fix_bad_apple.cpp) and [`fix_bad_apple.h`](file:///home/bgkang/Projects/BadApple-LAMMPS/fix_bad_apple.h) into `lammps/src/`.
2. Edit your `lammps/src/MAKE/Makefile.mpi` (or `Makefile.serial`) to include OpenCV compiler and linker flags:
   ```makefile
   EXTRA_INC += $(shell pkg-config --cflags opencv4)
   EXTRA_LIB += $(shell pkg-config --libs opencv4)
   ```
3. Compile:
   ```bash
   cd lammps/src
   make mpi -j$(nproc)
   ```

---

## 🚀 Running the Simulation

1. Place `bad_apple.mp4` and [`bad_apple.in`](file:///home/bgkang/Projects/BadApple-LAMMPS/bad_apple.in) in your working directory.
2. Run the simulation using the built LAMMPS binary:
   ```bash
   lmp -in bad_apple.in
   ```
   Or in parallel with MPI:
   ```bash
   mpirun -np 8 lmp -in bad_apple.in
   ```

---

## ⚙️ Simulation Details & Parameters

From [`bad_apple.in`](file:///home/bgkang/Projects/BadApple-LAMMPS/bad_apple.in):

| Parameter | Value | Description |
| :--- | :--- | :--- |
| **Dimensions** | $480 \times 360$ Å (2D) | Matches $480 \times 360$ pixel grid aspect ratio |
| **Number of Atoms** | 100,000 | Randomly seeded carbon atoms ($m = 12.011$ g/mol) |
| **Pair Potential** | `yukawa 1.0 4.0` | Screened Coulomb potential ($\kappa = 1.0\text{ Å}^{-1}, r_c = 4.0\text{ Å}$) |
| **Pair Coeff** | `1 1 5.0 4.0` | Repulsive energy prefactor $A = 5.0\text{ eV}\cdot\text{Å}$ |
| **Fix Bad Apple** | `bad_apple.mp4 100 5.0 480 360` | Video path, update every 100 steps, $k_{\text{spring}} = 5.0$, $480\times 360$ res |
| **Thermostat** | `langevin 2500 2500 0.1 424242` | Damping friction with $\tau_{\text{damp}} = 0.1\text{ ps}$ at $T = 2500\text{ K}$ |
| **Integrator** | `nve/limit 0.1` | NVE with maximum displacement capped at $0.1\text{ Å/step}$ |
| **Timestep** | $0.001\text{ ps}$ ($1\text{ fs}$) | Integration timestep in `metal` units |
| **Output** | `dump 1 all xyz 50 bad_apple_sim.xyz` | Writes trajectory every 50 steps |

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
