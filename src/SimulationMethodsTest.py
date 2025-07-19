import os
import sys
import time
import logging
import numpy as np
import h5py
import matplotlib.pyplot as plt
import imageio
from tqdm import tqdm
from numba import njit, prange
import pyfftw
from scipy.fft import fftfreq
from scipy.interpolate import interp1d
from colossus.cosmology import cosmology

# -----------------------------
# User Settings
# -----------------------------
N_PARTICLES = 5e4
BOX_SIZE = 100.0  # arbitrary units
SOFTENING = 0.05  # softening length
TOTAL_STEPS = 1e3
SAVE_EVERY = 5
SEED = 42

# Output directory
BASE_DIR = os.path.expanduser('~/bulk-flow-Rockstar/Simulation-Methods-Test')

METHODS = ['DirectSum', 'PM', 'AdaptiveMesh', 'BarnesHutTree', 'TreePM']

# -----------------------------
# Setup Logging
# -----------------------------
if not os.path.exists(BASE_DIR):
    os.makedirs(BASE_DIR)

log_file = os.path.join(BASE_DIR, 'simulation.log')
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler(sys.stdout)
    ]
)

def log(msg):
    logging.info(msg)

log("Simulation started.")

# -----------------------------
# Setup Cosmology
# -----------------------------
cosmo_params = {
    'flat': True,
    'H0': 70.0,
    'Om0': 0.3,
    'Ob0': 0.05,
    'sigma8': 0.8,
    'ns': 0.96
}
cosmology.addCosmology('myCosmo', cosmo_params)
cosmo = cosmology.setCosmology('myCosmo')

# -----------------------------
# Utility Functions
# -----------------------------
def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def save_snapshot_hdf5(filename, positions, velocities):
    with h5py.File(filename, 'w') as f:
        f.create_dataset('positions', data=positions)
        f.create_dataset('velocities', data=velocities)

def plot_particles(positions, out_path, step, method):
    plt.style.use('dark_background')
    fig, ax = plt.subplots(figsize=(8,8))
    plt.hist2d(positions[:,0], positions[:,1], bins=500, cmap='magma', range=[[0, BOX_SIZE], [0, BOX_SIZE]])
    ax.set_title(f"{method} - Step {step}")
    plt.axis('off')
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)

def make_video(image_folder, video_path):
    images = sorted([os.path.join(image_folder, img)
                     for img in os.listdir(image_folder)
                     if img.endswith('.png')])
    with imageio.get_writer(video_path, mode='I', fps=10) as writer:
        for filename in images:
            image = imageio.imread(filename)
            writer.append_data(image)

# -----------------------------
# Initialize Particles (Zel'dovich)
# -----------------------------
np.random.seed(SEED)
positions = np.random.uniform(0, BOX_SIZE, size=(N_PARTICLES, 2))
velocities = np.zeros_like(positions)

# Linear growth factor for Zel'dovich displacement
z_init = 50.0
z_final = 0.0
a_init = 1.0 / (1.0 + z_init)
a_final = 1.0

D_init = cosmo.growthFactor(a_init)
D_final = cosmo.growthFactor(a_final)
delta_D = D_final - D_init

# Simple initial displacement (placeholder Zel'dovich)
displacements = np.random.normal(0, 0.5, size=(N_PARTICLES, 2))
positions += displacements * (delta_D)

positions %= BOX_SIZE  # Enforce periodic BCs

log("Initial conditions generated using Zel'dovich approximation.")

# -----------------------------
# Timer Utility
# -----------------------------
class Timer:
    def __init__(self):
        self.start_time = None

    def start(self):
        self.start_time = time.time()

    def elapsed(self):
        return time.time() - self.start_time

    def eta(self, current_step, total_steps):
        elapsed = self.elapsed()
        rate = elapsed / max(1, current_step)
        remaining = (total_steps - current_step) * rate
        return remaining

# -----------------------------
# Prepare Output Folders
# -----------------------------
for method in METHODS:
    method_dir = os.path.join(BASE_DIR, method)
    ensure_dir(method_dir)
    ensure_dir(os.path.join(method_dir, 'snapshots'))
    ensure_dir(os.path.join(method_dir, 'plots'))

log("Output directories created.")


# -----------------------------
# Gravitational Force (Direct Sum)
# -----------------------------
@njit(parallel=True)
def compute_forces_direct_sum(positions, softening, box_size):
    N = positions.shape[0]
    forces = np.zeros_like(positions)
    for i in prange(N):
        fx, fy = 0.0, 0.0
        xi, yi = positions[i, 0], positions[i, 1]
        for j in range(N):
            if i != j:
                dx = xi - positions[j, 0]
                dy = yi - positions[j, 1]

                # Periodic BCs (minimum image convention)
                dx -= box_size * np.round(dx / box_size)
                dy -= box_size * np.round(dy / box_size)

                r2 = dx * dx + dy * dy + softening * softening
                inv_r3 = 1.0 / (r2 * np.sqrt(r2))

                fx -= dx * inv_r3
                fy -= dy * inv_r3
        forces[i, 0] = fx
        forces[i, 1] = fy
    return forces


# -----------------------------
# Leapfrog Integrator
# -----------------------------
def leapfrog_direct_sum(positions, velocities, dt, total_steps, save_every, method_dir):
    step_timer = Timer()
    total_timer = Timer()
    total_timer.start()

    images_folder = os.path.join(method_dir, 'plots')
    snapshots_folder = os.path.join(method_dir, 'snapshots')

    for step in range(1, total_steps + 1):
        step_timer.start()

        # Half kick
        forces = compute_forces_direct_sum(positions, SOFTENING, BOX_SIZE)
        velocities += 0.5 * dt * forces

        # Drift
        positions += dt * velocities
        positions %= BOX_SIZE  # Periodic BCs

        # Full kick
        forces = compute_forces_direct_sum(positions, SOFTENING, BOX_SIZE)
        velocities += 0.5 * dt * forces

        # Save and plot
        if step % save_every == 0:
            snapshot_file = os.path.join(snapshots_folder, f'step_{step:04d}.h5')
            save_snapshot_hdf5(snapshot_file, positions, velocities)

            plot_file = os.path.join(images_folder, f'step_{step:04d}.png')
            plot_particles(positions, plot_file, step, 'Direct Sum')

        # Logging
        eta_sec = total_timer.eta(step, total_steps)
        eta_str = time.strftime('%H:%M:%S', time.gmtime(eta_sec))

        log(f"[Direct Sum] Step {step}/{total_steps} complete. "
            f"Elapsed: {step_timer.elapsed():.2f}s. ETA: {eta_str}")

    # Generate video
    video_path = os.path.join(method_dir, 'simulation.mp4')
    make_video(images_folder, video_path)
    log("[Direct Sum] Video generated.")

    total_elapsed = total_timer.elapsed()
    log(f"[Direct Sum] Simulation complete. Total time: {total_elapsed:.2f} seconds.")

# -----------------------------
# Particle Mesh Method (using pyFFTW)
# -----------------------------
def leapfrog_particle_mesh(positions, velocities, dt, total_steps, save_every, method_dir):
    step_timer = Timer()
    total_timer = Timer()
    total_timer.start()

    images_folder = os.path.join(method_dir, 'plots')
    snapshots_folder = os.path.join(method_dir, 'snapshots')

    GRID_SIZE = 512  # Resolution of mesh grid

    density = pyfftw.empty_aligned((GRID_SIZE, GRID_SIZE), dtype='float64')
    potential = pyfftw.empty_aligned((GRID_SIZE, GRID_SIZE), dtype='complex128')
    fft_forward = pyfftw.builders.fft2(density)
    fft_backward = pyfftw.builders.ifft2(potential)

    grid_spacing = BOX_SIZE / GRID_SIZE

    def assign_density(positions):
        density.fill(0.0)
        factor = GRID_SIZE / BOX_SIZE
        for x, y in positions:
            ix = int(x * factor) % GRID_SIZE
            iy = int(y * factor) % GRID_SIZE
            density[ix, iy] += 1.0  # Cloud-in-cell could be implemented here

    def compute_forces():
        # Solve Poisson equation in Fourier space
        rho_k = fft_forward()
        kx = fftfreq(GRID_SIZE, d=grid_spacing) * 2.0 * np.pi
        ky = fftfreq(GRID_SIZE, d=grid_spacing) * 2.0 * np.pi
        kx, ky = np.meshgrid(kx, ky, indexing='ij')
        k2 = kx**2 + ky**2
        k2[0, 0] = 1.0  # Avoid division by zero at k=0

        potential[:, :] = -rho_k / k2
        potential[0, 0] = 0.0  # Zero mean field

        phi = fft_backward()
        phi_real = np.real(phi)

        fx = np.real(np.fft.ifft2(1j * kx * potential))
        fy = np.real(np.fft.ifft2(1j * ky * potential))

        return fx, fy

    def interpolate_forces(positions, fx_grid, fy_grid):
        forces = np.zeros_like(positions)
        factor = GRID_SIZE / BOX_SIZE
        for idx, (x, y) in enumerate(positions):
            ix = int(x * factor) % GRID_SIZE
            iy = int(y * factor) % GRID_SIZE
            forces[idx, 0] = fx_grid[ix, iy]
            forces[idx, 1] = fy_grid[ix, iy]
        return forces

    for step in range(1, total_steps + 1):
        step_timer.start()

        # Half kick
        assign_density(positions)
        fx_grid, fy_grid = compute_forces()
        forces = interpolate_forces(positions, fx_grid, fy_grid)
        velocities += 0.5 * dt * forces

        # Drift
        positions += dt * velocities
        positions %= BOX_SIZE  # Periodic BCs

        # Full kick
        assign_density(positions)
        fx_grid, fy_grid = compute_forces()
        forces = interpolate_forces(positions, fx_grid, fy_grid)
        velocities += 0.5 * dt * forces

        # Save and plot
        if step % save_every == 0:
            snapshot_file = os.path.join(snapshots_folder, f'step_{step:04d}.h5')
            save_snapshot_hdf5(snapshot_file, positions, velocities)

            plot_file = os.path.join(images_folder, f'step_{step:04d}.png')
            plot_particles(positions, plot_file, step, 'PM')

        # Logging
        eta_sec = total_timer.eta(step, total_steps)
        eta_str = time.strftime('%H:%M:%S', time.gmtime(eta_sec))

        log(f"[PM] Step {step}/{total_steps} complete. "
            f"Elapsed: {step_timer.elapsed():.2f}s. ETA: {eta_str}")

    # Generate video
    video_path = os.path.join(method_dir, 'simulation.mp4')
    make_video(images_folder, video_path)
    log("[PM] Video generated.")

    total_elapsed = total_timer.elapsed()
    log(f"[PM] Simulation complete. Total time: {total_elapsed:.2f} seconds.")

# -----------------------------
# Adaptive Mesh Method (Simplified Single-Level Refinement)
# -----------------------------
def leapfrog_adaptive_mesh(positions, velocities, dt, total_steps, save_every, method_dir):
    step_timer = Timer()
    total_timer = Timer()
    total_timer.start()

    images_folder = os.path.join(method_dir, 'plots')
    snapshots_folder = os.path.join(method_dir, 'snapshots')

    BASE_GRID = 256
    REFINED_GRID = 512  # Refinement level
    DENSITY_THRESHOLD = 2.0  # Arbitrary refinement trigger

    coarse_density = pyfftw.empty_aligned((BASE_GRID, BASE_GRID), dtype='float64')
    refined_density = pyfftw.empty_aligned((REFINED_GRID, REFINED_GRID), dtype='float64')

    coarse_potential = pyfftw.empty_aligned((BASE_GRID, BASE_GRID), dtype='complex128')
    refined_potential = pyfftw.empty_aligned((REFINED_GRID, REFINED_GRID), dtype='complex128')

    coarse_fft_forward = pyfftw.builders.fft2(coarse_density)
    coarse_fft_backward = pyfftw.builders.ifft2(coarse_potential)

    refined_fft_forward = pyfftw.builders.fft2(refined_density)
    refined_fft_backward = pyfftw.builders.ifft2(refined_potential)

    coarse_spacing = BOX_SIZE / BASE_GRID
    refined_spacing = BOX_SIZE / REFINED_GRID

    def assign_density(grid, positions, grid_size):
        grid.fill(0.0)
        factor = grid_size / BOX_SIZE
        for x, y in positions:
            ix = int(x * factor) % grid_size
            iy = int(y * factor) % grid_size
            grid[ix, iy] += 1.0

    def solve_poisson(grid, potential, fft_forward, fft_backward, grid_size, spacing):
        rho_k = fft_forward()
        kx = fftfreq(grid_size, d=spacing) * 2.0 * np.pi
        ky = fftfreq(grid_size, d=spacing) * 2.0 * np.pi
        kx, ky = np.meshgrid(kx, ky, indexing='ij')
        k2 = kx**2 + ky**2
        k2[0, 0] = 1.0  # avoid division by zero

        potential[:, :] = -rho_k / k2
        potential[0, 0] = 0.0

        phi = fft_backward()
        return np.real(phi)

    def interpolate_forces(grid_potential, grid_size):
        kx = fftfreq(grid_size, d=BOX_SIZE / grid_size) * 2.0 * np.pi
        ky = fftfreq(grid_size, d=BOX_SIZE / grid_size) * 2.0 * np.pi
        kx, ky = np.meshgrid(kx, ky, indexing='ij')

        grad_x = np.real(np.fft.ifft2(1j * kx * grid_potential))
        grad_y = np.real(np.fft.ifft2(1j * ky * grid_potential))

        return grad_x, grad_y

    def compute_forces(positions):
        assign_density(coarse_density, positions, BASE_GRID)
        coarse_phi = solve_poisson(coarse_density, coarse_potential, coarse_fft_forward, coarse_fft_backward, BASE_GRID, coarse_spacing)
        grad_x_c, grad_y_c = interpolate_forces(coarse_potential, BASE_GRID)

        # Detect refinement zones (high-density regions)
        refinement_zones = coarse_density > DENSITY_THRESHOLD

        assign_density(refined_density, positions, REFINED_GRID)
        refined_phi = solve_poisson(refined_density, refined_potential, refined_fft_forward, refined_fft_backward, REFINED_GRID, refined_spacing)
        grad_x_r, grad_y_r = interpolate_forces(refined_potential, REFINED_GRID)

        forces = np.zeros_like(positions)
        factor_c = BASE_GRID / BOX_SIZE
        factor_r = REFINED_GRID / BOX_SIZE

        for idx, (x, y) in enumerate(positions):
            ix_c = int(x * factor_c) % BASE_GRID
            iy_c = int(y * factor_c) % BASE_GRID

            ix_r = int(x * factor_r) % REFINED_GRID
            iy_r = int(y * factor_r) % REFINED_GRID

            if refinement_zones[ix_c, iy_c]:
                forces[idx, 0] = grad_x_r[ix_r, iy_r]
                forces[idx, 1] = grad_y_r[ix_r, iy_r]
            else:
                forces[idx, 0] = grad_x_c[ix_c, iy_c]
                forces[idx, 1] = grad_y_c[ix_c, iy_c]

        return forces

    for step in range(1, total_steps + 1):
        step_timer.start()

        # Half kick
        forces = compute_forces(positions)
        velocities += 0.5 * dt * forces

        # Drift
        positions += dt * velocities
        positions %= BOX_SIZE  # Periodic BCs

        # Full kick
        forces = compute_forces(positions)
        velocities += 0.5 * dt * forces

        # Save and plot
        if step % save_every == 0:
            snapshot_file = os.path.join(snapshots_folder, f'step_{step:04d}.h5')
            save_snapshot_hdf5(snapshot_file, positions, velocities)

            plot_file = os.path.join(images_folder, f'step_{step:04d}.png')
            plot_particles(positions, plot_file, step, 'Adaptive Mesh')

        # Logging
        eta_sec = total_timer.eta(step, total_steps)
        eta_str = time.strftime('%H:%M:%S', time.gmtime(eta_sec))

        log(f"[Adaptive Mesh] Step {step}/{total_steps} complete. "
            f"Elapsed: {step_timer.elapsed():.2f}s. ETA: {eta_str}")

    # Generate video
    video_path = os.path.join(method_dir, 'simulation.mp4')
    make_video(images_folder, video_path)
    log("[Adaptive Mesh] Video generated.")

    total_elapsed = total_timer.elapsed()
    log(f"[Adaptive Mesh] Simulation complete. Total time: {total_elapsed:.2f} seconds.")

# -----------------------------
# Barnes-Hut Tree Method (2D)
# -----------------------------
class QuadTreeNode:
    def __init__(self, x_min, x_max, y_min, y_max, particles_idx):
        self.x_min, self.x_max = x_min, x_max
        self.y_min, self.y_max = y_min, y_max
        self.particles_idx = particles_idx

        self.children = []
        self.center_of_mass = None
        self.total_mass = None

    def is_leaf(self):
        return len(self.children) == 0

    def subdivide(self, positions, max_particles=1):
        if len(self.particles_idx) <= max_particles:
            self.compute_mass_properties(positions)
            return

        mx = 0.5 * (self.x_min + self.x_max)
        my = 0.5 * (self.y_min + self.y_max)

        quadrants = [
            (self.x_min, mx, self.y_min, my),  # lower-left
            (mx, self.x_max, self.y_min, my),  # lower-right
            (self.x_min, mx, my, self.y_max),  # upper-left
            (mx, self.x_max, my, self.y_max)   # upper-right
        ]

        for x0, x1, y0, y1 in quadrants:
            idx = [i for i in self.particles_idx
                   if x0 <= positions[i,0] < x1 and y0 <= positions[i,1] < y1]
            if idx:
                child = QuadTreeNode(x0, x1, y0, y1, idx)
                child.subdivide(positions)
                self.children.append(child)

        self.compute_mass_properties(positions)

    def compute_mass_properties(self, positions):
        if len(self.particles_idx) == 0:
            self.center_of_mass = np.array([0.0, 0.0])
            self.total_mass = 0.0
        else:
            pts = positions[self.particles_idx]
            self.center_of_mass = np.mean(pts, axis=0)
            self.total_mass = len(self.particles_idx)

    def compute_force(self, particle_idx, positions, softening, theta):
        pos = positions[particle_idx]
        force = np.array([0.0, 0.0])

        if len(self.particles_idx) == 1 and self.particles_idx[0] == particle_idx:
            return force  # No self-force

        dx = self.center_of_mass[0] - pos[0]
        dy = self.center_of_mass[1] - pos[1]

        # Periodic BCs (minimum image convention)
        dx -= BOX_SIZE * np.round(dx / BOX_SIZE)
        dy -= BOX_SIZE * np.round(dy / BOX_SIZE)

        r2 = dx * dx + dy * dy + softening * softening
        d = max(self.x_max - self.x_min, self.y_max - self.y_min)

        if self.is_leaf() or d / np.sqrt(r2) < theta:
            inv_r3 = 1.0 / (r2 * np.sqrt(r2))
            f = self.total_mass * np.array([dx, dy]) * inv_r3
            return f
        else:
            for child in self.children:
                force += child.compute_force(particle_idx, positions, softening, theta)
            return force


def leapfrog_barnes_hut(positions, velocities, dt, total_steps, save_every, method_dir, theta=0.5):
    step_timer = Timer()
    total_timer = Timer()
    total_timer.start()

    images_folder = os.path.join(method_dir, 'plots')
    snapshots_folder = os.path.join(method_dir, 'snapshots')

    N = positions.shape[0]

    for step in range(1, total_steps + 1):
        step_timer.start()

        # Build QuadTree
        root = QuadTreeNode(0.0, BOX_SIZE, 0.0, BOX_SIZE, list(range(N)))
        root.subdivide(positions)

        # Half kick
        forces = np.zeros_like(positions)
        for i in range(N):
            forces[i] = root.compute_force(i, positions, SOFTENING, theta)
        velocities += 0.5 * dt * forces

        # Drift
        positions += dt * velocities
        positions %= BOX_SIZE

        # Full kick
        root = QuadTreeNode(0.0, BOX_SIZE, 0.0, BOX_SIZE, list(range(N)))
        root.subdivide(positions)
        forces = np.zeros_like(positions)
        for i in range(N):
            forces[i] = root.compute_force(i, positions, SOFTENING, theta)
        velocities += 0.5 * dt * forces

        # Save and plot
        if step % save_every == 0:
            snapshot_file = os.path.join(snapshots_folder, f'step_{step:04d}.h5')
            save_snapshot_hdf5(snapshot_file, positions, velocities)

            plot_file = os.path.join(images_folder, f'step_{step:04d}.png')
            plot_particles(positions, plot_file, step, 'Barnes-Hut Tree')

        eta_sec = total_timer.eta(step, total_steps)
        eta_str = time.strftime('%H:%M:%S', time.gmtime(eta_sec))

        log(f"[Barnes-Hut Tree] Step {step}/{total_steps} complete. "
            f"Elapsed: {step_timer.elapsed():.2f}s. ETA: {eta_str}")

    video_path = os.path.join(method_dir, 'simulation.mp4')
    make_video(images_folder, video_path)
    log("[Barnes-Hut Tree] Video generated.")

    total_elapsed = total_timer.elapsed()
    log(f"[Barnes-Hut Tree] Simulation complete. Total time: {total_elapsed:.2f} seconds.")

# -----------------------------
# TreePM Method (Hybrid)
# -----------------------------
def leapfrog_tree_pm(positions, velocities, dt, total_steps, save_every, method_dir, theta=0.5, rs=1.0):
    step_timer = Timer()
    total_timer = Timer()
    total_timer.start()

    images_folder = os.path.join(method_dir, 'plots')
    snapshots_folder = os.path.join(method_dir, 'snapshots')

    GRID_SIZE = 512
    density = pyfftw.empty_aligned((GRID_SIZE, GRID_SIZE), dtype='float64')
    potential = pyfftw.empty_aligned((GRID_SIZE, GRID_SIZE), dtype='complex128')
    fft_forward = pyfftw.builders.fft2(density)
    fft_backward = pyfftw.builders.ifft2(potential)

    grid_spacing = BOX_SIZE / GRID_SIZE

    def assign_density(positions):
        density.fill(0.0)
        factor = GRID_SIZE / BOX_SIZE
        for x, y in positions:
            ix = int(x * factor) % GRID_SIZE
            iy = int(y * factor) % GRID_SIZE
            density[ix, iy] += 1.0

    def compute_long_range_forces():
        rho_k = fft_forward()
        kx = fftfreq(GRID_SIZE, d=grid_spacing) * 2.0 * np.pi
        ky = fftfreq(GRID_SIZE, d=grid_spacing) * 2.0 * np.pi
        kx, ky = np.meshgrid(kx, ky, indexing='ij')
        k2 = kx**2 + ky**2
        k2[0, 0] = 1.0

        # Gaussian smoothing (long-range)
        f_gauss = np.exp(-k2 * rs**2)
        potential[:, :] = -rho_k / k2 * f_gauss
        potential[0, 0] = 0.0

        fx = np.real(np.fft.ifft2(1j * kx * potential))
        fy = np.real(np.fft.ifft2(1j * ky * potential))

        return fx, fy

    def interpolate_long_forces(positions, fx_grid, fy_grid):
        forces = np.zeros_like(positions)
        factor = GRID_SIZE / BOX_SIZE
        for idx, (x, y) in enumerate(positions):
            ix = int(x * factor) % GRID_SIZE
            iy = int(y * factor) % GRID_SIZE
            forces[idx, 0] = fx_grid[ix, iy]
            forces[idx, 1] = fy_grid[ix, iy]
        return forces

    N = positions.shape[0]

    for step in range(1, total_steps + 1):
        step_timer.start()

        # --- Long-range (PM) ---
        assign_density(positions)
        fx_grid, fy_grid = compute_long_range_forces()
        long_forces = interpolate_long_forces(positions, fx_grid, fy_grid)

        # --- Short-range (Tree) ---
        root = QuadTreeNode(0.0, BOX_SIZE, 0.0, BOX_SIZE, list(range(N)))
        root.subdivide(positions)

        short_forces = np.zeros_like(positions)
        for i in range(N):
            short_forces[i] = root.compute_force(i, positions, SOFTENING, theta)

        # Combine Forces
        total_forces = long_forces + short_forces

        # Half kick
        velocities += 0.5 * dt * total_forces

        # Drift
        positions += dt * velocities
        positions %= BOX_SIZE

        # Recompute forces after drift
        assign_density(positions)
        fx_grid, fy_grid = compute_long_range_forces()
        long_forces = interpolate_long_forces(positions, fx_grid, fy_grid)

        root = QuadTreeNode(0.0, BOX_SIZE, 0.0, BOX_SIZE, list(range(N)))
        root.subdivide(positions)
        short_forces = np.zeros_like(positions)
        for i in range(N):
            short_forces[i] = root.compute_force(i, positions, SOFTENING, theta)

        total_forces = long_forces + short_forces

        # Full kick
        velocities += 0.5 * dt * total_forces

        # Save and plot
        if step % save_every == 0:
            snapshot_file = os.path.join(snapshots_folder, f'step_{step:04d}.h5')
            save_snapshot_hdf5(snapshot_file, positions, velocities)

            plot_file = os.path.join(images_folder, f'step_{step:04d}.png')
            plot_particles(positions, plot_file, step, 'TreePM')

        eta_sec = total_timer.eta(step, total_steps)
        eta_str = time.strftime('%H:%M:%S', time.gmtime(eta_sec))

        log(f"[TreePM] Step {step}/{total_steps} complete. "
            f"Elapsed: {step_timer.elapsed():.2f}s. ETA: {eta_str}")

    video_path = os.path.join(method_dir, 'simulation.mp4')
    make_video(images_folder, video_path)
    log("[TreePM] Video generated.")

    total_elapsed = total_timer.elapsed()
    log(f"[TreePM] Simulation complete. Total time: {total_elapsed:.2f} seconds.")

import shutil
import psutil

def get_folder_size_mb(folder):
    total_size = 0
    for dirpath, _, filenames in os.walk(folder):
        for f in filenames:
            fp = os.path.join(dirpath, f)
            total_size += os.path.getsize(fp)
    return total_size / (1024 * 1024)

def run_method(method_name, func, positions_init, velocities_init, dt, total_steps, save_every):
    method_dir = os.path.join(BASE_DIR, method_name)
    log(f"Starting {method_name} simulation...")

    # Deep copy initial conditions
    positions = np.copy(positions_init)
    velocities = np.copy(velocities_init)

    start_time = time.time()
    func(positions, velocities, dt, total_steps, save_every, method_dir)
    elapsed = time.time() - start_time

    size_mb = get_folder_size_mb(method_dir)
    log(f"{method_name} completed in {elapsed:.2f} seconds. "
        f"Disk usage: {size_mb:.2f} MB")

    return elapsed, size_mb

def main():
    # Simulation parameters
    a_init = 1.0 / (1.0 + 50.0)  # z=50 initial scale factor
    a_final = 1.0  # z=0
    steps_for_structure = TOTAL_STEPS  # Adjust to suit your needs
    save_every = SAVE_EVERY
    dt = 0.01  # timestep size (adjust if needed)

    # Generate initial conditions again with Zel'dovich for safety
    np.random.seed(SEED)
    positions_init = np.random.uniform(0, BOX_SIZE, size=(N_PARTICLES, 2))
    velocities_init = np.zeros_like(positions_init)
    # For simplicity, no linear displacement this time, can add if needed.

    timings = {}
    disk_usages = {}

    # Run methods one by one
    timings['DirectSum'], disk_usages['DirectSum'] = run_method(
        'DirectSum', leapfrog_direct_sum, positions_init, velocities_init, dt, steps_for_structure, save_every)

    timings['PM'], disk_usages['PM'] = run_method(
        'PM', leapfrog_particle_mesh, positions_init, velocities_init, dt, steps_for_structure, save_every)

    timings['AdaptiveMesh'], disk_usages['AdaptiveMesh'] = run_method(
        'AdaptiveMesh', leapfrog_adaptive_mesh, positions_init, velocities_init, dt, steps_for_structure, save_every)

    timings['BarnesHutTree'], disk_usages['BarnesHutTree'] = run_method(
        'BarnesHutTree', leapfrog_barnes_hut, positions_init, velocities_init, dt, steps_for_structure, save_every)

    timings['TreePM'], disk_usages['TreePM'] = run_method(
        'TreePM', leapfrog_tree_pm, positions_init, velocities_init, dt, steps_for_structure, save_every)

    log("\n--- Summary ---")
    for method in METHODS:
        log(f"{method}: Time = {timings[method]:.2f} s, Disk usage = {disk_usages[method]:.2f} MB")

if __name__ == "__main__":
    main()
