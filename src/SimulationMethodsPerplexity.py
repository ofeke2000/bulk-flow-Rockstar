import os
import sys
import time
import logging
import numpy as np
import h5py
import matplotlib.pyplot as plt
import imageio.v2 as imageio
from tqdm import tqdm
from numba import njit, prange
import pyfftw
from scipy.fft import fftfreq
from colossus.cosmology import cosmology

# -----------------------------
# User Settings
# -----------------------------
N_PARTICLES = 50_000
BOX_SIZE = 100.0  # arbitrary units
SOFTENING = 0.05  # softening length
TOTAL_STEPS = 1000
SAVE_EVERY = 5
TIME_STEP = 0.01
SEED = 42

# Output directory
BASE_DIR = os.path.expanduser('~/bulk-flow-Rockstar/Simulation-Methods-Test-Perplexity')

METHODS = ['PM', 'AdaptiveMesh', 'BarnesHutTree', 'TreePM']

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
    # Enable compression to reduce disk usage
    with h5py.File(filename, 'w') as f:
        f.create_dataset('positions', data=positions, compression="gzip", compression_opts=4)
        f.create_dataset('velocities', data=velocities, compression="gzip", compression_opts=4)

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

def get_folder_size_mb(folder):
    total_size = 0
    for dirpath, _, filenames in os.walk(folder):
        for f in filenames:
            fp = os.path.join(dirpath, f)
            total_size += os.path.getsize(fp)
    return total_size / (1024 * 1024)

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
# Cloud-In-Cell Density Assignment (Vectorized)
# -----------------------------
def assign_density_cic(positions, grid_size, box_size):
    """Assign particle mass to a 2D grid using Cloud-In-Cell interpolation."""
    density = np.zeros((grid_size, grid_size), dtype=np.float64)
    cell_size = box_size / grid_size
    coords = (positions / cell_size)  # particle positions in grid units
    i = np.floor(coords).astype(int) % grid_size
    f = coords - i

    for offset_x in [0, 1]:
        wx = 1 - f[:, 0] if offset_x == 0 else f[:, 0]
        ix = (i[:, 0] + offset_x) % grid_size
        for offset_y in [0, 1]:
            wy = 1 - f[:, 1] if offset_y == 0 else f[:, 1]
            iy = (i[:, 1] + offset_y) % grid_size
            w = wx * wy
            np.add.at(density, (ix, iy), w)
    return density

# -----------------------------
# FFT Poisson Solver for 2D gravity (Periodic BC)
# -----------------------------
def solve_poisson_2d(density, box_size):
    """Solve Poisson equation on 2D grid with periodic BC using Fourier method."""
    grid_size = density.shape[0]
    grid_spacing = box_size / grid_size

    rho_k = np.fft.fft2(density)
    kx = fftfreq(grid_size, d=grid_spacing) * 2.0 * np.pi
    ky = fftfreq(grid_size, d=grid_spacing) * 2.0 * np.pi
    kx, ky = np.meshgrid(kx, ky, indexing='ij')

    k2 = kx**2 + ky**2
    k2[0, 0] = 1.0  # avoid division by zero for zero freq

    potential_k = -rho_k / k2
    potential_k[0, 0] = 0.0  # zero mean of potential

    potential = np.fft.ifft2(potential_k)
    potential = np.real(potential)

    # Force = -grad(phi)
    fx_k = 1j * kx * potential_k
    fy_k = 1j * ky * potential_k
    fx = np.real(np.fft.ifft2(fx_k))
    fy = np.real(np.fft.ifft2(fy_k))

    return fx, fy

# -----------------------------
# Interpolate Forces from Grid to Particles (CIC)
# -----------------------------
def interpolate_forces_cic(positions, fx_grid, fy_grid, box_size):
    grid_size = fx_grid.shape[0]
    cell_size = box_size / grid_size
    coords = (positions / cell_size)
    i = np.floor(coords).astype(int) % grid_size
    f = coords - i

    forces = np.zeros_like(positions)
    for offset_x in [0,1]:
        wx = 1 - f[:, 0] if offset_x==0 else f[:, 0]
        ix = (i[:,0] + offset_x) % grid_size
        for offset_y in [0,1]:
            wy = 1 - f[:, 1] if offset_y==0 else f[:, 1]
            iy = (i[:,1] + offset_y) % grid_size
            w = wx * wy
            forces[:,0] += w * fx_grid[ix, iy]
            forces[:,1] += w * fy_grid[ix, iy]
    return forces

# -----------------------------
# Leapfrog Integrator with PM Method
# -----------------------------
def leapfrog_particle_mesh(positions, velocities, dt, total_steps, save_every, method_dir):
    step_timer = Timer()
    total_timer = Timer()
    total_timer.start()

    images_folder = os.path.join(method_dir, 'plots')
    snapshots_folder = os.path.join(method_dir, 'snapshots')

    GRID_SIZE = 512

    for step in range(1, total_steps + 1):
        step_timer.start()

        density = assign_density_cic(positions, GRID_SIZE, BOX_SIZE)
        fx_grid, fy_grid = solve_poisson_2d(density, BOX_SIZE)
        forces = interpolate_forces_cic(positions, fx_grid, fy_grid, BOX_SIZE)

        # Half kick
        velocities += 0.5 * dt * forces

        # Drift
        positions += dt * velocities
        positions %= BOX_SIZE

        # Full kick
        density = assign_density_cic(positions, GRID_SIZE, BOX_SIZE)
        fx_grid, fy_grid = solve_poisson_2d(density, BOX_SIZE)
        forces = interpolate_forces_cic(positions, fx_grid, fy_grid, BOX_SIZE)
        velocities += 0.5 * dt * forces

        if step % save_every == 0:
            snapshot_file = os.path.join(snapshots_folder, f'step_{step:04d}.h5')
            save_snapshot_hdf5(snapshot_file, positions, velocities)
            plot_file = os.path.join(images_folder, f'step_{step:04d}.png')
            plot_particles(positions, plot_file, step, 'PM')

        eta_sec = total_timer.eta(step, total_steps)
        eta_str = time.strftime('%H:%M:%S', time.gmtime(eta_sec))
        log(f"[PM] Step {step}/{total_steps} complete. Elapsed: {step_timer.elapsed():.2f}s. ETA: {eta_str}")

    video_path = os.path.join(method_dir, 'simulation.mp4')
    make_video(images_folder, video_path)
    log("[PM] Video generated.")

    total_elapsed = total_timer.elapsed()
    log(f"[PM] Simulation complete. Total time: {total_elapsed:.2f} seconds.")

# -----------------------------
# Adaptive Mesh Method (Simplified with CIC and Poisson)
# -----------------------------
def leapfrog_adaptive_mesh(positions, velocities, dt, total_steps, save_every, method_dir):
    step_timer = Timer()
    total_timer = Timer()
    total_timer.start()

    images_folder = os.path.join(method_dir, 'plots')
    snapshots_folder = os.path.join(method_dir, 'snapshots')

    BASE_GRID = 256
    REFINED_GRID = 512
    DENSITY_THRESHOLD = 2.0  # tuning parameter

    def assign_density(grid, positions, grid_size):
        return assign_density_cic(positions, grid_size, BOX_SIZE)

    def solve_poisson(grid):
        return solve_poisson_2d(grid, BOX_SIZE)

    def compute_forces(positions):
        coarse_density = assign_density_cic(positions, BASE_GRID, BOX_SIZE)
        refined_density = assign_density_cic(positions, REFINED_GRID, BOX_SIZE)

        coarse_phi = np.fft.fft2(coarse_density)
        refined_phi = np.fft.fft2(refined_density)

        coarse_potential = np.zeros_like(coarse_phi, dtype=complex)
        refined_potential = np.zeros_like(refined_phi, dtype=complex)

        # Compute potential on coarse grid
        grid_spacing_c = BOX_SIZE / BASE_GRID
        kx = fftfreq(BASE_GRID, d=grid_spacing_c) * 2.0 * np.pi
        ky = fftfreq(BASE_GRID, d=grid_spacing_c) * 2.0 * np.pi
        kx, ky = np.meshgrid(kx, ky, indexing='ij')
        k2_c = kx**2 + ky**2
        k2_c[0,0] = 1.0
        coarse_potential = -coarse_phi / k2_c
        coarse_potential[0,0] = 0.0

        # Compute potential on refined grid
        grid_spacing_r = BOX_SIZE / REFINED_GRID
        kx_r = fftfreq(REFINED_GRID, d=grid_spacing_r) * 2.0 * np.pi
        ky_r = fftfreq(REFINED_GRID, d=grid_spacing_r) * 2.0 * np.pi
        kx_r, ky_r = np.meshgrid(kx_r, ky_r, indexing='ij')
        k2_r = kx_r**2 + ky_r**2
        k2_r[0,0] = 1.0
        refined_potential = -refined_phi / k2_r
        refined_potential[0,0] = 0.0

        # Inverse FFT to get potential real space
        phi_coarse = np.real(np.fft.ifft2(coarse_potential))
        phi_refined = np.real(np.fft.ifft2(refined_potential))

        # Compute gradients (forces)
        grad_x_c = np.real(np.fft.ifft2(1j * kx * coarse_potential))
        grad_y_c = np.real(np.fft.ifft2(1j * ky * coarse_potential))
        grad_x_r = np.real(np.fft.ifft2(1j * kx_r * refined_potential))
        grad_y_r = np.real(np.fft.ifft2(1j * ky_r * refined_potential))

        # Determine refinement zones
        refinement_zones = coarse_density > DENSITY_THRESHOLD

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
        forces = compute_forces(positions)
        velocities += 0.5 * dt * forces

        positions += dt * velocities
        positions %= BOX_SIZE

        forces = compute_forces(positions)
        velocities += 0.5 * dt * forces

        if step % save_every == 0:
            snapshot_file = os.path.join(snapshots_folder, f'step_{step:04d}.h5')
            save_snapshot_hdf5(snapshot_file, positions, velocities)
            plot_file = os.path.join(images_folder, f'step_{step:04d}.png')
            plot_particles(positions, plot_file, step, 'Adaptive Mesh')

        eta_sec = total_timer.eta(step, total_steps)
        eta_str = time.strftime('%H:%M:%S', time.gmtime(eta_sec))
        log(f"[Adaptive Mesh] Step {step}/{total_steps} complete. Elapsed: {step_timer.elapsed():.2f}s. ETA: {eta_str}")

    video_path = os.path.join(method_dir, 'simulation.mp4')
    make_video(images_folder, video_path)
    log("[Adaptive Mesh] Video generated.")

    total_elapsed = total_timer.elapsed()
    log(f"[Adaptive Mesh] Simulation complete. Total time: {total_elapsed:.2f} seconds.")

# -----------------------------
# Barnes-Hut Tree (2D)
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
                   if x0 <= positions[i, 0] < x1 and y0 <= positions[i, 1] < y1]
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
            # Apply minimum image convention for center of mass
            pts_wrapped = pts % BOX_SIZE
            self.center_of_mass = np.mean(pts_wrapped, axis=0)
            self.total_mass = len(self.particles_idx)

    def compute_force(self, particle_idx, positions, softening, theta):
        pos = positions[particle_idx]
        force = np.zeros(2, dtype=np.float64)

        if len(self.particles_idx) == 1 and self.particles_idx[0] == particle_idx:
            return force  # No self-force

        dx = self.center_of_mass[0] - pos[0]
        dy = self.center_of_mass[1] - pos[1]

        # Periodic BC minimum image
        dx -= BOX_SIZE * np.round(dx / BOX_SIZE)
        dy -= BOX_SIZE * np.round(dy / BOX_SIZE)

        r2 = dx*dx + dy*dy + softening*softening
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

        root = QuadTreeNode(0.0, BOX_SIZE, 0.0, BOX_SIZE, list(range(N)))
        root.subdivide(positions)

        forces = np.zeros_like(positions)
        for i in range(N):
            forces[i] = root.compute_force(i, positions, SOFTENING, theta)
        velocities += 0.5 * dt * forces

        positions += dt * velocities
        positions %= BOX_SIZE

        root = QuadTreeNode(0.0, BOX_SIZE, 0.0, BOX_SIZE, list(range(N)))
        root.subdivide(positions)
        forces.fill(0)
        for i in range(N):
            forces[i] = root.compute_force(i, positions, SOFTENING, theta)
        velocities += 0.5 * dt * forces

        if step % save_every == 0:
            snapshot_file = os.path.join(snapshots_folder, f'step_{step:04d}.h5')
            save_snapshot_hdf5(snapshot_file, positions, velocities)
            plot_file = os.path.join(images_folder, f'step_{step:04d}.png')
            plot_particles(positions, plot_file, step, 'Barnes-Hut Tree')

        eta_sec = total_timer.eta(step, total_steps)
        eta_str = time.strftime('%H:%M:%S', time.gmtime(eta_sec))
        log(f"[Barnes-Hut Tree] Step {step}/{total_steps} complete. Elapsed: {step_timer.elapsed():.2f}s. ETA: {eta_str}")

    video_path = os.path.join(method_dir, 'simulation.mp4')
    make_video(images_folder, video_path)
    log("[Barnes-Hut Tree] Video generated.")

    total_elapsed = total_timer.elapsed()
    log(f"[Barnes-Hut Tree] Simulation complete. Total time: {total_elapsed:.2f} seconds.")

# -----------------------------
# TreePM Method (Hybrid PM + Barnes-Hut)
# -----------------------------
def leapfrog_tree_pm(positions, velocities, dt, total_steps, save_every, method_dir, theta=0.5, rs=1.0):
    step_timer = Timer()
    total_timer = Timer()
    total_timer.start()

    images_folder = os.path.join(method_dir, 'plots')
    snapshots_folder = os.path.join(method_dir, 'snapshots')

    GRID_SIZE = 512

    for step in range(1, total_steps + 1):
        step_timer.start()

        density = assign_density_cic(positions, GRID_SIZE, BOX_SIZE)

        # Long-range force via Particle-Mesh with Gaussian smoothing
        potential_grid = np.fft.fft2(density)
        grid_spacing = BOX_SIZE / GRID_SIZE
        kx = fftfreq(GRID_SIZE, d=grid_spacing) * 2.0 * np.pi
        ky = fftfreq(GRID_SIZE, d=grid_spacing) * 2.0 * np.pi
        kx, ky = np.meshgrid(kx, ky, indexing='ij')

        k2 = kx**2 + ky**2
        k2[0,0] = 1.0

        f_gauss = np.exp(-k2 * rs**2)
        potential_k = -potential_grid / k2 * f_gauss
        potential_k[0,0] = 0.0

        fx_k = 1j * kx * potential_k
        fy_k = 1j * ky * potential_k

        fx_grid = np.real(np.fft.ifft2(fx_k))
        fy_grid = np.real(np.fft.ifft2(fy_k))

        long_forces = interpolate_forces_cic(positions, fx_grid, fy_grid, BOX_SIZE)

        # Short-range force via Barnes-Hut
        N = positions.shape[0]
        root = QuadTreeNode(0.0, BOX_SIZE, 0.0, BOX_SIZE, list(range(N)))
        root.subdivide(positions)
        short_forces = np.zeros_like(positions)
        for i in range(N):
            short_forces[i] = root.compute_force(i, positions, SOFTENING, theta)

        total_forces = long_forces + short_forces

        velocities += 0.5 * dt * total_forces

        positions += dt * velocities
        positions %= BOX_SIZE

        # Recompute forces after drift
        density = assign_density_cic(positions, GRID_SIZE, BOX_SIZE)
        potential_grid = np.fft.fft2(density)
        potential_k = -potential_grid / k2 * f_gauss
        potential_k[0,0] = 0.0

        fx_k = 1j * kx * potential_k
        fy_k = 1j * ky * potential_k
        fx_grid = np.real(np.fft.ifft2(fx_k))
        fy_grid = np.real(np.fft.ifft2(fy_k))
        long_forces = interpolate_forces_cic(positions, fx_grid, fy_grid, BOX_SIZE)

        root = QuadTreeNode(0.0, BOX_SIZE, 0.0, BOX_SIZE, list(range(N)))
        root.subdivide(positions)
        short_forces.fill(0)
        for i in range(N):
            short_forces[i] = root.compute_force(i, positions, SOFTENING, theta)
        total_forces = long_forces + short_forces

        velocities += 0.5 * dt * total_forces

        if step % save_every == 0:
            snapshot_file = os.path.join(snapshots_folder, f'step_{step:04d}.h5')
            save_snapshot_hdf5(snapshot_file, positions, velocities)
            plot_file = os.path.join(images_folder, f'step_{step:04d}.png')
            plot_particles(positions, plot_file, step, 'TreePM')

        eta_sec = total_timer.eta(step, total_steps)
        eta_str = time.strftime('%H:%M:%S', time.gmtime(eta_sec))
        log(f"[TreePM] Step {step}/{total_steps} complete. Elapsed: {step_timer.elapsed():.2f}s. ETA: {eta_str}")

    video_path = os.path.join(method_dir, 'simulation.mp4')
    make_video(images_folder, video_path)
    log("[TreePM] Video generated.")

    total_elapsed = total_timer.elapsed()
    log(f"[TreePM] Simulation complete. Total time: {total_elapsed:.2f} seconds.")

# -----------------------------
# Run Simulation Wrapper
# -----------------------------
def run_method(method_name, func, positions_init, velocities_init, dt, total_steps, save_every):
    method_dir = os.path.join(BASE_DIR, method_name)
    ensure_dir(method_dir)
    ensure_dir(os.path.join(method_dir, 'snapshots'))
    ensure_dir(os.path.join(method_dir, 'plots'))

    log(f"Starting {method_name} simulation...")

    positions = np.copy(positions_init)
    velocities = np.copy(velocities_init)

    start_time = time.time()
    try:
        func(positions, velocities, dt, total_steps, save_every, method_dir)
    except Exception as e:
        log(f"[{method_name}] ERROR: {e}")
        return float('nan'), float('nan')

    elapsed = time.time() - start_time
    size_mb = get_folder_size_mb(method_dir)

    log(f"{method_name} completed in {elapsed:.2f} seconds. Disk usage: {size_mb:.2f} MB")

    return elapsed, size_mb

# -----------------------------
# Main Execution
# -----------------------------
def main():
    np.random.seed(SEED)  # reproducible initial conditions

    # Zel'dovich approx for initial positions - simplified here by Gaussian displacements
    positions_init = np.random.uniform(0, BOX_SIZE, size=(N_PARTICLES, 2))
    velocities_init = np.zeros_like(positions_init)

    timings = {}
    disk_usages = {}

    METHODS_SEQUENCE = [
        # ('DirectSum', leapfrog_direct_sum),  # Uncomment and implement with njit if desired
        ('PM', leapfrog_particle_mesh),
        ('AdaptiveMesh', leapfrog_adaptive_mesh),
        ('BarnesHutTree', leapfrog_barnes_hut),
        ('TreePM', leapfrog_tree_pm),
    ]

    for method_name, func in METHODS_SEQUENCE:
        elapsed, size_mb = run_method(
            method_name, func, positions_init, velocities_init, TIME_STEP, TOTAL_STEPS, SAVE_EVERY)
        timings[method_name] = elapsed
        disk_usages[method_name] = size_mb

    log("\n--- Summary ---")
    for method_name in timings:
        log(f"{method_name}: Time = {timings[method_name]:.2f} s, Disk usage = {disk_usages[method_name]:.2f} MB")

if __name__ == "__main__":
    main()
