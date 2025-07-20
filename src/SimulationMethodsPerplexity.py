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
N_PARTICLES = 10_000
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

###########################################################################################
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
###################################################################################################

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

class BHNode:
    def __init__(self, center, size):
        self.center = np.array(center, dtype=np.float64)
        self.size = float(size)
        self.mass = 0.0
        self.com = np.zeros(2)
        self.children = [None, None, None, None]
        self.p_idx = None  # index if leaf (better for memory)

    def is_leaf(self):
        return all(child is None for child in self.children)


def get_quadrant(center, pos):
    dx = pos[0] > center[0]
    dy = pos[1] > center[1]
    return int(dx) + 2 * int(dy)


def insert_particle(node, pos, m, p_idx, positions, masses, indices):
    if node.is_leaf():
        if node.p_idx is None:
            node.p_idx = p_idx
            node.mass = m
            node.com[:] = pos
            return
        else:
            subdivide(node)
            # Reinsert existing
            rein_pos = positions[node.p_idx]
            rein_m = masses[node.p_idx]
            quadrant = get_quadrant(node.center, rein_pos)
            if node.children[quadrant] is None:
                node.children[quadrant] = make_child(node, quadrant)
            insert_particle(node.children[quadrant], rein_pos, rein_m, node.p_idx, positions, masses, indices)
            node.p_idx = None

    quadrant = get_quadrant(node.center, pos)
    if node.children[quadrant] is None:
        node.children[quadrant] = make_child(node, quadrant)
    insert_particle(node.children[quadrant], pos, m, p_idx, positions, masses, indices)

    node.mass += m
    node.com = (node.com*(node.mass-m) + m*pos) / node.mass


def make_child(node, quadrant):
    offset = np.array([
        -0.5 if quadrant in [0, 2] else 0.5,
        -0.5 if quadrant in [0, 1] else 0.5
    ]) * node.size
    return BHNode(node.center + offset, node.size/2)


def subdivide(node):
    node.children = [None, None, None, None]


def build_bh_tree(positions, masses, whole_box_size):
    root = BHNode((whole_box_size/2, whole_box_size/2), whole_box_size/2)
    indices = np.arange(len(positions))
    for idx in indices:
        insert_particle(root, positions[idx], masses[idx], idx, positions, masses, indices)
    return root

def minimum_image(dx, box_size):
    return dx - np.round(dx / box_size) * box_size

def compute_force_from_tree(node, pos, G, theta, epsilon, box_size, positions):
    # Recursively evaluate force from current node to single particle at pos
    if node.mass == 0.0 or (node.is_leaf() and node.p_idx is not None and np.allclose(positions[node.p_idx], pos)):
        return np.zeros(2)
    dx = minimum_image(node.com - pos, box_size)
    r = np.linalg.norm(dx) + epsilon
    if node.is_leaf() or (node.size / r) < theta:
        return G * node.mass * dx / (r**3)
    else:
        force = np.zeros(2)
        for child in node.children:
            if child is not None:
                force += compute_force_from_tree(child, pos, G, theta, epsilon, box_size, positions)
        return force

# ---- Leapfrog Barnes-Hut Simulation ----

def leapfrog_barnes_hut(positions, velocities, dt, total_steps, save_every, output_dir,
                        box_size, softening, G=1.0, theta=0.7):
    N = positions.shape[0]
    masses = np.full(N, 1.0 / N)
    os.makedirs(output_dir, exist_ok=True)

    for step in range(1, total_steps+1):
        root = build_bh_tree(positions, masses, box_size)
        forces = np.zeros_like(positions)
        # Compute all forces
        for i in range(N):
            forces[i] = compute_force_from_tree(root, positions[i], G, theta, softening, box_size, positions)
        # Half kick
        velocities += 0.5 * dt * forces
        positions += dt * velocities
        positions %= box_size

        # Recompute forces after drift
        root = build_bh_tree(positions, masses, box_size)
        for i in range(N):
            forces[i] = compute_force_from_tree(root, positions[i], G, theta, softening, box_size, positions)
        velocities += 0.5 * dt * forces

        if (step % save_every) == 0:
            save_snapshot(step, positions, velocities, output_dir)
            plot_particles(step, positions, output_dir)

        if step % 10 == 0 or step == total_steps:
            print(f'Barnes-Hut step {step}/{total_steps} complete.')

    generate_heatmap_video(output_dir)
    log('[BarnesHutTree] Simulation complete.')

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
        ('DirectSum', leapfrog_direct_sum),  # Uncomment and implement with njit if desired
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
