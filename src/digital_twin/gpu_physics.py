"""
gpu_physics.py
--------------
High-performance GPU compute backend for the Digital Twin ecosystem.
Uses PyTorch to run vectorized particle kinematics (spray drift) and
agronomic terrain updates at massive scale on CUDA.
"""

import torch
import numpy as np
from typing import Tuple, List
import threading
import math

class GPUPhysicsEngine:
    def __init__(self, max_particles: int = 50000):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.max_particles = max_particles
        self.lock = threading.Lock()
        
        # Tensors for Particle System (Spray Drift)
        # xyz coordinates: [lon, lat, altitude] (altitude in meters, lon/lat mapped locally)
        self.p_pos = torch.zeros((self.max_particles, 3), dtype=torch.float32, device=self.device)
        self.p_vel = torch.zeros((self.max_particles, 3), dtype=torch.float32, device=self.device)
        self.p_age = torch.zeros(self.max_particles, dtype=torch.float32, device=self.device)
        self.p_active = torch.zeros(self.max_particles, dtype=torch.bool, device=self.device)
        self.p_tau = torch.zeros(self.max_particles, dtype=torch.float32, device=self.device)
        
        self.gravity = -9.81
        
        print(f"[GPU Physics] Initialized on device: {self.device} with max_particles: {self.max_particles}")

    def emit_particles(self, count: int, source_pos: Tuple[float, float, float], initial_velocity: Tuple[float, float, float], spread: float = 2.0):
        """
        Emit `count` new particles from a drone's position.
        source_pos: (lon, lat, alt)
        """
        with self.lock:
            if count <= 0:
                return

            inactive_indices = (~self.p_active).nonzero(as_tuple=True)[0]
            
            if len(inactive_indices) == 0:
                return # Particle buffer full
                
            emit_count = min(count, len(inactive_indices))
            indices = inactive_indices[:emit_count]
            
            # Base positions with slight randomized spatial scatter
            # Using extremely small scatter for lon/lat (since it's degrees, e.g. 1e-5)
            lon_scatter = (torch.rand(emit_count, device=self.device) - 0.5) * 1e-5 * spread
            lat_scatter = (torch.rand(emit_count, device=self.device) - 0.5) * 1e-5 * spread
            alt_scatter = (torch.rand(emit_count, device=self.device) - 0.5) * 0.5 * spread
            
            self.p_pos[indices, 0] = source_pos[0] + lon_scatter
            self.p_pos[indices, 1] = source_pos[1] + lat_scatter
            self.p_pos[indices, 2] = source_pos[2] + alt_scatter
            
            # Velocity scatter (convert X and Y from m/s to roughly degrees/s)
            vx_scatter = (torch.rand(emit_count, device=self.device) - 0.5) * spread * 1e-5
            vy_scatter = (torch.rand(emit_count, device=self.device) - 0.5) * spread * 1e-5
            vz_scatter = (torch.rand(emit_count, device=self.device) - 1.0) * spread # Bias downwards
            
            self.p_vel[indices, 0] = (initial_velocity[0] * 1e-5) + vx_scatter
            self.p_vel[indices, 1] = (initial_velocity[1] * 1e-5) + vy_scatter
            self.p_vel[indices, 2] = initial_velocity[2] + vz_scatter
            
            # Particle-specific Stokes relaxation times (log-normal droplet size distribution)
            # Volume Median Diameter (VMD) centered at 250 micrometers
            log_dp = torch.randn(emit_count, device=self.device) * 0.35 + math.log(2.5e-4)
            dp = torch.exp(log_dp)
            
            # Stokes relaxation: tau = (rho_liquid * dp^2) / (18 * mu_air)
            # rho_liquid = 1000 kg/m^3, mu_air = 1.81e-5 Pa.s
            # (1000 * dp^2) / (18 * 1.81e-5) = 3.0693e6 * dp^2
            self.p_tau[indices] = 3.0693e6 * (dp ** 2)
            
            self.p_age[indices] = 0.0
            self.p_active[indices] = True

    def update_particles(self, dt: float, wind_vector: tuple = (0.0, 0.0, 0.0)):
        """
        Step the physics simulation for all active particles using GPU tensors.
        wind_vector: (wind_lon_deg_per_sec, wind_lat_deg_per_sec, wind_alt_m_per_sec)
        """
        with self.lock:
            active = self.p_active
            if not active.any():
                return
                
            # Kinematic updates using Stokes drag with individual droplet relaxation times
            wind_tensor = torch.tensor(wind_vector, dtype=torch.float32, device=self.device)
            drag_acc = (wind_tensor - self.p_vel[active]) / self.p_tau[active].unsqueeze(1)
            
            # Gravity acceleration along Z axis
            gravity_acc = torch.zeros_like(drag_acc)
            gravity_acc[:, 2] = self.gravity
            
            # Update velocities
            self.p_vel[active] += (drag_acc + gravity_acc) * dt
            
            # 3. Update positions based on velocity
            self.p_pos[active] += self.p_vel[active] * dt
            
            # 4. Aging and death
            self.p_age[active] += dt
            
            # Kill particles that hit the ground (alt < 0) or get too old
            hit_ground = self.p_pos[:, 2] < 0.0
            too_old = self.p_age > 10.0
            self.p_active[hit_ground | too_old] = False

    def get_active_particles_numpy(self):
        with self.lock:
            active = self.p_active
            if not active.any():
                return np.zeros((0, 3), dtype=np.float32)
                
            return self.p_pos[active].cpu().numpy()

    def generate_terrain_heatmap(self, width: int = 100, height: int = 100, center_lat: float = 37.7749, center_lon: float = -122.4194, extent_deg: float = 0.01):
        """
        Generates a synthetic ground moisture/stress heatmap on the GPU.
        Returns a flat list of dicts (lon, lat, weight) for Deck.gl HeatmapLayer.
        """
        # Create a meshgrid
        x = torch.linspace(center_lon - extent_deg/2, center_lon + extent_deg/2, width, device=self.device)
        y = torch.linspace(center_lat - extent_deg/2, center_lat + extent_deg/2, height, device=self.device)
        grid_y, grid_x = torch.meshgrid(y, x, indexing='ij')
        
        # Generate some synthetic perlin-like noise using sine waves
        t = torch.tensor(1.0, device=self.device) # could be time-dependent
        noise = torch.sin(grid_x * 500) * torch.cos(grid_y * 500) + torch.sin((grid_x + grid_y) * 200)
        
        # Normalize to 0-1
        noise = (noise - noise.min()) / (noise.max() - noise.min())
        
        # Flatten for Deck.gl
        grid_x_flat = grid_x.flatten().cpu().numpy()
        grid_y_flat = grid_y.flatten().cpu().numpy()
        weight_flat = noise.flatten().cpu().numpy()
        
        # For performance in Streamlit, we shouldn't send 10,000 dicts over websocket if we can avoid it.
        # But returning a dict of lists is efficient for pandas.
        return {
            'lon': grid_x_flat,
            'lat': grid_y_flat,
            'weight': weight_flat
        }
