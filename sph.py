import numpy as np
import matplotlib.pyplot as plt
import time
from itertools import count
import taichi as ti
import math
from math import ceil

@ti.data_oriented
class Particles:
    
    def __init__(self, np_positions, np_is_wall, canvas_width, canvas_height, cell_width, cell_height, 
        radius, bounds):
        # np_positions: np.array
        self.dim = 3
        self.max_num_particles_per_cell = 100
        self.max_num_neighbors = 100
        self.np_positions = np_positions
        self.np_is_wall = np_is_wall
        self.num_particles = self.np_positions.shape[0]
        self.canvas_width = canvas_width
        self.canvas_height = canvas_height
        self.positions = ti.Vector(self.dim, dt=ti.f32)
        self.new_positions = ti.Vector(self.dim, dt=ti.f32)
        self.new_velocities = ti.Vector(self.dim, dt=ti.f32)
        
        self.mid_positions = ti.Vector(self.dim, dt=ti.f32)
        self.forces = ti.Vector(self.dim, dt=ti.f32) 
        self.velocities = ti.Vector(self.dim, dt=ti.f32)
        self.mid_velocities = ti.Vector(self.dim, dt=ti.f32)
        self.densities = ti.field(ti.f32)
        self.mid_densities = ti.field(ti.f32)
        self.new_densities = ti.field(ti.f32)
        self.tmp_densities = ti.field(ti.f32)
        self.is_wall = ti.field(ti.i32)
        # stores self.max_num_particles_per_cell particle ids
        self.num_particles_in_cell = ti.field(ti.i32)
        self.cells = ti.field(dtype=ti.i32) 
        ti.root.dense(ti.i, self.num_particles).place(self.positions,self.new_positions,self.new_velocities, self.velocities, self.densities, 
            self.mid_positions, self.mid_velocities, self.mid_densities, self.forces, self.new_densities, self.tmp_densities,
            self.is_wall)
        self.radius = radius
        self.smooth_len_h = radius * 1.3 # 0.08 # 2.6
        self.ncell_w = ceil(1.0 / (2.0 * self.smooth_len_h))
        self.ncell_h = ceil(1.0 / (2.0 * self.smooth_len_h))
        ti.root.dense(ti.i, self.ncell_w).dense(ti.j, self.ncell_h).place(self.num_particles_in_cell)
        ti.root.dense(ti.i, self.ncell_w).dense(ti.j, self.ncell_h).dense(ti.k, self.max_num_particles_per_cell).place(self.cells)
        self.particle_neighbors = ti.field(dtype=ti.i32)
        self.particle_num_neighbors = ti.field(ti.i32)
        ti.root.dense(ti.i, self.num_particles).dense(ti.j, self.max_num_neighbors).place(self.particle_neighbors)
        ti.root.dense(ti.i, self.num_particles).place(self.particle_num_neighbors)
        self.n_iter = ti.field(ti.f32,shape=())
        
        self.init_density = 1000.0
        self.init_fields(self.np_positions, self.np_is_wall)
        self.gravity = ti.Vector([0.0, -9.8 / 100.0])
        # self.delta_t = 0.001
       
        self.volume = math.pi * self.radius * self.radius
        # self.volume = 4.0 / 3.0 * math.pi * self.radius * self.radius * self.radius

        self.left_bound = bounds["left_bound"]
        self.right_bound = bounds["right_bound"]
        self.lower_bound = bounds["lower_bound"]        
        
        self.alpha = 1.0
        self.sound_speed = 20.0
        self.epsilon = 0.01
        self.gamma = 7.0
        self.kappa = 0.2
        self.delta_t = 0.0003 # 0.1 * self.smooth_len_h / self.sound_speed

    @ti.kernel
    def init_fields(self, _position: ti.ext_arr(), _is_wall: ti.ext_arr()): # np_positions: np.
        for i in range(self.num_particles):
            self.densities[i] = self.init_density
            self.is_wall[i] = _is_wall[i]
            for j in ti.static(range(self.dim)):
                self.positions[i][j] = _position[i, j]
                self.velocities[i][j] = 0
            # self.velocities[i] = ti.Vector([ti.random(), ti.random()]) * 0.1
        # # print(self.positions)
        for i, j in self.num_particles_in_cell:
            self.num_particles_in_cell[i, j] = 0
        for i in self.particle_num_neighbors:
            self.particle_num_neighbors[i] = 0
        

    def get_np_fluid_positions(self):
        # for i in range(self.num_particles):
        #     for j in range(self.dim):
        #         self.np_positions[i, j] = self.positions[i][j]
        self.np_positions = self.positions.to_numpy()
        self.np_positions = self.np_positions[self.np_is_wall == 0]
        return self.np_positions

    def get_np_wall_positions(self):
        self.np_positions = self.positions.to_numpy()
        self.np_positions = self.np_positions[self.np_is_wall == 1]
        return self.np_positions
    @ti.func
    def get_grid(self, position):
        grid_id = (position * ti.Vector([self.ncell_w, self.ncell_h])).cast(int)
        grid_id = min(grid_id, ti.Vector([self.ncell_w-1, self.ncell_h-1]))
        grid_id = max(grid_id, ti.Vector([0, 0]))
        return grid_id

    @ti.func
    def reset_cells_neighbors(self):
        for i, j in self.num_particles_in_cell:
            self.num_particles_in_cell[i, j] = 0
        for i in self.particle_num_neighbors:
            self.particle_num_neighbors[i] = 0

    @ti.func 
    def set_cells(self):
        for i in self.positions:
            grid_id = self.get_grid(self.positions[i])
            grid_r, grid_c = grid_id[0], grid_id[1]
            idx = ti.atomic_add(self.num_particles_in_cell[grid_r, grid_c], 1)
            if idx < self.max_num_particles_per_cell:
                self.cells[grid_r, grid_c, idx] = i
            # self.num_particles_in_cell[grid_r, grid_c] += 1

    @ti.func
    def cubicKernel(self, r, h):
        # https://github.com/erizmr/SPH_Taichi
        # value of cubic spline smoothing kernel
        k = 10. / (7. * np.pi * h ** 2)
        q = r / h
        # assert q >= 0.0
        res = ti.cast(0.0, ti.f32)
        if q <= 1.0:
            res = k * (1 - 1.5 * q ** 2 + 0.75 * q ** 3)
        elif q < 2.0:
            res = k * 0.25 * (2 - q) ** 3
        return res

    @ti.func
    def cubicKernelDerivative(self, r, h):
        # https://github.com/erizmr/SPH_Taichi
        # derivative of cubcic spline smoothing kernel
        k = 10. / (7. * np.pi * h ** 2)
        q = r / h
        # assert q > 0.0
        res = ti.cast(0.0, ti.f32)
        if q < 1.0:
            res = (k / h) * (-3 * q + 2.25 * q ** 2)
        elif q < 2.0:
            res = -0.75 * (k / h) * (2 - q) ** 2
        return res

    @ti.func
    def get_grad_W_ab(self, pid, neighbor_id):
        r_ab = self.positions[pid] - self.positions[neighbor_id]
        r_ab_norm = r_ab.norm()
        grad_ab = ti.Vector([0.0,0.0])
        if r_ab_norm != 0.0:
            grad_ab = self.cubicKernelDerivative(r_ab_norm, self.smooth_len_h) * r_ab / r_ab_norm
        return grad_ab

    @ti.func
    def dvelocity_dt(self, pid):
        dv_dt = ti.Vector([0.0, 0.0])
        dv_dt += self.dvelocity_dt_momentum(pid)
        dv_dt += self.dvelocity_dt_viscocity(pid)
        dv_dt += self.dvelocity_dt_tension(pid)
        return dv_dt

    @ti.func
    def get_pressure(self, pid):
        B = self.init_density * self.sound_speed * self.sound_speed / self.gamma
        P = B * (pow(self.densities[pid]/self.init_density, self.gamma) - 1.0)
        return P

    @ti.func
    def dvelocity_dt_momentum(self, pid):
        dv_dt = self.gravity #-0.098 * (self.positions[pid] - ti.Vector([0.5,0.5])) #.normalized()
        for neighbor_idx in range(ti.static(self.max_num_neighbors)):
            if neighbor_idx < self.particle_num_neighbors[pid]:
                neighbor_id = self.particle_neighbors[pid, neighbor_idx]
                P_a, P_b = self.get_pressure(pid), self.get_pressure(neighbor_id)
                rho_a_square = self.densities[pid] * self.densities[pid]
                rho_b_square = self.densities[neighbor_id] * self.densities[neighbor_id]
                m_b = self.init_density * self.volume
                grad_ab = self.get_grad_W_ab(pid, neighbor_id)
                denom = (P_a / rho_a_square + P_b / rho_b_square)
                if denom != 0.0:
                    dv_dt -= m_b * denom * grad_ab
        self.forces[pid] = dv_dt - self.gravity
        return dv_dt

    @ti.func
    def dvelocity_dt_viscocity(self, pid): #pid: particle id
        dv_dt = ti.Vector([0.0, 0.0])
        for neighbor_idx in range(ti.static(self.max_num_neighbors)):
            if neighbor_idx < self.particle_num_neighbors[pid]:
                neighbor_id = self.particle_neighbors[pid, neighbor_idx]
                m_b = self.volume * self.densities[neighbor_id]
                v_ab =  self.velocities[pid] - self.velocities[neighbor_id]
                r_ab = self.positions[pid] - self.positions[neighbor_id]
                if v_ab.dot(r_ab) < 0:
                    r_ab_norm = r_ab.norm()
                    v = 2.0 * self.alpha * self.smooth_len_h * self.sound_speed / (self.densities[pid] + self.densities[neighbor_id]) 
                    tmp = v * v_ab.dot(r_ab) / (r_ab_norm * r_ab_norm + self.epsilon * self.smooth_len_h * self.smooth_len_h)
                    grad_ab = self.get_grad_W_ab(pid, neighbor_id)
                    dv_dt += m_b  *  tmp * grad_ab
                    # if r_ab_norm < 1.5 * self.smooth_len_h and r_ab_norm > 0.0:  # TODO: why?
                    # dv_dt += r_ab / r_ab_norm * ti.min(1.0 / r_ab_norm, 10.0) * 0.01
        return dv_dt

    @ti.func
    def dvelocity_dt_tension(self, pid):
        dv_dt = ti.Vector([0.0, 0.0])
        for neighbor_idx in range(ti.static(self.max_num_neighbors)):
            if neighbor_idx < self.particle_num_neighbors[pid]:
                neighbor_id = self.particle_neighbors[pid, neighbor_idx]
                r_ab = self.positions[pid] - self.positions[neighbor_id]
                m_b = self.volume * self.densities[neighbor_id]
                dv_dt = m_b * self.cubicKernel(r_ab.norm(), self.smooth_len_h) * r_ab
        m_a = self.volume * self.densities[pid]
        dv_dt *= -self.kappa / m_a
        return dv_dt      

    @ti.func
    def ddensity_dt(self, pid):
        drho_dt = 0.0
        for neighbor_idx in range(ti.static(self.max_num_neighbors)):
            if neighbor_idx < self.particle_num_neighbors[pid]:
                neighbor_id = self.particle_neighbors[pid, neighbor_idx]
                v_ab =  self.velocities[pid] - self.velocities[neighbor_id]
                grad_ab = self.get_grad_W_ab(pid, neighbor_id)
                m_b = self.volume * self.densities[neighbor_id]
                drho_dt += m_b * v_ab.dot(grad_ab)
        return drho_dt
    
    @ti.func
    def set_particle_neighbors(self, pid):
        grid_id = self.get_grid(self.positions[pid])
        for offset in ti.static(ti.grouped(ti.ndrange((-1, 2), (-1, 2)))):      
            current_grid = grid_id + offset
            grid_r, grid_c = current_grid[0], current_grid[1]
            grid_r = ti.min(ti.max(0, grid_r), self.ncell_w - 1)
            grid_c = ti.min(ti.max(0, grid_c),self.ncell_h - 1)
            # if grid_r < 0 or grid_c < 0 or grid_r >= self.ncell_w or grid_c >= self.ncell_h:
            for i in range(ti.static(self.max_num_particles_per_cell)):
                if i < self.num_particles_in_cell[grid_r, grid_c] and self.cells[grid_r, grid_c, i] != pid:
                    neighbor_id = self.cells[grid_r, grid_c, i]
                    if self.particle_num_neighbors[pid] < self.max_num_neighbors and \
                         (self.positions[pid] - self.positions[neighbor_id]).norm() < 2.0 * self.smooth_len_h:
                        self.particle_neighbors[pid, self.particle_num_neighbors[pid]] = neighbor_id
                        self.particle_num_neighbors[pid] += 1
    
    @ti.func
    def set_all_particles_neighbors(self):
        for pid in self.positions:
            self.set_particle_neighbors(pid)

    def update(self):
        self.update1()
        self.update2()

    @ti.kernel
    def update1(self):
        self.reset_cells_neighbors()
        self.set_cells()
        self.set_all_particles_neighbors()
        for pid in self.positions:
            # test
            # self.positions[pid] += self.delta_t * self.velocities[pid]
            # self.velocities[pid] += self.delta_t * self.gravity 
            # test
            # grid_id = self.get_grid(self.positions[pid])
            # grid_r, grid_c = grid_id[0], grid_id[1]
            dv_dt = self.dvelocity_dt(pid)
            # self.forces[pid] = dv_dt
            self.new_positions[pid] = self.positions[pid] + self.delta_t * self.velocities[pid]
            self.new_velocities[pid] = self.velocities[pid] + self.delta_t * dv_dt
            # self.mid_velocities[pid] = self.velocities[pid] + self.delta_t / 2.0 * dv_dt
            
            # self.mid_positions[pid] = self.positions[pid] + self.delta_t / 2.0 * self.mid_velocities[pid]
            ddensity = self.ddensity_dt(pid)
            self.tmp_densities[pid] = ddensity
            self.new_densities[pid] = self.densities[pid] + self.delta_t * ddensity

    @ti.kernel
    def update2(self):
        # TODO: add wall
        for pid in self.positions:
            if self.is_wall[pid] == 0:
                self.velocities[pid] = self.new_velocities[pid]
                self.positions[pid] = self.new_positions[pid]
                self.densities[pid] = self.new_densities[pid]
                # collide with walls
                if self.positions[pid][1] < self.lower_bound:
                    if self.velocities[pid][1] < 0.0:
                        self.velocities[pid][1] += -1.5  * self.velocities[pid][1]
                    self.positions[pid][1] = self.lower_bound
                if self.positions[pid][0] < self.left_bound:
                    if self.velocities[pid][0] < 0.0:
                        self.velocities[pid][0] += -1.5  * self.velocities[pid][0]
                    self.positions[pid][0] = self.left_bound
                if self.positions[pid][0] >= self.right_bound:
                    if self.velocities[pid][0] > 0.0:
                        self.velocities[pid][0] += -1.5  * self.velocities[pid][0]
                    self.positions[pid][0] = self.right_bound
        self.n_iter[None] +=1         
            
def main():
    ti.init(arch=ti.cpu)
    # ti.init(debug=True)
    canvas_width, canvas_height, cell_width, cell_height = 512, 512, 10, 10
    positions = []
    is_wall = []
    half_w = cell_width // 2
    half_h = cell_height // 2
    # ball i: x, y: y, j:z
    for i in range(half_w * 3 + 100, canvas_width - half_w * 3 - 100, cell_width):
    # for _ in range(1):
        # i = 200 
        for y in range(half_w * 3 + 100, canvas_width - half_w * 3 - 100, cell_width):
            for j in range(half_h * 3, canvas_height - half_h, cell_height):
                positions.append([(i*1.0 + 0.01 * j)/canvas_width, 
                        (y*1.0 + 0.01 * j)/canvas_width,
                        j*1.0/canvas_height])
                is_wall.append(0)
                # positions.append([i*1.0/canvas_width / 2.0, j*1.0/canvas_height /2.0])
    # wall
    for j in range(half_h, canvas_height - half_h, cell_height): # left (y = half_w)
        for i in range(half_w * 3 + 100, canvas_width - half_w * 3 - 100, cell_width):
            y = half_w
            positions.append([(i*1.0 + 0.01 * j)/canvas_width, 
                    (y*1.0 + 0.01 * j)/canvas_width,
                    j*1.0/canvas_height])
            is_wall.append(1)
    for j in range(half_h, canvas_height- half_h, cell_height): # right (y = canvas_width - half_w)
        for i in range(half_w * 3 + 100, canvas_width - half_w * 3 - 100, cell_width):        
            y = canvas_width - half_w
            positions.append([(i*1.0 + 0.01 * j)/canvas_width, 
                    (y*1.0 + 0.01 * j)/canvas_width,
                    j*1.0/canvas_height])
            is_wall.append(1)    
    for i in range(half_w * 2, canvas_width - half_w * 2, cell_width): # bottom (j = half_h)
        for y in range(half_w * 2, canvas_width - half_w * 2, cell_width):
            j = canvas_width - half_w
            positions.append([(i*1.0 + 0.01 * j)/canvas_width, 
                    (y*1.0 + 0.01 * j)/canvas_width,
                    j*1.0/canvas_height])
            is_wall.append(1)           
    positions = np.array(positions)
    is_wall = np.array(is_wall)

    radius = 3 / 400
    left_bound = half_w * 1.0 / canvas_width + radius 
    right_bound = (canvas_width - half_w) * 1.0 / canvas_width - radius 
    lower_bound = half_h * 1.0 / canvas_height + radius 
    bounds = {"left_bound": left_bound, "right_bound": right_bound, "lower_bound": lower_bound}    
    # positions += [[0.5,0.5],[0.6,0.5]]
    # positions = np.array(positions)
    # is_wall = np.array([])
    
    particles = Particles(positions, is_wall, canvas_width, canvas_height, cell_width, cell_height, radius,
                        bounds)

    gui = ti.GUI('Window Title', (canvas_width, canvas_height))
    # print(positions.shape)
    # particles.set_cells()
    # particles.update()
    # print(particles.num_particles_in_cell.to_numpy())

    draw_radius = 3# radius * 400
    steps = 0
    while True: 
        F = particles.forces.to_numpy()
        fluid_p = particles.get_np_fluid_positions()
        wall_p = particles.get_np_wall_positions()
        particles.update()
        if steps % 10 == 0:
            gui.circles(fluid_p, color = 0xFF0000, radius = draw_radius)
            gui.circles(wall_p, color = 0x0000FF, radius = draw_radius)
            # gui.arrows(p, F * 0.001)
            gui.show()
            particles.update()
            # print(particles.num_particles_in_cell.to_numpy())
            # print(particles.densities.to_numpy())
        steps += 1
        # print(particles.volume * particles.init_density)

if __name__ == '__main__':
    main()    