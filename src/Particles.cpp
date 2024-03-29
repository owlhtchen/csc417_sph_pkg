#include<CompileOptions.h>
#include<Particles.h>
#include <Eigen/Core>
#include <math.h>
#include <vector>
#include <EigenTypes.h>
#include <algorithm>
#include <tbb/tbb.h>
#include <vector>
#include <AtomicInt.h>
#include <assert.h>
#include <iostream>

using namespace tbb;

using std::cout; using std::endl;

// class Particles {

    Particles::Particles(std::vector<double> _positions, std::vector<int> _is_wall,
     double radius, double x_lower, double x_upper, double y_lower): 
        num_particles(int(_positions.size()/3)), radius(radius),
        x_lower(x_lower), x_upper(x_upper), y_lower(y_lower), z_lower(x_lower), z_upper(x_upper) {
            volume = 4.0 / 3 * pi * radius * radius * radius;
            positions.resize(num_particles, 3);
            velocities.resize(num_particles, 3);
            densities.resize(num_particles);
            is_wall.resize(num_particles);

            new_positions.resize(num_particles, 3);
            new_velocities.resize(num_particles, 3);
            new_densities.resize(num_particles);      

            h_smooth_length = radius * 1.3;
            n_cells_x = ceil(x_upper / (2.0 * h_smooth_length));
            n_cells_y = ceil(1.0 / (2.0 * h_smooth_length));
            n_cells_z = ceil(z_upper / (2.0 * h_smooth_length));

            num_flattened_cells = n_cells_x * n_cells_y * n_cells_z;
            std::cout << num_flattened_cells << std::endl;
            cells.resize(num_flattened_cells, max_num_particles_per_cell);
            num_particles_per_cell.resize(num_flattened_cells);
            particle_neighbors.resize(num_particles, particle_max_num_neighbors);
            particle_num_neighbors.resize(num_particles);

            // positions init
            for(int particleId = 0; particleId < num_particles; particleId++) {
                positions(particleId, 0) = _positions[particleId*3];
                positions(particleId, 1) = _positions[particleId*3+1];
                positions(particleId, 2) = _positions[particleId*3+2];
                velocities(particleId, 0) = 0.0;
                velocities(particleId, 1) = 0.0;
                velocities(particleId, 2) = 0.0;
                densities(particleId) = init_density;
                is_wall[particleId] = _is_wall[particleId];
            }

            reset_cells_neighbors();
    }

    void Particles::reset_cells_neighbors() {
        for(int i = 0; i < n_cells_x * n_cells_y * n_cells_z; i++) {
            num_particles_per_cell[i] = AtomicInt(0);
        }
        for(int i = 0; i < num_particles; i++) {
            particle_num_neighbors(i) = 0;
        }
    }

    Eigen::Vector3i Particles::get_grid(Eigen::Vector3d position) {
        double normalized_x = position(0) /  x_upper;
        double normalized_y = position(1);
        double normalized_z = position(2) / z_upper;
        int grid_x = std::max(std::min(int(normalized_x * n_cells_x), n_cells_x-1), 0);
        int grid_y = std::max(std::min(int(normalized_y * n_cells_y), n_cells_y-1), 0);
        int grid_z = std::max(std::min(int(normalized_z * n_cells_z), n_cells_z-1), 0);
        Eigen::Vector3i grid;
        grid << grid_x, grid_y, grid_z;
        return grid;
    }  

    void Particles::set_cells() {
        parallel_for(0, num_particles, [&](int pid) {
            Eigen::Vector3i grid = get_grid(positions.row(pid));
            int grid_x = grid(0); int grid_y = grid(1); int grid_z = grid(2);
            int cid = grid_x * (n_cells_y * n_cells_z) + grid_y * n_cells_z + grid_z;
            assert(cid < num_flattened_cells );
            assert(cid >=0 );
            int pid_cell = num_particles_per_cell[cid].fetch_add(1);
            if(pid_cell < max_num_particles_per_cell) {
                cells(cid, pid_cell) = pid;
            }
        });
    }

    bool Particles::is_in_range(int i, int lower, int upper) {
        return i >= lower && i < upper;
    }

    void Particles::set_particles_neighbors() {
        parallel_for(0, num_particles, [&](int pid) {
            // visited the surrounding 9 cells of this pid and find its neighbors
            Eigen::Vector3i grid = get_grid(positions.row(pid));
            int grid_x = grid(0); int grid_y = grid(1); int grid_z = grid(2);
            for(int x_range = -1; x_range <= 1; x_range++){
                for(int y_range = -1; y_range <= 1; y_range++) {
                    for(int z_range = -1; z_range <= 1; z_range++) {
                        int tmp_x = grid_x + x_range;
                        int tmp_y = grid_y + y_range;
                        int tmp_z = grid_z + z_range;
                        if(is_in_range(tmp_x, 0, n_cells_x) && 
                        is_in_range(tmp_y, 0, n_cells_y) && is_in_range(tmp_z, 0, n_cells_z)) {
                            int cid = tmp_x * (n_cells_y * n_cells_z) + tmp_y * n_cells_z + tmp_z;
                            for(int i = 0; i < num_particles_per_cell[cid] && i < max_num_particles_per_cell; i++) {
                                int neighbor_id = cells(cid, i);
                                if(neighbor_id != pid && 
                                    (positions.row(pid) - positions.row(neighbor_id)).norm() < h_smooth_length * 2.0
                                    && particle_num_neighbors[pid] < particle_max_num_neighbors) {
                                    particle_neighbors(pid, particle_num_neighbors[pid]++) = neighbor_id;
                                }
                            }
                        }
                        
                    }
                }
            }
        });
    }

    double Particles::cubicKernel(double r, double h) {
        // https://github.com/erizmr/SPH_Taichi
        // value of cubic spline smoothing kernel
        double k = 10.0 / (7.0 * pi * h * h);
        double q = r / h;
        double res = 0.0;
        if(q <= 1.0) {
            res = k * (1 - 1.5 * q * q + 0.75 * q * q * q);
        } else if(q < 2.0) {
            res = k * 0.25 * (2 - q) * (2 - q) * (2 - q);
        }
        return res;
    }

    double Particles::cubicKernelDerivative(double r, double h) {
        // https://github.com/erizmr/SPH_Taichi
        // derivative of cubcic spline smoothing kernel
        double k = 10.0 / (7.0 * pi * h * h);
        double q = r / h;
        double res = 0.0;
        if(q < 1.0) {
            res = (k / h) * (-3 * q + 2.25 * q * q);
        } else if(q < 2.0) {
            res = -0.75 * (k / h) * (2 - q) * (2 - q);
        } 
        return res;
    }

    Eigen::Vector3d Particles::get_grad_W_ab(int pid, int neighbor_id){
        Eigen::Vector3d r_ab = positions.row(pid) - positions.row(neighbor_id);
        double r_ab_norm = r_ab.norm();
        Eigen::Vector3d grad_ab = Eigen::Vector3d(0.0, 0.0, 0.0);
        if(r_ab_norm != 0.0) {
            grad_ab = cubicKernelDerivative(r_ab_norm, h_smooth_length) * r_ab / r_ab_norm;
        }
        return grad_ab;
    }

    double Particles::get_pressure(int pid) {
        double B = init_density * sound_speed * sound_speed / gamma;
        double P = B * (std::pow(densities[pid] / init_density, gamma) - 1.0);
        return P;
    }

    Eigen::Vector3d Particles::dvelocity_dt_momentum(int pid) { // particle id
        Eigen::Vector3d dv_dt = gravity;
        // for neighbor_idx in range(ti.static(self.max_num_neighbors)):
        for(int i = 0; i < particle_num_neighbors[pid]; i++) {
            int neighbor_id = particle_neighbors(pid, i);
            double P_a  = get_pressure(pid);
            double P_b = get_pressure(pid);
            double rho_a_square = densities[pid] * densities[pid];
            double rho_b_square = densities[neighbor_id] * densities[neighbor_id];
            double m_b = init_density * volume;
            Eigen::Vector3d grad_ab = get_grad_W_ab(pid, neighbor_id);
            double denom = P_a / rho_a_square + P_b / rho_b_square;
            dv_dt -= m_b * denom * grad_ab;  // changed
        }
        return dv_dt;
    }    

    Eigen::Vector3d Particles::dvelocity_dt_viscocity(int pid) {
        Eigen::Vector3d dv_dt = Eigen::Vector3d(0.0, 0.0, 0.0);
        for(int i = 0; i < particle_num_neighbors[pid]; i++) {
            int neighbor_id = particle_neighbors(pid, i);
            double m_b = volume * densities[neighbor_id];
            Eigen::Vector3d v_ab =  velocities.row(pid) - velocities.row(neighbor_id);
            Eigen::Vector3d r_ab = positions.row(pid) - positions.row(neighbor_id);
            if(v_ab.dot(r_ab) < 0) {
                double r_ab_norm = r_ab.norm();
                double v = 2.0 * alpha * h_smooth_length * sound_speed / (densities[pid] + densities[neighbor_id]);
                double tmp = v * v_ab.dot(r_ab) / (r_ab_norm * r_ab_norm + epsilon * h_smooth_length * h_smooth_length);
                Eigen::Vector3d grad_ab = get_grad_W_ab(pid, neighbor_id);
                dv_dt += m_b  *  tmp * grad_ab;
            }
        }
        return dv_dt;
    }

    Eigen::Vector3d Particles::dvelocity_dt_tension(int pid) {
        Eigen::Vector3d dv_dt = Eigen::Vector3d(0.0, 0.0, 0.0);
        for(int i = 0; i < particle_num_neighbors[pid]; i++) {
            int neighbor_id = particle_neighbors(pid, i);
            Eigen::Vector3d r_ab = positions.row(pid) - positions.row(neighbor_id);
            double m_b = volume * densities[neighbor_id];
            // dv_dt = m_b * cubicKernel(r_ab.norm(), h_smooth_length) * r_ab;
            dv_dt += m_b * cubicKernel(r_ab.norm(), h_smooth_length) * r_ab;
        }    
        double m_a = volume * densities[pid];
        dv_dt *= -kappa / m_a;
        return dv_dt;
    }

    Eigen::Vector3d Particles::dvelocity_dt(int pid) {
        Eigen::Vector3d dv_dt = Eigen::Vector3d(0.0, 0.0, 0.0);
        dv_dt += dvelocity_dt_momentum(pid);
        dv_dt += dvelocity_dt_viscocity(pid);
        // dv_dt += dvelocity_dt_tension(pid);
        return dv_dt;
    }

    double Particles::ddensity_dt(int pid) {
        double drho_dt = 0.0;
        for(int i = 0; i < particle_num_neighbors[pid]; i++) {
            int neighbor_id = particle_neighbors(pid, i);
            Eigen::Vector3d v_ab =  velocities.row(pid) - velocities.row(neighbor_id);
            Eigen::Vector3d grad_ab = get_grad_W_ab(pid, neighbor_id);
            double m_b = volume * densities[neighbor_id];
            drho_dt += m_b * v_ab.dot(grad_ab);       
        }
        return drho_dt;
    }      

    void Particles::update1() {
        // cout << "update1 " << endl;
        reset_cells_neighbors();
        set_cells();
        set_particles_neighbors();
        parallel_for(0, num_particles, [&](int pid) {
            Eigen::Vector3d dv_dt = dvelocity_dt(pid);
            new_positions.row(pid) = positions.row(pid) + delta_t * velocities.row(pid);
            new_velocities.row(pid)= velocities.row(pid) + delta_t * dv_dt.transpose();
            
            double ddensity = ddensity_dt(pid);
            new_densities[pid] = densities[pid] + delta_t * ddensity;
        });
        if(isnan(new_positions.sum())) {
            cout << "has nan: \n";
            cout << new_positions << endl;
            abort();
        }
    }

    void Particles::collide_with_lower(int pid) {
        // cout << "x_lower " << x_lower << endl;
        // cout << "y_lower " << y_lower << endl;
        // cout << "z_lower " << z_lower << endl;
        if(positions.row(pid).x() < x_lower) {
            if (velocities.row(pid).x() < 0.0) {
                velocities.row(pid).x()  += -1.5  * velocities.row(pid).x();
            }
            positions.row(pid).x()  = x_lower;
        }    
        if(positions.row(pid).y() < y_lower) {
            if (velocities.row(pid).y() < 0.0) {
                velocities.row(pid).y()  += -1.5  * velocities.row(pid).y();
            }
            positions.row(pid).y()  = y_lower;
        }     
        if(positions.row(pid).z() < z_lower) {
            if (velocities.row(pid).z() < 0.0) {
                velocities.row(pid).z()  += -1.5  * velocities.row(pid).z();
            }
            positions.row(pid).z()  = z_lower;
        }                  
    }

    void Particles::collide_with_upper(int pid) {
        if(positions.row(pid).x() >= x_upper) {
            if(velocities.row(pid).x() > 0.0) {
                velocities.row(pid).x() += -1.5  * velocities.row(pid).x();
            }
            positions.row(pid).x() = x_upper;
        }
        if(positions.row(pid).z() >= z_upper) {
            if(velocities.row(pid).z() > 0.0) {
                velocities.row(pid).z() += -1.5  * velocities.row(pid).z();
            }
            positions.row(pid).z() = z_upper;
        }        
    }

    void Particles::update2() {
        parallel_for(0, num_particles, [&](int pid) {
            if(!is_wall[pid]) {
                velocities.row(pid) = new_velocities.row(pid);
                positions.row(pid) = new_positions.row(pid);
                densities[pid] = new_densities[pid];
                // collide with walls
                collide_with_lower(pid);
                collide_with_upper(pid);
            }
        });
    }

    void Particles::update() {
        update1();
        update2();
    }       