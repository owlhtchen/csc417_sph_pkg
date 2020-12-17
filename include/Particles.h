#include <Eigen/Core>
#include <math.h>
#include <vector>
#include <EigenTypes.h>
#include <algorithm>
#include <tbb/tbb.h>

using namespace tbb;

class Particles {
    
    public:
    int num_particles;
    Eigen::MatrixXd positions;
    Eigen::MatrixXd velocities;
    Eigen::VectorXd densities;
    Eigen::VectorXd is_wall;

    Eigen::MatrixXd new_positions;
    Eigen::MatrixXd new_velocities;
    Eigen::VectorXd new_densities;

    Eigen::MatrixXi cells;   // flatten x, y, z index of cell
    Eigen::VectorXi num_particles_per_cell;
    Eigen::MatrixXi paritcle_neighbors; 
    Eigen::VectorXi particle_num_neighbors;

    double radius;
    double volume;
    double h_smooth_length;
    int n_cells_x;
    int n_cells_y;
    int n_cells_z;

    const double init_density = 1000.0;
    const Eigen::Vector3d gravity = Eigen::Vector3d(0.0, -9.8 / 100.0, 0.0);
    const double alpha = 1.0;
    const double sound_speed = 20.0;
    const double epsilon = 0.01;
    const double gamma = 7.0;
    const double kappa = 0.2;
    const double delta_t = 0.0003;
    const int max_num_particles_per_cell = 200;
    const int particle_max_num_neighbors = 200;
    const double pi = 3.14159265358979323846;

    int x_lower, x_upper, y_lower, z_lower, z_upper;

    Particles(std::vector<double> _positions, double radius, int x_lower = 0.0, int x_upper = 1.0, int y_lower = 0.0): 
        num_particles(int(_positions.size()/3)), radius(radius), volume(pi * radius * radius),
        x_lower(x_lower), x_upper(x_upper), y_lower(y_lower), z_lower(x_lower), z_upper(x_upper) {
            positions.resize(num_particles, 3);
            velocities.resize(num_particles, 3);
            densities.resize(num_particles);
            is_wall.resize(num_particles);

            new_positions.resize(num_particles, 3);
            new_velocities.resize(num_particles, 3);
            new_densities.resize(num_particles);      

            h_smooth_length = radius * 1.3;
            n_cells_x = ceil(1.0 / (2.0 * h_smooth_length));
            n_cells_y = n_cells_x;
            n_cells_z = n_cells_x;

            cells.resize(n_cells_x * n_cells_y * n_cells_z, max_num_particles_per_cell);
            num_particles_per_cell.resize(n_cells_x * n_cells_y * n_cells_z);
            paritcle_neighbors.resize(num_particles, particle_max_num_neighbors);
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
            }

            reset_cells_neighbors();
    }

    void reset_cells_neighbors() {
        for(int i = 0; i < n_cells_x * n_cells_y * n_cells_z; i++) {
            num_particles_per_cell(i) = 0;
        }
        for(int i = 0; i < num_particles; i++) {
            particle_num_neighbors(i) = 0;
        }
    }

    Eigen::Vector3i get_grid(Eigen::Vector3d position) {
        int grid_x = std::max(std::min(int(position(0) * n_cells_x), n_cells_x), 0);
        int grid_y = std::max(std::min(int(position(1) * n_cells_y), n_cells_y), 0);
        int grid_z = std::max(std::min(int(position(2) * n_cells_z), n_cells_z), 0);
        Eigen::Vector3i grid;
        grid << grid_x, grid_y, grid_z;
        return grid;
    }  

    void set_cells() {
        parallel_for(0, num_particles, [=](int i) {std::cout << (100 + i) << std::endl;});
        for(int pid = 0; pid < num_particles; pid++) {
            Eigen::Vector3i grid = get_grid(positions.row(pid));
            int grid_x = grid(0); int grid_y = grid(1); int grid_z = grid(2);
            cells(grid_x * (n_cells_y * n_cells_z))
        }
    }

    void update() {
        reset_cells_neighbors();

    }       
};