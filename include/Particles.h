#ifndef PARTICLES_H
#define PARTICLES_H

#include<CompileOptions.h>
#include <Eigen/Core>
#include <math.h>
#include <vector>
#include <EigenTypes.h>
#include <tbb/tbb.h>
#include <AtomicInt.h>


class Particles {
    
    public:
    const double init_density = 1000.0;
    const Eigen::Vector3d gravity = Eigen::Vector3d(0.0, -9.8 / 10.0, 0.0);
    const double alpha = 1.0;
    const double sound_speed = 20.0;
    const double epsilon = 0.01;
    const double gamma = 7.0;
    const double kappa = 0.2;
    const double delta_t = 0.0003;
    const int max_num_particles_per_cell = 200;
    const int particle_max_num_neighbors = 200;
    const double pi = 3.14159265358979323846;

    int num_particles;
    Eigen::MatrixXd positions;
    Eigen::MatrixXd velocities;
    Eigen::VectorXd densities;
    std::vector<int> is_wall;

    Eigen::MatrixXd new_positions;
    Eigen::MatrixXd new_velocities;
    Eigen::VectorXd new_densities;

    Eigen::MatrixXi cells;   // flatten x, y, z index of cell
    std::vector<AtomicInt> num_particles_per_cell; // flatten x, y, z index of cell
    Eigen::MatrixXi particle_neighbors; 
    Eigen::VectorXi particle_num_neighbors;

    double radius;
    double volume;
    double h_smooth_length;
    int n_cells_x;
    int n_cells_y;
    int n_cells_z;

    double x_lower, x_upper, y_lower, z_lower, z_upper;

    // debug
    int num_flattened_cells;

    Particles(std::vector<double> _positions, std::vector<int> _is_wall,
     double radius, double x_lower = 0.0, double x_upper = 0.5, double y_lower = 0.0);

    void reset_cells_neighbors() ;

    Eigen::Vector3i get_grid(Eigen::Vector3d position) ; 

    void set_cells() ;

    bool is_in_range(int i, int lower, int upper) ;

    void set_particles_neighbors() ;

    double cubicKernel(double r, double h) ;

    double cubicKernelDerivative(double r, double h) ;

    Eigen::Vector3d get_grad_W_ab(int pid, int neighbor_id);

    double get_pressure(int pid) ;

    Eigen::Vector3d dvelocity_dt_momentum(int pid) ; 

    Eigen::Vector3d dvelocity_dt_viscocity(int pid);

    Eigen::Vector3d dvelocity_dt_tension(int pid);

    Eigen::Vector3d dvelocity_dt(int pid);

    double ddensity_dt(int pid);   

    void update1();

    void collide_with_lower(int pid);

    void collide_with_upper(int pid);

    void update2();

    void update();  
};

#endif