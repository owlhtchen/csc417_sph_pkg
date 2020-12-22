#include <CompileOptions.h>
#include <igl/cotmatrix.h>
#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <iostream>
#include <tbb/tbb.h>
#include <igl/opengl/glfw/Viewer.h>
#include <setup.h>
#include <Particles.h>

#include <atomic>
#include <condition_variable>
#include <thread>
#include <chrono>

#include <igl/copyleft/marching_cubes.h>

using namespace std::chrono_literals;
using namespace tbb;

std::condition_variable cv;
std::mutex cv_m;
int iter = 0;
Eigen::MatrixXd draw_positions;
bool updating;

void simulation(Particles& particles) {
    std::unique_lock<std::mutex> lk(cv_m, std::defer_lock);
    while(true) {
        if(updating == false) {
            std::this_thread::sleep_for(200ms);
        } else {
            particles.update();
            lk.lock();
            draw_positions = particles.positions;
            lk.unlock();
            cv.notify_one();
        }
    }
}

int main()
{
    igl::opengl::glfw::Viewer viewer;
    using Eigen::MatrixXd; using Eigen::VectorXd; using Eigen::MatrixXi;
    using std::cout; using std::endl; using std::vector;
    
    vector<double> _positions;
    vector<int> _is_wall;
    std::vector<double>  _colors;
    std::vector<double>  _velocities;
    double radius = 3.0 / 400;
    const double c = 0.7;
    
    // no velocities
    setup_ball_positions_1(_positions, _is_wall, _colors, 2 * radius);
    // setup_ball_positions_2(_positions, _is_wall, _colors, 2 * radius);
    Particles particles(_positions, _is_wall, _colors, radius);

    // velocities
    // setup_ball_positions_3(_positions, _is_wall, _colors, _velocities, 2 * radius);
    // Particles particles(_positions, _is_wall, _colors, _velocities,  radius);
    
    cout << "particles.positions: " << particles.positions.size() << endl;
    draw_positions.resizeLike(particles.positions);

    std::thread worker(simulation, std::ref(particles));
    updating = false;

    viewer.callback_post_draw = [&](igl::opengl::glfw::Viewer& viewer) -> bool {
        std::unique_lock<std::mutex> lk(cv_m);
        if(cv.wait_for(lk, 20ms, [] (){return true;})) {
            viewer.data().set_points(draw_positions, particles.colors);
            viewer.data().point_size = 5;
        }
        lk.unlock();
        return true;
    };
    viewer.callback_key_down = [&](igl::opengl::glfw::Viewer &viewer, unsigned char key, int modifiers) -> bool {
        if(key == 'B' || key == 'b') {
            updating = false;

            MatrixXd all_X; VectorXd all_phi;
            particles.get_X_phi(all_X, all_phi, c);
            MatrixXd V; MatrixXi F;
            igl::copyleft::marching_cubes(all_phi, all_X, 
                particles.grid_per_dim, particles.grid_per_dim, particles.grid_per_dim, V, F);
            viewer.core().is_animating = false;
            viewer.data().set_mesh(V, F);
            cout << "grid_per_dim: " << particles.grid_per_dim << endl;
            cout << "phi, X: " << all_phi.size() << ", " << all_X.size() << endl;
            cout << "phi, X: " << all_phi.mean() << ", " << all_X.mean() << endl;
            cout << "V, F " << V.size() << ", " << F.size() << endl;
        } else if (key == 'S' || key == 's') {
            viewer.data().clear();
             viewer.core().is_animating = true;
            updating = true;
        } else if(key == 'N' || key == 'n') {
            updating = true;
            viewer.core().is_animating = true;
        }
        return false;
    };
    viewer.core().is_animating = false;
    viewer.launch();
    worker.join();
}