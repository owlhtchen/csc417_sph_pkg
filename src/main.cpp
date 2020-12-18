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
using namespace std::chrono_literals;

using namespace tbb;

std::condition_variable cv;
std::mutex cv_m;
int iter = 0;
Eigen::MatrixXd draw_positions;

void simulation(Particles& particles) {
    std::unique_lock<std::mutex> lk(cv_m, std::defer_lock);
    while(true) {
        particles.update();
        lk.lock();
        draw_positions = particles.positions;
        lk.unlock();
        cv.notify_one();
    }
}

int main()
{
    igl::opengl::glfw::Viewer viewer;
    using Eigen::MatrixXd; using Eigen::VectorXd;
    using std::cout; using std::endl; using std::vector;
    
    vector<double> _positions;
    vector<int> _is_wall;
    double radius = 3.0 / 400;
    setup_ball_positions(_positions, _is_wall, 2 * radius);
    Particles particles(_positions, _is_wall, radius);
    draw_positions.resizeLike(particles.positions);
    // cout << "hello" << endl;
    // cout << "??? " << _positions.size() << endl;

    std::thread worker(simulation, std::ref(particles));

    viewer.callback_post_draw = [&](igl::opengl::glfw::Viewer& viewer) -> bool {
        std::unique_lock<std::mutex> lk(cv_m);
        if(cv.wait_for(lk, 20ms, [] (){return true;})) {
            viewer.data().set_points(draw_positions, Eigen::RowVector3d(1.0,1.0,1.0));
            viewer.data().point_size = 5;
        }
        lk.unlock();
        return true;
    };
    viewer.core().is_animating = true;
    viewer.launch();
    worker.join();
}