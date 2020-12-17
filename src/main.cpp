#include <igl/cotmatrix.h>
#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <iostream>
#include <tbb/tbb.h>
#include <igl/opengl/glfw/Viewer.h>
#include <setup.h>
#include <Particles.h>

using namespace tbb;

int main()
{
    igl::opengl::glfw::Viewer viewer;
    using Eigen::MatrixXd; using Eigen::VectorXd;
    using std::cout; using std::endl; using std::vector;
    
    vector<double> _positions;
    setup_positions(_positions, 0.08);
    double radius = 3.0 / 400;
    Particles particles(_positions, radius);
    cout << "hello" << endl;
    cout << "??? " << _positions.size() << endl;
    viewer.data().set_points(particles.positions, Eigen::RowVector3d(1.0,1.0,1.0));
    viewer.data().point_size = 10;
    viewer.launch();
}