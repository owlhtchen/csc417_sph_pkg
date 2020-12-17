#include <igl/cotmatrix.h>
#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <iostream>
#include <tbb/tbb.h>
#include <igl/opengl/glfw/Viewer.h>

using namespace tbb;

int main()
{
    igl::opengl::glfw::Viewer viewer;
    using Eigen::MatrixXd; using Eigen::VectorXd;
    using std::cout; using std::endl;
    MatrixXd points;
    points.resize(1, 3);
    // x right, y up, z 
    points << -1.0, -1.0, 1.0;
    viewer.data().set_points(points,Eigen::RowVector3d(1.0,1.0,1.0));
    viewer.data().point_size = 10;
    cout << viewer.data().point_size << endl;
    viewer.launch();
}