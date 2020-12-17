#include <igl/cotmatrix.h>
#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <iostream>
#include <tbb/tbb.h>

using namespace tbb;

int main()
{
  Eigen::MatrixXd V(4,2);
  V<<0,0,
     1,0,
     1,1,
     0,1;
  Eigen::MatrixXi F(2,3);
  F<<0,1,2,
     0,2,3;
  Eigen::SparseMatrix<double> L;
  igl::cotmatrix(V,F,L);
  std::cout<<"Hello, mesh: "<<std::endl<<L*V<<std::endl;

   int n = 100;
    parallel_for( blocked_range<size_t>(0,n),
     [=](const blocked_range<size_t>& r) {
                     for(size_t i=r.begin(); i!=r.end(); ++i)
                         std::cout << (100 + i) << std::endl;
                 }
  );

// https://software.intel.com/content/www/us/en/develop/documentation/onetbb-documentation/top/onetbb-developer-guide/parallelizing-simple-loops/parallel-for/lambda-expressions.html
    std::cout << " --- " << std::endl; 
    parallel_for(0, n, [=](size_t i) {std::cout << (100 + i) << std::endl;});
}