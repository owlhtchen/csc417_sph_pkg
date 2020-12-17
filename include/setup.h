
#include<vector>

void setup_positions(std::vector<double>& positions, double step, 
    double xBound=0.5, double yBound=0.7, double zBound=0.5) {
    // x, y, z: 0.0 to 1.0 for simulation canvas
    positions.clear();
    for(double x=0.0; x < xBound; x+=step) {
        for(double y=0.0; y < yBound; y+=step) {
            for(double z=0.0; z < zBound; z+=step) {
                positions.push_back(x);
                positions.push_back(y);
                positions.push_back(z);
            }
        }
    }
}