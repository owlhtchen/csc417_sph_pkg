#ifndef SETUP_H
#define SETUP_H

#include<vector>

void setup_ball_positions(std::vector<double>& positions, std::vector<int> & is_wall, double step, 
    double xBound=0.125, double yBound=0.3, double zBound=0.125) {
    // x, y, z: 0.0 to 1.0 for simulation canvas
    positions.clear();
    double start_offset = 0.000;
    for(double x=0.0 + start_offset; x < xBound + start_offset; x+=step) {
        for(double y=0.0; y < yBound; y+=step) {
            for(double z=0.0 + start_offset; z < zBound + start_offset; z+=step) {
                positions.push_back(x - 0.01 * step * y);
                positions.push_back(y);
                positions.push_back(z + 0.01 * step * y);
                is_wall.push_back(false);
            }
        }
    }
}

#endif