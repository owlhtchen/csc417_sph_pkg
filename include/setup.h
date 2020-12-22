#ifndef SETUP_H
#define SETUP_H

#include<vector>


void setup_ball_positions(std::vector<double>& positions, 
    std::vector<int> & is_wall, std::vector<double> & _colors, 
    std::vector<double> & velocities, double step,
    // double xBound=0.125, double yBound=0.3, double zBound=0.125) {
        // TWO BLOCK water horizontally collide
    double xBound=0.16, double yBound=0.6, double zBound=0.16) {
    // x, y, z: 0.0 to 1.0 for simulation canvas
    positions.clear();
    double start_offset = 0.000;
    for(double x=0.0 + start_offset; x < 0.1; x+=step) {
        for(double y=0.0; y < yBound; y+=step) {
            for(double z=0.0 + start_offset; z < zBound + start_offset; z+=step) {
                positions.push_back(x - 0.01 * step * y);
                positions.push_back(y);
                positions.push_back(z + 0.01 * step * y);
                velocities.push_back(0.6);
                velocities.push_back(0.0);
                velocities.push_back(0.0);
                _colors.push_back(1.0);
                _colors.push_back(0.0);
                _colors.push_back(0.0);
                is_wall.push_back(false);
            }
        }
    }

    for(double x=0.2; x < 0.3; x+=step) {
        for(double y=0.0; y < yBound; y+=step) {
            for(double z=0.0 + start_offset; z < zBound + start_offset; z+=step) {
                positions.push_back(x - 0.01 * step * y);
                positions.push_back(y);
                positions.push_back(z + 0.01 * step * y);
                velocities.push_back(-0.55);
                velocities.push_back(0.0);
                velocities.push_back(0.0);
                _colors.push_back(1.0);
                _colors.push_back(1.0);
                _colors.push_back(0.0);
                is_wall.push_back(false);
            }
        }
    }
}

void setup_ball_positions_2(std::vector<double>& positions, std::vector<int> & is_wall, 
    std::vector<double> & _colors, double step, 
    // double xBound=0.125, double yBound=0.3, double zBound=0.125) {
    double xBound=0.26, double yBound=0.6, double zBound=0.26) {
        // BLUE RED WHITE layers and ball

        using Eigen::Vector3d;
    // x, y, z: 0.0 to 1.0 for simulation canvas
    positions.clear();
    double start_offset = 0.000;
    for(double x=0.0 + start_offset; x < xBound + start_offset; x+=step) {
        for(double y=0.000; y < 0.05; y+=step) {
            for(double z=0.0 + start_offset; z < zBound + start_offset; z+=step) {
                positions.push_back(x - 0.01 * step * y);
                positions.push_back(y);
                positions.push_back(z + 0.01 * step * y);
                _colors.push_back(1.0);
                _colors.push_back(0.0);
                _colors.push_back(0.0);
                is_wall.push_back(false);
            }
        }
    }

    for(double x=0.0 + start_offset; x < xBound + start_offset; x+=step) {
        for(double y=0.2; y < 0.25; y+=step) {
            for(double z=0.0 + start_offset; z < zBound + start_offset; z+=step) {
                positions.push_back(x - 0.01 * step * y);
                positions.push_back(y);
                positions.push_back(z + 0.01 * step * y);
                _colors.push_back(1.0);
                _colors.push_back(1.0);
                _colors.push_back(1.0);
                is_wall.push_back(false);
            }
        }
    }

   Eigen::Vector3d center = Eigen::Vector3d(xBound/2.0, 0.35, zBound/2.0);    
   double radius = 0.045;
   for(double x = center.x() - radius; x <= center.x() + radius; x += step) {
       for(double y = center.y() - radius; y <= center.y() + radius; y += step) {
           for(double z = center.z() - radius; z <= center.z() + radius; z += step) {
               Vector3d dist = Eigen::Vector3d(x, y, z) - center;
               if(dist.norm() < radius) {
                    positions.push_back(x - 0.01 * step * y);
                    positions.push_back(y);
                    positions.push_back(z + 0.01 * step * y);
                    _colors.push_back(0.0);
                    _colors.push_back(0.0);
                    _colors.push_back(1.0);
                    is_wall.push_back(false);                   
               }
           }
       }
   }
}

void setup_ball_positions_1(std::vector<double>& positions, std::vector<int> & is_wall, 
    std::vector<double> & _colors, double step, 
    // double xBound=0.125, double yBound=0.3, double zBound=0.125) {
        // WHITE water block
    double xBound=0.16, double yBound=0.6, double zBound=0.16) {
    // x, y, z: 0.0 to 1.0 for simulation canvas
    positions.clear();
    double start_offset = 0.000;
    for(double x=0.0 + start_offset; x < xBound + start_offset; x+=step) {
        for(double y=0.0; y < yBound; y+=step) {
            for(double z=0.0 + start_offset; z < zBound + start_offset; z+=step) {
                positions.push_back(x - 0.01 * step * y);
                positions.push_back(y);
                positions.push_back(z + 0.01 * step * y);
                _colors.push_back(1.0);
                _colors.push_back(0.0);
                _colors.push_back(0.0);
                is_wall.push_back(false);
            }
        }
    }
}


#endif