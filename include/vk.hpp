#pragma once

#include <vector>

#include <opencv2/core/mat.hpp>

namespace VK {

void set_resolution(unsigned int width, unsigned int height);
void init();
void upload_faces(std::vector<float> faces);
void upload_edges(std::vector<float> edges);
void set_line_width(float w);
cv::Mat draw(float cam_x, float cam_y, float cam_z, float rot_x, float rot_y, float rot_z);

}
