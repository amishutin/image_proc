#ifndef IMPLEMENTATION_HPP
#define IMPLEMENTATION_HPP

#include "opencv2/core.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/highgui.hpp"

// RGB2GRAY conversion
void impl_rgb2gray(cv::Mat &src, cv::Mat &dst);

// Threshold function
void impl_threshold(cv::Mat &src, cv::Mat &dst, double thresh, double maxval, int type);

//Box filter
void impl_boxFilter(cv::Mat &src, cv::Mat &dst, int ddepth, cv::Size ksize);

// Morphology operation - opening
void impl_morphology(cv::Mat &src, cv::Mat &dst);

// Upscaling 2x
void impl_upscale2x(cv::Mat &src, cv::Mat &dst);

// Downscaling 2x
void impl_downscale2x(cv::Mat &src, cv::Mat &dst);

// Alpha compositing
void impl_alphaCompositing(cv::Mat& foreground, cv::Mat& background, cv::Mat& alpha, cv::Mat& dst);

#endif // IMPLEMENTATION_HPP