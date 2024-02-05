#ifndef REF_IMPLEMENTATION_HPP
#define REF_IMPLEMENTATION_HPP

#include "opencv2/core.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/highgui.hpp"

// RGB2GRAY conversion
void ref_rgb2gray(cv::Mat &src, cv::Mat &dst);

// Threshold function
void ref_threshold(cv::Mat &src, cv::Mat &dst, double thresh, double maxval, int type);

//Box filter
void ref_boxFilter(cv::Mat &src, cv::Mat &dst, int ddepth, cv::Size ksize);

// Morphology operation - erosion
void ref_morphology(cv::Mat &src, cv::Mat &dst);

// Upscaling 2x
void ref_upscale2x(cv::Mat &src, cv::Mat &dst);

// Downscaling 2x
void ref_downscale2x(cv::Mat &src, cv::Mat &dst);

// Alpha compositing
void ref_alphaCompositing(const cv::Mat& foreground, const cv::Mat& background, const cv::Mat& alpha, cv::Mat& dst);

#endif // REF_IMPLEMENTATION_HPP