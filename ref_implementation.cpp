#include "ref_implementation.hpp"

// RGB2GRAY conversion
void ref_rgb2gray(cv::Mat &src, cv::Mat &dst)
{
    cv::cvtColor(src, dst, cv::COLOR_RGB2GRAY);
}

// Threshold function
void ref_threshold(cv::Mat &src, cv::Mat &dst, double thresh, double maxval, int type)
{
    cv::threshold(src, dst, thresh, maxval, type);
}

//Box filter
void ref_boxFilter(cv::Mat &src, cv::Mat &dst, int ddepth, cv::Size ksize)
{
    cv::boxFilter(src, dst, ddepth, ksize);
}

// Morphology operation - erosion
void ref_morphology(cv::Mat &src, cv::Mat &dst)
{
    cv::erode(src, dst, cv::Mat());
}

// Upscaling 2x
void ref_upscale2x(cv::Mat &src, cv::Mat &dst)
{
    cv::resize(src, dst, cv::Size(src.cols * 2, src.rows * 2), 0, 0, cv::INTER_LINEAR);
}

// Downscaling 2x
void ref_downscale2x(cv::Mat &src, cv::Mat &dst)
{
    cv::resize(src, dst, cv::Size(src.cols * 0.5, src.rows * 0.5), 0, 0, cv::INTER_NEAREST);
}

// Alpha compositing
void ref_alphaCompositing(const cv::Mat& foreground, const cv::Mat& background, const cv::Mat& alpha, cv::Mat& dst)
{
    cv::Mat tmp;
    cv::multiply(alpha, foreground, tmp); 

    cv::multiply(cv::Scalar::all(1.0)-alpha, background, dst); 

    cv::add(tmp, dst, dst);  
}
