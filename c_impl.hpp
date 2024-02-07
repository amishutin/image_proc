#ifndef IMPL_MY_HPP
#define IMPL_MY_HPP

#include <stdint.h>
#include "opencv2/core.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/highgui.hpp"

// RGB2GRAY conversion
void c_rgb2gray_c3(uint8_t *pSrc, uint8_t *pDst, int height, int width);

// Threshold function
void c_threshold_c1(uint8_t *pSrc, uint8_t *pDst, int height, int width, uint8_t thresh);

// Downscaling 2x
void c_downscale2x_c1(uint8_t *pSrc, uint8_t *pDst, int height, int width);

// box filter
void c_blur_c3(uint8_t* src, uint8_t* dst, int st, int kh, int kw, int h, int w);

// morphology erode
void c_morphology(cv::Mat &src, cv::Mat &dst);

// alpha compositing
void c_impl_alphaCompositing(cv::Mat& foreground, cv::Mat& background, cv::Mat& alpha, cv::Mat& dst);

#endif // IMPLEMENTATION_HPP
