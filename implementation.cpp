#include "implementation.hpp"
#include <riscv_vector.h>

// RGB2GRAY conversion
void impl_rgb2gray(cv::Mat &src, cv::Mat &dst)
{
    static_cast<void>(src);
    static_cast<void>(dst);
}

// Threshold function
void impl_threshold(cv::Mat &src,cv:: Mat &dst, double thresh, double maxval, int type)
{
    static_cast<void>(src);
    static_cast<void>(dst);
    static_cast<void>(thresh);
    static_cast<void>(maxval);
    static_cast<void>(type);
}

//Box filter
void impl_boxFilter(cv::Mat &src, cv::Mat &dst, int ddepth, cv::Size ksize)
{
    static_cast<void>(src);
    static_cast<void>(dst);
    static_cast<void>(ddepth);
    static_cast<void>(ksize);
}

// Morphology operation - erosion
void impl_morphology(cv::Mat &src, cv::Mat &dst)
{
    static_cast<void>(src);
    static_cast<void>(dst);
}

// Upscaling 2x
void impl_upscale2x(cv::Mat &src, cv::Mat &dst)
{
    static_cast<void>(src);
    static_cast<void>(dst);
}

// Downscaling 2x
void impl_downscale2x(cv::Mat &src, cv::Mat &dst)
{
    static_cast<void>(src);
    static_cast<void>(dst);
}

// Alpha compositing
void impl_alphaCompositing(cv::Mat& foreground, cv::Mat& background, cv::Mat& alpha, cv::Mat& dst)
{
    int pixelNum = foreground.rows * foreground.cols * foreground.channels();

    float* pFg  = reinterpret_cast<float*>(foreground.data);
    float* pBg  = reinterpret_cast<float*>(background.data);
    float* pA   = reinterpret_cast<float*>(alpha.data);
    float* pDst = reinterpret_cast<float*>(dst.data);

    size_t vl = vsetvl_e32m1(pixelNum); 
    size_t leftovers = pixelNum % vl;
    vfloat32m1_t img1, img2, alphaVec;

    auto var1 = [](auto img1, auto img2, auto alphaVec, size_t vl){
        auto diff = vfsub_vv_f32m1(img1, img2, vl);
        auto partial = vfmul_vv_f32m1(alphaVec, diff, vl);
        auto res = vfadd_vv_f32m1(partial, img2, vl);

        return res;
    };

    auto var2 = [](auto img1, auto img2, auto alphaVec, size_t vl){
        auto diff = vfsub_vv_f32m1(img1, img2, vl);
        auto res = vfmadd_vv_f32m1(alphaVec, diff, img2, vl);

        return res;
    };

    auto var3 = [](auto img1, auto img2, auto alphaVec, size_t vl){
        auto tmp = vfmacc_vv_f32m1(img1, img2, alphaVec, vl);
        auto antiAlpha = vfmul_vf_f32m1(alphaVec, -1.0, vl);
        auto res = vfmacc_vv_f32m1(tmp, antiAlpha, img2, vl);

        return res;
    };

    for(int vec_block = 0; vec_block < pixelNum - leftovers; vec_block += vl, pFg += vl, pBg += vl, pA += vl, pDst += vl)
    {
        img1 = vle32_v_f32m1(pFg, vl);
        img2 = vle32_v_f32m1(pBg, vl);
        alphaVec = vle32_v_f32m1(pA, vl);

        vse32_v_f32m1(pDst, var1(img1, img2, alphaVec, vl), vl);
    }

    if (leftovers){
        img1 = vle32_v_f32m1(pFg, leftovers);
        img2 = vle32_v_f32m1(pBg, leftovers);
        alphaVec = vle32_v_f32m1(pA, leftovers);
        
        vse32_v_f32m1(pDst, var1(img1, img2, alphaVec, vl), vl);
    }
}
