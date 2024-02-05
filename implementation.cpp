#include "implementation.hpp"
#include <riscv_vector.h>
#include <iostream>

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
    std::cout << "morph start" << std::endl;
    int idx;
    bool result = 0;
    uint8_t* pSrc = reinterpret_cast<uint8_t*>(src.data);
    uint8_t* pDst = reinterpret_cast<uint8_t*>(dst.data);
    size_t vl = vsetvl_e8m1(4);
    for (int i = 1; i < src.rows - 1; i++)
    {
        for (int j = 1; j < src.cols - 1; j+=vl)
        {
            idx = (i * src.cols + j);
            vuint8m1_t vidx = vle8_v_u8m1(pSrc + idx, vl);
            vuint8m1_t vidx1 = vle8_v_u8m1(pSrc + idx - src.cols - 1, vl + 2);
            vuint8m1_t vidx2 = vle8_v_u8m1(pSrc + idx - 1, vl + 2);
            vuint8m1_t vidx3 = vle8_v_u8m1(pSrc + idx + src.cols - 1, vl + 2);
            vuint8m1_t vres = vand_vv_u8m1(vidx1, vidx2, vl + 2);
            vres = vand_vv_u8m1(vres, vidx3, vl + 2);
            uint8_t* pTmp = new uint8_t*[vl + 2];
            vse8_v_u8m1(pTmp, vres, vl + 2);
            for (uint8_t t : pTmp){
                std::cout << t << ' ';
            }
            //vidx1 = vcopy_v_u8m1(vres);
            // vidx1 = vsll_vx_u8m1(vres, 0, vl);
            // vidx2 = vsll_vx_u8m1(vres, 1, vl);
            // vidx3 = vsll_vx_u8m1(vres, 2, vl);
            // vres = vand_vv_u8m1(vidx1, vidx2, vl);
            // vres = vand_vv_u8m1(vres, vidx3, vl);
            //vse8_v_u8m1(pSrc + idx, vres, vl); 
        }
    }
    std::cout << "morph" << std::endl;
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

# if 0
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
#endif

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
