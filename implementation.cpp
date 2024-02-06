#include "implementation.hpp"
#include <riscv_vector.h>
//#include <iostream>

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
    // std::cout << "morph start" << std::endl;
    
    int idx;
    bool result = 0;
    uint8_t* pSrc = reinterpret_cast<uint8_t*>(src.data);
    uint8_t* pDst = reinterpret_cast<uint8_t*>(dst.data);
    size_t vl = 0;
    size_t size = src.cols;
    uint8_t* buffer[3];
    //vuint8m8_t* vbuffer[3];

    vuint8m8_t vidx1;
    vuint8m8_t vidx2;
    vuint8m8_t vidx3;

        for (int j = 1; j < size - 1; j+=vl - 2)
        {
            vl = vsetvl_e8m8(size - j + 1);

            //begin

            buffer[1] = pSrc;
            buffer[2] = pSrc + src.cols;

            // vidx1 = vle8_v_u8m8(pSrc, vl);
            // vidx2 = vle8_v_u8m8(pSrc + src.cols, vl);

            // vbuffer[1] = &vidx1;
            // vbuffer[2] = &vidx2;

            for (int i = 1; i < src.rows - 1; i++)
            {
            idx = (i * src.cols + j);

            buffer[0] = buffer[1];
            buffer[1] = buffer[2];
            buffer[2] = pSrc + idx + src.cols - 1;

            vuint8m8_t vidx1 = vle8_v_u8m8(buffer[0], vl);
            vuint8m8_t vidx2 = vle8_v_u8m8(buffer[1], vl);
            vuint8m8_t vidx3 = vle8_v_u8m8(buffer[2], vl);

            // vbuffer[0] = vbuffer[1];
            // vbuffer[1] = vbuffer[2];
            // vbuffer[2] = *(vle8_v_u8m8(pSrc + idx + src.cols - 1, vl));

            vuint8m8_t vres = vand_vv_u8m8(vidx1, vidx2, vl);
            vres = vand_vv_u8m8(vres, vidx3, vl);

            vidx1 = vmv_v_v_u8m8(vres, vl);

            vidx2 = vslideup_vx_u8m8(vidx2, vidx1, 1, vl);

            vidx3 = vslidedown_vx_u8m8(vidx3, vidx1, 1, vl);

            vres = vand_vv_u8m8(vidx1, vidx2, vl);
            vres = vand_vv_u8m8(vres, vidx3, vl);

            // vuint8m8_t vres = vand_vv_u8m8(vbuffer[0], vbuffer[1], vl);
            // vres = vand_vv_u8m8(vres, vbuffer[3], vl);

            // vbuffer[0] = vmv_v_v_u8m8(vres, vl);

            // vbuffer[1] = vslideup_vx_u8m8(vbuffer[1], vbuffer[0], 1, vl);

            // vbuffer[2] = vslidedown_vx_u8m8(vbuffer[2], vbuffer[0], 1, vl);

            // vres = vand_vv_u8m8(vbuffer[0], vbuffer[1], vl);
            // vres = vand_vv_u8m8(vres, vbuffer[2], vl);

            vres = vslidedown_vx_u8m8(vres, vres, 1, vl);

            vse8_v_u8m8(pDst + idx, vres, vl - 2);
            }
        }
    // std::cout << "morph" << std::endl;
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
