#include "implementation.hpp"
#include <riscv_vector.h>
#include <iostream>

#define setvl vsetvl_e8m1
#define lse vlse8_v_u8m2
#define fwcvt vfwcvt_f_xu_v_f16m4
#define fmul vfmul_vf_f16m4
#define fadd vfadd_vv_f16m4
#define fncvt vfncvt_xu_f_w_u8m2 

// RGB2GRAY conversion
void impl_rgb2gray(cv::Mat &src, cv::Mat &dst)
{
uint8_t* pSrc = reinterpret_cast<uint8_t*>(src.data);
uint8_t* pDst = reinterpret_cast<uint8_t*>(dst.data);

int height = src.rows, width = src.cols;
int len = width * height * src.channels();

size_t vl = setvl(len);
int tail = (width * height) % vl;

vuint8m2_t vRd, vGrn, vBl;
vfloat16m4_t vres, vsum;

for (uint64_t i = 0; i < len - 3 * tail; i += vl * 3, pDst += vl) {
    vRd = lse(pSrc + i , 3, vl);        //запись со сдвигом
    vGrn = lse(pSrc + i + 1, 3, vl);
    vBl = lse(pSrc + i + 2, 3, vl);
    
    auto vRdf = fwcvt(vRd, vl);              // конвертация
    auto vGrnf = fwcvt(vGrn, vl);
    auto vBlf = fwcvt(vBl, vl);

    auto vRdm = fmul(vRdf, 0.299f, vl);          //умножаем
    auto vGrnm = fmul(vGrnf, 0.587f, vl);
    auto vBlm = fmul(vBlf, 0.114f, vl);

    vsum = fadd(vRdm, vGrnm, vl);                //суммируем
    vres = fadd(vsum, vBlm, vl);

    auto vresu = fncvt(vres, vl);
    vse8_v_u8m2(pDst, vresu, vl);
}
if (tail) {
    vl = tail;
    vRd = lse(pSrc + len - 3 * tail, 3, vl);
    vGrn = lse(pSrc + len - 3 * tail + 1, 3, vl);
    vBl = lse(pSrc + len - 3 * tail + 2, 3, vl);

    auto vRdf = fwcvt(vRd, vl);
    auto vGrnf = fwcvt(vGrn, vl);
    auto vBlf = fwcvt(vBl, vl);

    auto vRdm = fmul(vRdf, 0.299f, vl);
    auto vGrnm = fmul(vGrnf, 0.587f, vl);
    auto vBlm = fmul(vBlf, 0.114f, vl);

    vsum = fadd(vRdm, vGrnm, vl);
    vres = fadd(vsum, vBlm, vl);

    auto vresu = fncvt(vres, vl);
    vse8_v_u8m2(pDst + (width * height) - tail, vresu, vl);
}
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
