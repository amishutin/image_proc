#include "implementation.hpp"
#include <riscv_vector.h>
#include <iostream>


#define VER2
#define VEC vuint8m4_t
#define VSE vse8_v_u8m4
#define VLE vle8_v_u8m4
#define VAND vand_vv_u8m4
#define VSU vslideup_vx_u8m4
#define VSD vslidedown_vx_u8m4
#define VMV vmv_v_v_u8m4
#define VMVX vmv_v_x_u8m4
#define VSET vsetvl_e8m4
#define sieve sieve3

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

//ver1

#ifdef VER1

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

    VEC vidx1;
    VEC vidx2;
    VEC vidx3;

        for (int j = 1; j < size - 1; j+=vl - 2)
        {
            vl = VSET(size - j + 1);

            //begin

            buffer[1] = pSrc;
            buffer[2] = pSrc + src.cols;

            vidx1 = VLE(pSrc, vl);
            vidx2 = VLE(pSrc + src.cols, vl);

            for (int i = 1; i < src.rows - 1; i++)
            {
            idx = (i * src.cols + j);

            buffer[0] = buffer[1];
            buffer[1] = buffer[2];
            buffer[2] = pSrc + idx + src.cols - 1;

            vidx1 = VLE(buffer[0], vl);
            vidx2 = VLE(buffer[1], vl);
            vidx3 = VLE(buffer[2], vl);

            // vidx1 = VLE(pSrc + idx - src.cols - 1, vl);
            // vidx2 = VLE(pSrc + idx - 1, vl);
            // vidx3 = VLE(pSrc + idx + src.cols - 1, vl);

            VEC vres = VAND(vidx1, vidx2, vl);
            vres = VAND(vres, vidx3, vl);

            vidx1 = VMV(vres, vl);

            vidx2 = VSU(vidx2, vidx1, 1, vl);

            vidx3 = VSD(vidx3, vidx1, 1, vl);

            vres = VAND(vidx1, vidx2, vl);
            vres = VAND(vres, vidx3, vl);

            vres = VSD(vres, vres, 1, vl);

            VSE(pDst + idx, vres, vl - 2);
            }
        }
    // std::cout << "morph" << std::endl;
}

#elif defined(VER2)

inline void sieve2(VEC& vSrc, VEC& v1, VEC& v2, size_t& vl)
{
    v1 = VSU(v1, vSrc, 1, vl);

    v1 = VAND(v1, vSrc, vl);

    vSrc = VSD(vSrc, vSrc, 1, vl);

    vSrc = VAND(vSrc, v1, vl);
}

inline void sieve3(VEC& vSrc, VEC& v1, VEC& v2, size_t& vl)
{
    v1 = VSU(v1, vSrc, 1, vl);

    v2 = VAND(v1, vSrc, vl);

    v1 = VSD(v1, vSrc, 1, vl);

    vSrc = VAND(v2, v1, vl);
}


void impl_morphology(cv::Mat &src, cv::Mat &dst)
{
    // std::cout << "morph start" << std::endl;
    
    int idx;
    bool result = 0;
    uint8_t* pSrc = reinterpret_cast<uint8_t*>(src.data);
    uint8_t* pDst = reinterpret_cast<uint8_t*>(dst.data);
    size_t vl = 0;
    size_t size = src.cols;
    // size_t cycles = (src.rows - 2) / 4;
    // uint8_t* buffer[3];
    // VEC vbuffer[4];

    VEC vidx0;
    VEC vidx1;
    VEC vidx2;
    VEC vidx3;

        for (int j = 1; j < size - 1; j+=vl - 2)
        {
            vl = VSET(size - j + 1);

            //begin

            vidx2 = VLE(pSrc + j - 1, vl);
            vidx3 = VLE(pSrc + src.cols + j - 1, vl);

            sieve(vidx2, vidx0, vidx1, vl);
            sieve(vidx3, vidx1, vidx0, vl);

            // for (int i = 1; i < src.rows - 1; i++)
            for (int i = 1; i < src.rows - 1; i+=4)
            {
            idx = (i * src.cols + j);
// 2 3 0 -1
            vidx0 = VLE(pSrc + idx + src.cols - 1, vl);
            sieve(vidx0, vidx1, vidx2, vl);
            vidx1 = VAND(vidx0, vidx2, vl);
            vidx1 = VAND(vidx1, vidx3, vl);

            vidx1 = VSD(vidx1, vidx1, 1, vl);
            VSE(pDst + idx, vidx1, vl - 2);


            idx += src.cols;
// 3 0 1 -2
            vidx1 = VLE(pSrc + idx + src.cols - 1, vl);
            sieve(vidx1, vidx2, vidx3, vl);
            vidx2 = VAND(vidx1, vidx0, vl);
            vidx2 = VAND(vidx2, vidx3, vl);

            vidx2 = VSD(vidx2, vidx2, 1, vl);
            VSE(pDst + idx, vidx2, vl - 2);


            idx += src.cols;
// 0 1 2 -3
            vidx2 = VLE(pSrc + idx + src.cols - 1, vl);
            sieve(vidx2, vidx3, vidx0, vl);
            vidx3 = VAND(vidx0, vidx1, vl);
            vidx3 = VAND(vidx3, vidx2, vl);

            vidx3 = VSD(vidx3, vidx3, 1, vl);
            VSE(pDst + idx, vidx3, vl - 2);


            idx += src.cols;
// 1 2 3 -0
            vidx3 = VLE(pSrc + idx + src.cols - 1, vl);
            sieve(vidx3, vidx0, vidx1, vl);
            vidx0 = VAND(vidx1, vidx2, vl);
            vidx0 = VAND(vidx0, vidx3, vl);

            vidx0 = VSD(vidx0, vidx0, 1, vl);
            VSE(pDst + idx, vidx0, vl - 2);
            }
        }
    // std::cout << "morph" << std::endl;
}

#elif defined(VER3)
inline void sieve2(VEC& vSrc, VEC& v1, VEC& v2, size_t& vl)
{
    v1 = VSU(v1, vSrc, 1, vl);

    v1 = VAND(v1, vSrc, vl);

    vSrc = VSD(vSrc, vSrc, 1, vl);

    vSrc = VAND(vSrc, v1, vl);
}

inline void sieve3(VEC& vDst, VEC& vSrc, VEC& v2, size_t& vl)
{
    vDst = VSU(vDst, vSrc, 1, vl);

    v2 = VAND(vDst, vSrc, vl);

    vSrc = VSD(vSrc, vSrc, 1, vl);

    vDst = VAND(v2, vSrc, vl);
}


void impl_morphology(cv::Mat &src, cv::Mat &dst)
{
    // std::cout << "morph start" << std::endl;
    
    int idx;
    bool result = 0;
    uint8_t* pSrc = reinterpret_cast<uint8_t*>(src.data);
    uint8_t* pDst = reinterpret_cast<uint8_t*>(dst.data);
    size_t vl = 0;
    size_t size = src.cols;
    // size_t cycles = (src.rows - 2) / 4;
    // uint8_t* buffer[3];
    // VEC vbuffer[4];

    VEC vidx0;
    VEC vidx1;
    VEC vidx2;
    VEC vidx3;
    VEC vide0;
    VEC vide1;
    VEC vide2;
    VEC vide3;

        for (int j = 1; j < size - 1; j+=vl - 2)
        {
            vl = VSET(size - j + 1);

            //begin

            vidx2 = VLE(pSrc + j - 1, vl);
            vidx3 = VLE(pSrc + src.cols + j - 1, vl);

            sieve(vide2, vidx2, vidx1, vl);
            sieve(vide3, vidx3, vidx0, vl);

            // for (int i = 1; i < src.rows - 1; i++)
            for (int i = 1; i < src.rows - 1; i+=4)
            {
            idx = (i * src.cols + j);
// 2 3 0 -1
            vidx0 = VLE(pSrc + idx + src.cols - 1, vl);
            sieve(vide0, vidx0, vidx2, vl);

            // uint8_t* t = new uint8_t[vl];
            // VSE(t, vide0, vl - 2);
            // for(int k = 0; k < vl; k++)
            // {
            //     if (t[k])
            //     std::cout << '1';
            //     else
            //     std::cout << '0';
            // }
            // std::cout << std::endl;
            // delete[] t;

            vidx1 = VAND(vide0, vide2, vl);
            vidx1 = VAND(vidx1, vide3, vl);

            vidx1 = VSD(vidx1, vidx1, 1, vl);

            VSE(pDst + idx, vidx1, vl - 2);


            idx += src.cols;
// 3 0 1 -2
            vidx1 = VLE(pSrc + idx + src.cols - 1, vl);
            sieve(vide1, vidx1, vidx3, vl);
            vidx2 = VAND(vide1, vide0, vl);
            vidx2 = VAND(vidx2, vide3, vl);

            vidx2 = VSD(vidx2, vidx2, 1, vl);
            VSE(pDst + idx, vidx2, vl - 2);


            idx += src.cols;
// 0 1 2 -3
            vidx2 = VLE(pSrc + idx + src.cols - 1, vl);
            sieve(vide2, vidx2, vidx0, vl);
            vidx3 = VAND(vide0, vide1, vl);
            vidx3 = VAND(vidx3, vide2, vl);

            vidx3 = VSD(vidx3, vidx3, 1, vl);
            VSE(pDst + idx, vidx3, vl - 2);


            idx += src.cols;
// 1 2 3 -0
            vidx3 = VLE(pSrc + idx + src.cols - 1, vl);
            sieve(vide3, vidx3, vidx1, vl);
            vidx0 = VAND(vide1, vide2, vl);
            vidx0 = VAND(vidx0, vide3, vl);

            vidx0 = VSD(vidx0, vidx0, 1, vl);
            VSE(pDst + idx, vidx0, vl - 2);

            }
        }
    // std::cout << "morph" << std::endl;
}

#elif defined(VER4)

    void impl_morphology(cv::Mat &src, cv::Mat &dst)
{
    // std::cout << "morph start" << std::endl;
    
    int idx;
    bool result = 0;
    uint8_t* pSrc = reinterpret_cast<uint8_t*>(src.data);
    uint8_t* pDst = reinterpret_cast<uint8_t*>(dst.data);
    size_t vl = 0;
    size_t size = src.cols;
    //uint8_t* buffer[3];

    VEC vidx1;
    VEC vidx2;
    VEC vidx3;

    VEC v0 = VMVX(0, vl);

        for (int j = 1; j < size - 1; j+=vl - 2)
        {
            vl = VSET(size - j + 1);

            //             buffer[1] = pSrc;
            // buffer[2] = pSrc + src.cols;

            // vidx1 = VLE(pSrc, vl);
            // vidx2 = VLE(pSrc + src.cols, vl);

            //begin

            for (int i = 1; i < src.rows - 1; i++)
            {
            idx = (i * src.cols + j);

            //             buffer[0] = buffer[1];
            // buffer[1] = buffer[2];
            // buffer[2] = pSrc + idx + src.cols - 1;

            // vidx1 = VLE(buffer[0], vl);
            // vidx2 = VLE(buffer[1], vl);
            // vidx3 = VLE(buffer[2], vl);

            vidx1 = VLE(pSrc + idx - src.cols - 1, vl);
            vidx2 = VLE(pSrc + idx - 1, vl);
            vidx3 = VLE(pSrc + idx + src.cols - 1, vl);



            VEC vres = VAND(vidx1, vidx2, vl);
            vres = VAND(vres, vidx3, vl);

            vidx1 = VMV(vres, vl);

            vidx2 = VSU(vidx2, vidx1, 1, vl);

            vidx3 = VSD(vidx3, vidx1, 1, vl);

            vres = VAND(vidx1, vidx2, vl);
            vres = VAND(vres, vidx3, vl);

            vres = VSD(vres, vres, 1, vl);

            VSE(pDst + idx, vres, vl - 2);
            }
        }
    // std::cout << "morph" << std::endl;
}

#elif defined(VER5)

    void impl_morphology(cv::Mat &src, cv::Mat &dst)
{
    // std::cout << "morph start" << std::endl;
    
    int idx;
    bool result = 0;
    uint8_t* pSrc = reinterpret_cast<uint8_t*>(src.data);
    uint8_t* pDst = reinterpret_cast<uint8_t*>(dst.data);
    size_t vl = 0;
    size_t size = src.cols;
    uint8_t buf[3 * size];

    VEC vidx1;
    VEC vidx2;
    VEC vidx3;

    VEC v0 = VMVX(0, vl);

        // for (int j = 1; j < size - 1; j+=vl - 2)
        // {
            vl = VSET(size - j + 1);

            // buffer[1] = pSrc;
            // buffer[2] = pSrc + src.cols;

            // vidx1 = VLE(pSrc, vl);
            // vidx2 = VLE(pSrc + src.cols, vl);

            //begin

            for (int i = 1; i < src.rows - 1; i++)
            {
                for (int j = 1; j < src.cols - 1; j++)
            idx = (i * src.cols + j);

            //             buffer[0] = buffer[1];
            // buffer[1] = buffer[2];
            // buffer[2] = pSrc + idx + src.cols - 1;

            // vidx1 = VLE(buffer[0], vl);
            // vidx2 = VLE(buffer[1], vl);
            // vidx3 = VLE(buffer[2], vl);

            vidx1 = VLE(pSrc + idx - src.cols - 1, vl);
            vidx2 = VLE(pSrc + idx - 1, vl);
            vidx3 = VLE(pSrc + idx + src.cols - 1, vl);



            VEC vres = VAND(vidx1, vidx2, vl);
            vres = VAND(vres, vidx3, vl);

            vidx1 = VMV(vres, vl);

            vidx2 = VSU(vidx2, vidx1, 1, vl);

            vidx3 = VSD(vidx3, vidx1, 1, vl);

            vres = VAND(vidx1, vidx2, vl);
            vres = VAND(vres, vidx3, vl);

            vres = VSD(vres, vres, 1, vl);

            VSE(pDst + idx, vres, vl - 2);
            }
        // }
    // std::cout << "morph" << std::endl;
}


#endif

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
