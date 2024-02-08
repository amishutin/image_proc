#include "implementation.h"
#include <riscv_vector.h>
#include <stdint.h>

// RGB2GRAY conversion
/*
void impl_rgb2gray(cv::Mat &src, cv::Mat &dst)
{
    CvMat iSrc = cvMat(src);
    CvMat iDst = cvMat(dst);
    c_rgb2gray_c3(iSrc.ptr, iDst.ptr, iSrc.height, iSrc.width);
}
*/

void c_rgb2gray_c3(uint8_t *pSrc, uint8_t *pDst, int height, int width)
{
    const float cR = 0.299f;
    const float cG = 0.587f;
    const float cB = 0.114f;

#if 0
    int rowIndex = 0;
    for (int y = 0; y < height; ++y) 
    {
        for (int x = 0; x < width; ++x) 
        {
            pDst[rowIndex + x] = (uint8_t)(cR * pSrc[(rowIndex + x) * 3] 
                                         + cG * pSrc[(rowIndex + x) * 3 + 1] 
                                         + cB * pSrc[(rowIndex + x) * 3 + 2]);
        }
        rowIndex += width;
    }
#elif 1
    uint64_t len = height * width;
    for (uint64_t i = 0; i < len; ++i) 
    {
        pDst[i] = (uint8_t)(cR * pSrc[i * 3] + cG * pSrc[i * 3 + 1] + cB * pSrc[i * 3 + 2]);
    }
#endif
}

static uint8_t redIndices[16] = {0, 3, 6, 9, 12, 15, 18, 21, 24, 27, 30, 33, 36, 39, 42, 45};
static uint8_t greenIndices[16] = {1, 4, 7, 10, 13, 16, 19, 22, 25, 28, 31, 34, 37, 40, 43, 46};
static uint8_t blueIndices[16] = {2, 5, 8, 11, 14, 17, 20, 23, 26, 29, 32, 35, 38, 41, 44, 47};

void riscv_rgb2gray_c3(uint8_t *pSrc, uint8_t *pDst, int height, int width)
{
    vuint8m1_t srcmm;
    vuint8m1_t xmm0, xmm1, xmm2;
    vfloat16m2_t xmm0f, xmm1f, xmm2f;
    vfloat16m2_t coeffsmmR, coeffsmmG, coeffsmmB;
    const size_t vl = 16;

    coeffsmmR = vfmv_v_f_f16m2((_Float16)0.299, vl);
    coeffsmmG = vfmv_v_f_f16m2((_Float16)0.587, vl);
    coeffsmmB = vfmv_v_f_f16m2((_Float16)0.114, vl);

    uint64_t len = height * width;    

    for (uint64_t i = 0; i < len; i += vl)
    {
        uint8_t *src = pSrc + i * 3;
        uint8_t *dst = pDst + i;

        // rrrr, gggg, bbbb
        xmm0 = vlse8_v_u8m1((uint8_t *)(src + 0), 3, vl);
        xmm1 = vlse8_v_u8m1((uint8_t *)(src + 1), 3, vl);
        xmm2 = vlse8_v_u8m1((uint8_t *)(src + 2), 3, vl);

        // convert uint8_t -> _Float16
        xmm0f = vfwcvt_f_xu_v_f16m2(xmm0, vl);
        xmm1f = vfwcvt_f_xu_v_f16m2(xmm1, vl);
        xmm2f = vfwcvt_f_xu_v_f16m2(xmm2, vl);

        // r * rc, r * rc + g * gc, r * rc + g * gc + b * bc
        xmm0f = vfmul_vv_f16m2(coeffsmmR, xmm0f, vl);
        xmm0f = vfmadd_vv_f16m2(coeffsmmG, xmm1f, xmm0f, vl);
        xmm0f = vfmadd_vv_f16m2(coeffsmmB, xmm2f, xmm0f, vl);

        // convert _Float16 -> uint8_t
        xmm0 = vfncvt_xu_f_w_u8m1(xmm0f, vl);

        // Store gray
        vse8_v_u8m1(dst, xmm0, vl);
    }
}

/*
void riscv_rgb2gray_c3(uint8_t *pSrc, uint8_t *pDst, int height, int width)
{
    vuint8m1_t srcmm;
    vuint8m1_t xmm0, xmm1, xmm2;
    vuint16m2_t xmm0m2, xmm1m2, xmm2m2;
    // vuint16m2_t coeffsmmR, coeffsmmG, coeffsmmB, divmm;
    vuint8m1_t redmm, greenmm, bluemm;
    size_t vl = 4;

    //coeffsmmR = vmv_v_x_u16m2(77, vl); // 77/256 ~= 0.299
    //coeffsmmG = vmv_v_x_u16m2(150, vl); // ~150/256 ~= 0.587
    //coeffsmmB = vmv_v_x_u16m2(29, vl); // ~29/256 ~= 0.114
    // divmm = vmv_v_x_u16m2(256, vl);

    redmm = vle8_v_u8m1(redIndices, vl);
    greenmm = vle8_v_u8m1(greenIndices, vl);
    bluemm = vle8_v_u8m1(blueIndices, vl);

    uint64_t len = height * width;    

    for (uint64_t i = 0; i < len; i += vl)
    {
        uint8_t *src = pSrc + i * 3;
        uint8_t *dst = pDst + i;

        // r0;g0;b0;r1;g1;b1;r2;g2;b2;r3;g3;b3;
        srcmm = vle8_v_u8m1(src, 3 * vl);

        // r0;r1;r2;r3, b0;b1;b2;b3, g0;g1;g2;g3
        xmm0 = vrgather_vv_u8m1(srcmm, redmm, vl);
        xmm1 = vrgather_vv_u8m1(srcmm, greenmm, vl);
        xmm2 = vrgather_vv_u8m1(srcmm, bluemm, vl);

        // convert uint8_t -> uint16_t
        xmm0m2 = vwcvtu_x_x_v_u16m2(xmm0, vl);
        xmm1m2 = vwcvtu_x_x_v_u16m2(xmm1, vl);
        xmm2m2 = vwcvtu_x_x_v_u16m2(xmm2, vl);

        // r * rc, r * rc + g * gc, r * rc + g * gc + b * bc
        xmm0m2 = vmul_vx_u16m2(xmm0m2, (uint16_t)77u, vl);
        xmm0m2 = vmadd_vx_u16m2(xmm1m2, (uint16_t)150u, xmm0m2, vl);
        xmm0m2 = vmadd_vx_u16m2(xmm2m2, (uint16_t)29u, xmm0m2, vl);

        // divide sum by 256
        xmm0m2 = vdivu_vx_u16m2(xmm0m2, (uint16_t)256u, vl);

        // convert uint16_t -> uint8_t        
        xmm0 = vncvt_x_x_w_u8m1(xmm0m2, vl);

        // Store gray
        vse8_v_u8m1(dst, xmm0, vl);
    }
}
*/

// Threshold function
/*
void impl_threshold(cv::Mat &src,cv:: Mat &dst, double thresh, double maxval, int type)
{
    static_cast<void>(maxval);
    static_cast<void>(type);

    CvMat iSrc = cvMat(src);
    CvMat iDst = cvMat(dst);
    c_threshold_c1(iSrc.ptr, iDst.ptr, iSrc.height, iSrc.width, (uint8_t)thresh);
}
*/

void c_threshold_c1(uint8_t *pSrc, uint8_t *pDst, int height, int width, uint8_t thresh)
{
#if 0
    int rowIndex = 0;
    for (int y = 0; y < height; ++y) 
    {
        for (int x = 0; x < width; ++x) 
        {
            pDst[rowIndex + x] = (pSrc[rowIndex + x] < thresh)? pSrc[rowIndex + x] : thresh;
        }
        rowIndex += width;
    }
#elif 1
    uint64_t len = height * width;
    for (uint64_t i = 0; i < len; ++i) 
    {
        pDst[i] = (pSrc[i] <= thresh)? pSrc[i] : thresh;
    }
#endif
}

void riscv_threshold_c1(uint8_t *pSrc, uint8_t *pDst, int height, int width, uint8_t thresh)
{
    vuint8m1_t srcmm, xmm, threshmm;
    const size_t vl = 16;
    threshmm = vmv_v_x_u8m1(thresh, vl);

    uint64_t len = height * width;

    for (uint64_t i = 0; i < len; i += vl) 
    {
        uint8_t *src = pSrc + i;
        uint8_t *dst = pDst + i;

        srcmm = vle8_v_u8m1(src, vl);
        //xmm = vminu_vx_u8m1(srcmm, thresh, vl);
        // xmm[k] = (srcmm[k] < thresh)? 0 : (srcmm[k] - thresh)
        xmm = vssubu_vv_u8m1(srcmm, threshmm, vl);
        // xmm[k] = (srcmm[k] < thresh)? (srcmm[k] - 0) : (srcmm[k] - srcmm[k] + thresh)
        xmm = vssubu_vv_u8m1(srcmm, xmm, vl); 

        vse8_v_u8m1(dst, xmm, vl);
    }
}

//Box filter
/*
void impl_boxFilter(cv::Mat &src, cv::Mat &dst, int ddepth, cv::Size ksize)
{
    static_cast<void>(src);
    static_cast<void>(dst);
    static_cast<void>(ddepth);
    static_cast<void>(ksize);
}

// Morphology operation - opening
void impl_morphology(cv::Mat &src, cv::Mat &dst, int op)
{
    static_cast<void>(src);
    static_cast<void>(dst);
    static_cast<void>(op);
}

// Upscaling 2x
void impl_upscale2x(cv::Mat &src, cv::Mat &dst)
{
    static_cast<void>(src);
    static_cast<void>(dst);
}
*/

// Downscaling 2x
/*
void impl_downscale2x(cv::Mat &src, cv::Mat &dst)
{
    CvMat iSrc = cvMat(src);
    CvMat iDst = cvMat(dst);
    c_downscale2x_c1(iSrc.ptr, iDst.ptr, iSrc.height, iSrc.width);
}
*/

void c_downscale2x_c1(uint8_t *pSrc, uint8_t *pDst, int height, int width)
{
    const int downscale = 2;
    int dstWidth = width / downscale;
    int dstHeight = height / downscale;

    for (int y = 0; y < dstHeight; y++) {
        for (int x = 0; x < dstWidth; x++) {
            pDst[x] = pSrc[x * downscale];
        }
        pSrc = pSrc + width * downscale;
        pDst = pDst + dstWidth;
    }
}

/*
static uint8_t evenIndices[32] = {0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32, 34, 36, 38, 40, 42, 44, 46, 48, 50, 52, 54, 56, 58, 60, 62};

void riscv_downscale2x_c1(uint8_t *pSrc, uint8_t *pDst, int height, int width)
{
    vuint8m1_t srcmm, xmm;
    vuint8m1_t idxmm;
    size_t vl = 8;

    idxmm = vle8_v_u8m1(evenIndices, vl * 2);

    const int downscale = 2;
    int dstWidth = width / downscale;
    int dstHeight = height / downscale;

    for (int y = 0; y < dstHeight; y++) {
        for (int x = 0; x < dstWidth; x += vl) {
            uint8_t* src = pSrc + x * downscale;
            uint8_t* dst = pDst + x;

            srcmm = vle8_v_u8m1(src, vl * 2);
            xmm = vrgather_vv_u8m1(srcmm, idxmm, vl);
            vse8_v_u8m1(dst, xmm, vl);
        }
        pSrc = pSrc + width * downscale;
        pDst = pDst + dstWidth;
    }    
}
*/

/*
void riscv_downscale2x_c1(uint8_t *pSrc, uint8_t *pDst, int height, int width)
{
    vuint8m1_t xmm;
    const size_t vl = vsetvlmax_e8m1();

    const int downscale = 2;
    int dstWidth = width / downscale;
    int dstHeight = height / downscale;

    for (int y = 0; y < dstHeight; y++) {
        for (int x = 0; x < dstWidth; x += vl) {
            uint8_t* src = pSrc + x * downscale;
            uint8_t* dst = pDst + x;

            xmm = vlse8_v_u8m1(src, downscale, vl);
            vse8_v_u8m1(dst, xmm, vl);
        }
        pSrc = pSrc + width * downscale;
        pDst = pDst + dstWidth;
    }    
}
*/

//static uint8_t evenIndices[16] = {0, 2, 4, 6, 0, 2, 4, 6, 16, 18, 20, 22, 16, 18, 20, 22};
//static uint8_t mask0[16] = {1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0};
//static uint8_t mask1[16] = {0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1};

static uint8_t evenIndices[16] = {0, 2, 4, 6, 8, 10, 12, 14, 0, 2, 4, 6, 8, 10, 12, 14};
static uint8_t mask0[16] = {1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0};
static uint8_t mask1[16] = {0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1};

void riscv_downscale2x_c1(uint8_t *pSrc, uint8_t *pDst, int height, int width)
{
    vuint8m1_t srcmm0, srcmm1, xmm0, xmm1;
    vuint8m1_t idxmm;
    vuint8m1_t maskmm0, maskmm1;
    size_t vl = 16;

    idxmm = vle8_v_u8m1(evenIndices, vl);

    maskmm0 = vle8_v_u8m1(mask0, vl);
    maskmm1 = vle8_v_u8m1(mask1, vl);

    const int downscale = 2;
    int dstWidth = width / downscale;
    int dstHeight = height / downscale;
    int tail = dstHeight % vl;
    int endRow = dstHeight - tail;

    for (int y = 0; y < dstHeight; y++) {
        for (int x = 0; x < dstWidth; x += vl) {
            uint8_t* src = pSrc + x * downscale;
            uint8_t* dst = pDst + x;

            // To avoid vsetvli we load two times and do some special moves
            // comments are for vl=8, idea for vl=16 is the same.
            // s0;s1;s2;s3;s4;s5;s6;s7, s8;s9;s10;s11;s12;s13;s14;s15
            srcmm0 = vle8_v_u8m1(src, vl);
            srcmm1 = vle8_v_u8m1((uint8_t *)(src + vl), vl);

            // s0;s2;s4;s6;s0;s2;s4;s6, s8;s10;s12;s14;s8;s10;s12;s14;
            xmm0 = vrgather_vv_u8m1(srcmm0, idxmm, vl);
            xmm1 = vrgather_vv_u8m1(srcmm1, idxmm, vl);

            // s0;s2;s4;s6;0;0;0;0, 0;0;0;0;s8;s10;s12;s14;
            xmm0 = vmul_vv_u8m1(xmm0, maskmm0, vl);
            xmm1 = vmul_vv_u8m1(xmm1, maskmm1, vl);

            // s0;s2;s4;s6;s8;s10;s12;s14
            xmm0 = vadd_vv_u8m1(xmm0, xmm1, vl);

            vse8_v_u8m1(dst, xmm0, vl);
        }
        pSrc = pSrc + width * downscale;
        pDst = pDst + dstWidth;
    }

    vl = tail;
    for (int y = 0; y < dstHeight; y++) {
        uint8_t* src = pSrc + endRow * downscale;
        uint8_t* dst = pDst + endRow;

        // To avoid vsetvli we load two times and do some special moves
        // comments are for vl=8, idea for vl=16 is the same.
        // s0;s1;s2;s3;s4;s5;s6;s7, s8;s9;s10;s11;s12;s13;s14;s15
        srcmm0 = vle8_v_u8m1(src, vl);
        srcmm1 = vle8_v_u8m1((uint8_t *)(src + vl), vl);

        // s0;s2;s4;s6;s0;s2;s4;s6, s8;s10;s12;s14;s8;s10;s12;s14;
        xmm0 = vrgather_vv_u8m1(srcmm0, idxmm, vl);
        xmm1 = vrgather_vv_u8m1(srcmm1, idxmm, vl);

        // s0;s2;s4;s6;0;0;0;0, 0;0;0;0;s8;s10;s12;s14;
        xmm0 = vmul_vv_u8m1(xmm0, maskmm0, vl);
        xmm1 = vmul_vv_u8m1(xmm1, maskmm1, vl);

        // s0;s2;s4;s6;s8;s10;s12;s14
        xmm0 = vadd_vv_u8m1(xmm0, xmm1, vl);

        vse8_v_u8m1(dst, xmm0, vl);
        pSrc = pSrc + width * downscale;
        pDst = pDst + dstWidth;
    }  
}


// Alpha compositing
/*
void impl_alphaCompositing(cv::Mat& foreground, cv::Mat& background, cv::Mat& alpha, cv::Mat& dst)
{
    static_cast<void>(foreground);
    static_cast<void>(background);
    static_cast<void>(alpha);
    static_cast<void>(dst);
}
*/
