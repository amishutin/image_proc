#include "c_impl.hpp"
#include <stdint.h>


void c_rgb2gray_c3(uint8_t *pSrc, uint8_t *pDst, int height, int width)
{
    const float cR = 0.299f;
    const float cG = 0.587f;
    const float cB = 0.114f;

    uint64_t len = height * width;
    for (uint64_t i = 0; i < len; ++i) 
    {
        pDst[i] = (uint8_t)(cR * pSrc[i * 3] + cG * pSrc[i * 3 + 1] + cB * pSrc[i * 3 + 2]);
    }
}

void c_threshold_c1(uint8_t *pSrc, uint8_t *pDst, int height, int width, uint8_t thresh)
{
    uint64_t len = height * width;
    for (uint64_t i = 0; i < len; ++i) 
    {
        pDst[i] = (pSrc[i] <= thresh)? pSrc[i] : thresh;
    }
}

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

void c_blur_c3(uint8_t* src, uint8_t* dst, int st, int kh, int kw, int h, int w)
{
  int j, i, y, x;
  uint16_t div = kw*kh;
  for (i = 0; i < h; i++) {
    for (j = 0; j < w; j++) {
      uint8_t* s = src + 3 * j;
      uint8_t* d = dst + 3 * j;
      uint16_t sumr = 0;
      uint16_t sumg = 0;
      uint16_t sumb = 0;
      for (y = 0; y < kh; y++) {
        for (x = 0; x < kw; x++) {
          sumr = sumr + s[st*y + 3 * x + 0];
          sumg = sumg + s[st*y + 3 * x + 1];
          sumb = sumb + s[st*y + 3 * x + 2];
        }
      }
      d[0] = (uint8_t)(sumr / div);
      d[1] = (uint8_t)(sumg / div);
      d[2] = (uint8_t)(sumb / div);
    }
    src = src + st;
    dst = dst + st;
  }
}

// erode with 3x3 kernel for binary images
void c_morphology(cv::Mat &src, cv::Mat &dst)
{
    int idx;
    bool result = 0;
    uint8_t* pSrc = reinterpret_cast<uint8_t*>(src.data);
    uint8_t* pDst = reinterpret_cast<uint8_t*>(dst.data);
    for (int i = 1; i < src.rows - 1; i++)
    {
        for (int j = 1; j < src.cols - 1; j++)
        {
            idx = (i * src.cols + j);
            dst.at<uint8_t>(idx) = (src.at<uint8_t>(idx - src.cols - 1) & src.at<uint8_t>(idx - src.cols) &
                      src.at<uint8_t>(idx - src.cols + 1) & src.at<uint8_t>(idx - 1) &
                      src.at<uint8_t>(idx) & src.at<uint8_t>(idx + 1) & src.at<uint8_t>(idx + src.cols - 1) &
                      src.at<uint8_t>(idx + src.cols) & src.at<uint8_t>(idx + src.cols + 1));
        }
    }
}

// Alpha compositing
void c_impl_alphaCompositing(cv::Mat& foreground, cv::Mat& background, cv::Mat& alpha, cv::Mat& dst)
{
    int pixelNum = foreground.rows * foreground.cols * foreground.channels();

    float* pFg = reinterpret_cast<float*>(foreground.data);
    float* pBg = reinterpret_cast<float*>(background.data);
    float* pA = reinterpret_cast<float*>(alpha.data);
    float* pDst = reinterpret_cast<float*>(dst.data);

    for(uint32_t i = 0; i < pixelNum; ++i, ++pFg, ++pBg, ++pA, ++pDst)
    {
        *pDst = (*pFg)*(*pA) + (*pBg)*(1 - *pA);
    }
}