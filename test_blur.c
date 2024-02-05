#include <stdint.h>
#include <stdio.h>
#include <malloc.h>
#include <def.h>
#include <riscv_vector.h>

typedef unsigned char uchar; 

void print_c3(char* s, uchar* src, int st, int h, int w)
{
    printf("%s:\n", s);
    int j, i, y, x;
    for (i = 0; i < h; i++) {
        for (j = 0; j < w; j++) {
            uchar* s = src + 3 * j;
            printf("(%3d %3d %3d) ", s[0],s[1],s[2]);
        }
        src = src + st;
        printf("\n");
    }
}
void blur_c3(uint8_t* src, uint8_t* dst, int st, int kh, int kw, int h, int w)
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



void blur_c3_rvv(uchar* src, uchar* dst, int st, int kh, int kw, int h, int w)
{
  int j, i, y, x;
  size_t vl = vsetvl_e8m1(12);

  vuint16m2_t  vn = vmv_v_x_u16m2(0, vl);
  vuint16m2_t  vd = vmv_v_x_u16m2(kh*kw, vl);

  vuint8m1_t  v0;
  vuint16m2_t v1;
  vuint16m2_t vs;

  for (i = 0; i < h; i++) {
    for (j = 0; j < w; j+=4) {
      uchar* s = src + 3 * j;
      uchar* d = dst + 3 * j;
      vs = vn;
      for (y = 0; y < kh; y++) {
        for (x = 0; x < kw; x++) {
          uchar* p= s+st*y + 3 *x;
          v0 = vle8_v_u8m1 (p, vl);
          v1 = vwcvtu_x_x_v_u16m2(v0, vl);
          vs = vadd_vv_u16m2 (vs, v1, vl);
        }
      }
      vs = vdivu_vv_u16m2(vs, vd, vl);
      v0 = vncvt_x_x_w_u8m1(vs, vl);
      vse8_v_u8m1((uint8_t*)d, v0, vl);
    }
    src = src + st;
    dst = dst + st;
  }
}


void run_blur(void)
{
   
   printf("\n\n**********\n");
   printf("blur\n");
   int kh=3;
   int kw=3;
   //int width=9;
   //int height=9;

   int width =128;
   int height=128;
   
   uchar* src;
   uchar* dst;
   uchar* dst1;
   int st = width*3;

   
   src  = malloc(height*st+1024);//buffer for input image
   dst  = malloc(height*st+1024);//buffer for output image
   dst1 = malloc(height*st+1024);//buffer for output image
   int i, j, n, c;
   for (i = 0; i < height; i++) {
       for (j = 0; j < 3*width; j++) {
           src[3 * width*i + j] = 1+i+j;
           dst[3 * width*i + j] = 0;
       }
   }
   int dwidth =width -2;
   int dheight=height-2;
   double time_c=0.0f, time_rvv=0.0f;
   int64_t start_clock, stop_clock;
   int nloop = 100;

   blur_c3(src, dst, st, kh, kw, dheight, dwidth);
   PERF(blur_c3(src, dst, st, kh, kw, dheight, dwidth), time_c)
   printf("time_c  =%15.4f\n", time_c/(double)nloop);

   blur_c3_rvv(src, dst1, st, kh, kw, dheight, dwidth);
   PERF(blur_c3_rvv(src, dst1, st, kh, kw, dheight, dwidth), time_rvv)
   printf("time_rvv=%15.4f\n", time_rvv/(double)nloop);
   if(width<=12){
   print_c3("src", src, st, height, width);
   print_c3("dst", dst, st, dheight, dwidth);
   print_c3("dst1", dst1, st, dheight, dwidth);
   }
}