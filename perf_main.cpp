#include "opencv2/core.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/videoio.hpp"
#include "implementation.hpp"
#include "ref_implementation.hpp"
#include "c_impl.hpp"
#include <iostream>
#include <string>
#include <chrono>

#define TIME_SETUP()\
    auto start = std::chrono::steady_clock::now();     \
    auto end = std::chrono::steady_clock::now();       \
    std::chrono::duration<double> diff;                \

#define TIME_MEASURE(FUNC, NAME)                       \
    start = std::chrono::steady_clock::now();          \
    for (uint32_t i = 0U; i < repetitions; ++i)        \
    {                                                  \
        FUNC;                                          \
    }                                                  \
    end = std::chrono::steady_clock::now();            \
    diff = end - start;                                \
    std::cout << "\n" << NAME << " took " << diff.count() / static_cast<double>(repetitions) << '\n'; \

using namespace cv;
static std::string img_path = "";

static constexpr uint32_t repetitions = 10U;

// RGB2GRAY conversion
void run_rgb2gray_perf(Mat& src)
{
    Mat dst_ref;
    Mat c_dst_ref = Mat::zeros(src.size(), src.type());
    Mat dst = Mat::zeros(src.size(), src.type());
    TIME_SETUP();

    TIME_MEASURE(ref_rgb2gray(src, dst_ref), "Reference rgb2gray");

    TIME_MEASURE(c_rgb2gray_c3(reinterpret_cast<uint8_t*>(src.data), reinterpret_cast<uint8_t*>(c_dst_ref.data), src.rows, src.cols), "C_Impl rgb2gray");

    TIME_MEASURE(impl_rgb2gray(src, dst), "RVV0.7 rgb2gray");
}

// Threshold function
void run_threshold_perf(cv::Mat &src)
{
    Mat dst_ref;
    Mat c_dst_ref = Mat::zeros(src.size(), src.type());
    Mat dst = Mat::zeros(src.size(), src.type());
    const double thresh = 125., maxval = 0.;
    const int type = THRESH_TRUNC;
    
    TIME_SETUP();

    TIME_MEASURE(ref_threshold(src, dst_ref, thresh, maxval, type), "Reference threshold");

    TIME_MEASURE(c_threshold_c1(reinterpret_cast<uint8_t*>(src.data), reinterpret_cast<uint8_t*>(c_dst_ref.data), src.rows, src.cols, 1), "C_Impl threshold");

    TIME_MEASURE(impl_threshold(src, dst, thresh, maxval, type), "RVV0.7 threshold");
}

// Morphology operation - opening
void run_morphology_perf(cv::Mat &src)
{
    Mat dst_ref;
    Mat c_dst_ref = Mat::zeros(src.size(), src.type());
    Mat dst = Mat::zeros(src.size(), src.type());
    const int op = MORPH_OPEN;

    TIME_SETUP();

    TIME_MEASURE(ref_morphology(src, dst_ref), "Reference morphology");

    TIME_MEASURE(c_morphology(src, c_dst_ref), "C_Impl morphology");

    TIME_MEASURE(impl_morphology(src, dst), "RVV0.7 morphology");
}

// Upscaling 2x
void run_upscale2x_perf(cv::Mat &src)
{
    Mat dst_ref;
    Mat dst = Mat::zeros(src.size(), src.type());

    TIME_SETUP();

    TIME_MEASURE(ref_upscale2x(src, dst_ref), "Reference upscale2x");

    TIME_MEASURE(impl_upscale2x(src, dst), "RVV0.7 upscale2x");
}

// Downscaling 2x
void run_downscale2x_perf(cv::Mat &src)
{
    Mat dst_ref;
    Mat c_dst_ref = Mat::zeros(src.rows / 2, src.cols / 2, src.type());
    Mat dst = Mat::zeros(src.rows / 2, src.cols / 2, src.type());

    TIME_SETUP();

    TIME_MEASURE(ref_downscale2x(src, dst_ref), "Reference downscale2x");

    TIME_MEASURE(c_downscale2x_c1(reinterpret_cast<uint8_t*>(src.data), reinterpret_cast<uint8_t*>(c_dst_ref.data), src.rows, src.cols), "C_Impl downscale2x");

    TIME_MEASURE(impl_downscale2x(src, dst), "RVV0.7 downscale2x");
}

// Alpha compositing
void run_alphaCompositing_perf(cv::Mat& foreground, cv::Mat& background, cv::Mat& alpha)
{
    Mat dst_ref;
    Mat c_dst_ref = Mat::zeros(foreground.size(), foreground.type());
    Mat dst = Mat::zeros(foreground.size(), foreground.type());

    TIME_SETUP();

    TIME_MEASURE(ref_alphaCompositing(foreground, background, alpha, dst_ref), "Reference alphacompositing");

    TIME_MEASURE(c_impl_alphaCompositing(foreground, background, alpha, c_dst_ref), "C_Impl alphacompositing");

    TIME_MEASURE(impl_alphaCompositing(foreground, background, alpha, dst), "RVV0.7 alphacompositing");    
}


int main(int argc, char **argv)
{
    // set images path
    if (argc > 1)
    {
        int argi = 1;
        for (argi; argi < argc; argi++)
        {
            if (strstr(argv[argi], "--img_path="))
            {
                img_path = argv[argi];
                const std::string ext("--img_path=");
                if (img_path != ext &&
                    img_path.size() > ext.size() &&
                    img_path.substr(0,ext.size()) == "--img_path=")
                {
                    img_path = img_path.substr(ext.size(), img_path.size() - ext.size());
                }
                std::cout << "\nimg_path=" << img_path << std::endl;
            }
        }
    }
    else
    {
        std::cout << "\nSet path to the images!\n";
        std::cout << "\nUsage: perf_main --img_path=\n";
        return -1;
    }

    // [[for alpha compositing]]
    Mat foreground = imread(img_path + "/pochita.png");
    Mat background = imread(img_path + "/factory.png");
    Mat alpha = imread(img_path + "/alpha.png");
    // Convert Mat to float data type
    foreground.convertTo(foreground, CV_32FC3);
    background.convertTo(background, CV_32FC3); 
    // Normalize the alpha mask to keep intensity between 0 and 1
    alpha.convertTo(alpha, CV_32FC3, 1.0/255);

    // [[for morphology and threshold]]
    Mat src = imread(img_path + "/pochita.png");
    Mat src_gray;
    cvtColor(src, src_gray, COLOR_RGB2GRAY);

    std::cout.precision(10);
    // run_rgb2gray_perf(src);
    // run_threshold_perf(src_gray);
    // run_morphology_perf(src_gray);
    // run_upscale2x_perf(src);
    run_downscale2x_perf(src_gray);
    // run_alphaCompositing_perf(foreground, background, alpha);
    return 0;
}