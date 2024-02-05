#include "opencv2/core.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/videoio.hpp"
#include "implementation.hpp"
#include "ref_implementation.hpp"
#include <iostream>
#include <string>
#include <chrono>

using namespace cv;
static std::string img_path = "";

static constexpr uint32_t repetitions = 1000U;

// RGB2GRAY conversion
void run_rgb2gray_perf(Mat& src)
{
    Mat dst_ref;
    auto start = std::chrono::steady_clock::now();
    for (uint32_t i = 0U; i < repetitions; ++i)
    {
        ref_rgb2gray(src, dst_ref);
    }
    auto end = std::chrono::steady_clock::now();
    std::chrono::duration<double> diff = end - start;
    std::cout << "\nReference rgb2gray took " << diff.count() / static_cast<double>(repetitions) << '\n';

    Mat dst = Mat::zeros(dst_ref.size(), dst_ref.type());
    start = std::chrono::steady_clock::now();
    for (uint32_t i = 0U; i < repetitions; ++i)
    {
        impl_rgb2gray(src, dst_ref);
    }
    end = std::chrono::steady_clock::now();
    diff = end - start;
    std::cout << "\nRVV0.7 rgb2gray took " << diff.count() / static_cast<double>(repetitions) << '\n';
}

// Threshold function
void run_threshold_perf(cv::Mat &src)
{
    Mat dst_ref;
    const double thresh = 125., maxval = 0.;
    const int type = THRESH_TRUNC;
    auto start = std::chrono::steady_clock::now();
    for (uint32_t i = 0U; i < repetitions; ++i)
    {
        ref_threshold(src, dst_ref, thresh, maxval, type);
    }
    auto end = std::chrono::steady_clock::now();
    std::chrono::duration<double> diff = end - start;
    std::cout << "\nReference threshold took " << diff.count() / static_cast<double>(repetitions) << '\n';

    Mat dst = Mat::zeros(dst_ref.size(), dst_ref.type());
     start = std::chrono::steady_clock::now();
    for (uint32_t i = 0U; i < repetitions; ++i)
    {
        impl_threshold(src, dst, thresh, maxval, type);
    }
    end = std::chrono::steady_clock::now();
    diff = end - start;
    std::cout << "\nRVV0.7 threshold took " << diff.count() / static_cast<double>(repetitions) << '\n';
}

//Box filter
void run_boxFilter_perf(cv::Mat &src)
{
    Mat dst_ref;
    const int ddepth = -1;
    const Size size = Size(5,5);

    auto start = std::chrono::steady_clock::now();
    for (uint32_t i = 0U; i < repetitions; ++i)
    {
        ref_boxFilter(src, dst_ref, ddepth, size);
    }
    auto end = std::chrono::steady_clock::now();
    std::chrono::duration<double> diff = end - start;
    std::cout << "\nReference boxFilter took " << diff.count() / static_cast<double>(repetitions) << '\n';

    Mat dst = Mat::zeros(dst_ref.size(), dst_ref.type());
    start = std::chrono::steady_clock::now();
    for (uint32_t i = 0U; i < repetitions; ++i)
    {
        impl_boxFilter(src, dst, ddepth, size);
    }
    end = std::chrono::steady_clock::now();
    diff = end - start;
    std::cout << "\nRVV0.7 boxFilter took " << diff.count() / static_cast<double>(repetitions) << '\n';
}

// Morphology operation - opening
void run_morphology_perf(cv::Mat &src)
{
    Mat dst_ref;
    const int op = MORPH_OPEN;

    auto start = std::chrono::steady_clock::now();
    for (uint32_t i = 0U; i < repetitions; ++i)
    {
        ref_morphology(src, dst_ref);
    }
    auto end = std::chrono::steady_clock::now();
    std::chrono::duration<double> diff = end - start;
    std::cout << "\nReference morphology took " << diff.count() / static_cast<double>(repetitions) << '\n';

    Mat dst = Mat::zeros(dst_ref.size(), dst_ref.type());
    start = std::chrono::steady_clock::now();
    for (uint32_t i = 0U; i < repetitions; ++i)
    {
        impl_morphology(src, dst);
    }
    end = std::chrono::steady_clock::now();
    diff = end - start;
    std::cout << "\nRVV0.7 morphology took " << diff.count() / static_cast<double>(repetitions) << '\n';
    
}

// Upscaling 2x
void run_upscale2x_perf(cv::Mat &src)
{
    Mat dst_ref;

    auto start = std::chrono::steady_clock::now();
    for (uint32_t i = 0U; i < repetitions; ++i)
    {
        ref_upscale2x(src, dst_ref);
    }
    auto end = std::chrono::steady_clock::now();
    std::chrono::duration<double> diff = end - start;
    std::cout << "\nReference upscale2x took " << diff.count() / static_cast<double>(repetitions) << '\n';

    Mat dst = Mat::zeros(dst_ref.size(), dst_ref.type());
    start = std::chrono::steady_clock::now();
    for (uint32_t i = 0U; i < repetitions; ++i)
    {
        impl_upscale2x(src, dst);
    }
    end = std::chrono::steady_clock::now();
    diff = end - start;
    std::cout << "\nRVV0.7 upscale2x took " << diff.count() / static_cast<double>(repetitions) << '\n';
}

// Downscaling 2x
void run_downscale2x_perf(cv::Mat &src)
{
    Mat dst_ref;

    auto start = std::chrono::steady_clock::now();
    for (uint32_t i = 0U; i < repetitions; ++i)
    {
        ref_downscale2x(src, dst_ref);
    }
    auto end = std::chrono::steady_clock::now();
    std::chrono::duration<double> diff = end - start;
    std::cout << "\nReference downscale2x took " << diff.count() / static_cast<double>(repetitions) << '\n';

    Mat dst = Mat::zeros(dst_ref.size(), dst_ref.type());
    start = std::chrono::steady_clock::now();
    for (uint32_t i = 0U; i < repetitions; ++i)
    {
        impl_downscale2x(src, dst);
    }
    end = std::chrono::steady_clock::now();
    diff = end - start;
    std::cout << "\nRVV0.7 downscale2x took " << diff.count() / static_cast<double>(repetitions) << '\n';

}

// Alpha compositing
void run_alphaCompositing_perf(cv::Mat& foreground, cv::Mat& background, cv::Mat& alpha)
{
    Mat dst_ref;

    auto start = std::chrono::steady_clock::now();
    for (uint32_t i = 0U; i < repetitions; ++i)
    {
        ref_alphaCompositing(foreground, background, alpha, dst_ref);
    }
    auto end = std::chrono::steady_clock::now();
    std::chrono::duration<double> diff = end - start;
    std::cout << "\nReference alphaCompositing took " << diff.count() / static_cast<double>(repetitions) << '\n';

    Mat dst = Mat::zeros(dst_ref.size(), dst_ref.type());
    start = std::chrono::steady_clock::now();
    for (uint32_t i = 0U; i < repetitions; ++i)
    {
        impl_alphaCompositing(foreground, background, alpha, dst);
    }
    end = std::chrono::steady_clock::now();
    diff = end - start;
    std::cout << "\nRVV0.7 alphaCompositing took " << diff.count() / static_cast<double>(repetitions) << '\n';
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
    run_rgb2gray_perf(src);
    run_threshold_perf(src_gray);
    run_boxFilter_perf(src);
    run_morphology_perf(src_gray);
    run_upscale2x_perf(src);
    run_downscale2x_perf(src);
    run_alphaCompositing_perf(foreground, background, alpha);
    return 0;
}