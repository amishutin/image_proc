#include "opencv2/core.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/videoio.hpp"
#include "implementation.hpp"
#include "ref_implementation.hpp"
#include <iostream>
#include <string>

using namespace cv;
static std::string img_path = "";

// RGB2GRAY conversion
void run_rgb2gray_functional(Mat& src)
{
    Mat dst_ref, diff_image;
    ref_rgb2gray(src, dst_ref);
    Mat dst = Mat::zeros(dst_ref.size(), dst_ref.type());
    impl_rgb2gray(src, dst);
    imwrite(img_path + "/dst_image_rgb2gray.png", dst);
    absdiff(dst, dst_ref, diff_image);
    imwrite(img_path + "/diff_image_rgb2gray.png", diff_image);

}

// Threshold function
void run_threshold_functional(cv::Mat &src)
{
    Mat dst_ref, diff_image;
    const double thresh = 125., maxval = 0.;
    const int type = THRESH_TRUNC;
    ref_threshold(src, dst_ref, thresh, maxval, type);
    Mat dst = Mat::zeros(dst_ref.size(), dst_ref.type());
    impl_threshold(src, dst, thresh, maxval, type);
    imwrite(img_path + "/dst_image_threshold.png", dst);
    absdiff(dst, dst_ref, diff_image);
    imwrite(img_path + "/diff_image_threshold.png", diff_image);
}

//Box filter
void run_boxFilter_functional(cv::Mat &src)
{
    Mat dst_ref, diff_image;
    const int ddepth = -1;
    const Size size = Size(5,5);
    ref_boxFilter(src, dst_ref, ddepth, size);
    Mat dst = Mat::zeros(dst_ref.size(), dst_ref.type());
    impl_boxFilter(src, dst, ddepth, size);
    imwrite(img_path + "/dst_image_boxFilter.png", dst);
    absdiff(dst, dst_ref, diff_image);
    imwrite(img_path + "/diff_image_boxFilter.png", diff_image);
}

// Morphology operation - erosion
void run_morphology_functional(cv::Mat &src)
{
    Mat dst_ref, diff_image;
    ref_morphology(src, dst_ref);
    Mat dst = Mat::zeros(dst_ref.size(), dst_ref.type());
    impl_morphology(src, dst);
    imwrite(img_path + "/dst_image_morphology.png", dst);
    absdiff(dst, dst_ref, diff_image);
    imwrite(img_path + "/diff_image_morphology.png", diff_image);
}

// Upscaling 2x
void run_upscale2x_functional(cv::Mat &src)
{
    Mat dst_ref, diff_image;
    ref_upscale2x(src, dst_ref);
    Mat dst = Mat::zeros(dst_ref.size(), dst_ref.type());
    impl_upscale2x(src, dst);
    imwrite(img_path + "/dst_image_upscale2x.png", dst);
    absdiff(dst, dst_ref, diff_image);
    imwrite(img_path + "/diff_image_upscale2x.png", diff_image);
}

// Downscaling 2x
void run_downscale2x_functional(cv::Mat &src)
{
    Mat dst_ref, diff_image;
    ref_downscale2x(src, dst_ref);
    Mat dst = Mat::zeros(dst_ref.size(), dst_ref.type());
    impl_downscale2x(src, dst);
    imwrite(img_path + "/dst_image_downscale2x.png", dst);
    absdiff(dst, dst_ref, diff_image);
    imwrite(img_path + "/diff_image_downscale2x.png", diff_image);
}

// Alpha compositing
void run_alphaCompositing_functional(cv::Mat& foreground, cv::Mat& background, cv::Mat& alpha)
{
    Mat dst_ref, diff_image;
    ref_alphaCompositing(foreground, background, alpha, dst_ref);
    imwrite(img_path + "/ref_image_alpha_compose.png", dst_ref);
    Mat dst = Mat::zeros(foreground.size(), foreground.type());
    impl_alphaCompositing(foreground, background, alpha, dst);
    imwrite(img_path + "/dst_image_alpha_compose.png", dst);
    absdiff(dst, dst_ref, diff_image);
    imwrite(img_path + "/diff_image_alpha_compose.png", diff_image);
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
        std::cout << "\nUsage: functional_main --img_path=\n";
        return -1;
    }

    // [[for alpha compositing]]
    Mat foreground = imread(img_path + "/pochita.png");
    Mat background = imread(img_path + "/factory.png");
    Mat alpha = imread(img_path + "/alpha.png");
    Mat erode_img = imread(img_path + "/erode_img.png", IMREAD_GRAYSCALE);
    // Convert Mat to float data type
    foreground.convertTo(foreground, CV_32FC3);
    background.convertTo(background, CV_32FC3); 
    // Normalize the alpha mask to keep intensity between 0 and 1
    alpha.convertTo(alpha, CV_32FC3, 1.0/255);

    Mat src = imread(img_path + "/pochita.png");
    Mat src_gray;
    cvtColor(src, src_gray, COLOR_RGB2GRAY);


    // run_rgb2gray_functional(src);
    // run_threshold_functional(src_gray);
    // run_boxFilter_functional(src);
    run_morphology_functional(erode_img);
    // run_upscale2x_functional(src);
    // run_downscale2x_functional(src);
    // run_alphaCompositing_functional(foreground, background, alpha);

    return 0;
}