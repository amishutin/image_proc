#include "opencv2/core.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/videoio.hpp"
#include <riscv_vector.h>
#include <iostream>

using namespace cv;
static std::string img_path = "";
using namespace std;

void refAlphaCompositing(Mat& foreground, Mat& background, Mat& alpha, Mat& outImage)
{
    multiply(alpha, foreground, foreground);
    // Multiply the foreground with the alpha
    multiply(alpha, foreground, foreground); 
 
    // Multiply the background with ( 1 - alpha )
    multiply(Scalar::all(1.0)-alpha, background, background); 
 
    // Add the masked foreground and background.
    add(foreground, background, outImage);  
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
    auto img = imread(img_path + "/pochita.png");

    // RGB2GRAY conversion
    Mat img_gray, img_thr;
    cvtColor(img, img_gray, COLOR_RGB2GRAY);
    imwrite(img_path + "/pochita_gray.png", img_gray);

    // Threshold function
    threshold(img_gray, img_thr, 125, 0, THRESH_TRUNC);
    imwrite(img_path + "/pochita_thr.png", img_thr);

    //Box filter
    Mat img_box_filter;
    boxFilter(img_gray, img_box_filter, -1, Size(5,5));
    imwrite(img_path + "/pochita_box_filt.png", img_box_filter);

    // Morphology operation - opening
    Mat img_morph;
    morphologyEx(img_gray, img_morph, MORPH_OPEN, getStructuringElement(MORPH_RECT, Size(5,5)));
    imwrite(img_path + "/pochita_morph_open.png", img_morph);

    // Up/Down scaling 2x
    Mat img_up2, img_dn2;
    resize(img_gray, img_up2, Size(img_gray.cols * 2, img_gray.rows * 2), 0, 0, INTER_LINEAR);
    imwrite(img_path + "/pochita_up2.png", img_up2);
    resize(img_gray, img_dn2, Size(img_gray.cols * 0.5, img_gray.rows * 0.5), 0, 0, INTER_LINEAR);
    imwrite(img_path + "/pochita_dn2.png", img_dn2);

    // Alpha compositing
    auto factory = imread(img_path + "/factory.png");
    Mat foreground = imread(img_path + "/pochita.png");
    Mat alpha, background;
    factory.copyTo(background);
    alpha = imread(img_path + "/alpha.png");
    // Convert Mat to float data type
    foreground.convertTo(foreground, CV_32FC3);
    background.convertTo(background, CV_32FC3);
 
    // Normalize the alpha mask to keep intensity between 0 and 1
    alpha.convertTo(alpha, CV_32FC3, 1.0/255);
 
    // Storage for output image
    Mat refOutImage = Mat::zeros(foreground.size(), foreground.type());
    refAlphaCompositing(foreground, background, alpha, refOutImage);
    imwrite(img_path + "/composed.png", refOutImage);

    return 0;
}

