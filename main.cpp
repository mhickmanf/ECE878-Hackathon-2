#include <iostream>
#include <cstdio>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cassert>
#include <cstdio>
#include <string>
#include <opencv2/opencv.hpp>
#include <cmath>
#include <chrono>

#include "utils.h"
#include "gaussian_kernel.h"

/* 
 * Compute if the two images are correctly 
 * computed. The reference image can 
 * either be produced by a software or by 
 * your own serial implementation.
 * */
void checkApproxResults(unsigned char *ref, unsigned char *gpu, size_t numElems){

    for (int i = 0; i < numElems; i++){
        if (ref[i] - gpu[i] > 1){
            std::cerr << "Error at position " << i << "\n";

            std::cerr << "Reference:: " << std::setprecision(17) << +ref[i] << "\n";
            std::cerr << "GPU:: " << +gpu[i] << "\n";

            exit(1);
        }
    }
}

void checkResult(const std::string &reference_file, const std::string &output_file, float eps){
    cv::Mat ref_img, out_img;

    ref_img = cv::imread(reference_file, -1);
    out_img = cv::imread(output_file, -1);

    unsigned char *refPtr = ref_img.ptr<unsigned char>(0);
    unsigned char *oPtr = out_img.ptr<unsigned char>(0);

    checkApproxResults(refPtr, oPtr, ref_img.rows * ref_img.cols * ref_img.channels());
    std::cout << "PASSED!\n";
}


void gaussian_blur_filter(float *arr, const int f_sz, const float f_sigma = 0.2){
    float filterSum = 0.f;
    float norm_const = 0.0; // normalization const for the kernel

    for (int r = -f_sz / 2; r <= f_sz / 2; r++){
        for (int c = -f_sz / 2; c <= f_sz / 2; c++){ // assuming c_sz was supposed to be f_sz
            float fSum = expf(-(float)(r * r + c * c) / (2 * f_sigma * f_sigma));
            arr[(r + f_sz / 2) * f_sz + (c + f_sz / 2)] = fSum;
            filterSum += fSum;
        }
    }

    norm_const = 1.f / filterSum;

    for (int r = -f_sz / 2; r <= f_sz / 2; ++r){
        for (int c = -f_sz / 2; c <= f_sz / 2; ++c){
            arr[(r + f_sz / 2) * f_sz + (c + f_sz / 2)] *= norm_const;
        }
    }
}



// Serial implementations of kernel functions
void serialGaussianBlur(unsigned char *in, unsigned char *out, const int rows, const int cols, float *filter, const int filterWidth){

    int filterRadius = (filterWidth-1)/2;

    // get pixel
    for (int row = 0; row < rows; row++){
        for (int col = 0; col < cols; col++){
            int offset = row*cols + col;
            
            float blur_value = 0;

            // calculate gaussian blur from filter
            int filter_offset = 0;
            for (int i = row - filterRadius; i <= row + filterRadius; i++){
                for (int j = col - filterRadius; j <= col + filterRadius; j++){

                    // check bounds
                    if ((i < rows) && (j < cols) && (i >= 0) && (j >= 0)){
                        int pixel_offset = i*cols + j;

                        float pixel_value = (float)in[pixel_offset];

                        // get filter pixel
                        float filter_value = filter[filter_offset];                        
                        blur_value = blur_value + pixel_value*filter_value;

                    }

                    filter_offset = filter_offset + 1;
                }
            }
            
            out[offset] = (unsigned char)blur_value;
        }
    }


}

// Separate the input RGBA image into three channels R, G, B.
void serialSeparateChannels(uchar4 *imrgba, unsigned char *r, unsigned char *g, unsigned char *b,
    const int rows, const int cols){
    
    for (int i = 0; i < rows; i++){
        for (int j = 0; j < cols; j++){
            int offset = i*cols + j;

            uchar4 rgb_pixel = imrgba[offset];
            r[offset] = rgb_pixel.x; // red value
            g[offset] = rgb_pixel.y; // green value
            b[offset] = rgb_pixel.z; // blue value
        }
    }
}

// Recombine  the  blurred  channel  into  one  single  uchar4  image.
void serialRecombineChannels(unsigned char *r, unsigned char *g, unsigned char *b, uchar4 *orgba, const int rows, const int cols){
    int offset;
    for (int i = 0; i < rows; i++){
        for (int j = 0; j < cols; j++){
            offset = i*cols + j;
            uchar4 rgba = make_uchar4(b[offset], g[offset], r[offset], 255);
            orgba[offset] = rgba;
        }
    }
}

void serial_gauss_blur(uchar4* imrgba, uchar4 *oimrgba, size_t rows, size_t cols, 
    unsigned char *red, unsigned char *green, unsigned char *blue, 
    unsigned char *rblurred, unsigned char *gblurred, unsigned char *bblurred, float *filter,  int filterWidth){

    // separate
    auto serial_start = std::chrono::steady_clock::now();
    serialSeparateChannels(imrgba, red, green, blue, rows, cols);
    auto serial_end = std::chrono::steady_clock::now();
    std::chrono::duration<double> serial_elapsed_seconds = serial_end-serial_start;
    std::cout << "serial elapsed time: Separate Channels: " << serial_elapsed_seconds.count()*(1e+6) << " us\n";

    // gauss r
    serial_start = std::chrono::steady_clock::now();
    serialGaussianBlur(red, rblurred, rows, cols, filter, filterWidth);
    // g
    serialGaussianBlur(green, gblurred, rows, cols, filter, filterWidth);
    //b
    serialGaussianBlur(blue, bblurred, rows, cols, filter, filterWidth);
    serial_end = std::chrono::steady_clock::now();
    serial_elapsed_seconds = serial_end-serial_start;
    std::cout << "serial elapsed time: Avg. Gaussian Blur:  " << (serial_elapsed_seconds.count()/3)*(1e+6) << " us\n";
    
    // gather
    serial_start = std::chrono::steady_clock::now();
    serialRecombineChannels(rblurred, gblurred, bblurred, oimrgba, rows, cols);
    serial_end = std::chrono::steady_clock::now();
    serial_elapsed_seconds = serial_end-serial_start;
    std::cout << "serial elapsed time: Gather Channels:  " << serial_elapsed_seconds.count()*(1e+6) << " us\n";

}

int main(int argc, char const *argv[]){

    uchar4 *h_in_img, *h_o_img; // pointers to the actual image input and output pointers
    uchar4 *d_in_img, *d_o_img;
    uchar4 *r_o_img; // added by me

    unsigned char *h_red, *h_green, *h_blue;
    unsigned char *d_red, *d_green, *d_blue;
    unsigned char *h_red_blurred, *h_green_blurred, *h_blue_blurred; // added by me
    unsigned char *d_red_blurred, *d_green_blurred, *d_blue_blurred;

    float *h_filter, *d_filter;
    cv::Mat imrgba, o_img, ro_img;

    const int fWidth = 9;
    const float fDev = 2;
    std::string infile;
    std::string outfile;
    std::string reference;

    switch (argc){
    case 2:
        infile = std::string(argv[1]);
        outfile = "blurred_gpu.png";
        reference = "blurred_serial.png";
        break;
    case 3:
        infile = std::string(argv[1]);
        outfile = std::string(argv[2]);
        reference = "blurred_serial.png";
        break;
    case 4:
        infile = std::string(argv[1]);
        outfile = std::string(argv[2]);
        reference = std::string(argv[3]);
        break;
    default:
        std::cerr << "Usage ./gblur <in_image> <out_image> <reference_file> \n";
        exit(1);
    }

    // preprocess
    cv::Mat img = cv::imread(infile.c_str(), cv::IMREAD_COLOR);
    if (img.empty())
    {
        std::cerr << "Image file couldn't be read, exiting\n";
        exit(1);
    }

    cv::cvtColor(img, imrgba, cv::COLOR_BGR2RGBA);

    o_img.create(img.rows, img.cols, CV_8UC4);
    ro_img.create(img.rows, img.cols, CV_8UC4);

    const size_t numPixels = img.rows * img.cols;

    h_in_img = imrgba.ptr<uchar4>(0);
    h_o_img = (uchar4 *)o_img.ptr<unsigned char>(0);
    r_o_img = (uchar4 *)ro_img.ptr<unsigned char>(0);

    // below added by me
    h_red = new unsigned char[numPixels];
    h_green = new unsigned char[numPixels];
    h_blue = new unsigned char[numPixels];
    h_red_blurred = new unsigned char[numPixels];
    h_green_blurred = new unsigned char[numPixels];
    h_blue_blurred = new unsigned char[numPixels];
    
    // allocate the memories for the device pointers

    // filter allocation
    h_filter = new float[fWidth * fWidth];
    gaussian_blur_filter(h_filter, fWidth, fDev); // create a filter of 9x9 with std_dev = 0.2

    printArray<float>(h_filter, 81); // printUtility.

    // copy the image and filter over to GPU here
    checkCudaErrors(cudaMalloc((void**)&d_in_img, sizeof(uchar4)*numPixels));
    checkCudaErrors(cudaMalloc((void**)&d_o_img, sizeof(uchar4)*numPixels));
    checkCudaErrors(cudaMalloc((void**)&d_red, sizeof(unsigned char)*numPixels));
    checkCudaErrors(cudaMalloc((void**)&d_green, sizeof(unsigned char)*numPixels));
    checkCudaErrors(cudaMalloc((void**)&d_blue, sizeof(unsigned char)*numPixels));
    checkCudaErrors(cudaMalloc((void**)&d_red_blurred, sizeof(unsigned char)*numPixels));
    checkCudaErrors(cudaMalloc((void**)&d_green_blurred, sizeof(unsigned char)*numPixels));
    checkCudaErrors(cudaMalloc((void**)&d_blue_blurred, sizeof(unsigned char)*numPixels));
    checkCudaErrors(cudaMalloc((void**)&d_filter, sizeof(float)*fWidth*fWidth));
    
    checkCudaErrors(cudaMemcpy(d_in_img, h_in_img, sizeof(uchar4)*numPixels, cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_filter, h_filter, sizeof(float)*fWidth*fWidth, cudaMemcpyHostToDevice));

    // kernel launch code
    //your_gauss_blur(d_in_img, d_o_img, img.rows, img.cols, d_red, d_green, d_blue, d_red_blurred, d_green_blurred, d_blue_blurred, d_filter, fWidth);
    your_gauss_blur_shared(d_in_img, d_o_img, img.rows, img.cols, d_red, d_green, d_blue, d_red_blurred, d_green_blurred, d_blue_blurred, d_filter, fWidth);

    cudaDeviceSynchronize();
    checkCudaErrors(cudaGetLastError());
    std::cout << "Finished kernel launch \n";
    
    // memcpy the output image to the host side.
    checkCudaErrors(cudaMemcpy(h_o_img, d_o_img, numPixels*sizeof(uchar4), cudaMemcpyDeviceToHost));

    // perform serial memory allocation and function calls, final output should be stored in *r_o_img
    //  ** there are many ways to perform timing in c++ such as std::chrono **
    serial_gauss_blur(h_in_img, r_o_img, img.rows, img.cols, h_red, h_green, h_blue, h_red_blurred, h_green_blurred, h_blue_blurred,  h_filter, fWidth);

    // create the image with the output data
    cv::Mat output(img.rows, img.cols, CV_8UC4, (void *)h_o_img); // generate GPU output image.
    bool suc = cv::imwrite(outfile.c_str(), output);
    if (!suc){
        std::cerr << "Couldn't write GPU image!\n";
        exit(1);
    }
    
    cv::Mat output_s(img.rows, img.cols, CV_8UC4, (void *)r_o_img); // generate serial output image.
    suc = cv::imwrite(reference.c_str(), output_s);
    if (!suc){
        std::cerr << "Couldn't write serial image!\n";
        exit(1);
    }

    // check if the caclulation was correct to a degree of tolerance
    checkResult(reference, outfile, 1e-5);

    // free any necessary memory.
    cudaFree(d_in_img);
    cudaFree(d_o_img);
    cudaFree(d_red);
    cudaFree(d_green);
    cudaFree(d_blue);
    cudaFree(d_red_blurred);
    cudaFree(d_green_blurred);
    cudaFree(d_blue_blurred);
    
    delete[] h_filter;
    delete[] h_red;
    delete[] h_green;
    delete[] h_blue;
    delete[] h_red_blurred;
    delete[] h_green_blurred;
    delete[] h_blue_blurred;

    return 0;
}