#include <cuda_runtime.h> 
#include <cuda.h> 
#include "utils.h"

/*
 * The launcher for your kernels. 
 * This is a single entry point and 
 * all arrays **MUST** be pre-allocated 
 * on device. you must implement all other 
 * kernels in the respective files.
 * */ 



void your_gauss_blur(uchar4* d_imrgba, uchar4 *d_oimrgba, size_t rows, size_t cols, 
        unsigned char *d_red, unsigned char *d_green, unsigned char *d_blue, 
        unsigned char *d_rblurred, unsigned char *d_gblurred, unsigned char *d_bblurred,  
        float *d_filter, int filterWidth);

void your_gauss_blur_shared(uchar4* d_imrgba, uchar4 *d_oimrgba, size_t rows, size_t cols, 
        unsigned char *d_red, unsigned char *d_green, unsigned char *d_blue, 
        unsigned char *d_rblurred, unsigned char *d_gblurred, unsigned char *d_bblurred,  
        float *d_filter, int filterWidth);

void your_gauss_blur_separable_row_megan(uchar4* d_imrgba, uchar4 *d_oimrgba, size_t rows, size_t cols, 
        unsigned char *d_red, unsigned char *d_green, unsigned char *d_blue, 
        unsigned char *d_rblurred, unsigned char *d_gblurred, unsigned char *d_bblurred,  
        float *d_filter, int filterWidth, float* temp);

void your_gauss_blur_separable_row_max(uchar4* d_imrgba, uchar4 *d_oimrgba, size_t rows, size_t cols, 
        unsigned char *d_red, unsigned char *d_green, unsigned char *d_blue, 
        unsigned char *d_rblurred, unsigned char *d_gblurred, unsigned char *d_bblurred,  
        float *d_filter, int filterWidth, float* partial_rsum, float* partial_gsum, float* partial_bsum);