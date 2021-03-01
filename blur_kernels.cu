#include "./gaussian_kernel.h"

#define BLOCK 32
#define TILE_WIDTH 40 // BLOCK + (filterWidth - 1)

/*
The actual gaussian blur kernel to be implemented by 
you. Keep in mind that the kernel operates on a 
single channel.
 */
__global__ 
void gaussianBlur_global(unsigned char *d_in, unsigned char *d_out, 
        const int rows, const int cols, float *d_filter, const int filterWidth){

  int filterRadius = (filterWidth-1)/2;

  // get current pixel at (col, row) 
  int col=blockIdx.x*blockDim.x+threadIdx.x;
  int row=blockIdx.y*blockDim.y+threadIdx.y;

  if(col<cols && row<rows) {
    int offset = row*cols + col;
    float blur_value = 0;

    // calculate gaussian blur from filter
    int filter_offset = 0;
    for (int i = row - filterRadius; i <= row + filterRadius; i++){
      for (int j = col - filterRadius; j <= col + filterRadius; j++){

      // check bounds
      if ((i < rows) && (j < cols) && (i >= 0) && (j >= 0)){
        int pixel_offset = i*cols + j;
        float pixel_value = (float)d_in[pixel_offset];
        // get filter pixel
        float filter_value = d_filter[filter_offset]; 
        blur_value = blur_value + pixel_value*filter_value;
      }
      filter_offset = filter_offset + 1;
      }
    }
    d_out[offset] = (unsigned char)blur_value;
  }
}

__global__ 
void gaussianBlur_shared(unsigned char *d_in, unsigned char *d_out, 
        const int rows, const int cols, float *d_filter, const int filterWidth){

}


/*
  Computes the convolution along the row.
*/
__global__ 
void gaussianBlurSeparableRow(float *d_in, unsigned char *d_out, 
	const int rows, const int cols, float *d_filter, const int filterWidth){

  int filterRadius = (filterWidth-1)/2;

  // get current pixel at (col, row)
  int col=blockIdx.x*blockDim.x+threadIdx.x;
  int row=blockIdx.y*blockDim.y+threadIdx.y;



  __syncthreads();
  if(col<cols && row<rows) {
    int offset = row*cols + col;
    float blur_value = 0;

    // calculate gaussian blur from filter
    int filter_offset = 0;
    int i = 0;
    for (int j = -(filterWidth/2) ; j < filterWidth/2;j++){
      // check bounds
      if ((i < rows) && (i >= 0)){
        int pixel_offset = (row+i)*cols + (j+col);
        float pixel_value = (float)d_in[pixel_offset];
        // get filter pixel
        float filter_value = d_filter[(i+(filterWidth/2))*filterWidth + (j+(filterWidth/2))];
        blur_value = blur_value + pixel_value*filter_value;
      }
      filter_offset = filter_offset + 1;
    }
    d_out[offset] = (unsigned char)blur_value;
    //d_out[offset] = (unsigned char)d_in[offset];
  }


}

/*
  Computes the convolution along the column.
*/
__global__ 
void gaussianBlurSeparableColumn(unsigned char *d_in, float *d_out, 
        const int rows, const int cols, float *d_filter, const int filterWidth){

int filterRadius = (filterWidth-1)/2;

  // get current pixel at (col, row)
  int col=blockIdx.x*blockDim.x+threadIdx.x;
  int row=blockIdx.y*blockDim.y+threadIdx.y;

  if(col<cols && row<rows) {
    int offset = row*cols + col;
    float blur_value = 0;

    // calculate gaussian blur from filter
    int filter_offset = 0;
    int j = 0;
    for(int i = -(filterWidth/2) ; i < filterWidth/2;i++){
      // check bounds
      if ((j < cols) && (j >= 0)){

        int pixel_offset = (row+i)*cols + (j+col);
        float pixel_value = (float)d_in[pixel_offset];
        // get filter pixel
        float filter_value = d_filter[(i+(filterWidth/2)) * filterWidth + (filterWidth/2 + j)];
        //float filter_value = d_filter[filter_offset]; 
        blur_value = blur_value + pixel_value*filter_value;
      }
      filter_offset = filter_offset + 1;
    }
    d_out[offset] = blur_value;
    //d_out[offset] = (float)d_in[offset];
    //test
  }
}



/*
  Given an input RGBA image separate 
  that into appropriate rgba channels.
 */
__global__ 
void separateChannels(uchar4 *d_imrgba, unsigned char *d_r, unsigned char *d_g, unsigned char *d_b, const int rows, const int cols){

  // get current pixel at (i,j)
  int i=blockIdx.x*blockDim.x+threadIdx.x;
  int j=blockIdx.y*blockDim.y+threadIdx.y;
        
  if(i<cols && j<rows) {
    
    // get 1D coordinate for the grayscale image
    int offset= j*cols + i;
    uchar4 rgb_pixel = d_imrgba[offset];

    d_r[offset] = rgb_pixel.x; // red value
    d_g[offset] = rgb_pixel.y; // green value
    d_b[offset] = rgb_pixel.z; // blue value
  }
} 
 

/*
  Given input channels combine them 
  into a single uchar4 channel. 

  You can use some handy constructors provided by the 
  cuda library i.e. 
  make_int2(x, y) -> creates a vector of type int2 having x,y components 
  make_uchar4(x,y,z,255) -> creates a vector of uchar4 type x,y,z components 
  the last argument being the transperency value. 
 */
__global__ 
void recombineChannels(unsigned char *d_r, unsigned char *d_g, unsigned char *d_b, uchar4 *d_orgba, const int rows, const int cols){

  // get current pixel at (i,j)
  int i=blockIdx.x*blockDim.x+threadIdx.x;
  int j=blockIdx.y*blockDim.y+threadIdx.y;
        
  if(i<cols && j<rows) {
    // get 1D coordinate for the grayscale image
    int offset= j*cols + i;
    uchar4 rgba = make_uchar4(d_b[offset], d_g[offset], d_r[offset], 255);
    d_orgba[offset] = rgba;
  }
} 


void your_gauss_blur(uchar4* d_imrgba, uchar4 *d_oimrgba, size_t rows, size_t cols, 
        unsigned char *d_red, unsigned char *d_green, unsigned char *d_blue, 
        unsigned char *d_rblurred, unsigned char *d_gblurred, unsigned char *d_bblurred,
        float *d_filter,  int filterWidth,  float *partial_rsum, 
	float *partial_gsum, float *partial_bsum){

        dim3 blockSize(BLOCK,BLOCK,1); // For 2D image
        dim3 gridSize((cols/BLOCK)+1,(rows/BLOCK)+1,1);

        separateChannels<<<gridSize, blockSize>>>(d_imrgba, d_red, d_green, d_blue, rows, cols);
        cudaDeviceSynchronize();
        checkCudaErrors(cudaGetLastError());

        gaussianBlurSeparableColumn<<<gridSize, blockSize>>>(d_red, partial_rsum, rows, cols, d_filter, filterWidth);
        cudaDeviceSynchronize();
        checkCudaErrors(cudaGetLastError());

        gaussianBlurSeparableRow<<<gridSize, blockSize>>>(partial_rsum, d_rblurred, rows, cols, d_filter, filterWidth);
        cudaDeviceSynchronize();
        checkCudaErrors(cudaGetLastError());

	gaussianBlurSeparableColumn<<<gridSize, blockSize>>>(d_green, partial_gsum, rows, cols, d_filter, filterWidth);
        cudaDeviceSynchronize();
        checkCudaErrors(cudaGetLastError());

        gaussianBlurSeparableRow<<<gridSize, blockSize>>>(partial_gsum, d_gblurred, rows, cols, d_filter, filterWidth);
        cudaDeviceSynchronize();
        checkCudaErrors(cudaGetLastError());

	gaussianBlurSeparableColumn<<<gridSize, blockSize>>>(d_blue, partial_bsum, rows, cols, d_filter, filterWidth);
        cudaDeviceSynchronize();
        checkCudaErrors(cudaGetLastError());

        gaussianBlurSeparableRow<<<gridSize, blockSize>>>(partial_bsum, d_bblurred, rows, cols, d_filter, filterWidth);
        cudaDeviceSynchronize();
        checkCudaErrors(cudaGetLastError());
/*
        gaussianBlur_global<<<gridSize, blockSize>>>(d_blue, d_bblurred, rows, cols, d_filter, filterWidth);
        cudaDeviceSynchronize();
        checkCudaErrors(cudaGetLastError());
*/
        recombineChannels<<<gridSize, blockSize>>>(d_rblurred, d_gblurred, d_bblurred, d_oimrgba, rows, cols);
        cudaDeviceSynchronize();
        checkCudaErrors(cudaGetLastError());

}

void your_gauss_blur_shared(uchar4* d_imrgba, uchar4 *d_oimrgba, size_t rows, size_t cols, 
  unsigned char *d_red, unsigned char *d_green, unsigned char *d_blue, 
  unsigned char *d_rblurred, unsigned char *d_gblurred, unsigned char *d_bblurred,
  float *d_filter,  int filterWidth){

  dim3 blockSize(BLOCK,BLOCK,1); // For 2D image
  dim3 gridSize((cols/BLOCK)+1,(rows/BLOCK)+1,1);

  separateChannels<<<gridSize, blockSize>>>(d_imrgba, d_red, d_green, d_blue, rows, cols);
  cudaDeviceSynchronize();
  checkCudaErrors(cudaGetLastError());

  gaussianBlur_shared<<<gridSize, blockSize>>>(d_red, d_rblurred, rows, cols, d_filter, filterWidth);
  cudaDeviceSynchronize();
  checkCudaErrors(cudaGetLastError());

  gaussianBlur_shared<<<gridSize, blockSize>>>(d_green, d_gblurred, rows, cols, d_filter, filterWidth);
  cudaDeviceSynchronize();
  checkCudaErrors(cudaGetLastError());

  gaussianBlur_shared<<<gridSize, blockSize>>>(d_blue, d_bblurred, rows, cols, d_filter, filterWidth);
  cudaDeviceSynchronize();
  checkCudaErrors(cudaGetLastError());

  recombineChannels<<<gridSize, blockSize>>>(d_rblurred, d_gblurred, d_bblurred, d_oimrgba, rows, cols);

  cudaDeviceSynchronize();
  checkCudaErrors(cudaGetLastError());   

}



