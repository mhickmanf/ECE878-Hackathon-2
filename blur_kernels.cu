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

  int filterRadius = (filterWidth-1)/2;

  //shared memory array
  __shared__ unsigned char local_in[TILE_WIDTH * TILE_WIDTH];
  
  // global data id
  int global_col=blockIdx.x*blockDim.x+threadIdx.x;
  int global_row=blockIdx.y*blockDim.y+threadIdx.y;
  int global_offset;

  // thread block id
  int local_col = threadIdx.x;
  int local_row = threadIdx.y;

  // shifted shared data id
  int shared_col = local_col + filterRadius;
  int shared_row = local_row + filterRadius;
  int shared_offset;

  // load global data into shared memory
  int x,y;

  if (global_col < cols && global_row < rows){

    // CASE 1: UPPER LEFT CORNER
    x = global_col - filterRadius;
    y = global_row - filterRadius;
    
    global_offset = y*cols + x;
    if(global_offset >= 0 && global_offset < cols*rows){
      shared_offset = (shared_row - filterRadius) * TILE_WIDTH + (shared_col - filterRadius);
      local_in[shared_offset] = d_in[global_offset];
    } else{
      shared_offset = (shared_row - filterRadius) * TILE_WIDTH + (shared_col - filterRadius);
      local_in[shared_offset] = 0.0;
    }

  }
  __syncthreads();

  if (global_col < cols && global_row < rows){

    // CASE 2: UPPER RIGHT CORNER
    x = global_col + filterRadius;
    y = global_row - filterRadius;
    global_offset = y*cols + x;
    if(global_offset >= 0 && global_offset < cols*rows){
      shared_offset = (shared_row - filterRadius) * TILE_WIDTH + (shared_col + filterRadius);
      local_in[shared_offset] = d_in[global_offset];
    }
  }

  __syncthreads();

  if (global_col < cols && global_row < rows){

    // CASE 3: LOWER LEFT CORNER
    x = global_col - filterRadius;
    y = global_row + filterRadius;
    global_offset = y*cols + x;
    if(global_offset >= 0 && global_offset < cols*rows){

      shared_offset = (shared_row + filterRadius) * TILE_WIDTH + (shared_col - filterRadius);
      local_in[shared_offset] = d_in[global_offset];
    }
  }

  __syncthreads();

  if (global_col < cols && global_row < rows){
    // CASE 4: LOWER RIGHT CORNER
    x = global_col + filterRadius;
    y = global_row + filterRadius;
    global_offset = y*cols + x;

    if(global_offset >= 0 && global_offset < cols*rows){  
      shared_offset = (shared_row + filterRadius) * TILE_WIDTH + (shared_col + filterRadius);
      local_in[shared_offset] = d_in[global_offset];
    }
  }

  __syncthreads();

  if (global_col < cols && global_row < rows){
    // CASE: ITSELF
    x = global_col;
    y = global_row;
    global_offset = y*cols + x;
    if(global_offset >= 0 && global_offset < cols*rows){ 
      shared_offset = (shared_row) * TILE_WIDTH + (shared_col);
      local_in[shared_offset] = d_in[global_offset];
    }
  }
  __syncthreads(); // Memory Finished Loading

  int row_offset, col_offset;
  float blur_value = 0;
  int filter_offset = 0;

  if (global_col < cols && global_row < rows){
    
    for (row_offset = -filterRadius; row_offset <= filterRadius; row_offset++){
      for (col_offset = -filterRadius; col_offset <= filterRadius; col_offset++){
        shared_offset = (shared_row + row_offset) * TILE_WIDTH + (shared_col + col_offset);
        if (shared_offset >= 0 && shared_offset < TILE_WIDTH*TILE_WIDTH){
          float pixel_value = (float)local_in[shared_offset];
          float filter_value = d_filter[filter_offset]; 
          blur_value = blur_value + pixel_value*filter_value;

        }
        filter_offset++;

      }
    }
  }
  __syncthreads();


  if (global_col < cols && global_row < rows){
    d_out[global_row * cols + global_col] = (unsigned char)blur_value;
  }
}



/*
  Computes the convolution along the row.
*/
__global__ 
void gaussianBlurSeparableRow(){

}

/*
  Computes the convolution along the column.
*/
__global__ 
void gaussianBlurSeparableColumn(){

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
        float *d_filter,  int filterWidth){

        dim3 blockSize(BLOCK,BLOCK,1); // For 2D image
        dim3 gridSize((cols/BLOCK)+1,(rows/BLOCK)+1,1);

        separateChannels<<<gridSize, blockSize>>>(d_imrgba, d_red, d_green, d_blue, rows, cols);
        cudaDeviceSynchronize();
        checkCudaErrors(cudaGetLastError());
        
        gaussianBlur_global<<<gridSize, blockSize>>>(d_red, d_rblurred, rows, cols, d_filter, filterWidth);
        cudaDeviceSynchronize();
        checkCudaErrors(cudaGetLastError());

        gaussianBlur_global<<<gridSize, blockSize>>>(d_green, d_gblurred, rows, cols, d_filter, filterWidth);
        cudaDeviceSynchronize();
        checkCudaErrors(cudaGetLastError());

        gaussianBlur_global<<<gridSize, blockSize>>>(d_blue, d_bblurred, rows, cols, d_filter, filterWidth);
        cudaDeviceSynchronize();
        checkCudaErrors(cudaGetLastError());

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



