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


    // CASE 2: UPPER RIGHT CORNER
    x = global_col + filterRadius;
    y = global_row - filterRadius;
    global_offset = y*cols + x;
    if(global_offset >= 0 && global_offset < cols*rows){
      shared_offset = (shared_row - filterRadius) * TILE_WIDTH + (shared_col + filterRadius);
      local_in[shared_offset] = d_in[global_offset];
    }


    // CASE 3: LOWER LEFT CORNER
    x = global_col - filterRadius;
    y = global_row + filterRadius;
    global_offset = y*cols + x;
    if(global_offset >= 0 && global_offset < cols*rows){

      shared_offset = (shared_row + filterRadius) * TILE_WIDTH + (shared_col - filterRadius);
      local_in[shared_offset] = d_in[global_offset];
    }

    // CASE 4: LOWER RIGHT CORNER
    x = global_col + filterRadius;
    y = global_row + filterRadius;
    global_offset = y*cols + x;

    if(global_offset >= 0 && global_offset < cols*rows){  
      shared_offset = (shared_row + filterRadius) * TILE_WIDTH + (shared_col + filterRadius);
      local_in[shared_offset] = d_in[global_offset];
    }
  }
  __syncthreads(); // Memory Finished Loading


  // Do the blur
  float blur_value = 0;
  int filter_offset = 0;

  for (int i = -filterRadius; i <= filterRadius; i++){
    for (int j = -filterRadius; j <= filterRadius; j++){

      shared_offset = (shared_row + i) * TILE_WIDTH + (shared_col + j);

      // check bounds
      if ((filter_offset >=0) && (filter_offset<filterWidth*filterWidth) && ((global_row + i) < rows) && ((global_col + j) < cols) && ((global_row + i) >= 0) && ((global_col + j) >= 0) && (shared_offset >= 0) && (shared_offset < TILE_WIDTH*TILE_WIDTH) && (shared_row+i<TILE_WIDTH) && (shared_col+j<TILE_WIDTH) && (shared_row+i >= 0) && (shared_col+j >= 0)){
        float pixel_value = (float)local_in[shared_offset];

        // get filter pixel
        float filter_value = d_filter[filter_offset];                        
        blur_value = blur_value + pixel_value*filter_value;

      }
      filter_offset++;
    }
  }
  
  __syncthreads();


  if (global_col < cols && global_row < rows){
    shared_offset = (shared_row) * TILE_WIDTH + (shared_col);
    d_out[global_row * cols + global_col] = (unsigned char)blur_value;
  }
}



/*
Computes convolution along the row, using shared memory.  Megan's Version.
 */
 __global__ 
 void gaussianBlur_separable_row(unsigned char *d_in, unsigned char *d_out, 
  const int rows, const int cols, float *d_filter, const int filterWidth, float *temp){

  int filterRadius = (filterWidth-1)/2;

  // TODO implement shared memory
  //shared memory array
  //__shared__ unsigned char local_in[TILE_WIDTH_ROW];

  // get current pixel at (col, row)
  int col=blockIdx.x*blockDim.x+threadIdx.x;
  int row=blockIdx.y*blockDim.y+threadIdx.y;
  int filter_row = blockIdx.z;

  float blur_value = 0;
  if(col<cols && row<rows) {
    //int offset = row*cols + col;
    
    // calculate gaussian blur from filter
    //int filter_offset = 0;
    int filter_offset = filter_row*filterWidth;
    //for (int i = row - filterRadius; i <= row + filterRadius; i++){
    for (int j = col - filterRadius; j <= col + filterRadius; j++){

      // check bounds
      if ((j < cols) && (j >= 0)){
        int pixel_offset = row*cols + j;
        float pixel_value = (float)d_in[pixel_offset];
        // get filter pixel
        float filter_value = d_filter[filter_offset]; 
        blur_value = blur_value + pixel_value*filter_value;
      }
      filter_offset = filter_offset + 1;
      //}
    }
  }

  //__syncthreads(); // if using shared memory, we have to sync threads here
  if(col<cols && row<rows) {
    int global_row_offset = row + filterRadius - filter_row;
    int global_col_offset = col;
    if ((global_row_offset < rows) && (global_row_offset >= 0)){
      int global_offset = global_row_offset*cols + global_col_offset;
      atomicAdd(&temp[global_offset], blur_value);
    }
  }
}

/* 
  Function used with gaussianBlur_separable_row_megan.
  In order to use atomicAdd, inputs must be float arrays.
  Use this function to convert temporary float array back to uchar
*/
__global__
 void float_to_uchar_array(float *temp, unsigned char *d_out, const int rows, const int cols){
  // get current pixel at (i,j)
  int i=blockIdx.x*blockDim.x+threadIdx.x;
  int j=blockIdx.y*blockDim.y+threadIdx.y;
        
  if(i<cols && j<rows) {
    // get 1D coordinate for the grayscale image
    int offset= j*cols + i;
    unsigned char pixel = static_cast<unsigned char>(temp[offset]);
    d_out[offset] = pixel;
    temp[offset] = 0;
  }
}


/*
  Computes the convolution along the row. Max's Version.
*/
__global__ 
void gaussianBlurSeparableRow(float *d_in, unsigned char *d_out, 
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
    int i = -(filterWidth/2);
    for (int j = -(filterWidth/2) ; j < filterWidth/2;j++){
      // check bounds
      if ((j+col < cols) && (j+col >= 0)){
        int pixel_offset = (row*cols) + (j+col);
        float pixel_value = d_in[pixel_offset];

        // get filter pixel
        float filter_value = d_filter[(i+(filterWidth/2))*filterWidth + (j+(filterWidth/2))];
        
        blur_value = blur_value + pixel_value*filter_value;
      }
      filter_offset = filter_offset + 1;
    }
    //printf("%f ",blur_value);
    d_out[offset] = (unsigned char)(blur_value*1000)+10;
    //d_out[offset] = (unsigned char)d_in[offset];
  }
}

/*
  Computes the convolution along the column. Max's Version.
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
    int j = -(filterWidth/2);
    for(int i = -(filterWidth/2) ; i < filterWidth/2;i++){
      // check bounds
      if ((i+row < rows) && (i+row >= 0)){

        int pixel_offset = (row+i)*cols + (col);
        float pixel_value = (float)d_in[pixel_offset];
        // get filter pixel
        float filter_value = d_filter[(i+(filterWidth/2)) * filterWidth + (filterWidth/2 + j)];
        //float filter_value = d_filter[filter_offset]; 
        blur_value = blur_value + pixel_value*filter_value;

      }
      filter_offset = filter_offset + 1;
    }
    d_out[offset] = blur_value;

    __syncthreads();
    //d_out[offset] = (float)d_in[offset];
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

/*
  Original CUDA implementation using Global Memory.
*/
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

/*
  Implementation using Shared Memory.
*/
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


/*
  Seperable Row Version 1 - Max
*/

void your_gauss_blur_separable_row_max(uchar4* d_imrgba, uchar4 *d_oimrgba, size_t rows, size_t cols, 
  unsigned char *d_red, unsigned char *d_green, unsigned char *d_blue, 
  unsigned char *d_rblurred, unsigned char *d_gblurred, unsigned char *d_bblurred,  
  float *d_filter, int filterWidth, float* partial_rsum, float* partial_gsum, float* partial_bsum){

  dim3 blockSize(BLOCK,BLOCK,1); // For 2D image
  dim3 gridSize((cols/BLOCK)+1,(rows/BLOCK)+1,1);

  separateChannels<<<gridSize, blockSize>>>(d_imrgba, d_red, d_green, d_blue, rows, cols);
  cudaDeviceSynchronize();
  checkCudaErrors(cudaGetLastError());

  gaussianBlurSeparableColumn<<<gridSize, blockSize>>>(d_red, partial_sum, rows, cols, d_filter, filterWidth);
  cudaDeviceSynchronize();
  checkCudaErrors(cudaGetLastError());

  gaussianBlurSeparableRow<<<gridSize, blockSize>>>(partial_sum, d_rblurred, rows, cols, d_filter, filterWidth);
  cudaDeviceSynchronize();
  checkCudaErrors(cudaGetLastError());

  gaussianBlurSeparableColumn<<<gridSize, blockSize>>>(d_green, partial_sum, rows, cols, d_filter, filterWidth);
  cudaDeviceSynchronize();
  checkCudaErrors(cudaGetLastError());

  gaussianBlurSeparableRow<<<gridSize, blockSize>>>(partial_sum, d_gblurred, rows, cols, d_filter, filterWidth);
  cudaDeviceSynchronize();
  checkCudaErrors(cudaGetLastError());

  gaussianBlurSeparableColumn<<<gridSize, blockSize>>>(d_blue, partial_sum, rows, cols, d_filter, filterWidth);
  cudaDeviceSynchronize();
  checkCudaErrors(cudaGetLastError());

  gaussianBlurSeparableRow<<<gridSize, blockSize>>>(partial_sum, d_bblurred, rows, cols, d_filter, filterWidth);
  cudaDeviceSynchronize();
  checkCudaErrors(cudaGetLastError());

  recombineChannels<<<gridSize, blockSize>>>(d_rblurred, d_gblurred, d_bblurred, d_oimrgba, rows, cols);
  cudaDeviceSynchronize();
  checkCudaErrors(cudaGetLastError());

}




/*
  Seperable Row Version 2 - Megan
*/
void your_gauss_blur_separable_row_megan(uchar4* d_imrgba, uchar4 *d_oimrgba, size_t rows, size_t cols, 
  unsigned char *d_red, unsigned char *d_green, unsigned char *d_blue, 
  unsigned char *d_rblurred, unsigned char *d_gblurred, unsigned char *d_bblurred,
  float *d_filter,  int filterWidth, float *temp){

  dim3 blockSize(BLOCK,BLOCK,1); // For 2D image
  dim3 gridSize((cols/BLOCK)+1,(rows/BLOCK)+1,filterWidth);
  dim3 gridSize_2D((cols/BLOCK)+1,(rows/BLOCK)+1,1);


  separateChannels<<<gridSize_2D, blockSize>>>(d_imrgba, d_red, d_green, d_blue, rows, cols);
  cudaDeviceSynchronize();
  checkCudaErrors(cudaGetLastError());
  
  gaussianBlur_separable_row<<<gridSize, blockSize>>>(d_red, d_rblurred, rows, cols, d_filter, filterWidth, temp);
  cudaDeviceSynchronize();
  checkCudaErrors(cudaGetLastError());

  float_to_uchar_array<<<gridSize_2D, blockSize>>>(temp, d_rblurred, rows, cols);
  cudaDeviceSynchronize();
  checkCudaErrors(cudaGetLastError());

  gaussianBlur_separable_row<<<gridSize, blockSize>>>(d_green, d_gblurred, rows, cols, d_filter, filterWidth, temp);
  cudaDeviceSynchronize();
  checkCudaErrors(cudaGetLastError());

  float_to_uchar_array<<<gridSize_2D, blockSize>>>(temp, d_gblurred, rows, cols);
  cudaDeviceSynchronize();
  checkCudaErrors(cudaGetLastError());

  gaussianBlur_separable_row<<<gridSize, blockSize>>>(d_blue, d_bblurred, rows, cols, d_filter, filterWidth, temp);
  cudaDeviceSynchronize();
  checkCudaErrors(cudaGetLastError());

  float_to_uchar_array<<<gridSize_2D, blockSize>>>(temp, d_bblurred, rows, cols);
  cudaDeviceSynchronize();
  checkCudaErrors(cudaGetLastError());

  recombineChannels<<<gridSize_2D, blockSize>>>(d_rblurred, d_gblurred, d_bblurred, d_oimrgba, rows, cols);

  cudaDeviceSynchronize();
  checkCudaErrors(cudaGetLastError());   

}

