## Gaussian Blur - Hackathon Extension

In this lab, you’ll be implementing a Gaussian Blur algorithm.  Gaussian blur is an example of a convolution operation where the kernel is Gaussian. Visually,  the  convolution  operation  with  a  filter  generated  using equation above results in a smooth image with reduced visual detail.  

Algorithmically, there are three steps to the procedure:

1.  Separate the input RGBA image into three channels R, G, B. This is a simple map operation from a uchar4 pixel to unsigned char.
2.  Apply  Gaussian  blur  filter  to  each  channel  by  convolving  a  region  of  the image with the filter.  The filter is provided in code.
3.  Recombine  the  blurred  channel  into  one  single  uchar4  image.   This  is  the reverse operation of the first one and is called a gather operation.

### Hackathon Extension

In this extension, we look into ways to get the maximum performance out of our blurring kernel.

#### Part A: Shared Memory Kernel

Extend the kernel you wrote to incorporate shared memory. Specifically, set up a shared memory of size TILE WIDTH and use it to compute the local sums. Compare and contrast the performance with the original kernel and shared memory kernel. Provide visual proof of performance improvement.

#### Part B: Separable Convolution

We have been treating convolution as a weighted sum overa  2D region. In practice this is actually a computatationally intensive task. Since convolution is a separable operation (as we can independently convolve along the x and y dimensions). Implement kernels 'gaussian blur separable row' that computes the convolution along the row and a corresponding convolution that computes along the columns. You may use shared memory if that suits the need. Compare the performance with the other kernels from part A.

### Report Deliverables:

In addition, a LaTeX generated report detailing the design and the profiling results is necessary.

Profile on:
    • at least two different GPU hardware models
    • at least three different image sizes

Profile the following parameters:
    • Memcpy times:  Profile how much time it took to copy the data over to theGPU.
    • Kernel  Execution  Time:  Profile  the  kernel  execution  time  ofall three kernels.
    • Effect  of  varying  threads  per  block:   Profile  if  varying  the  number  of threads per block froms ∈ [16,1024] affects the kenrel run times. This data must be compiled  in  form  of  either  neat  tables  or  visual  graphs. Screenshots will not be provided with any credit. 


### Setup on Palmetto: 

There's no setup required. Copy this folder to your home directory and request an interactive job with a GPU like so:
`qsub -I -l select=1:ncpus=20:ngpus=1:mem=16gb:gpu_model=p100,walltime=48:00:00`. 

This shall give you a node with a GPU and `nvcc`. Use the Makefile provided to compile your code.


### Local Setup:

You will need CUDA 9.0 or above, OpenCV4 and access to linux shell. If you're using Visual Studio, then you shall need to install CUDA Toolkit and OpenCV4 and integrate them in your project. It is advised you take advantage of Palmetto to avoid significant delays.

module add gcc/7.1.0
module add opencv/4.2.0-gcc
module add cuda/9.2.88-gcc