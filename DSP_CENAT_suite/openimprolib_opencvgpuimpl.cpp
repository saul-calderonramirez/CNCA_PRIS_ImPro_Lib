#include "openimprolib_opencvgpuimpl.h"
using namespace gpu;
OpenImProLib_OpenCVGPUimpl::OpenImProLib_OpenCVGPUimpl(){
}

extern void convert_to_hsv_wrapper(uchar4 *rgb, float4 *hsv, int width, int height);

ImageImPro* OpenImProLib_OpenCVGPUimpl::convertRGBToHSV(ImageImPro* input){
    // host
        uchar4 *host_image, *host_out;


        // device
        uchar4 *image;
        float4 *hsv;

        // timing
        cudaEvent_t	start, stop;

        cudaEventCreate(&start);
        cudaEventCreate(&stop);
        size_t number_of_bytes_rgb = sizeof(uchar4) * input->getSize().width * input->getSize().height;
        size_t number_of_bytes_hsv = sizeof(float4) * input->getSize().width * input->getSize().height;


        host_out = (uchar4*)malloc(number_of_bytes_hsv);

        cudaMalloc((void **) &image, number_of_bytes_rgb);
        cudaMalloc((void **) &hsv, number_of_bytes_hsv);

        assert(cudaGetLastError() == cudaSuccess);

        cudaEventRecord(start, 0);

        cudaMemcpy(image, host_image, number_of_bytes_rgb, cudaMemcpyHostToDevice);
        assert(cudaGetLastError() == cudaSuccess);

   //     convert_to_hsv_wrapper(image, hsv, input->getSize().width, input->getSize().height);

        cudaEventRecord(stop, 0);

        cudaMemcpy(host_out, hsv, number_of_bytes_hsv, cudaMemcpyDeviceToHost);
        cudaEventSynchronize(stop);

        float elapsed_time;
        cudaEventElapsedTime(&elapsed_time, start, stop);
        printf("Time to compute hsv image with GPU: %3.1f ms\n", elapsed_time);
        cudaEventDestroy(start);
        cudaEventDestroy(stop);




        cudaFree(hsv);
        cudaFree(image);
        free(host_image);
        free(host_out);
}

/*
*The Canny filter detects the edges on an images
*Calculates the first derivative in the four directions, and applies an hysteresis threshold
*Hysteresis threshold: Has an Upper and Lower boundary, U and L. If the gradient Gi > U at pixel Pi, then Pi is classified as an edge
*if  L < Gi < U, and Gi is connected to a Pixel Pj classified as an edge, then Pi is also classified as an edge
*Otherwise the Pixel Pi is part of the background
*@param input, input image, should be in grayscale (one channel)
*@param output, binary image, with one channel. It is an empty pointer, the image is created with 32F, 1 channel format
*@param limInf, Lower boundary L
*@param limSup, Upper boundary U
*@param appertureSize, defines the size of the convolution window. Odd numbers 3, 5 and 7 must be used
*A smaller window is more sensitive to noise
*@return Binary Image with the edges white and the rest of the image black
*/
ImageImPro* OpenImProLib_OpenCVGPUimpl::filterCanny(ImageImPro* ptrInput, double limInf, double limSup, int apertureSize){
    ImageImPro* ptrGrayScaleInput = ptrInput;
    bool converted = false;
    if(ptrInput->getChannels() > 1){
            ptrGrayScaleInput = ptrInput->getGrayScale();
            converted = true;
    }
    GpuMat* ptrGpuMatOutput = new GpuMat();
    GpuMat* ptrGpuMatInput = ptrGrayScaleInput->getGPUMat();
    gpu::Canny(*ptrGpuMatInput, *ptrGpuMatOutput, limInf, limSup, apertureSize);
    ImageImPro* ptrImProOutput = new ImageImPro_OpenCvImpl(ptrGpuMatOutput);
    //frees memory
    if(converted)delete ptrGrayScaleInput;
    delete ptrGpuMatOutput;
    delete ptrGpuMatInput;
    return ptrImProOutput;
}
/*
*The sobel filter is an approximation to a derivative, it can apply first or second order in both coordinates in an image
*First, the sobel operator applies a gaussian filter, in order to smooth the image
*Then it calculates the derivative, and umbralizes the image
*@param input, the input image, could be grayscale or RGB
*@param output, the output image must have at least 16 bit pixel representation, to avoid overflow
*@param xOrder, the derivative order for the X axis
*@param yOrder, the derivative order for the Y axis
*@param apertureSize, size of the filter window, if the size is 3, the scharr filter is used, less sensitive to noise
*/
ImageImPro* OpenImProLib_OpenCVGPUimpl::filterSobel(ImageImPro* ptrInput, int xOrder, int yOrder, int apertureSize){
    ImageImPro* ptrGrayScaleInput = ptrInput;
    bool converted = false;
    if(ptrInput->getChannels() > 1){
            ptrGrayScaleInput = ptrInput->getGrayScale();
            converted = true;
    }
    GpuMat* ptrGpuMatOutput = new GpuMat();
    GpuMat* ptrGpuMatInput = ptrGrayScaleInput->getGPUMat();
    gpu::Sobel(*ptrGpuMatInput, *ptrGpuMatOutput, CV_8U, xOrder, yOrder, apertureSize);
    ImageImPro* ptrImProOutput = new ImageImPro_OpenCvImpl(ptrGpuMatOutput);
    if(converted)delete ptrGrayScaleInput;
    delete ptrGpuMatOutput;
    delete ptrGpuMatInput;
    return ptrImProOutput;
}

ImageImPro* OpenImProLib_OpenCVGPUimpl::applyThreshold(ImageImPro* ptrInput, double threshold, double maxValue, ThresholdType typeThresh){
    ImageImPro* ptrGrayScaleInput = ptrInput;
    bool converted = false;
    if(ptrInput->getChannels() > 1){
            ptrGrayScaleInput = ptrInput->getGrayScale();
            converted = true;
    }
    GpuMat* ptrGpuMatOutput = new GpuMat();
    GpuMat* ptrGpuMatInput = ptrGrayScaleInput->getGPUMat();
    //CORREGIR, EL TYPETHRESH DEBE SER DEL TIPO DEL OCV GPU
    gpu::threshold(*ptrGpuMatInput, *ptrGpuMatOutput, threshold, maxValue, typeThresh);
    ImageImPro* ptrImProOutput = new ImageImPro_OpenCvImpl(ptrGpuMatOutput);
    if(converted)delete ptrGrayScaleInput;
    delete ptrGpuMatOutput;
    delete ptrGpuMatInput;
    return ptrImProOutput;
}

ImageImPro* OpenImProLib_OpenCVGPUimpl::filterGauss(ImageImPro* ptrInput, double sigmaX, double sigmaY, int apertureSize){
    Size size;
    size.width = apertureSize;
    size.height = apertureSize;
    GpuMat* ptrGpuMatOutput = new GpuMat();
    GpuMat* ptrGpuMatInput = ptrInput->getGPUMat();
    gpu::GaussianBlur(*ptrGpuMatInput, *ptrGpuMatOutput, size, sigmaX, sigmaY);
    ImageImPro* ptrImProOutput = new ImageImPro_OpenCvImpl(ptrGpuMatOutput);
    delete ptrGpuMatOutput;
    delete ptrGpuMatInput;
    return ptrImProOutput;
}
