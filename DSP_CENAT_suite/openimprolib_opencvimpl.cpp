#include "openimprolib_opencvimpl.h"


OpenImProLib_OpenCvImpl::OpenImProLib_OpenCvImpl(){


}



int OpenImProLib_OpenCvImpl::imProThresh2CvThresh(ThresholdType thresholdType){
    int cvThreshType = -1;
    switch(thresholdType){
           case BINARY_THRESH:
                cvThreshType = CV_THRESH_BINARY;
           break;
           case BINARY_INV_THRESH:
                cvThreshType = CV_THRESH_BINARY_INV;
           break;
           case TRUNC_THRESH:
                cvThreshType = CV_THRESH_TRUNC;
           break;
           case TO_ZERO_THRESH:
                cvThreshType = CV_THRESH_TOZERO;
           break;
           case TO_ZERO_INV_THRESH:
                cvThreshType = CV_THRESH_TOZERO_INV;
           break;
    }
    return cvThreshType;

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
ImageImPro* OpenImProLib_OpenCvImpl::filterCanny(ImageImPro* ptrInput, double limInf, double limSup, int apertureSize){    
    IplImage* ptrInputCv = ptrInput->getOpenCvImage();
    IplImage* ptrOutputCv = cvCreateImage(cvGetSize(ptrInputCv), IPL_DEPTH_8U, 1);
    cvCanny(ptrInputCv, ptrOutputCv, limInf, limSup, apertureSize);
    ImageImPro* ptrOutput = new ImageImPro_OpenCvImpl(ptrOutputCv);    
    cvReleaseImage(&ptrOutputCv);
    cvReleaseImage(&ptrInputCv);
    return ptrOutput;
}

ImageImPro* OpenImProLib_OpenCvImpl::filterGauss(ImageImPro* ptrInput, double sigmaX, double sigmaY, int apertureSize){    
    Size size;
    size.width = apertureSize;
    size.height = apertureSize;
    Mat* ptrMatOutput = new Mat();
    Mat* ptrMatInput = ptrInput->getMat();
    GaussianBlur(*ptrMatInput, *ptrMatOutput, size, sigmaX, sigmaY);
    ImageImPro* ptrOutput = new ImageImPro_OpenCvImpl(ptrMatOutput);    
    delete ptrMatOutput;
    delete ptrMatInput;
    return ptrOutput;
}
/*
* The threshold operator classifies each pixel on the input image Pi, comparing Pi < threshold, and in the output applyng the type
* of threshold selected by the user
*@param input, input image must have ideally a dimmension of 1, otherwise the grayscaled image will be calculated
*@param output, output image, with the threshold applied, It is an empty pointer, the image is created with 8U, 1 channel format
*@param limInf, Lower boundary L
*@param threshold, the threshold checked in every pixel
*@param maxValue, the maxValue of the output image, assigned according to the type of threshold
*@param typeThresh, the type of the threshold could be:
*BINARY_THRESH: Output_i = (Input_i > Thresh) Max:0
*BINARY_INV_THRESH: Output_i = (Input_i > Thresh) 0:Max
*TRUNC_THRESH: Output_i = (Input_i > Thresh) M:Input_i
*TO_ZERO_INV_THRESH: Output_i = (Input_i > Thresh) 0:Input_i
*TO_ZERO_INV_THRESH: Output_i = (Input_i > Thresh) Input_i:0
*/
ImageImPro* OpenImProLib_OpenCvImpl::applyThreshold(ImageImPro* ptrInput, double threshold, double maxValue, ThresholdType typeThresh){     
     IplImage* ptrCvInput = ptrInput->getOpenCvImage();
     IplImage* ptrCvOutput = cvCreateImage(cvGetSize(ptrCvInput), IPL_DEPTH_8U, 1);
     int cvThresholdType = imProThresh2CvThresh(typeThresh);
     if(ptrInput->getChannels() != 1){
         IplImage* ptrCvInputGray = cvCreateImage(cvSize(ptrCvInput->width, ptrCvInput->height),IPL_DEPTH_8U,1);
         cvCvtColor(ptrCvInput,ptrCvInputGray, CV_RGB2GRAY);
         cvThreshold(ptrCvInputGray, ptrCvOutput, threshold, maxValue, cvThresholdType);
         cvReleaseImage(&ptrCvInputGray);
     }
     else{
        cvThreshold(ptrCvInput, ptrCvOutput, threshold, maxValue, cvThresholdType);
     }
     ImageImPro* ptrOutput = new ImageImPro_OpenCvImpl(ptrCvOutput);
     cvReleaseImage(&ptrCvOutput);
     cvReleaseImage(&ptrCvInput);
     return ptrOutput;
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
ImageImPro* OpenImProLib_OpenCvImpl::filterSobel(ImageImPro* ptrInput, int xOrder, int yOrder, int apertureSize){  
    IplImage* ptrCvInput = ptrInput->getOpenCvImage();
    //buffer for sobel result needing more bits per pixel for the result, then, rescaling is necesary to get it back to 8 bits per pixel
    IplImage* ptrCvTemp = cvCreateImage(cvGetSize(ptrCvInput),IPL_DEPTH_32F,1);
    IplImage* ptrCvOutput = cvCreateImage(cvGetSize(ptrCvInput), IPL_DEPTH_8U, 1);
    if(ptrInput->getChannels() != 1){
        IplImage* ptrCvInputGray = cvCreateImage(cvSize(ptrCvInput->width,ptrCvInput->height),IPL_DEPTH_8U,1);
        cvCvtColor(ptrCvInput,ptrCvInputGray, CV_RGB2GRAY);
        cvSobel(ptrCvInputGray,ptrCvTemp, xOrder, yOrder, apertureSize);
        cvReleaseImage(&ptrCvInputGray);
    }
    else{
        cvSobel(ptrCvInput,ptrCvTemp, xOrder, yOrder, apertureSize);
    }
    cvConvertScaleAbs(ptrCvTemp, ptrCvOutput, 1, 0);
    ImageImPro* ptrOutput = new ImageImPro_OpenCvImpl(ptrCvOutput);
    cvReleaseImage(&ptrCvOutput);
    cvReleaseImage(&ptrCvInput);
    cvReleaseImage(&ptrCvTemp);
    return ptrOutput;
}

ImageImPro* OpenImProLib_OpenCvImpl::thresholdEqualsRGB(ImageImPro* ptrInput, RGB_VALUE threshold){
    ImageImPro* ptrOutput = new ImageImPro_OpenCvImpl(ptrInput->getSize(), ImageImPro::BIT_8_U, 1);
    ImageImPro* ptrTemp = new ImageImPro_OpenCvImpl(ptrInput->getSize(), ImageImPro::BIT_8_U, 1);
    vector<vector<Point> > contours;
    vector<Vec4i> hierarchy;

    //thresholds using the flag received to create a mask
    for(int y = 0; y < ptrInput->getSize().height; ++y){
        for(int x = 0; x < ptrInput->getSize().width; ++x){
            unsigned char r = ptrInput->getPV(x, y, R);
            unsigned char g = ptrInput->getPV(x, y, G);
            unsigned char b = ptrInput->getPV(x, y, B);
            if( r == threshold.r && b == threshold.b && g == threshold.g){
                ptrOutput->setPV(x, y, 255);
            }
            else{
                ptrOutput->setPV(x, y, 0);
            }
        }
    }
    Mat* ptrOutputMat = ptrOutput->getMat();
    findContours( *ptrOutputMat, contours, hierarchy, CV_RETR_LIST, CV_CHAIN_APPROX_SIMPLE, Point(0, 0) );

      /// Draw contours
    Mat drawing = Mat::zeros( ptrOutputMat->size(), CV_8U);
    int k = 0;
    for( int i = 0; i < contours.size(); i++ ){
        if(i % 2 == 0){
            Scalar color = Scalar(255 - k);
            cout << "Objeto encontrado: " << k << endl;
            k++;
            //-1 to fill the countour
            drawContours( drawing, contours, i, color, -1, 8, hierarchy, 0, Point() );
        }
          /// Show in a window

    }
    delete ptrOutput;
    ptrOutput = new ImageImPro_OpenCvImpl(&drawing);
    //Eliminates the borders marked from the mask, to avoid taking them into account
    for(int y = 0; y < ptrInput->getSize().height; ++y){
        for(int x = 0; x < ptrInput->getSize().width; ++x){
            unsigned char r = ptrInput->getPV(x, y, R);
            unsigned char g = ptrInput->getPV(x, y, G);
            unsigned char b = ptrInput->getPV(x, y, B);
            if( r == threshold.r && b == threshold.b && g == threshold.g){
                ptrOutput->setPV(x, y, 0);
            }
        }
    }

    return ptrOutput;
}

ImageImPro** OpenImProLib_OpenCvImpl::getCounturedObjectMask(ImageImPro* ptrMask, ImageImPro* ptrInput, int objectMaskTag){
    ImageImPro* ptrOutput = NULL;
    ImageImPro* ptrOutputMask = NULL;
    int minX = DBL_MAX;
    int minY = DBL_MAX;
    int maxX = -1;
    int maxY = -1;
    int deltaX, deltaY;
    for(int x = 0; x < ptrInput->getSize().width; ++x ){
        for(int y = 0; y < ptrInput->getSize().height; ++y){
            unsigned char value = ptrMask->getPV(x, y);
            if(value == objectMaskTag){
                if(minX > x)
                    minX = x;
                if(minY > y)
                    minY = y;
                if(maxX < x)
                    maxX = x;
                if(maxY < y)
                    maxY = y;
            }
        }
    }
    deltaX = maxX - minX;
    deltaY = maxY - minY;
    ptrOutput = new ImageImPro_OpenCvImpl(ImSize(deltaX + 1, deltaY + 1), ptrInput->getDepth(), 3);
    ptrOutputMask = new ImageImPro_OpenCvImpl(ImSize(deltaX + 1, deltaY + 1), ptrInput->getDepth(), 1);
    for(int x = 0; x < ptrInput->getSize().width; ++x ){
        for(int y = 0; y < ptrInput->getSize().height; ++y){
            unsigned char value = ptrMask->getPV(x, y);
            if( value == objectMaskTag){
                ptrOutput->setPV(x - minX, y - minY, ptrInput->getPV_RGB(x, y));
                ptrOutputMask->setPV(x - minX, y - minY, 255);
            }
        }
    }
    ImageImPro** ptrArray = (ImageImPro**)malloc(sizeof(ImageImPro*)*2);
    ptrArray[0] = ptrOutput;
    ptrArray[1] = ptrOutputMask;
    return ptrArray;
}

vector<double> OpenImProLib_OpenCvImpl::getDensityFunction(ImageImPro* ptrInput, ImageImPro* ptrMask, int layer){
    vector<double> histogram(256);
    unsigned char value;
    for(int x = 0; x < ptrInput->getSize().width; ++x){
        for(int y = 0; y < ptrInput->getSize().height; ++y){
            if(ptrMask->getPV(x, y) == 255){
                value = (unsigned char)ptrInput->getPV(x, y, layer);
                histogram[value]++;
            }
        }
    }
    int N = ptrInput->getSize().width * ptrInput->getSize().height;
    //Von mises approximation
    for(int i = 0; i < 256; ++i){
        histogram[i] = histogram[i]/N;
    }

    return histogram;
}
