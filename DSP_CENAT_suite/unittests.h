#ifndef UnitTests_OCL_H
#define UnitTests_OCL_H
#include "includes.h"
#include "imageimpro_opencvimpl.h"
#include "openimprolib_opencvgpuimpl.h"
#include "openimprolib_opencvimpl.h"
using namespace cv;


class UnitTests
{

public:
    UnitTests();

    static int getMilliCount(){
        timeb tb;
        ftime(&tb);
        int nCount = tb.millitm + (tb.time & 0xfffff) * 1000;
        return nCount;
    }

    static int getMilliSpan(int nTimeStart){
        int nSpan = getMilliCount() - nTimeStart;
        if(nSpan < 0)
            nSpan += 0x100000 * 1000;
        return nSpan;
    }


    static void testOCV_OCL_1(){
        /*const char* wndname = "Original";
        const char* wndname2 = "Pyrdown";
        ocl::setBinpath("/usr/lib/x86_64-linux-gnu/");
        vector<ocl::Info> info;
        CV_Assert(ocl::getDevice(info));
        std::cout<<"Init hecho!"<< std::endl;
        Mat image = imread("baboon.jpg", CV_LOAD_IMAGE_GRAYSCALE);
        cv::ocl::oclMat pyr_ocl;
        cv::ocl::oclMat input(image);
        int startMilli = getMilliCount();
       // ocl::pyrDown(ocl::oclMat(image), pyr_ocl);
        ocl::Canny(input, pyr_ocl, 100, 300, 3);
        int endMilli = getMilliSpan(startMilli);
        std::cout<< "Time OCL pyrDown: " << endMilli << std::endl;
        Mat down = Mat(pyr_ocl);
        namedWindow( wndname, 1 );
        imshow(wndname, image);
        int c = waitKey();
        namedWindow( wndname2, 1 );
        imshow(wndname2, down);
        c = waitKey();*/
    }

    static void testBenchmarks1(ImageImPro* ptrImage){
        OpenImProLib* ptrLib = new OpenImProLib_OpenCvImpl();
        OpenImProLib*  ptrLibGPU = new OpenImProLib_OpenCVGPUimpl();

        cout << "NEW TEST:" << endl << endl;

        int startMilli, endMilli;
        startMilli = UnitTests::getMilliCount();
        ImageImPro* ptrOutput = ptrLibGPU->filterCanny(ptrImage, 10, 500, 3);
        delete ptrOutput;
        endMilli = UnitTests::getMilliSpan(startMilli);
        cout << "GPU canny time ms: " << endMilli << endl;
        startMilli = UnitTests::getMilliCount();
        ptrOutput = ptrLib->filterCanny(ptrImage, 10, 500, 3);
        delete ptrOutput;
        endMilli = UnitTests::getMilliSpan(startMilli);
        cout << "TBB canny time ms: " << endMilli << endl;

        startMilli = UnitTests::getMilliCount();
        ptrOutput = ptrLib->applyThreshold(ptrImage, 100, 255, OpenImProLib::BINARY_THRESH);
        delete ptrOutput;
        endMilli = UnitTests::getMilliSpan(startMilli);
        cout << "TBB threshold time ms: " << endMilli << endl;

        startMilli = UnitTests::getMilliCount();
        ptrOutput = ptrLibGPU->applyThreshold(ptrImage, 100, 255, OpenImProLib::BINARY_THRESH);
        delete ptrOutput;
        endMilli = UnitTests::getMilliSpan(startMilli);
        cout << "GPU threshold time ms: " << endMilli << endl;


        startMilli = UnitTests::getMilliCount();
        ptrOutput = ptrLibGPU->filterSobel(ptrImage, 1, 1, 3);
        delete ptrOutput;
        endMilli = UnitTests::getMilliSpan(startMilli);
        cout << "GPU sobel time ms: " << endMilli << endl;

        startMilli = UnitTests::getMilliCount();
        ptrOutput = ptrLib->filterSobel(ptrImage, 1, 1, 3);
        delete ptrOutput;
        endMilli = UnitTests::getMilliSpan(startMilli);
        cout << "TBB sobel time ms: " << endMilli << endl;


        startMilli = UnitTests::getMilliCount();
        ptrOutput = ptrLibGPU->filterGauss(ptrImage, 0, 0, 11);
        delete ptrOutput;
        endMilli = UnitTests::getMilliSpan(startMilli);
        cout << "GPU gauss time ms: " << endMilli << endl;


        startMilli = UnitTests::getMilliCount();
        ptrOutput = ptrLib->filterGauss(ptrImage, 0, 0, 11);
        delete ptrOutput;
        endMilli = UnitTests::getMilliSpan(startMilli);
        cout << "TBB gauss time ms: " << endMilli << endl;

    }

    static void test_GPU_2(){
        try{
            Mat src_host = cv::imread("baboon.jpg", CV_LOAD_IMAGE_GRAYSCALE);
            gpu::GpuMat dst, src;
            int startMilli = getMilliCount();
            src.upload(src_host);
            gpu::threshold(src, dst, 128.0, 255.0, CV_THRESH_BINARY);
            Mat result_host = (Mat)dst;
            int endMilli = getMilliSpan(startMilli);
            std::cout<< "Time OCL pyrDown: " << endMilli << std::endl;
            namedWindow( "Result", 1 );
            imshow("Result", result_host);
            waitKey();
        }
        catch(const cv::Exception& ex){
                std::cout << "Error: " << ex.what() << std::endl;
        }
    }

    static void testContours(ImageImPro* ptrImage){
        OpenImProLib* ptrLib = new OpenImProLib_OpenCvImpl();
        ImageImPro* ptrImageCanny =  ptrLib->applyThreshold(ptrImage, 100, 255, OpenImProLib::BINARY_THRESH);
        ImageImPro* ptrGrayRes = ptrImage->getGrayScale();
        IplImage* ptrGrayCv =  ptrGrayRes->getOpenCvImage();
        cvZero(ptrGrayCv);
        CvMemStorage* ptrMemStorage = cvCreateMemStorage(0);
        CvSeq* ptrContours = 0;
        IplImage* ptrIm = ptrImageCanny->getOpenCvImage();
        cvFindContours(ptrIm, ptrMemStorage, &ptrContours, sizeof(CvContour), CV_RETR_LIST, CV_CHAIN_APPROX_SIMPLE);
        if(ptrContours != NULL){
            cvDrawContours(ptrGrayCv, ptrContours, cvScalarAll(255), cvScalarAll(255), 100 );
            cvShowImage("Contours", ptrGrayCv);
        }

    }


};

#endif // UnitTests_OCL_H
