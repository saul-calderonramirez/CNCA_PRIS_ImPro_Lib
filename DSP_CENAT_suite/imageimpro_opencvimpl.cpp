#include "imageimpro_opencvimpl.h"

ImageImPro_OpenCvImpl::ImageImPro_OpenCvImpl(IplImage* ptrInput){
  this->ptrImage = cvCloneImage(ptrInput);
}

ImageImPro_OpenCvImpl::ImageImPro_OpenCvImpl(ImageImPro* ptrInput){
  IplImage* ptrCvImage = ptrInput->getOpenCvImage();
  this->ptrImage = cvCloneImage(ptrCvImage);
  cvReleaseImage(&ptrCvImage);
}

ImageImPro_OpenCvImpl::ImageImPro_OpenCvImpl(GpuMat* ptrGpuMat){
   //Buffer Matrix necessary
   Mat* ptrMat = new Mat(*ptrGpuMat);
   //copies the memory
   IplImage iplTemp = *ptrMat;
   this->ptrImage = cvCloneImage(&iplTemp);
   delete ptrMat;
}



ImageImPro_OpenCvImpl::ImageImPro_OpenCvImpl(Mat* ptrMat){
    //copies the memory
    IplImage iplTemp = *ptrMat;
    this->ptrImage = cvCloneImage(&iplTemp);
}

ImageImPro_OpenCvImpl::ImageImPro_OpenCvImpl(){
    this->ptrImage = NULL;
}

ImageImPro_OpenCvImpl::ImageImPro_OpenCvImpl(char* ptrName){
      loadImage(ptrName);
}

ImageImPro_OpenCvImpl::ImageImPro_OpenCvImpl(ImSize size, ImageImProDepth depth, int channels){
     createImage(size, depth, channels);
}

void ImageImPro_OpenCvImpl::setPV(int x, int y, RGB_VALUE value){
    assert(x < this->getSize().width && y < this->getSize().height && x >= 0 && y >= 0||!( std::cerr << "x: " << x << " y: "<< y << endl));
    int cy = (this->ptrImage->height - 1) - y;
    char* ptrRowCv = (char *)(this->ptrImage->imageData + cy * this->ptrImage->widthStep);
    //translation to default axis origin
    ptrRowCv[this->ptrImage->nChannels * x + R] = value.r;
    ptrRowCv[this->ptrImage->nChannels * x + G] = value.g;
    ptrRowCv[this->ptrImage->nChannels * x + B] = value.b;
}

void ImageImPro_OpenCvImpl::setPV(int x, int y, int value, int layer){
    assert(x < this->getSize().width && y < this->getSize().height && x >= 0 && y >= 0 && layer < this->getChannels()||!( std::cerr << "x: " << x << " y: "<< y << endl));
    int cy = (this->ptrImage->height - 1) - y;
    char* ptrRowCv = (char *)(this->ptrImage->imageData + cy * this->ptrImage->widthStep);
    //translation to default axis origin
    ptrRowCv[this->ptrImage->nChannels * x + layer] = value;
}


int ImageImPro_OpenCvImpl::getPV(int x, int y, int layer){
    assert(x < this->getSize().width && y < this->getSize().height && x >= 0 && y >= 0 && layer < this->getChannels() ||!( std::cerr << "x: " << x << " y: "<< y << endl) );
    int value = 0;
    int cy = (this->ptrImage->height - 1) - y;
    char* ptrRowCv = (char *)(this->ptrImage->imageData + cy * this->ptrImage->widthStep);
    //translation to default axis origin
    value = ptrRowCv[this->ptrImage->nChannels * x + layer];
    return value;
}

RGB_VALUE ImageImPro_OpenCvImpl::getPV_RGB(int x, int y){
    assert(x < this->getSize().width && y < this->getSize().height && x >= 0 && y >= 0 ||!( std::cerr << "x: " << x << " y: "<< y << endl));
    int cy = (this->ptrImage->height - 1) - y;
    char* ptrRowCv = (char *)(this->ptrImage->imageData + cy * this->ptrImage->widthStep);
    //translation to default axis origin
    int r = ptrRowCv[this->ptrImage->nChannels * x + R];
    int g = ptrRowCv[this->ptrImage->nChannels * x + G];
    int b = ptrRowCv[this->ptrImage->nChannels * x + B];
    RGB_VALUE value(r, g, b);
    return value;
}





ImageImPro::ImageImProDepth ImageImPro_OpenCvImpl::getDepth(){
    int depthCv = IPL_DEPTH_8U;
    ImageImProDepth depthImPro = ImageImPro::BIT_8_U;
    if(this->ptrImage != NULL){
        depthCv = this->ptrImage->depth;
        if(depthCv == IPL_DEPTH_32F){
            depthImPro = ImageImPro::BIT_32_F;
        }
    }   
    return depthImPro;
}

int ImageImPro_OpenCvImpl::getChannels(){
    int channels = 0;
    if(this->ptrImage != NULL){
        channels = this->ptrImage->nChannels;
    }
    return channels;
}

void ImageImPro_OpenCvImpl::loadImage(char* ptrName){
    this->ptrImage = cvLoadImage(ptrName);

}

int ImageImPro_OpenCvImpl::depthImPro2Cv(ImageImPro::ImageImProDepth depth){
    int depthCv = 0;
    switch(depth){
        case ImageImPro::BIT_8_U:
            depthCv = IPL_DEPTH_8U;
        break;
        case ImageImPro::BIT_16_S:
            depthCv = IPL_DEPTH_16S;
        break;
        case ImageImPro::BIT_32_F:
            depthCv = IPL_DEPTH_32F;
        break;

    }
    return depthCv;
}

ImageImPro_OpenCvImpl::ImageImProDepth ImageImPro_OpenCvImpl::depthCv2ImPro(int depth){
    ImageImProDepth depthImPro;
    switch(depth){
        case IPL_DEPTH_8U:
            depthImPro = ImageImPro_OpenCvImpl::BIT_8_U;
        break;
        case IPL_DEPTH_16S:
            depthImPro = ImageImPro_OpenCvImpl::BIT_16_S;
        break;
        case IPL_DEPTH_32F:
            depthImPro = ImageImPro_OpenCvImpl::BIT_32_F;
        break;
    }
    return depthImPro;
}

ImageImPro_OpenCvImpl::ImageImProDepth ImageImPro_OpenCvImpl::depthMat2ImPro(int depth){
    ImageImProDepth depthImPro;
    switch(depth){
        case CV_8U:
            depthImPro = ImageImPro_OpenCvImpl::BIT_8_U;
        break;
        case CV_16S:
            depthImPro = ImageImPro_OpenCvImpl::BIT_16_S;
        break;
        case CV_32F:
            depthImPro = ImageImPro_OpenCvImpl::BIT_32_F;
        break;
    }
    return depthImPro;
}

void ImageImPro_OpenCvImpl::createImage(ImSize size, ImageImProDepth depth, int channels){
    int depthCv = depthImPro2Cv(depth);
    CvSize cvSize;
    cvSize.height = size.height;
    cvSize.width = size.width;
    this->ptrImage = cvCreateImage(cvSize, depthCv, channels);
    if(channels == 3)
        cvSet(this->ptrImage, cvScalar(0,0,0));
    if(channels == 1)
        cvSet(this->ptrImage, cvScalar(0));
}

ImSize ImageImPro_OpenCvImpl::getSize(){
    ImSize size(0, 0);
    if(this->ptrImage != NULL){
        CvSize cv = cvGetSize(this->ptrImage);
        size.height = cv.height;
        size.width = cv.width;
    }
    return size;
}

IplImage* ImageImPro_OpenCvImpl:: getOpenCvImage(){    
    return cvCloneImage(this->ptrImage);
}

GpuMat* ImageImPro_OpenCvImpl::getGPUMat(){
    Mat* ptrMat = this->getMat();
    GpuMat* ptrGpuMat = new GpuMat();
    ptrGpuMat->upload(*ptrMat);
    delete ptrMat;
    return ptrGpuMat;
}

ImageImPro* ImageImPro_OpenCvImpl::getGrayScale(){
    IplImage *ptrCvImGray = cvCreateImage(cvGetSize(this->ptrImage),IPL_DEPTH_8U,1);
    cvCvtColor(this->ptrImage, ptrCvImGray, CV_RGB2GRAY);
    ImageImPro* ptrImGray = new  ImageImPro_OpenCvImpl(ptrCvImGray);
    return ptrImGray;
}

void ImageImPro_OpenCvImpl::showImageOnWindow(char* ptrWinName){
    IplImage* ptrCvImage = getOpenCvImage();
    //Displaying image on opencv window
     //First, the windows name will be set. In this case the window's name is
     //the same as the image name.
     cvNamedWindow(ptrWinName, 1);
     //Second, the image will be displayed
     cvShowImage(ptrWinName, ptrCvImage);
     //Finally, the window will be destroyed right after you press any key
     cvWaitKey();
     cvDestroyWindow(ptrWinName);
     cvReleaseImage(&ptrCvImage);
}

ImageImPro* ImageImPro_OpenCvImpl::getHSV(){
    ImageImPro* ptrImHSV;
    Mat img_hsv,img_rgb;
    img_rgb = *(this->getMat());
    cvtColor(img_rgb, img_hsv,CV_RGB2HSV);

    ptrImHSV = new ImageImPro_OpenCvImpl(&img_hsv);

    ptrImHSV->showImageOnWindow("HSV");
    return ptrImHSV;
}

ImageImPro* ImageImPro_OpenCvImpl::getLayer(int layer){
    vector<Mat> planes;
    Mat* ptrImage = this->getMat();
    Mat image = *ptrImage;
    split( image, planes );
    planes[0]; // H channel
    planes[1]; // S channel
    planes[2]; // V channel
  /*  namedWindow("HSV", CV_WINDOW_AUTOSIZE);
    imshow("HSV",  planes[1]);
    imwrite( "../capaH.jpg", planes[1]);*/
    ImageImPro* ptrLayer;
    ptrLayer = new ImageImPro_OpenCvImpl(&planes[layer]);
    //delete ptrImage;
    return ptrLayer;
}

ImageImPro_OpenCvImpl::~ImageImPro_OpenCvImpl(){
    if(this->ptrImage != NULL){
        cvReleaseImage(&this->ptrImage);
    }
}

QImage* ImageImPro_OpenCvImpl::MatToQImage(const Mat& mat){
    // 8-bits unsigned, NO. OF CHANNELS=1
    if(mat.type()==CV_8UC1){
        // Set the color table (used to translate colour indexes to qRgb values)
        QVector<QRgb> colorTable;
        for (int i=0; i<256; i++)
            colorTable.push_back(qRgb(i,i,i));
        // Copy input Mat
        const uchar *qImageBuffer = (const uchar*)mat.data;
        // Create QImage with same dimensions as input Mat
        QImage* img = new QImage(qImageBuffer, mat.cols, mat.rows, mat.step, QImage::Format_Indexed8);
        img->setColorTable(colorTable);
        return img;
    }
    // 8-bits unsigned, NO. OF CHANNELS=3
    if(mat.type()==CV_8UC3){
        // Copy input Mat
        const uchar *qImageBuffer = (const uchar*)mat.data;
        // Create QImage with same dimensions as input Mat
        QImage* img = new QImage(qImageBuffer, mat.cols, mat.rows, mat.step, QImage::Format_RGB888);
        QImage* img1 = new QImage(img->rgbSwapped());
        return img1;//img->rgbSwapped();
    }
    else
    {
        qDebug() << "ERROR: Mat could not be converted to QImage.";
        return NULL;
    }
}

QImage* ImageImPro_OpenCvImpl::getQImage(){
    QImage* ptrImg = NULL;
    Mat* ptrMat = this->getMat();
    ptrImg = this->MatToQImage(*ptrMat);
    delete ptrMat;
    return ptrImg;
}

Mat* ImageImPro_OpenCvImpl::getMat(){
    //copies the data
    Mat* ptrMat = new Mat(this->ptrImage, true);    
    return ptrMat;
}

