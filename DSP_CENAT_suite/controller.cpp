#include "controller.h"
using namespace std;
Controller::Controller(){
     this->ptrImage = NULL;
     this->ptrLib = new OpenImProLib_OpenCvImpl();
     this->ptrLibGPU = new OpenImProLib_OpenCVGPUimpl();
}



void Controller::loadImage(char* ptrName)throw (ControllerException){
    if(this->ptrImage != NULL){
        delete this->ptrImage;
    }
    this->ptrImage = new ImageImPro_OpenCvImpl(ptrName);
    if(this->ptrImage == NULL){
        throw ControllerException("Invalid image name");
    }

}

void Controller::applyFilterCanny()throw (ControllerException){
    cudaEvent_t	start, stop;
    cudaEventCreate(&start);
        cudaEventCreate(&stop);
    if(this->ptrImage != NULL){
        ImageImPro* ptrImageCanny = this->ptrLibGPU->filterCanny(this->ptrImage, 10, 500, 3);
        delete this->ptrImage;
        ptrImage = ptrImageCanny;
    }
    else{
         throw ControllerException("No image loaded");
    }
 }


void Controller::runBenchmarks()throw (ControllerException){
    if(this->ptrImage != NULL){
        UnitTests::testBenchmarks1(this->ptrImage);
    }
    else{
         throw ControllerException("No image loaded");
    }
}

 void Controller::findCountour()throw (ControllerException){
     if(this->ptrImage != NULL){
        ImageImPro* ptrTemp = this->ptrLib->thresholdEqualsRGB(this->ptrImage, RGB_VALUE(FUCSIA));
        ptrTemp->showImageOnWindow("AfterThreshEquals");
        ImageImPro** ptrTemp2 = this->ptrLib->getCounturedObjectMask(ptrTemp, this->ptrImage, 253);
        ptrTemp2[0]->showImageOnWindow("AfterGettingObject");
        ptrTemp2[1]->showImageOnWindow("AfterGettingObject");
        delete ptrTemp2[0];
        delete ptrTemp2[1];
        delete ptrTemp;

     }
     else{
          throw ControllerException("No image loaded");
     }
 }

void Controller::applyFilterSobel()throw (ControllerException){
     if(this->ptrImage != NULL){         
         ImageImPro* ptrImageSobel = this->ptrLibGPU->filterSobel(ptrImage, 1, 1, 3);
         delete this->ptrImage;
         ptrImage = ptrImageSobel;
     }
     else{
         throw ControllerException("No image loaded");
     }
 }

void Controller::applyFilterGauss()throw (ControllerException){
    if(this->ptrImage != NULL){
        ImageImPro* ptrImageGauss = this->ptrLibGPU->filterGauss(this->ptrImage, 0, 0, 11);
        delete this->ptrImage;
        ptrImage = ptrImageGauss;
    }
    else{
        throw ControllerException("No image loaded");
    }

}
void Controller::convertToHSV()throw (ControllerException){
    if(this->ptrImage != NULL){
        ImageImPro* ptrImageHSV = this->ptrImage->getHSV();
        cout<<"hizo la transcripcion RNA"<<endl;
        delete this->ptrImage;
        ptrImage = ptrImageHSV;
    }
    else{
        throw ControllerException("No image loaded");
    }
}

void Controller::applyBinaryThreshold()throw (ControllerException){
     if(this->ptrImage != NULL){        
         ImageImPro* ptrImageBin = this->ptrLibGPU->applyThreshold(this->ptrImage, 100, 255, OpenImProLib::BINARY_THRESH);
         delete this->ptrImage;
         ptrImage = ptrImageBin;
     }
     else{
         throw ControllerException("No image loaded");
     }
 }


 ImageImPro* Controller::getImage(){
    return this->ptrImage;
 }


Controller::~Controller(){
    if(this->ptrImage != NULL)
        delete this->ptrImage;
    if(this->ptrLib != NULL)
        delete this->ptrLib;
    if(this->ptrLibGPU != NULL)
        delete this->ptrLibGPU;
}
