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
//El color FUCSIA ES IGUAL AL MAGENTA
 vector<float> Controller::findCountour()throw (ControllerException){
     vector<float> densityFunction;
     if(this->ptrImage != NULL){
        ImageImPro* ptrThresholded = this->ptrLib->thresholdEqualsRGB(this->ptrImage, RGB_VALUE(FUCSIA));
        ptrThresholded->showImageOnWindow("AfterThreshEquals");
        ImageImPro** ptrMaskedObj = this->ptrLib->getCounturedObjectMask(ptrThresholded, this->ptrImage, 255);
       /* ptrMaskedObj[0]->showImageOnWindow("AfterGettingObject_Original_image");
        ptrMaskedObj[1]->showImageOnWindow("AfterGettingObject_Mask");*/
        //Translated to HSV
        ImageImPro* ptrMaskedObjHSV = ptrMaskedObj[0]->getHSV();
        ImageImPro* ptrMaskedObjH = ptrMaskedObjHSV->getLayer(0);
       // ptrMaskedObjH->showImageOnWindow("Object H layer");
        densityFunction = this->ptrLib->getDensityFunction(ptrMaskedObjH,ptrMaskedObj[1], 0);
     }
     else{
          throw ControllerException("No image loaded");
     }
     return densityFunction;
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
