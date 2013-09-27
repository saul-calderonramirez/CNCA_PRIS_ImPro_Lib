#ifndef CONTROLLER_H
#define CONTROLLER_H
#include "includes.h"
#include "openimprolib_opencvimpl.h"
#include "imageimpro_opencvimpl.h"

#include "openimprolib_opencvgpuimpl.h"
#include "unittests.h"



#define DEF_IMAGE "../default_images/marked1.bmp"

class Controller{

private:   
    OpenImProLib* ptrLib;
    OpenImProLib* ptrLibGPU;
    ImageImPro* ptrImage;


public:   
    class ControllerException: public exception{
      public:
      const char* ptrMessage;
      ControllerException(){}
      ControllerException(const char* ptrErrorMessage):ptrMessage(ptrErrorMessage){
        this->ptrMessage = ptrMessage;
      }
      virtual const char* what() const throw(){
        return this->ptrMessage;
      }
    }controllerException;
    Controller();
    ImageImPro* getImage();
    void loadImage(char* ptrName)throw (ControllerException);
    void applyFilterCanny()throw (ControllerException);
    void applyBinaryThreshold()throw (ControllerException);
    void applyFilterSobel()throw (ControllerException);
    void applyFilterGauss()throw (ControllerException);
    void convertToHSV()throw (ControllerException);
    vector<float> findCountour()throw (ControllerException);
    void runBenchmarks()throw (ControllerException);

    ~Controller();
};

#endif
