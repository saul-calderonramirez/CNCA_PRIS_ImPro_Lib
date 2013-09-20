#ifndef IMAGE_IMPRO_H
#define IMAGE_IMPRO_H
#include "includes.h"
using namespace::cv;
using namespace::std;
using namespace gpu;

#define R 2
#define G 1
#define B 0
#define WHITE 0
#define BLACK 1
#define FUCSIA 2
class ImSize{
    public:
        int width;
        int height;
        ImSize(int width, int height){
            this->width = width;
            this->height = height;
        }
};


class RGB_VALUE{
    public:
        int r;
        int g;
        int b;

    RGB_VALUE(int color){
        if(color == WHITE)
            setWhite();
        if(color == BLACK)
            setBlack();
        if(color == FUCSIA)
            setFucsia();
    }

    RGB_VALUE(int r, int g, int b){
        this->r = r;
        this->g = g;
        this->b = b;
    }
    void setWhite(){
        r = 255;
        g = 255;
        b = 255;
    }
    void setBlack(){
        r = 0;
        g = 0;
        b = 0;
    }
    void setFucsia(){
        r = 255;
        g = 0;
        b = 255;
    }
};

class ImageImPro{
    public:
        enum ImageImProDepth { BIT_8_U, BIT_16_S, BIT_32_F };
        ImageImPro();
        virtual void loadImage(char* ptrName)=0;
        virtual void createImage(ImSize size, ImageImProDepth depth, int channels)=0;
        virtual ImageImPro* getGrayScale() = 0;
        virtual ImageImPro* getHSV() = 0;
        virtual QImage* getQImage() = 0;
        virtual Mat* getMat() = 0;
        virtual GpuMat* getGPUMat() = 0;
        virtual ImSize getSize() = 0;
        virtual IplImage* getOpenCvImage() = 0;
        virtual ~ImageImPro(){}
        virtual ImageImProDepth getDepth() = 0;
        virtual int getChannels() = 0;
        virtual void setPV(int x, int y, int value, int layer = 0) = 0;
        virtual void setPV(int x, int y, RGB_VALUE value) = 0;
        virtual int getPV(int x, int y, int layer = 0) = 0;

        virtual RGB_VALUE getPV_RGB(int x, int y) = 0;
        virtual void showImageOnWindow(char* ptrWinName) = 0;

};
#endif
