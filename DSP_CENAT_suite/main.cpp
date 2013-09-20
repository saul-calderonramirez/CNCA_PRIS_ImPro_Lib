#include <QApplication>
#include "mainwindow.h"




int main(int argc, char *argv[]){    
   // UnitTestsOCV_OCL::testOCV_OCL_2();
    QApplication a(argc, argv);
    MainWindow w;
    w.show();
    return a.exec();
    //return 0;
}
