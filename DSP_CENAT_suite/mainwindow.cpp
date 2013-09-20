#include "mainwindow.h"
#include "ui_mainwindow.h"

MainWindow::MainWindow(QWidget *parent) :
    QMainWindow(parent),
    ui(new Ui::MainWindow){
    this->ptrScene = new QGraphicsScene;
    this->ptrController = new Controller();
    this->ui->setupUi(this);
    connect(ui->actionOtro_Abrir, SIGNAL(triggered()), this, SLOT(on_mnOpenImage_clicked()));
}

void MainWindow::showImage(QImage* qImage){
     QPixmap pixMap = QPixmap::fromImage(*qImage);
     this->ptrScene->addPixmap(pixMap);
     this->ui->imgDisplay->setScene(this->ptrScene);
     this->ui->imgDisplay->show();
}

MainWindow::~MainWindow(){
    delete ui;
    delete this->ptrController;
}

void MainWindow::on_mnOpenImage_clicked(){
    QString fileName = QFileDialog::getOpenFileName(this, tr("Open File"),"", tr("Files (*.*)"));
    if(fileName != NULL){
        string nameStd = fileName.toStdString();
        char *ptrName = new char[nameStd.size() + 1];
        ptrName[nameStd.size()] = 0;
        memcpy(ptrName, nameStd.c_str(), nameStd.size());
        cout << "Image to load: " << ptrName << endl;
        this->ptrController->loadImage(ptrName);
        QImage* qImage = this->ptrController->getImage()->getQImage();
        showImage(qImage);
    }
}

void MainWindow::on_btnCannyFilter_clicked(){
    try{
        this->ptrController->applyFilterCanny();
        QImage* qImage = this->ptrController->getImage()->getQImage();
        showImage(qImage);
    }
    catch(Controller::ControllerException contExc){
        cout << contExc.what() << endl;
    }
}

void MainWindow::on_btnSobelFilter_clicked(){
    try{
        this->ptrController->applyFilterSobel();
        QImage* qImage = this->ptrController->getImage()->getQImage();
        showImage(qImage);
    }
    catch(Controller::ControllerException contExc){
        cout << contExc.what() << endl;
    }
}

void MainWindow::on_btnUmbBin_clicked(){
    try{
        this->ptrController->applyBinaryThreshold();
        QImage* qImage = this->ptrController->getImage()->getQImage();
        showImage(qImage);
    }
    catch(Controller::ControllerException contExc){
        cout << contExc.what() << endl;
    }
}

void MainWindow::on_btnLoadDefImage_clicked(){
    try{
        this->ptrController->loadImage((char*)DEF_IMAGE);
        QImage* qImage = this->ptrController->getImage()->getQImage();
        showImage(qImage);
    }
    catch(Controller::ControllerException contExc){
        cout << contExc.what() << endl;
    }

}

void MainWindow::on_btnContour_clicked(){
    try{
        cout << "hizo click en find contour" << endl;
        this->ptrController->findCountour();
    }
    catch(Controller::ControllerException contExc){
        cout << contExc.what() << endl;
    }

}

void MainWindow::on_btnSegWaterSheds_2_clicked(){
    try{
        this->ptrController->runBenchmarks();
    }
    catch(Controller::ControllerException contExc){
        cout << contExc.what() << endl;
    }
}

void MainWindow::on_btnGaussFilter_clicked(){
    try{
        this->ptrController->applyFilterGauss();
        QImage* qImage = this->ptrController->getImage()->getQImage();
        showImage(qImage);
    }
    catch(Controller::ControllerException contExc){
        cout << contExc.what() << endl;
    }

}




void MainWindow::displayHOGHistogram(vector<float> descriptorValues){

    QVector<double> x(descriptorValues.size()), y(descriptorValues.size()); // initialize with entries 0..100
    float max = 0;
    for(int i = 0; i < descriptorValues.size(); ++i){
        y[i] = descriptorValues[i];
        x[i] = i;
        if(descriptorValues[i] > max){
            max = descriptorValues[i];
        }

    }
    // create graph and assign data to it:
    this->ui->plotter->addGraph();
    this->ui->plotter->graph(0)->setData(x, y);
    // give the axes some labels:
    this->ui->plotter->xAxis->setLabel("x");
    this->ui->plotter->yAxis->setLabel("y");
    // set axes ranges, so we see all data:
    this->ui->plotter->xAxis->setRange(0, descriptorValues.size());
    this->ui->plotter->yAxis->setRange(0, 1);
    this->ui->plotter->replot();
}

void MainWindow::on_btnKittler_clicked()
{
    cout << "hog button!" << endl;
    // generate some data:


}



void MainWindow::on_btnHSV_clicked(){
    try{
        cout << "hizo click en convertir a HSV" << endl;
        this->ptrController->convertToHSV();
    }
    catch(Controller::ControllerException contExc){
        cout << contExc.what() << endl;
    }
}
