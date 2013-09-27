#ifndef MAINWINDOW_H
#define MAINWINDOW_H
#include "controller.h"
#include "includes.h"

namespace Ui {
class MainWindow;
}

class MainWindow : public QMainWindow
{
    Q_OBJECT
    
public:
    explicit MainWindow(QWidget *parent = 0);
    ~MainWindow();
    Controller* ptrController;
    QGraphicsScene* ptrScene;
private slots:
    void on_btnCannyFilter_clicked();
    void showImage(QImage* qImage);
    void on_btnSobelFilter_clicked();

    void on_btnUmbBin_clicked();
    void on_mnOpenImage_clicked();
    void on_btnLoadDefImage_clicked();

    void on_btnContour_clicked();

    void on_btnSegWaterSheds_2_clicked();

    void on_btnGaussFilter_clicked();

    void on_btnContour_2_clicked();

    void on_btnContour_3_clicked();

    void on_btnKittler_clicked();
    void displayHistogram(vector<float> descriptorValues);

    void on_btnHSV_clicked();

private:
    Ui::MainWindow *ui;
};

#endif
