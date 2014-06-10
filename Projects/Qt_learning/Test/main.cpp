#include "mainwindow.h"
#include <QApplication>
#include <QHBoxLayout>
#include <QSpinBox>
#include <QSlider>

int main(int argc, char *argv[])
{
    QApplication a(argc, argv);
    MainWindow w;
    w.setWindowTitle("Enter your age!");
    QWidget *window = new QWidget(&w);
    QSpinBox *spinBox = new QSpinBox(window);
    QSlider *slider = new QSlider(Qt::Horizontal,window);
    spinBox->setRange(0,130);
    slider->setRange(0,130);

    QObject::connect(spinBox,SIGNAL(valueChanged(int)),slider,SLOT(setValue(int)));
    QObject::connect(slider, SIGNAL(valueChanged(int)),spinBox,SLOT(setValue(int)));

    spinBox->setValue(35);

    QHBoxLayout *layout = new QHBoxLayout;
    layout->addWidget(spinBox);
    layout->addWidget(slider);

    window->setLayout(layout);

    window->show();
    w.show();

    return a.exec();
}
