#ifndef VC_MAINWINDOW_H
#define VC_MAINWINDOW_H

#include <QtWidgets/QMainWindow>
#include "ui_vc_mainwindow.h"

class VC_MainWindow : public QMainWindow
{
	Q_OBJECT

public:
	VC_MainWindow(QWidget *parent = 0);
	~VC_MainWindow();

private:
	Ui::VC_MainWindowClass ui;
};

#endif // VC_MAINWINDOW_H
