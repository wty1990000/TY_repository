#include "vc_mainwindow.h"
#include <QtWidgets/QApplication>

int main(int argc, char *argv[])
{
	QApplication a(argc, argv);
	VC_MainWindow w;
	w.show();
	return a.exec();
}
