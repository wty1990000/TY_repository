#include "nrf_viewer.h"
#include <QtWidgets/QApplication>

int main(int argc, char *argv[])
{
	QApplication a(argc, argv);
	NRF_Viewer w;
	w.show();
	return a.exec();
}
