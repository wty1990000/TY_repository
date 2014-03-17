#include "helloworld.h"
#include <QtWidgets/QApplication>

int main(int argc, char *argv[])
{
	QApplication a(argc, argv);
	helloworld w;
	w.show();
	return a.exec();
}
