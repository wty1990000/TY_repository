#include "gotodialog.h"
#include <QApplication>

int main(int argc, char *argv[])
{
    QApplication a(argc, argv);
    gotoDialog w;
    w.show();

    return a.exec();
}
