#include "extentiondialog.h"
#include <QApplication>

int main(int argc, char *argv[])
{
    QApplication a(argc, argv);
    ExtentionDialog w;
    w.show();

    return a.exec();
}
