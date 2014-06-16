#include "extentiondialog.h"
#include <QApplication>

int main(int argc, char *argv[])
{
    QApplication a(argc, argv);
    ExtentionDialog w;
    w.setColumnRange('C','F');
    w.show();

    return a.exec();
}
