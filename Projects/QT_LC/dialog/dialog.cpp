#include "dialog.h"
#include <QMessageBox>
#include <QAction>


dialog::dialog(QWidget *parent)
	: QMainWindow(parent)
{
	ui.setupUi(this);
	setWindowTitle(tr("Main Window"));
	openAction = new QAction(QIcon(":/images/doc-open"), tr("&Open..."), this);
}

dialog::~dialog()
{

}
void dialog::open()
{
	QMessageBox::information(this,tr("Information"),tr("Open"));
}