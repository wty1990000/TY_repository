#ifndef QTFRAMEWORK_H
#define QTFRAMEWORK_H

#include <QtWidgets/QMainWindow>
#include "ui_qtframework.h"

class QtFramework : public QMainWindow
{
	Q_OBJECT

public:
	QtFramework(QWidget *parent = 0);
	~QtFramework();

private:
	Ui::QtFrameworkClass ui;

	public slots:
		void on_btnHello_clicked(){
			ui.btnHello->setText("Hi!");
		}
};

#endif // QTFRAMEWORK_H
