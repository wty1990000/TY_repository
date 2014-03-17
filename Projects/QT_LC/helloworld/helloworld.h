#ifndef HELLOWORLD_H
#define HELLOWORLD_H

#include <QtWidgets/QMainWindow>
#include "ui_helloworld.h"

class helloworld : public QMainWindow
{
	Q_OBJECT

public:
	helloworld(QWidget *parent = 0);
	~helloworld();

private:
	Ui::helloworldClass ui;
};

#endif // HELLOWORLD_H
