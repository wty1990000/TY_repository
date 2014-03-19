#ifndef DIALOG_H
#define DIALOG_H

#include <QtWidgets/QMainWindow>
#include "ui_dialog.h"

class dialog : public QMainWindow
{
	Q_OBJECT

public:
	dialog(QWidget *parent = 0);
	~dialog();

private:
	Ui::dialogClass ui;
	QAction *openAction;
	void open();
};

#endif // DIALOG_H
