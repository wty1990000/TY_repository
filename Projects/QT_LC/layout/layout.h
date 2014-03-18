#ifndef LAYOUT_H
#define LAYOUT_H

#include <QtWidgets/QWidget>
#include "ui_layout.h"

class layout : public QWidget
{
	Q_OBJECT

public:
	layout(QWidget *parent = 0);
	~layout();

private:
	Ui::layoutClass ui;
};

#endif // LAYOUT_H
