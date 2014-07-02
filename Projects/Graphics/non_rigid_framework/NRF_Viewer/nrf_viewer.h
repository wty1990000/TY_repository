#ifndef NRF_VIEWER_H
#define NRF_VIEWER_H

#include <QtWidgets/QMainWindow>
#include "ui_nrf_viewer.h"

class NRF_Viewer : public QMainWindow
{
	Q_OBJECT

public:
	NRF_Viewer(QWidget *parent = 0);
	~NRF_Viewer();

private:
	Ui::NRF_ViewerClass ui;
};

#endif // NRF_VIEWER_H
