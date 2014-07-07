#include "mainwindow.h"
#include "ui_mainwindow.h"

MainWindow::MainWindow(QWidget *parent) :
    QMainWindow(parent),
    ui(new Ui::MainWindow)
{
    ui->setupUi(this);
	createActions();
}

void MainWindow::createActions()
{
	ui->action_New->setStatusTip(tr("Create a new Spreadsheat file"));
	connect(ui->action_New, SIGNAL(ui->action_New->triggered()), this, SLOT(newFile()));
	
	ui->action_Save->setStatusTip(tr("Save the spreadsheat to file"));
	connect(ui->action_Save, SIGNAL(ui->action_Save->triggered()),this,SLOT(saveFile()));
	
	ui->actionSave_As->setStatusTip(tr("Save as"));
	connect(ui->actionSave_As,SIGNAL(ui->actionSave_As->triggered()),this,SLOT(saveAs()));

	
}

MainWindow::~MainWindow()
{
    delete ui;
}
