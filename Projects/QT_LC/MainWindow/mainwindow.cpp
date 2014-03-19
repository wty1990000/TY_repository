#include <QAction>
#include <QMenuBar>
#include <QMessageBox>
#include <QStatusBar>
#include <QToolBar>

#include "mainwindow.h"

MainWindow::MainWindow(QWidget *parent)
	: QMainWindow(parent)
{
	ui.setupUi(this);
	setWindowTitle(tr("Main Window"));
	openAction = new QAction(QIcon(tr(":/images/doc_open")), tr("&Open..."), this);
    openAction->setShortcuts(QKeySequence::Open);
    openAction->setStatusTip(tr("Open an existing file"));
    connect(openAction, &QAction::triggered, this, &MainWindow::open);
 
    QMenu *file = menuBar()->addMenu(tr("&File"));
    file->addAction(openAction);
 
    QToolBar *toolBar = addToolBar(tr("&File"));
    toolBar->addAction(openAction);
	toolBar->setMovable(false);
 
    statusBar() ;
}

MainWindow::~MainWindow()
{

}
void MainWindow::open()
{
    QMessageBox::information(this, tr("Information"), tr("Open"));
}