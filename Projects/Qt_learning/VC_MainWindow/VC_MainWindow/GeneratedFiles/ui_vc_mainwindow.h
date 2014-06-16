/********************************************************************************
** Form generated from reading UI file 'vc_mainwindow.ui'
**
** Created by: Qt User Interface Compiler version 5.3.0
**
** WARNING! All changes made in this file will be lost when recompiling UI file!
********************************************************************************/

#ifndef UI_VC_MAINWINDOW_H
#define UI_VC_MAINWINDOW_H

#include <QtCore/QVariant>
#include <QtWidgets/QAction>
#include <QtWidgets/QApplication>
#include <QtWidgets/QButtonGroup>
#include <QtWidgets/QHeaderView>
#include <QtWidgets/QMainWindow>
#include <QtWidgets/QMenuBar>
#include <QtWidgets/QStatusBar>
#include <QtWidgets/QToolBar>
#include <QtWidgets/QWidget>

QT_BEGIN_NAMESPACE

class Ui_VC_MainWindowClass
{
public:
    QMenuBar *menuBar;
    QToolBar *mainToolBar;
    QWidget *centralWidget;
    QStatusBar *statusBar;

    void setupUi(QMainWindow *VC_MainWindowClass)
    {
        if (VC_MainWindowClass->objectName().isEmpty())
            VC_MainWindowClass->setObjectName(QStringLiteral("VC_MainWindowClass"));
        VC_MainWindowClass->resize(600, 400);
        menuBar = new QMenuBar(VC_MainWindowClass);
        menuBar->setObjectName(QStringLiteral("menuBar"));
        VC_MainWindowClass->setMenuBar(menuBar);
        mainToolBar = new QToolBar(VC_MainWindowClass);
        mainToolBar->setObjectName(QStringLiteral("mainToolBar"));
        VC_MainWindowClass->addToolBar(mainToolBar);
        centralWidget = new QWidget(VC_MainWindowClass);
        centralWidget->setObjectName(QStringLiteral("centralWidget"));
        VC_MainWindowClass->setCentralWidget(centralWidget);
        statusBar = new QStatusBar(VC_MainWindowClass);
        statusBar->setObjectName(QStringLiteral("statusBar"));
        VC_MainWindowClass->setStatusBar(statusBar);

        retranslateUi(VC_MainWindowClass);

        QMetaObject::connectSlotsByName(VC_MainWindowClass);
    } // setupUi

    void retranslateUi(QMainWindow *VC_MainWindowClass)
    {
        VC_MainWindowClass->setWindowTitle(QApplication::translate("VC_MainWindowClass", "VC_MainWindow", 0));
    } // retranslateUi

};

namespace Ui {
    class VC_MainWindowClass: public Ui_VC_MainWindowClass {};
} // namespace Ui

QT_END_NAMESPACE

#endif // UI_VC_MAINWINDOW_H
