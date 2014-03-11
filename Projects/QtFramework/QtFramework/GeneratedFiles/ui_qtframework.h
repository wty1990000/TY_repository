/********************************************************************************
** Form generated from reading UI file 'qtframework.ui'
**
** Created by: Qt User Interface Compiler version 5.1.1
**
** WARNING! All changes made in this file will be lost when recompiling UI file!
********************************************************************************/

#ifndef UI_QTFRAMEWORK_H
#define UI_QTFRAMEWORK_H

#include <QtCore/QVariant>
#include <QtWidgets/QAction>
#include <QtWidgets/QApplication>
#include <QtWidgets/QButtonGroup>
#include <QtWidgets/QHeaderView>
#include <QtWidgets/QMainWindow>
#include <QtWidgets/QMenuBar>
#include <QtWidgets/QPushButton>
#include <QtWidgets/QStatusBar>
#include <QtWidgets/QToolBar>
#include <QtWidgets/QWidget>

QT_BEGIN_NAMESPACE

class Ui_QtFrameworkClass
{
public:
    QWidget *centralWidget;
    QPushButton *btnHello;
    QMenuBar *menuBar;
    QToolBar *mainToolBar;
    QStatusBar *statusBar;

    void setupUi(QMainWindow *QtFrameworkClass)
    {
        if (QtFrameworkClass->objectName().isEmpty())
            QtFrameworkClass->setObjectName(QStringLiteral("QtFrameworkClass"));
        QtFrameworkClass->resize(600, 400);
        centralWidget = new QWidget(QtFrameworkClass);
        centralWidget->setObjectName(QStringLiteral("centralWidget"));
        btnHello = new QPushButton(centralWidget);
        btnHello->setObjectName(QStringLiteral("btnHello"));
        btnHello->setGeometry(QRect(260, 130, 75, 23));
        QtFrameworkClass->setCentralWidget(centralWidget);
        menuBar = new QMenuBar(QtFrameworkClass);
        menuBar->setObjectName(QStringLiteral("menuBar"));
        menuBar->setGeometry(QRect(0, 0, 600, 21));
        QtFrameworkClass->setMenuBar(menuBar);
        mainToolBar = new QToolBar(QtFrameworkClass);
        mainToolBar->setObjectName(QStringLiteral("mainToolBar"));
        QtFrameworkClass->addToolBar(Qt::TopToolBarArea, mainToolBar);
        statusBar = new QStatusBar(QtFrameworkClass);
        statusBar->setObjectName(QStringLiteral("statusBar"));
        QtFrameworkClass->setStatusBar(statusBar);

        retranslateUi(QtFrameworkClass);

        QMetaObject::connectSlotsByName(QtFrameworkClass);
    } // setupUi

    void retranslateUi(QMainWindow *QtFrameworkClass)
    {
        QtFrameworkClass->setWindowTitle(QApplication::translate("QtFrameworkClass", "QtFramework", 0));
        btnHello->setText(QApplication::translate("QtFrameworkClass", "PushButton", 0));
    } // retranslateUi

};

namespace Ui {
    class QtFrameworkClass: public Ui_QtFrameworkClass {};
} // namespace Ui

QT_END_NAMESPACE

#endif // UI_QTFRAMEWORK_H
