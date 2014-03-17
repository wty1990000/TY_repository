/********************************************************************************
** Form generated from reading UI file 'helloworld.ui'
**
** Created by: Qt User Interface Compiler version 5.1.1
**
** WARNING! All changes made in this file will be lost when recompiling UI file!
********************************************************************************/

#ifndef UI_HELLOWORLD_H
#define UI_HELLOWORLD_H

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

class Ui_helloworldClass
{
public:
    QMenuBar *menuBar;
    QToolBar *mainToolBar;
    QWidget *centralWidget;
    QStatusBar *statusBar;

    void setupUi(QMainWindow *helloworldClass)
    {
        if (helloworldClass->objectName().isEmpty())
            helloworldClass->setObjectName(QStringLiteral("helloworldClass"));
        helloworldClass->resize(600, 400);
        menuBar = new QMenuBar(helloworldClass);
        menuBar->setObjectName(QStringLiteral("menuBar"));
        helloworldClass->setMenuBar(menuBar);
        mainToolBar = new QToolBar(helloworldClass);
        mainToolBar->setObjectName(QStringLiteral("mainToolBar"));
        helloworldClass->addToolBar(mainToolBar);
        centralWidget = new QWidget(helloworldClass);
        centralWidget->setObjectName(QStringLiteral("centralWidget"));
        helloworldClass->setCentralWidget(centralWidget);
        statusBar = new QStatusBar(helloworldClass);
        statusBar->setObjectName(QStringLiteral("statusBar"));
        helloworldClass->setStatusBar(statusBar);

        retranslateUi(helloworldClass);

        QMetaObject::connectSlotsByName(helloworldClass);
    } // setupUi

    void retranslateUi(QMainWindow *helloworldClass)
    {
        helloworldClass->setWindowTitle(QApplication::translate("helloworldClass", "helloworld", 0));
    } // retranslateUi

};

namespace Ui {
    class helloworldClass: public Ui_helloworldClass {};
} // namespace Ui

QT_END_NAMESPACE

#endif // UI_HELLOWORLD_H
