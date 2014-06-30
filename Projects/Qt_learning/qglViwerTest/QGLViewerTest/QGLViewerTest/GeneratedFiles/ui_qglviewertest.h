/********************************************************************************
** Form generated from reading UI file 'qglviewertest.ui'
**
** Created by: Qt User Interface Compiler version 5.3.0
**
** WARNING! All changes made in this file will be lost when recompiling UI file!
********************************************************************************/

#ifndef UI_QGLVIEWERTEST_H
#define UI_QGLVIEWERTEST_H

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

class Ui_QGLViewerTestClass
{
public:
    QMenuBar *menuBar;
    QToolBar *mainToolBar;
    QWidget *centralWidget;
    QStatusBar *statusBar;

    void setupUi(QMainWindow *QGLViewerTestClass)
    {
        if (QGLViewerTestClass->objectName().isEmpty())
            QGLViewerTestClass->setObjectName(QStringLiteral("QGLViewerTestClass"));
        QGLViewerTestClass->resize(600, 400);
        menuBar = new QMenuBar(QGLViewerTestClass);
        menuBar->setObjectName(QStringLiteral("menuBar"));
        QGLViewerTestClass->setMenuBar(menuBar);
        mainToolBar = new QToolBar(QGLViewerTestClass);
        mainToolBar->setObjectName(QStringLiteral("mainToolBar"));
        QGLViewerTestClass->addToolBar(mainToolBar);
        centralWidget = new QWidget(QGLViewerTestClass);
        centralWidget->setObjectName(QStringLiteral("centralWidget"));
        QGLViewerTestClass->setCentralWidget(centralWidget);
        statusBar = new QStatusBar(QGLViewerTestClass);
        statusBar->setObjectName(QStringLiteral("statusBar"));
        QGLViewerTestClass->setStatusBar(statusBar);

        retranslateUi(QGLViewerTestClass);

        QMetaObject::connectSlotsByName(QGLViewerTestClass);
    } // setupUi

    void retranslateUi(QMainWindow *QGLViewerTestClass)
    {
        QGLViewerTestClass->setWindowTitle(QApplication::translate("QGLViewerTestClass", "QGLViewerTest", 0));
    } // retranslateUi

};

namespace Ui {
    class QGLViewerTestClass: public Ui_QGLViewerTestClass {};
} // namespace Ui

QT_END_NAMESPACE

#endif // UI_QGLVIEWERTEST_H
