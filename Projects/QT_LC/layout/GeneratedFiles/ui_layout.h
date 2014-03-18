/********************************************************************************
** Form generated from reading UI file 'layout.ui'
**
** Created by: Qt User Interface Compiler version 5.1.1
**
** WARNING! All changes made in this file will be lost when recompiling UI file!
********************************************************************************/

#ifndef UI_LAYOUT_H
#define UI_LAYOUT_H

#include <QtCore/QVariant>
#include <QtWidgets/QAction>
#include <QtWidgets/QApplication>
#include <QtWidgets/QButtonGroup>
#include <QtWidgets/QHeaderView>
#include <QtWidgets/QWidget>

QT_BEGIN_NAMESPACE

class Ui_layoutClass
{
public:

    void setupUi(QWidget *layoutClass)
    {
        if (layoutClass->objectName().isEmpty())
            layoutClass->setObjectName(QStringLiteral("layoutClass"));
        layoutClass->resize(600, 400);

        retranslateUi(layoutClass);

        QMetaObject::connectSlotsByName(layoutClass);
    } // setupUi

    void retranslateUi(QWidget *layoutClass)
    {
        layoutClass->setWindowTitle(QApplication::translate("layoutClass", "layout", 0));
    } // retranslateUi

};

namespace Ui {
    class layoutClass: public Ui_layoutClass {};
} // namespace Ui

QT_END_NAMESPACE

#endif // UI_LAYOUT_H
