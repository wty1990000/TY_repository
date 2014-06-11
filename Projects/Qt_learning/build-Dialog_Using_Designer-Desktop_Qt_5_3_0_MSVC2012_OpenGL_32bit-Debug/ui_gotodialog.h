/********************************************************************************
** Form generated from reading UI file 'gotodialog.ui'
**
** Created by: Qt User Interface Compiler version 5.3.0
**
** WARNING! All changes made in this file will be lost when recompiling UI file!
********************************************************************************/

#ifndef UI_GOTODIALOG_H
#define UI_GOTODIALOG_H

#include <QtCore/QVariant>
#include <QtWidgets/QAction>
#include <QtWidgets/QApplication>
#include <QtWidgets/QButtonGroup>
#include <QtWidgets/QDialog>
#include <QtWidgets/QHBoxLayout>
#include <QtWidgets/QHeaderView>
#include <QtWidgets/QLabel>
#include <QtWidgets/QLineEdit>
#include <QtWidgets/QPushButton>
#include <QtWidgets/QSpacerItem>
#include <QtWidgets/QVBoxLayout>

QT_BEGIN_NAMESPACE

class Ui_gotoDialog
{
public:
    QVBoxLayout *verticalLayout;
    QHBoxLayout *horizontalLayout;
    QLabel *label;
    QLineEdit *lineEdit;
    QHBoxLayout *horizontalLayout_2;
    QSpacerItem *horizontalSpacer;
    QPushButton *okButton;
    QPushButton *cancelButton;

    void setupUi(QDialog *gotoDialog)
    {
        if (gotoDialog->objectName().isEmpty())
            gotoDialog->setObjectName(QStringLiteral("gotoDialog"));
        gotoDialog->resize(222, 71);
        verticalLayout = new QVBoxLayout(gotoDialog);
        verticalLayout->setSpacing(6);
        verticalLayout->setContentsMargins(11, 11, 11, 11);
        verticalLayout->setObjectName(QStringLiteral("verticalLayout"));
        horizontalLayout = new QHBoxLayout();
        horizontalLayout->setSpacing(6);
        horizontalLayout->setObjectName(QStringLiteral("horizontalLayout"));
        label = new QLabel(gotoDialog);
        label->setObjectName(QStringLiteral("label"));

        horizontalLayout->addWidget(label);

        lineEdit = new QLineEdit(gotoDialog);
        lineEdit->setObjectName(QStringLiteral("lineEdit"));

        horizontalLayout->addWidget(lineEdit);


        verticalLayout->addLayout(horizontalLayout);

        horizontalLayout_2 = new QHBoxLayout();
        horizontalLayout_2->setSpacing(6);
        horizontalLayout_2->setObjectName(QStringLiteral("horizontalLayout_2"));
        horizontalSpacer = new QSpacerItem(40, 20, QSizePolicy::Expanding, QSizePolicy::Minimum);

        horizontalLayout_2->addItem(horizontalSpacer);

        okButton = new QPushButton(gotoDialog);
        okButton->setObjectName(QStringLiteral("okButton"));
        okButton->setEnabled(false);
        okButton->setAutoDefault(true);
        okButton->setDefault(true);

        horizontalLayout_2->addWidget(okButton);

        cancelButton = new QPushButton(gotoDialog);
        cancelButton->setObjectName(QStringLiteral("cancelButton"));

        horizontalLayout_2->addWidget(cancelButton);


        verticalLayout->addLayout(horizontalLayout_2);

#ifndef QT_NO_SHORTCUT
        label->setBuddy(lineEdit);
#endif // QT_NO_SHORTCUT

        retranslateUi(gotoDialog);

        QMetaObject::connectSlotsByName(gotoDialog);
    } // setupUi

    void retranslateUi(QDialog *gotoDialog)
    {
        gotoDialog->setWindowTitle(QApplication::translate("gotoDialog", "gotoDialog", 0));
        label->setText(QApplication::translate("gotoDialog", "&Cell Location", 0));
        okButton->setText(QApplication::translate("gotoDialog", "OK", 0));
        cancelButton->setText(QApplication::translate("gotoDialog", "Cancel", 0));
    } // retranslateUi

};

namespace Ui {
    class gotoDialog: public Ui_gotoDialog {};
} // namespace Ui

QT_END_NAMESPACE

#endif // UI_GOTODIALOG_H
