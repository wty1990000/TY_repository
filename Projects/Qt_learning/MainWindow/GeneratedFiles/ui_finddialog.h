/********************************************************************************
** Form generated from reading UI file 'finddialog.ui'
**
** Created by: Qt User Interface Compiler version 5.3.0
**
** WARNING! All changes made in this file will be lost when recompiling UI file!
********************************************************************************/

#ifndef UI_FINDDIALOG_H
#define UI_FINDDIALOG_H

#include <QtCore/QVariant>
#include <QtWidgets/QAction>
#include <QtWidgets/QApplication>
#include <QtWidgets/QButtonGroup>
#include <QtWidgets/QCheckBox>
#include <QtWidgets/QDialog>
#include <QtWidgets/QFormLayout>
#include <QtWidgets/QHBoxLayout>
#include <QtWidgets/QHeaderView>
#include <QtWidgets/QLabel>
#include <QtWidgets/QLineEdit>
#include <QtWidgets/QPushButton>
#include <QtWidgets/QVBoxLayout>

QT_BEGIN_NAMESPACE

class Ui_findDialog
{
public:
    QFormLayout *formLayout;
    QVBoxLayout *verticalLayout;
    QHBoxLayout *horizontalLayout;
    QLabel *label;
    QLineEdit *lineEdit;
    QCheckBox *caseCheckBox;
    QCheckBox *backwardCheckBox;
    QVBoxLayout *verticalLayout_2;
    QPushButton *findButton;
    QPushButton *closeButton;

    void setupUi(QDialog *findDialog)
    {
        if (findDialog->objectName().isEmpty())
            findDialog->setObjectName(QStringLiteral("findDialog"));
        findDialog->setWindowModality(Qt::NonModal);
        findDialog->resize(295, 88);
        formLayout = new QFormLayout(findDialog);
        formLayout->setObjectName(QStringLiteral("formLayout"));
        verticalLayout = new QVBoxLayout();
        verticalLayout->setObjectName(QStringLiteral("verticalLayout"));
        horizontalLayout = new QHBoxLayout();
        horizontalLayout->setObjectName(QStringLiteral("horizontalLayout"));
        label = new QLabel(findDialog);
        label->setObjectName(QStringLiteral("label"));

        horizontalLayout->addWidget(label);

        lineEdit = new QLineEdit(findDialog);
        lineEdit->setObjectName(QStringLiteral("lineEdit"));

        horizontalLayout->addWidget(lineEdit);


        verticalLayout->addLayout(horizontalLayout);

        caseCheckBox = new QCheckBox(findDialog);
        caseCheckBox->setObjectName(QStringLiteral("caseCheckBox"));
        caseCheckBox->setChecked(true);

        verticalLayout->addWidget(caseCheckBox);

        backwardCheckBox = new QCheckBox(findDialog);
        backwardCheckBox->setObjectName(QStringLiteral("backwardCheckBox"));

        verticalLayout->addWidget(backwardCheckBox);


        formLayout->setLayout(0, QFormLayout::LabelRole, verticalLayout);

        verticalLayout_2 = new QVBoxLayout();
        verticalLayout_2->setObjectName(QStringLiteral("verticalLayout_2"));
        findButton = new QPushButton(findDialog);
        findButton->setObjectName(QStringLiteral("findButton"));
        findButton->setDefault(true);

        verticalLayout_2->addWidget(findButton);

        closeButton = new QPushButton(findDialog);
        closeButton->setObjectName(QStringLiteral("closeButton"));

        verticalLayout_2->addWidget(closeButton);


        formLayout->setLayout(0, QFormLayout::FieldRole, verticalLayout_2);

#ifndef QT_NO_SHORTCUT
        label->setBuddy(lineEdit);
#endif // QT_NO_SHORTCUT

        retranslateUi(findDialog);

        QMetaObject::connectSlotsByName(findDialog);
    } // setupUi

    void retranslateUi(QDialog *findDialog)
    {
        findDialog->setWindowTitle(QApplication::translate("findDialog", "Find", 0));
        label->setText(QApplication::translate("findDialog", "Find &what:", 0));
        caseCheckBox->setText(QApplication::translate("findDialog", "Match &case", 0));
        backwardCheckBox->setText(QApplication::translate("findDialog", "Search &backward", 0));
        findButton->setText(QApplication::translate("findDialog", "&Find", 0));
        closeButton->setText(QApplication::translate("findDialog", "Close", 0));
    } // retranslateUi

};

namespace Ui {
    class findDialog: public Ui_findDialog {};
} // namespace Ui

QT_END_NAMESPACE

#endif // UI_FINDDIALOG_H
