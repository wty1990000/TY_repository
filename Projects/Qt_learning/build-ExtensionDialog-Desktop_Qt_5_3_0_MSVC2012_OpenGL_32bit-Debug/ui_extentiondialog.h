/********************************************************************************
** Form generated from reading UI file 'extentiondialog.ui'
**
** Created by: Qt User Interface Compiler version 5.3.0
**
** WARNING! All changes made in this file will be lost when recompiling UI file!
********************************************************************************/

#ifndef UI_EXTENTIONDIALOG_H
#define UI_EXTENTIONDIALOG_H

#include <QtCore/QVariant>
#include <QtWidgets/QAction>
#include <QtWidgets/QApplication>
#include <QtWidgets/QButtonGroup>
#include <QtWidgets/QComboBox>
#include <QtWidgets/QDialog>
#include <QtWidgets/QGridLayout>
#include <QtWidgets/QGroupBox>
#include <QtWidgets/QHeaderView>
#include <QtWidgets/QLabel>
#include <QtWidgets/QPushButton>
#include <QtWidgets/QSpacerItem>
#include <QtWidgets/QVBoxLayout>

QT_BEGIN_NAMESPACE

class Ui_ExtentionDialog
{
public:
    QGridLayout *gridLayout_4;
    QGroupBox *primaryGroupBox;
    QGridLayout *gridLayout;
    QLabel *label;
    QComboBox *primaryColumnCombo;
    QSpacerItem *horizontalSpacer;
    QLabel *label_2;
    QComboBox *primaryOrderCombo;
    QVBoxLayout *verticalLayout;
    QPushButton *okButton;
    QPushButton *cancelButton;
    QSpacerItem *verticalSpacer;
    QPushButton *moreButton;
    QSpacerItem *verticalSpacer_2;
    QGroupBox *secondaryGroupBox;
    QGridLayout *gridLayout_2;
    QLabel *label_3;
    QComboBox *secondaryColumnCombo;
    QSpacerItem *horizontalSpacer_2;
    QLabel *label_4;
    QComboBox *secondaryOrderCombo;
    QGroupBox *tertiaryGroupBox;
    QGridLayout *gridLayout_3;
    QLabel *label_5;
    QComboBox *tertiaryColumnCombo;
    QSpacerItem *horizontalSpacer_3;
    QLabel *label_6;
    QComboBox *tertiaryOrderCombo;

    void setupUi(QDialog *ExtentionDialog)
    {
        if (ExtentionDialog->objectName().isEmpty())
            ExtentionDialog->setObjectName(QStringLiteral("ExtentionDialog"));
        ExtentionDialog->resize(263, 306);
        gridLayout_4 = new QGridLayout(ExtentionDialog);
        gridLayout_4->setSpacing(6);
        gridLayout_4->setContentsMargins(11, 11, 11, 11);
        gridLayout_4->setObjectName(QStringLiteral("gridLayout_4"));
        primaryGroupBox = new QGroupBox(ExtentionDialog);
        primaryGroupBox->setObjectName(QStringLiteral("primaryGroupBox"));
        gridLayout = new QGridLayout(primaryGroupBox);
        gridLayout->setSpacing(6);
        gridLayout->setContentsMargins(11, 11, 11, 11);
        gridLayout->setObjectName(QStringLiteral("gridLayout"));
        label = new QLabel(primaryGroupBox);
        label->setObjectName(QStringLiteral("label"));

        gridLayout->addWidget(label, 0, 0, 1, 1);

        primaryColumnCombo = new QComboBox(primaryGroupBox);
        primaryColumnCombo->setObjectName(QStringLiteral("primaryColumnCombo"));

        gridLayout->addWidget(primaryColumnCombo, 0, 1, 1, 1);

        horizontalSpacer = new QSpacerItem(41, 20, QSizePolicy::Expanding, QSizePolicy::Minimum);

        gridLayout->addItem(horizontalSpacer, 0, 2, 1, 1);

        label_2 = new QLabel(primaryGroupBox);
        label_2->setObjectName(QStringLiteral("label_2"));

        gridLayout->addWidget(label_2, 1, 0, 1, 1);

        primaryOrderCombo = new QComboBox(primaryGroupBox);
        primaryOrderCombo->setObjectName(QStringLiteral("primaryOrderCombo"));

        gridLayout->addWidget(primaryOrderCombo, 1, 1, 1, 2);


        gridLayout_4->addWidget(primaryGroupBox, 0, 0, 1, 1);

        verticalLayout = new QVBoxLayout();
        verticalLayout->setSpacing(6);
        verticalLayout->setObjectName(QStringLiteral("verticalLayout"));
        okButton = new QPushButton(ExtentionDialog);
        okButton->setObjectName(QStringLiteral("okButton"));
        okButton->setDefault(true);

        verticalLayout->addWidget(okButton);

        cancelButton = new QPushButton(ExtentionDialog);
        cancelButton->setObjectName(QStringLiteral("cancelButton"));

        verticalLayout->addWidget(cancelButton);

        verticalSpacer = new QSpacerItem(20, 0, QSizePolicy::Minimum, QSizePolicy::Expanding);

        verticalLayout->addItem(verticalSpacer);

        moreButton = new QPushButton(ExtentionDialog);
        moreButton->setObjectName(QStringLiteral("moreButton"));
        moreButton->setCheckable(true);

        verticalLayout->addWidget(moreButton);


        gridLayout_4->addLayout(verticalLayout, 0, 1, 2, 1);

        verticalSpacer_2 = new QSpacerItem(20, 27, QSizePolicy::Minimum, QSizePolicy::Expanding);

        gridLayout_4->addItem(verticalSpacer_2, 1, 0, 1, 1);

        secondaryGroupBox = new QGroupBox(ExtentionDialog);
        secondaryGroupBox->setObjectName(QStringLiteral("secondaryGroupBox"));
        gridLayout_2 = new QGridLayout(secondaryGroupBox);
        gridLayout_2->setSpacing(6);
        gridLayout_2->setContentsMargins(11, 11, 11, 11);
        gridLayout_2->setObjectName(QStringLiteral("gridLayout_2"));
        label_3 = new QLabel(secondaryGroupBox);
        label_3->setObjectName(QStringLiteral("label_3"));

        gridLayout_2->addWidget(label_3, 0, 0, 1, 1);

        secondaryColumnCombo = new QComboBox(secondaryGroupBox);
        secondaryColumnCombo->setObjectName(QStringLiteral("secondaryColumnCombo"));

        gridLayout_2->addWidget(secondaryColumnCombo, 0, 1, 1, 1);

        horizontalSpacer_2 = new QSpacerItem(40, 20, QSizePolicy::Expanding, QSizePolicy::Minimum);

        gridLayout_2->addItem(horizontalSpacer_2, 0, 2, 1, 1);

        label_4 = new QLabel(secondaryGroupBox);
        label_4->setObjectName(QStringLiteral("label_4"));

        gridLayout_2->addWidget(label_4, 1, 0, 1, 1);

        secondaryOrderCombo = new QComboBox(secondaryGroupBox);
        secondaryOrderCombo->setObjectName(QStringLiteral("secondaryOrderCombo"));

        gridLayout_2->addWidget(secondaryOrderCombo, 1, 1, 1, 2);


        gridLayout_4->addWidget(secondaryGroupBox, 2, 0, 1, 1);

        tertiaryGroupBox = new QGroupBox(ExtentionDialog);
        tertiaryGroupBox->setObjectName(QStringLiteral("tertiaryGroupBox"));
        gridLayout_3 = new QGridLayout(tertiaryGroupBox);
        gridLayout_3->setSpacing(6);
        gridLayout_3->setContentsMargins(11, 11, 11, 11);
        gridLayout_3->setObjectName(QStringLiteral("gridLayout_3"));
        label_5 = new QLabel(tertiaryGroupBox);
        label_5->setObjectName(QStringLiteral("label_5"));

        gridLayout_3->addWidget(label_5, 0, 0, 1, 1);

        tertiaryColumnCombo = new QComboBox(tertiaryGroupBox);
        tertiaryColumnCombo->setObjectName(QStringLiteral("tertiaryColumnCombo"));

        gridLayout_3->addWidget(tertiaryColumnCombo, 0, 1, 1, 1);

        horizontalSpacer_3 = new QSpacerItem(40, 20, QSizePolicy::Expanding, QSizePolicy::Minimum);

        gridLayout_3->addItem(horizontalSpacer_3, 0, 2, 1, 1);

        label_6 = new QLabel(tertiaryGroupBox);
        label_6->setObjectName(QStringLiteral("label_6"));

        gridLayout_3->addWidget(label_6, 1, 0, 1, 1);

        tertiaryOrderCombo = new QComboBox(tertiaryGroupBox);
        tertiaryOrderCombo->setObjectName(QStringLiteral("tertiaryOrderCombo"));

        gridLayout_3->addWidget(tertiaryOrderCombo, 1, 1, 1, 2);


        gridLayout_4->addWidget(tertiaryGroupBox, 3, 0, 1, 1);

        cancelButton->raise();
        primaryGroupBox->raise();
        okButton->raise();
        moreButton->raise();
        secondaryGroupBox->raise();
        tertiaryGroupBox->raise();
        QWidget::setTabOrder(primaryColumnCombo, primaryOrderCombo);
        QWidget::setTabOrder(primaryOrderCombo, secondaryColumnCombo);
        QWidget::setTabOrder(secondaryColumnCombo, secondaryOrderCombo);
        QWidget::setTabOrder(secondaryOrderCombo, tertiaryColumnCombo);
        QWidget::setTabOrder(tertiaryColumnCombo, tertiaryOrderCombo);
        QWidget::setTabOrder(tertiaryOrderCombo, okButton);
        QWidget::setTabOrder(okButton, cancelButton);
        QWidget::setTabOrder(cancelButton, moreButton);

        retranslateUi(ExtentionDialog);
        QObject::connect(okButton, SIGNAL(clicked()), ExtentionDialog, SLOT(accept()));
        QObject::connect(cancelButton, SIGNAL(clicked()), ExtentionDialog, SLOT(reject()));
        QObject::connect(moreButton, SIGNAL(toggled(bool)), secondaryGroupBox, SLOT(setVisible(bool)));
        QObject::connect(moreButton, SIGNAL(toggled(bool)), tertiaryGroupBox, SLOT(setVisible(bool)));

        QMetaObject::connectSlotsByName(ExtentionDialog);
    } // setupUi

    void retranslateUi(QDialog *ExtentionDialog)
    {
        ExtentionDialog->setWindowTitle(QApplication::translate("ExtentionDialog", "ExtentionDialog", 0));
        primaryGroupBox->setTitle(QApplication::translate("ExtentionDialog", "&Primary Key", 0));
        label->setText(QApplication::translate("ExtentionDialog", "Column", 0));
        primaryColumnCombo->clear();
        primaryColumnCombo->insertItems(0, QStringList()
         << QApplication::translate("ExtentionDialog", "None", 0)
        );
        label_2->setText(QApplication::translate("ExtentionDialog", "Order", 0));
        primaryOrderCombo->clear();
        primaryOrderCombo->insertItems(0, QStringList()
         << QApplication::translate("ExtentionDialog", "Ascending", 0)
         << QApplication::translate("ExtentionDialog", "Descending", 0)
        );
        okButton->setText(QApplication::translate("ExtentionDialog", "OK", 0));
        cancelButton->setText(QApplication::translate("ExtentionDialog", "Cancel", 0));
        moreButton->setText(QApplication::translate("ExtentionDialog", "&More", 0));
        secondaryGroupBox->setTitle(QApplication::translate("ExtentionDialog", "&Secondary Key", 0));
        label_3->setText(QApplication::translate("ExtentionDialog", "Column", 0));
        secondaryColumnCombo->clear();
        secondaryColumnCombo->insertItems(0, QStringList()
         << QApplication::translate("ExtentionDialog", "None", 0)
        );
        label_4->setText(QApplication::translate("ExtentionDialog", "Order", 0));
        secondaryOrderCombo->clear();
        secondaryOrderCombo->insertItems(0, QStringList()
         << QApplication::translate("ExtentionDialog", "Ascending", 0)
         << QApplication::translate("ExtentionDialog", "Descending", 0)
        );
        tertiaryGroupBox->setTitle(QApplication::translate("ExtentionDialog", "&Tertiary Key", 0));
        label_5->setText(QApplication::translate("ExtentionDialog", "Column", 0));
        tertiaryColumnCombo->clear();
        tertiaryColumnCombo->insertItems(0, QStringList()
         << QApplication::translate("ExtentionDialog", "None", 0)
        );
        label_6->setText(QApplication::translate("ExtentionDialog", "Order", 0));
        tertiaryOrderCombo->clear();
        tertiaryOrderCombo->insertItems(0, QStringList()
         << QApplication::translate("ExtentionDialog", "Ascending", 0)
         << QApplication::translate("ExtentionDialog", "Descending", 0)
        );
    } // retranslateUi

};

namespace Ui {
    class ExtentionDialog: public Ui_ExtentionDialog {};
} // namespace Ui

QT_END_NAMESPACE

#endif // UI_EXTENTIONDIALOG_H
