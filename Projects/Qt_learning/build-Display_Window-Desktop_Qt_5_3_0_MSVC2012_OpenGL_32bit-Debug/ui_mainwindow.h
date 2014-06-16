/********************************************************************************
** Form generated from reading UI file 'mainwindow.ui'
**
** Created by: Qt User Interface Compiler version 5.3.0
**
** WARNING! All changes made in this file will be lost when recompiling UI file!
********************************************************************************/

#ifndef UI_MAINWINDOW_H
#define UI_MAINWINDOW_H

#include <QtCore/QVariant>
#include <QtWidgets/QAction>
#include <QtWidgets/QApplication>
#include <QtWidgets/QButtonGroup>
#include <QtWidgets/QHeaderView>
#include <QtWidgets/QMainWindow>
#include <QtWidgets/QMenu>
#include <QtWidgets/QMenuBar>
#include <QtWidgets/QStatusBar>
#include <QtWidgets/QToolBar>
#include <QtWidgets/QWidget>

QT_BEGIN_NAMESPACE

class Ui_MainWindow
{
public:
    QAction *action_New;
    QAction *action_Open;
    QAction *action_Save;
    QAction *actionSave_As;
    QAction *actionE_xit;
    QAction *actionCu_t;
    QAction *action_Copy;
    QAction *action_Paste;
    QAction *action_Delete;
    QAction *action_Find;
    QAction *action_Go_to_Cell;
    QAction *action_Row;
    QAction *action_Column;
    QAction *action_All;
    QAction *action_Recalculate;
    QAction *action_Sort;
    QAction *action_Show_Grid;
    QAction *action_Aoto_recalculate;
    QAction *action_About;
    QAction *actionAbout_Qt;
    QWidget *centralWidget;
    QMenuBar *menuBar;
    QMenu *menu_File;
    QMenu *menu_Edit;
    QMenu *menu_Select;
    QMenu *menu_Tools;
    QMenu *menu_Options;
    QMenu *menu_Hellp;
    QToolBar *mainToolBar;
    QStatusBar *statusBar;

    void setupUi(QMainWindow *MainWindow)
    {
        if (MainWindow->objectName().isEmpty())
            MainWindow->setObjectName(QStringLiteral("MainWindow"));
        MainWindow->resize(897, 500);
        action_New = new QAction(MainWindow);
        action_New->setObjectName(QStringLiteral("action_New"));
        action_Open = new QAction(MainWindow);
        action_Open->setObjectName(QStringLiteral("action_Open"));
        action_Save = new QAction(MainWindow);
        action_Save->setObjectName(QStringLiteral("action_Save"));
        actionSave_As = new QAction(MainWindow);
        actionSave_As->setObjectName(QStringLiteral("actionSave_As"));
        actionE_xit = new QAction(MainWindow);
        actionE_xit->setObjectName(QStringLiteral("actionE_xit"));
        actionCu_t = new QAction(MainWindow);
        actionCu_t->setObjectName(QStringLiteral("actionCu_t"));
        action_Copy = new QAction(MainWindow);
        action_Copy->setObjectName(QStringLiteral("action_Copy"));
        action_Paste = new QAction(MainWindow);
        action_Paste->setObjectName(QStringLiteral("action_Paste"));
        action_Delete = new QAction(MainWindow);
        action_Delete->setObjectName(QStringLiteral("action_Delete"));
        action_Find = new QAction(MainWindow);
        action_Find->setObjectName(QStringLiteral("action_Find"));
        action_Go_to_Cell = new QAction(MainWindow);
        action_Go_to_Cell->setObjectName(QStringLiteral("action_Go_to_Cell"));
        action_Row = new QAction(MainWindow);
        action_Row->setObjectName(QStringLiteral("action_Row"));
        action_Column = new QAction(MainWindow);
        action_Column->setObjectName(QStringLiteral("action_Column"));
        action_All = new QAction(MainWindow);
        action_All->setObjectName(QStringLiteral("action_All"));
        action_Recalculate = new QAction(MainWindow);
        action_Recalculate->setObjectName(QStringLiteral("action_Recalculate"));
        action_Sort = new QAction(MainWindow);
        action_Sort->setObjectName(QStringLiteral("action_Sort"));
        action_Show_Grid = new QAction(MainWindow);
        action_Show_Grid->setObjectName(QStringLiteral("action_Show_Grid"));
        action_Show_Grid->setCheckable(true);
        action_Aoto_recalculate = new QAction(MainWindow);
        action_Aoto_recalculate->setObjectName(QStringLiteral("action_Aoto_recalculate"));
        action_Aoto_recalculate->setCheckable(true);
        action_About = new QAction(MainWindow);
        action_About->setObjectName(QStringLiteral("action_About"));
        actionAbout_Qt = new QAction(MainWindow);
        actionAbout_Qt->setObjectName(QStringLiteral("actionAbout_Qt"));
        centralWidget = new QWidget(MainWindow);
        centralWidget->setObjectName(QStringLiteral("centralWidget"));
        MainWindow->setCentralWidget(centralWidget);
        menuBar = new QMenuBar(MainWindow);
        menuBar->setObjectName(QStringLiteral("menuBar"));
        menuBar->setGeometry(QRect(0, 0, 897, 21));
        menu_File = new QMenu(menuBar);
        menu_File->setObjectName(QStringLiteral("menu_File"));
        menu_Edit = new QMenu(menuBar);
        menu_Edit->setObjectName(QStringLiteral("menu_Edit"));
        menu_Select = new QMenu(menu_Edit);
        menu_Select->setObjectName(QStringLiteral("menu_Select"));
        menu_Tools = new QMenu(menuBar);
        menu_Tools->setObjectName(QStringLiteral("menu_Tools"));
        menu_Options = new QMenu(menuBar);
        menu_Options->setObjectName(QStringLiteral("menu_Options"));
        menu_Hellp = new QMenu(menuBar);
        menu_Hellp->setObjectName(QStringLiteral("menu_Hellp"));
        MainWindow->setMenuBar(menuBar);
        mainToolBar = new QToolBar(MainWindow);
        mainToolBar->setObjectName(QStringLiteral("mainToolBar"));
        MainWindow->addToolBar(Qt::TopToolBarArea, mainToolBar);
        statusBar = new QStatusBar(MainWindow);
        statusBar->setObjectName(QStringLiteral("statusBar"));
        MainWindow->setStatusBar(statusBar);

        menuBar->addAction(menu_File->menuAction());
        menuBar->addAction(menu_Edit->menuAction());
        menuBar->addAction(menu_Tools->menuAction());
        menuBar->addAction(menu_Options->menuAction());
        menuBar->addAction(menu_Hellp->menuAction());
        menu_File->addAction(action_New);
        menu_File->addAction(action_Open);
        menu_File->addAction(action_Save);
        menu_File->addAction(actionSave_As);
        menu_File->addSeparator();
        menu_File->addSeparator();
        menu_File->addAction(actionE_xit);
        menu_Edit->addAction(actionCu_t);
        menu_Edit->addAction(action_Copy);
        menu_Edit->addAction(action_Paste);
        menu_Edit->addAction(action_Delete);
        menu_Edit->addAction(menu_Select->menuAction());
        menu_Edit->addSeparator();
        menu_Edit->addAction(action_Find);
        menu_Edit->addAction(action_Go_to_Cell);
        menu_Select->addAction(action_Row);
        menu_Select->addAction(action_Column);
        menu_Select->addAction(action_All);
        menu_Tools->addAction(action_Recalculate);
        menu_Tools->addAction(action_Sort);
        menu_Options->addAction(action_Show_Grid);
        menu_Options->addAction(action_Aoto_recalculate);
        menu_Hellp->addAction(action_About);
        menu_Hellp->addAction(actionAbout_Qt);

        retranslateUi(MainWindow);

        QMetaObject::connectSlotsByName(MainWindow);
    } // setupUi

    void retranslateUi(QMainWindow *MainWindow)
    {
        MainWindow->setWindowTitle(QApplication::translate("MainWindow", "MainWindow", 0));
        action_New->setText(QApplication::translate("MainWindow", "&New", 0));
        action_New->setShortcut(QApplication::translate("MainWindow", "Ctrl+N", 0));
        action_Open->setText(QApplication::translate("MainWindow", "&Open...", 0));
        action_Open->setShortcut(QApplication::translate("MainWindow", "Ctrl+O", 0));
        action_Save->setText(QApplication::translate("MainWindow", "&Save", 0));
        action_Save->setShortcut(QApplication::translate("MainWindow", "Ctrl+S", 0));
        actionSave_As->setText(QApplication::translate("MainWindow", "Save &As...", 0));
        actionE_xit->setText(QApplication::translate("MainWindow", "E&xit", 0));
        actionE_xit->setShortcut(QApplication::translate("MainWindow", "Ctrl+Q", 0));
        actionCu_t->setText(QApplication::translate("MainWindow", "Cu&t", 0));
        actionCu_t->setShortcut(QApplication::translate("MainWindow", "Ctrl+X", 0));
        action_Copy->setText(QApplication::translate("MainWindow", "&Copy", 0));
        action_Copy->setShortcut(QApplication::translate("MainWindow", "Ctrl+C", 0));
        action_Paste->setText(QApplication::translate("MainWindow", "&Paste", 0));
        action_Paste->setShortcut(QApplication::translate("MainWindow", "Ctrl+V", 0));
        action_Delete->setText(QApplication::translate("MainWindow", "&Delete", 0));
        action_Delete->setShortcut(QApplication::translate("MainWindow", "Del", 0));
        action_Find->setText(QApplication::translate("MainWindow", "&Find...", 0));
        action_Find->setShortcut(QApplication::translate("MainWindow", "Ctrl+F", 0));
        action_Go_to_Cell->setText(QApplication::translate("MainWindow", "&Go to Cell...", 0));
        action_Go_to_Cell->setShortcut(QApplication::translate("MainWindow", "F5", 0));
        action_Row->setText(QApplication::translate("MainWindow", "&Row", 0));
        action_Column->setText(QApplication::translate("MainWindow", "&Column", 0));
        action_All->setText(QApplication::translate("MainWindow", "&All", 0));
        action_All->setShortcut(QApplication::translate("MainWindow", "Ctrl+A", 0));
        action_Recalculate->setText(QApplication::translate("MainWindow", "&Recalculate", 0));
        action_Recalculate->setShortcut(QApplication::translate("MainWindow", "F9", 0));
        action_Sort->setText(QApplication::translate("MainWindow", "&Sort...", 0));
        action_Show_Grid->setText(QApplication::translate("MainWindow", "&Show Grid", 0));
        action_Aoto_recalculate->setText(QApplication::translate("MainWindow", "&Aoto-recalculate", 0));
        action_About->setText(QApplication::translate("MainWindow", "&About", 0));
        actionAbout_Qt->setText(QApplication::translate("MainWindow", "About &Qt", 0));
        menu_File->setTitle(QApplication::translate("MainWindow", "&File", 0));
        menu_Edit->setTitle(QApplication::translate("MainWindow", "&Edit", 0));
        menu_Select->setTitle(QApplication::translate("MainWindow", "&Select", 0));
        menu_Tools->setTitle(QApplication::translate("MainWindow", "&Tools", 0));
        menu_Options->setTitle(QApplication::translate("MainWindow", "&Options", 0));
        menu_Hellp->setTitle(QApplication::translate("MainWindow", "&Help", 0));
    } // retranslateUi

};

namespace Ui {
    class MainWindow: public Ui_MainWindow {};
} // namespace Ui

QT_END_NAMESPACE

#endif // UI_MAINWINDOW_H
