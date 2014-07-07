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
    QAction *actionCut;
    QAction *actionCopy;
    QAction *actionPaste;
    QAction *actionDelete;
    QAction *actionRow;
    QAction *actionColumn;
    QAction *actionAll;
    QAction *action_Find;
    QAction *actionGotoCell;
    QAction *actionRecalculate;
    QAction *actionSort;
    QAction *actionShowGrid;
    QAction *actionAutorecalculate;
    QAction *actionAbout;
    QAction *actionAboutQt;
    QWidget *centralWidget;
    QMenuBar *menuBar;
    QMenu *menu_File;
    QMenu *menu_Edit;
    QMenu *menu_Select;
    QMenu *menuType_Here;
    QMenu *menu_Tools;
    QMenu *menu_Options;
    QMenu *menu_Help;
    QToolBar *fileToolBar;
    QStatusBar *statusBar;
    QToolBar *editToolBar;

    void setupUi(QMainWindow *MainWindow)
    {
        if (MainWindow->objectName().isEmpty())
            MainWindow->setObjectName(QStringLiteral("MainWindow"));
        MainWindow->resize(771, 523);
        action_New = new QAction(MainWindow);
        action_New->setObjectName(QStringLiteral("action_New"));
        QIcon icon;
        icon.addFile(QStringLiteral(":/myresourses/new.png"), QSize(), QIcon::Normal, QIcon::Off);
        action_New->setIcon(icon);
        action_Open = new QAction(MainWindow);
        action_Open->setObjectName(QStringLiteral("action_Open"));
        QIcon icon1;
        icon1.addFile(QStringLiteral(":/myresourses/open.png"), QSize(), QIcon::Normal, QIcon::Off);
        action_Open->setIcon(icon1);
        action_Save = new QAction(MainWindow);
        action_Save->setObjectName(QStringLiteral("action_Save"));
        QIcon icon2;
        icon2.addFile(QStringLiteral(":/myresourses/save.png"), QSize(), QIcon::Normal, QIcon::Off);
        action_Save->setIcon(icon2);
        actionSave_As = new QAction(MainWindow);
        actionSave_As->setObjectName(QStringLiteral("actionSave_As"));
        actionE_xit = new QAction(MainWindow);
        actionE_xit->setObjectName(QStringLiteral("actionE_xit"));
        actionCut = new QAction(MainWindow);
        actionCut->setObjectName(QStringLiteral("actionCut"));
        QIcon icon3;
        icon3.addFile(QStringLiteral(":/myresourses/cut.png"), QSize(), QIcon::Normal, QIcon::Off);
        actionCut->setIcon(icon3);
        actionCopy = new QAction(MainWindow);
        actionCopy->setObjectName(QStringLiteral("actionCopy"));
        QIcon icon4;
        icon4.addFile(QStringLiteral(":/myresourses/copy.png"), QSize(), QIcon::Normal, QIcon::Off);
        actionCopy->setIcon(icon4);
        actionPaste = new QAction(MainWindow);
        actionPaste->setObjectName(QStringLiteral("actionPaste"));
        QIcon icon5;
        icon5.addFile(QStringLiteral(":/myresourses/paste.png"), QSize(), QIcon::Normal, QIcon::Off);
        actionPaste->setIcon(icon5);
        actionDelete = new QAction(MainWindow);
        actionDelete->setObjectName(QStringLiteral("actionDelete"));
        actionRow = new QAction(MainWindow);
        actionRow->setObjectName(QStringLiteral("actionRow"));
        actionColumn = new QAction(MainWindow);
        actionColumn->setObjectName(QStringLiteral("actionColumn"));
        actionAll = new QAction(MainWindow);
        actionAll->setObjectName(QStringLiteral("actionAll"));
        action_Find = new QAction(MainWindow);
        action_Find->setObjectName(QStringLiteral("action_Find"));
        QIcon icon6;
        icon6.addFile(QStringLiteral(":/myresourses/find.png"), QSize(), QIcon::Normal, QIcon::Off);
        action_Find->setIcon(icon6);
        actionGotoCell = new QAction(MainWindow);
        actionGotoCell->setObjectName(QStringLiteral("actionGotoCell"));
        QIcon icon7;
        icon7.addFile(QStringLiteral(":/myresourses/next.png"), QSize(), QIcon::Normal, QIcon::Off);
        actionGotoCell->setIcon(icon7);
        actionRecalculate = new QAction(MainWindow);
        actionRecalculate->setObjectName(QStringLiteral("actionRecalculate"));
        actionSort = new QAction(MainWindow);
        actionSort->setObjectName(QStringLiteral("actionSort"));
        actionShowGrid = new QAction(MainWindow);
        actionShowGrid->setObjectName(QStringLiteral("actionShowGrid"));
        actionShowGrid->setCheckable(true);
        actionAutorecalculate = new QAction(MainWindow);
        actionAutorecalculate->setObjectName(QStringLiteral("actionAutorecalculate"));
        actionAutorecalculate->setCheckable(true);
        actionAbout = new QAction(MainWindow);
        actionAbout->setObjectName(QStringLiteral("actionAbout"));
        actionAboutQt = new QAction(MainWindow);
        actionAboutQt->setObjectName(QStringLiteral("actionAboutQt"));
        centralWidget = new QWidget(MainWindow);
        centralWidget->setObjectName(QStringLiteral("centralWidget"));
        MainWindow->setCentralWidget(centralWidget);
        menuBar = new QMenuBar(MainWindow);
        menuBar->setObjectName(QStringLiteral("menuBar"));
        menuBar->setGeometry(QRect(0, 0, 771, 21));
        menu_File = new QMenu(menuBar);
        menu_File->setObjectName(QStringLiteral("menu_File"));
        menu_Edit = new QMenu(menuBar);
        menu_Edit->setObjectName(QStringLiteral("menu_Edit"));
        menu_Select = new QMenu(menu_Edit);
        menu_Select->setObjectName(QStringLiteral("menu_Select"));
        menuType_Here = new QMenu(menu_Edit);
        menuType_Here->setObjectName(QStringLiteral("menuType_Here"));
        menu_Tools = new QMenu(menuBar);
        menu_Tools->setObjectName(QStringLiteral("menu_Tools"));
        menu_Options = new QMenu(menuBar);
        menu_Options->setObjectName(QStringLiteral("menu_Options"));
        menu_Help = new QMenu(menuBar);
        menu_Help->setObjectName(QStringLiteral("menu_Help"));
        MainWindow->setMenuBar(menuBar);
        fileToolBar = new QToolBar(MainWindow);
        fileToolBar->setObjectName(QStringLiteral("fileToolBar"));
        MainWindow->addToolBar(Qt::TopToolBarArea, fileToolBar);
        statusBar = new QStatusBar(MainWindow);
        statusBar->setObjectName(QStringLiteral("statusBar"));
        MainWindow->setStatusBar(statusBar);
        editToolBar = new QToolBar(MainWindow);
        editToolBar->setObjectName(QStringLiteral("editToolBar"));
        MainWindow->addToolBar(Qt::TopToolBarArea, editToolBar);

        menuBar->addAction(menu_File->menuAction());
        menuBar->addAction(menu_Edit->menuAction());
        menuBar->addAction(menu_Tools->menuAction());
        menuBar->addAction(menu_Options->menuAction());
        menuBar->addAction(menu_Help->menuAction());
        menu_File->addAction(action_New);
        menu_File->addAction(action_Open);
        menu_File->addAction(action_Save);
        menu_File->addAction(actionSave_As);
        menu_File->addSeparator();
        menu_File->addSeparator();
        menu_File->addAction(actionE_xit);
        menu_Edit->addAction(actionCut);
        menu_Edit->addAction(actionCopy);
        menu_Edit->addAction(actionPaste);
        menu_Edit->addAction(actionDelete);
        menu_Edit->addAction(menu_Select->menuAction());
        menu_Edit->addSeparator();
        menu_Edit->addAction(action_Find);
        menu_Edit->addAction(actionGotoCell);
        menu_Edit->addAction(menuType_Here->menuAction());
        menu_Select->addAction(actionRow);
        menu_Select->addAction(actionColumn);
        menu_Select->addAction(actionAll);
        menu_Tools->addAction(actionRecalculate);
        menu_Tools->addAction(actionSort);
        menu_Options->addAction(actionShowGrid);
        menu_Options->addAction(actionAutorecalculate);
        menu_Help->addAction(actionAbout);
        menu_Help->addAction(actionAboutQt);
        fileToolBar->addAction(action_New);
        fileToolBar->addAction(action_Open);
        fileToolBar->addAction(action_Save);
        editToolBar->addAction(actionCut);
        editToolBar->addAction(actionCopy);
        editToolBar->addAction(actionPaste);

        retranslateUi(MainWindow);

        QMetaObject::connectSlotsByName(MainWindow);
    } // setupUi

    void retranslateUi(QMainWindow *MainWindow)
    {
        MainWindow->setWindowTitle(QApplication::translate("MainWindow", "MainWindow", 0));
        action_New->setText(QApplication::translate("MainWindow", "&New", 0));
#ifndef QT_NO_TOOLTIP
        action_New->setToolTip(QApplication::translate("MainWindow", "<html><head/><body><p><span style=\" font-weight:600;\">Create a new spreadsheet file</span></p></body></html>", 0));
#endif // QT_NO_TOOLTIP
        action_New->setShortcut(QApplication::translate("MainWindow", "Ctrl+N", 0));
        action_Open->setText(QApplication::translate("MainWindow", "&Open...", 0));
#ifndef QT_NO_TOOLTIP
        action_Open->setToolTip(QApplication::translate("MainWindow", "<html><head/><body><p>Open an existing spreadsheet file</p></body></html>", 0));
#endif // QT_NO_TOOLTIP
        action_Open->setShortcut(QApplication::translate("MainWindow", "Ctrl+O", 0));
        action_Save->setText(QApplication::translate("MainWindow", "&Save", 0));
        action_Save->setShortcut(QApplication::translate("MainWindow", "Ctrl+S", 0));
        actionSave_As->setText(QApplication::translate("MainWindow", "Save &As...", 0));
        actionE_xit->setText(QApplication::translate("MainWindow", "E&xit", 0));
        actionE_xit->setShortcut(QApplication::translate("MainWindow", "Ctrl+Q", 0));
        actionCut->setText(QApplication::translate("MainWindow", "Cu&t", 0));
        actionCut->setShortcut(QApplication::translate("MainWindow", "Ctrl+X", 0));
        actionCopy->setText(QApplication::translate("MainWindow", "&Copy", 0));
        actionCopy->setShortcut(QApplication::translate("MainWindow", "Ctrl+C", 0));
        actionPaste->setText(QApplication::translate("MainWindow", "&Paste", 0));
        actionPaste->setShortcut(QApplication::translate("MainWindow", "Ctrl+V", 0));
        actionDelete->setText(QApplication::translate("MainWindow", "&Delete", 0));
        actionDelete->setShortcut(QApplication::translate("MainWindow", "Del", 0));
        actionRow->setText(QApplication::translate("MainWindow", "&Row", 0));
        actionColumn->setText(QApplication::translate("MainWindow", "&Column", 0));
        actionAll->setText(QApplication::translate("MainWindow", "&All", 0));
        actionAll->setShortcut(QApplication::translate("MainWindow", "Ctrl+A", 0));
        action_Find->setText(QApplication::translate("MainWindow", "&Find...", 0));
        action_Find->setShortcut(QApplication::translate("MainWindow", "Ctrl+F", 0));
        actionGotoCell->setText(QApplication::translate("MainWindow", "&Go to Cell...", 0));
        actionGotoCell->setShortcut(QApplication::translate("MainWindow", "F5", 0));
        actionRecalculate->setText(QApplication::translate("MainWindow", "&Recalculate", 0));
        actionRecalculate->setShortcut(QApplication::translate("MainWindow", "F9", 0));
        actionSort->setText(QApplication::translate("MainWindow", "&Sort...", 0));
        actionShowGrid->setText(QApplication::translate("MainWindow", "&Show Grid", 0));
        actionAutorecalculate->setText(QApplication::translate("MainWindow", "&Auto-recalculate", 0));
        actionAbout->setText(QApplication::translate("MainWindow", "&About", 0));
        actionAboutQt->setText(QApplication::translate("MainWindow", "About &Qt", 0));
        menu_File->setTitle(QApplication::translate("MainWindow", "&File", 0));
        menu_Edit->setTitle(QApplication::translate("MainWindow", "&Edit", 0));
        menu_Select->setTitle(QApplication::translate("MainWindow", "&Select", 0));
        menuType_Here->setTitle(QApplication::translate("MainWindow", "Type Here", 0));
        menu_Tools->setTitle(QApplication::translate("MainWindow", "&Tools", 0));
        menu_Options->setTitle(QApplication::translate("MainWindow", "&Options", 0));
        menu_Help->setTitle(QApplication::translate("MainWindow", "&Help", 0));
        editToolBar->setWindowTitle(QApplication::translate("MainWindow", "toolBar", 0));
    } // retranslateUi

};

namespace Ui {
    class MainWindow: public Ui_MainWindow {};
} // namespace Ui

QT_END_NAMESPACE

#endif // UI_MAINWINDOW_H
