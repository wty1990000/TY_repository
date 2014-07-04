#-------------------------------------------------
#
# Project created by QtCreator 2014-06-16T22:19:48
#
#-------------------------------------------------

QT       += core gui

greaterThan(QT_MAJOR_VERSION, 4): QT += widgets

TARGET = MainWindow
TEMPLATE = app


SOURCES += main.cpp\
        mainwindow.cpp \
    finddialog.cpp \
    gotocelldialog.cpp \
    sortdialog.cpp

HEADERS  += mainwindow.h \
    finddialog.h \
    gotocelldialog.h \
    sortdialog.h

FORMS    += mainwindow.ui \
    finddialog.ui \
    gotocelldialog.ui \
    sortdialog.ui

RESOURCES += \
    rc_MainWindow.qrc
