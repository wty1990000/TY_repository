#ifndef GOTODIALOG_H
#define GOTODIALOG_H

#include <QDialog>
#include "ui_gotodialog.h"

namespace Ui {
class gotoDialog;
}

class gotoDialog : public QDialog
{
    Q_OBJECT

public:
    explicit gotoDialog(QWidget *parent = 0);
    ~gotoDialog();

private:
    Ui::gotoDialog *ui;
private slots:
    void on_lineEdit_textChanged();
};

#endif // GOTODIALOG_H
