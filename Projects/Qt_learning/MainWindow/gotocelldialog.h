#ifndef GOTOCELLDIALOG_H
#define GOTOCELLDIALOG_H

#include <QDialog>

namespace Ui {
class gotocellDialog;
}

class gotocellDialog : public QDialog
{
    Q_OBJECT

public:
    explicit gotocellDialog(QWidget *parent = 0);
    ~gotocellDialog();
private slots:
    void on_lineEdit_textChanged();     //automatically done by setupUI
private:
    Ui::gotocellDialog *ui;
};

#endif // GOTOCELLDIALOG_H
