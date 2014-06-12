#ifndef EXTENTIONDIALOG_H
#define EXTENTIONDIALOG_H

#include <QDialog>

namespace Ui {
class ExtentionDialog;
}

class ExtentionDialog : public QDialog
{
    Q_OBJECT

public:
    explicit ExtentionDialog(QWidget *parent = 0);
    ~ExtentionDialog();

private:
    Ui::ExtentionDialog *ui;
};

#endif // EXTENTIONDIALOG_H
