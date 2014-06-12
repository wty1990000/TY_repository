#include "extentiondialog.h"
#include "ui_extentiondialog.h"

ExtentionDialog::ExtentionDialog(QWidget *parent) :
    QDialog(parent),
    ui(new Ui::ExtentionDialog)
{
    ui->setupUi(this);
}

ExtentionDialog::~ExtentionDialog()
{
    delete ui;
}
