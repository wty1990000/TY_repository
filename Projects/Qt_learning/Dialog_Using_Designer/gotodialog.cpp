#include "gotodialog.h"
#include "ui_gotodialog.h"

gotoDialog::gotoDialog(QWidget *parent) :
    QDialog(parent),
    ui(new Ui::gotoDialog)
{
    ui->setupUi(this);

    QRegExp regExp("[A-Za-z][1-9][0-9]{0,2}");
    ui->lineEdit->setValidator(new QRegExpValidator(regExp,this));

    connect(ui->okButton,SIGNAL(clicked()),this,SLOT(accept()));
    connect(ui->cancelButton,SIGNAL(clicked()),this,SLOT(reject()));
}

void gotoDialog::on_lineEdit_textChanged()
{
    ui->okButton->setEnabled(ui->lineEdit->hasAcceptableInput());
}

gotoDialog::~gotoDialog()
{
    delete ui;
}
