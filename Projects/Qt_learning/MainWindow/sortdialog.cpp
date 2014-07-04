#include "sortdialog.h"
#include "ui_sortdialog.h"

sortDialog::sortDialog(QWidget *parent) :
    QDialog(parent),
    ui(new Ui::sortDialog)
{
    ui->setupUi(this);
	ui->secondarygroupBox->hide();
	ui->tertiarygroupBox->hide();
	layout()->setSizeConstraint(QLayout::SetFixedSize);	//The form size cannot be changed.

	setColumnRange('A','Z');
}

void sortDialog::setColumnRange(QChar first, QChar last)
{
	ui->primaryColumnCombo->clear();
	ui->secondaryColumnCombo->clear();
	ui->tertiaryColumnCombo->clear();

	ui->secondaryColumnCombo->addItem(tr("None"));
	ui->tertiaryColumnCombo->addItem(tr("None"));

	ui->primaryColumnCombo->setMinimumSize(ui->secondaryColumnCombo->sizeHint());
	QChar ch = first;
	while(ch <= last){
		ui->primaryColumnCombo->addItem(QString(ch));
		ui->secondaryColumnCombo->addItem(QString(ch));
		ui->tertiaryColumnCombo->addItem(QString(ch));
		ch = ch.unicode() + 1;
	}
}

sortDialog::~sortDialog()
{
    delete ui;
}
