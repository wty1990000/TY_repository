#ifndef FINDDIALOG_H
#define FINDDIALOG_H

#include <QDialog>

class QCheckBox;
class QLabel;
class QLineEdit;
class QPushButton;

namespace Ui {
class FindDialog;
}

class FindDialog : public QDialog
{
    Q_OBJECT

public:
    explicit FindDialog(QWidget *parent = 0);   //Avoid implicit conversion
    ~FindDialog();

signals:
    void findNext(const QString& str, Qt::CaseSensitivity cs);
    void findPrevious(const QString& str, Qt::CaseSensitivity cs);

private slots:
    void findClicked();
    void enableFindButton(const QString& text);

private:
    Ui::FindDialog *ui;
    QLabel *label;
    QLineEdit* lineEdit;
    QCheckBox* caseCheckBox;
    QCheckBox* backwardCheckBox;
    QPushButton* findButton;
    QPushButton* closeButton;
};

#endif // FINDDIALOG_H
