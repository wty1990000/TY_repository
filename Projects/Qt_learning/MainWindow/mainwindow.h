#ifndef MAINWINDOW_H
#define MAINWINDOW_H

#include <QMainWindow>

namespace Ui {
class MainWindow;
}

class MainWindow : public QMainWindow
{
    Q_OBJECT

public:
    explicit MainWindow(QWidget *parent = 0);
    ~MainWindow();
protected:
	void closeEvent(QCloseEvent *event);

//Slots
private slots:
	void newFile();
	void open();
	bool save();
	bool saveAs();
	void find();
	void goToCell();
	void sort();
	void about();
	void openRecentFile();
	void updateStatusBar();
	void spreadsheetModified();
private:
	void createActions();
	void createMenus();
	void createContextMenu();
	void createToolBars();
	void readSettings();
	void writeSettings();
	bool okToContinue();
	bool loadFile(const QString &fileName);
	bool saveFile(const QString &fileName);
	void setCurrentFile(const QString &fileName);
	void updateRecentFileActions();
	QString strippedName(const QString &fullFileName);

private:
    Ui::MainWindow *ui;
	enum{MaxRecentFiles = 5};
	QAction *recentFileActions[MaxRecentFiles];
};

#endif // MAINWINDOW_H
