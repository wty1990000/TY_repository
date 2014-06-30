#include <QDebug>
#include <QtCore/QObject>

class Reader : public QObject
{
	Q_OBJECT
public:
	Reader(){}

	void receiveNewspaper(const QString &name)
	{
		qDebug()<<"Receives Nespaper:"<<name;
	}
};