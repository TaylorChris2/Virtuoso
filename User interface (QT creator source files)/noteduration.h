#ifndef NOTEDURATION_H
#define NOTEDURATION_H


#include "QThread"
#include "QObject"
#include "qaudiooutput.h"

class noteduration : public QThread
{
    Q_OBJECT
public:
    explicit noteduration(QObject *parent = 0);
    void run();
    QString path = 0;
    int duration;

signals:
    void progressPercentUpdated(int i);
    void progressLabelUpdated(QString text);
    void nextStep();
    void loadingFailed();
    void lockWindow(bool lock);

public slots:

    //void cleanUpandClose();


};

#endif // NOTEDURATION_H
