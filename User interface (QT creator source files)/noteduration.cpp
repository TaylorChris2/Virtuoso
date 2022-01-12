#include "noteduration.h"
#include "noteplayer.h"
#include "QtCore"
#include "QDateTime"
#include "QObject"
#include "qsound.h"
#include "qaudiooutput.h"
#include "qfile.h"
#include "qtimer.h"
#include "mainwindow.h"

noteduration::noteduration(QObject *parent) :
    QThread(parent)
{
    //threadtimer = new QTimer(this);
    //QObject::moveToThread(this);
    //connect(threadtimer, SIGNAL(timeout()), this, SLOT(hold()));
}

void printblu(QString string, bool nextLine = true)
{
    QTextStream out(stdout);
    out << string;
    if (nextLine)
    {
        out << Qt::endl;
    }

}

void noteduration::run()
{
    printblu("started");
    for (int i=0;i<100;i++){
        noteplayer *bluf = new noteplayer;
        bluf->path = path;
        //bluf->duration = 500;
        bluf->start();
        QThread::msleep(30);
    }

}
