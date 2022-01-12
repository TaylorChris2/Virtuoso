#include "renderedplayer.h"
#include "QtCore"
#include "QDateTime"
#include "QObject"
#include "qsound.h"
#include "qaudiooutput.h"
#include "qfile.h"
#include "qtimer.h"
#include "mainwindow.h"
#include "noteplayer.h"

renderedplayer::renderedplayer(QObject *parent) :
    QThread(parent)
{
    //threadtimer = new QTimer(this);
    //QObject::moveToThread(this);
    //connect(threadtimer, SIGNAL(timeout()), this, SLOT(hold()));
}


void printasdf(QString string, bool nextLine = true)
{
    QTextStream out(stdout);
    out << string;
    if (nextLine)
    {
        out << Qt::endl;
    }

}

void renderedplayer::hold(){
    pos +=100;
    emit update(pos);
}

void renderedplayer::handleStateChanged(QAudio::State newState)
{
    switch (newState) {
        case QAudio::IdleState:
            // Finished playing (no more data)
            printasdf("oofing ");
            audio->stop();
            exit();
    }
}

void renderedplayer::pause(){
    audio->suspend();
}

void renderedplayer::resume(){
    audio->resume();
}


void renderedplayer::run()
{
    printasdf("started");
    QAudioFormat format = noteplayer::getWaveFormat(path);
    audio = new QAudioOutput(format);
    printasdf("dos");
    QFile file;
    file.setFileName(path);
    file.open(QIODevice::ReadOnly);
    audio->start(&file);
    audio->setNotifyInterval(100);
    connect(audio, SIGNAL(notify()), this, SLOT(hold()));
    connect(audio, SIGNAL(stateChanged(QAudio::State)), this, SLOT(handleStateChanged(QAudio::State)));
    //QThread::msleep(2000);
    exec();


    printasdf("tres");
    quit();

}
