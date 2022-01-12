#ifndef RENDEREDPLAYER_H
#define RENDEREDPLAYER_H

#include "QThread"
#include "QObject"
#include "qaudiooutput.h"

class renderedplayer : public QThread
{
    Q_OBJECT
public:
    explicit renderedplayer(QObject *parent = 0);
    void run();
    QString path = 0;
    int duration;
    QAudioOutput * audio;
    static QAudioFormat getWaveFormat(QString);
    void pause();
    void resume();
    int pos=0;

signals:
    void progressPercentUpdated(int i);
    void progressLabelUpdated(QString text);
    void nextStep();
    void loadingFailed();
    void lockWindow(bool lock);
    void update(int);

public slots:
    void hold();
    void handleStateChanged(QAudio::State);
    //void cleanUpandClose();


};

#endif // RENDEREDPLAYER_H
