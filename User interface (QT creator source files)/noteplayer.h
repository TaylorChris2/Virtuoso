#ifndef NOTEPLAYER_H
#define NOTEPLAYER_H


#include "QThread"
#include "QObject"
#include "qaudiooutput.h"
#include "qmediaplayer.h"

class noteplayer : public QThread
{
    Q_OBJECT
public:
    explicit noteplayer(QObject *parent = 0);
    void run();
    QString path;
    int duration;
    static QAudioFormat getWaveFormat(QString);
    int volume;
    QMediaPlayer *player;

signals:
    void progressPercentUpdated(int i);
    void progressLabelUpdated(QString text);
    void nextStep();
    void loadingFailed();
    void lockWindow(bool lock);

public slots:
    void hold();
    void handleStateChanged(QMediaPlayer::MediaStatus);
    //void cleanUpandClose();


};

#endif // NOTEPLAYER_H
