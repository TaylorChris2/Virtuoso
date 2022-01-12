#ifndef SOUNDPLAYER_H
#define SOUNDPLAYER_H

#include <QWidget>
#include <QAudioOutput>
#include <QFile>
#include <QTimer>

class SoundPlayer : public QWidget
{
    Q_OBJECT
public:
    explicit SoundPlayer(QAudioOutput* audio, QWidget *parent = 0);
    ~SoundPlayer();
    QAudioFormat getWaveFormat(QString fileName);
    QByteArray reverseByteArray(QByteArray reverseMe);
    QByteArray reverseHexByteArray(QByteArray reverseMe);
    int hexToUDec(QByteArray hexNum, bool isReversed = false);
    void setFile(QString fileName);

signals:
    void soundDone();

public slots:
    void play();
    void stop();
    void pause();
    void resume();


private slots:
    void finishedPlaying(QAudio::State state);

private:
    QFile inputFile;
    QAudioOutput *audio;
    QTimer *timerA;
    qint64 startPos;
    bool started=false;

};

#endif // SOUNDPLAYER_H
