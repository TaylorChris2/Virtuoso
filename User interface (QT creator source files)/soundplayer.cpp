#include "soundplayer.h"
#include "math.h"
#include <QDebug>
#include "qdatetime.h"

SoundPlayer::SoundPlayer(QAudioOutput *audio, QWidget *parent) :
    QWidget(parent)
{
    this->startPos = 0;






    this->audio = audio;

    //audio

    /*this->timerA = new QTimer(this);
    this->timerA->setSingleShot(true);
    this->timerA->setInterval(0);
    connect(this->timerA, SIGNAL(timeout()), SLOT(play()));*/
    //timerA->start();
}
void SoundPlayer::setFile(QString fileName){
    this->inputFile.setFileName(fileName);
    this->inputFile.open(QIODevice::ReadOnly);
    //this->inputFile.seek(this->startPos);

    /*for(int i = 0; i < this->inputFile.size(); ++i) {
        QByteArray ba = this->inputFile.peek(4);
        //qDebug() << ba;
        if (ba == "data") {
            qDebug() << "FOUND THE DATA MARKER";
            this->inputFile.read(8);
            break;
        } else {
            this->inputFile.read(1);
        }
    }*/
}

SoundPlayer::~SoundPlayer()
{
    if (audio != 0) {
        audio->stop();
        delete audio;
    }
    inputFile.close();
}

void SoundPlayer::play()
{
    int start_time = (int)QDateTime::currentMSecsSinceEpoch();
    /*if (this->timerA != 0)
    delete this->timerA;
    this->timerA = 0;
    this->started = true;*/


    audio->start(&inputFile);
    qDebug()<<((int)QDateTime::currentMSecsSinceEpoch()-start_time);
    qDebug("abjdi");

    //QAudio::Error err = audio->error();
    /*if (err == 0) {
        qDebug("No Error");
    } else {
        if (err == 1)
            qDebug("Error opening audio device");
        else if (err == 2)
            qDebug("Error occurred during read/write of audio device");
        else if (err == 3)
            qDebug("Underrun error");
        else if (err == 4)
            qDebug("Fatal Error");
        emit soundDone();
    }*/
}

void SoundPlayer::stop()
{
    int start_time = (int)QDateTime::currentMSecsSinceEpoch();
    audio->stop();
    qDebug()<<((int)QDateTime::currentMSecsSinceEpoch()-start_time);
    qDebug("abjdi2");
}

void SoundPlayer::pause()
{
    audio->suspend();
}

void SoundPlayer::resume()
{
    audio->resume();
}

void SoundPlayer::finishedPlaying(QAudio::State state)
{
    if (state == QAudio::IdleState) {
        audio->stop();
        inputFile.close();
        delete audio;
        audio = 0;
        emit soundDone();
    }
}



