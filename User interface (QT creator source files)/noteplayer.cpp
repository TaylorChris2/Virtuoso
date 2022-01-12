#include "noteplayer.h"
#include "QtCore"
#include "QDateTime"
#include "QObject"
#include "qsound.h"
#include "qaudiooutput.h"
#include "qfile.h"
#include "qtimer.h"
#include "mainwindow.h"

noteplayer::noteplayer(QObject *parent) :
    QThread(parent)
{
    //threadtimer = new QTimer(this);
    //QObject::moveToThread(this);
    //connect(threadtimer, SIGNAL(timeout()), this, SLOT(hold()));
}


QByteArray reverseByteArray(QByteArray reverseMe)
{
    QByteArray out(reverseMe.length(), 0);
    for(int i = 0; i < reverseMe.length(); ++i) {
        out[reverseMe.length()-i-1] = reverseMe.at(i);
    }
    return out;
}

QByteArray reverseHexByteArray(QByteArray reverseMe)
{
    QByteArray out(reverseMe.length(), 0);
    for(int i = 0; i < reverseMe.length()-1; i += 2) {
        out[reverseMe.length()-i-1] = reverseMe.at(i+1);
        out[reverseMe.length()-i-2] = reverseMe.at(i);
    }
    return out;
}

int hexToUDec(QByteArray hexNum, bool isReversed)
{
    QByteArray hexNumR = hexNum;
    if (isReversed) {
        while(hexNumR.at(0) == '0') {
            hexNumR.remove(0, 1);
        }
        hexNumR = reverseByteArray(hexNumR);
    } else {
        while(hexNumR.at(hexNumR.length()-1) == '0') {
            hexNumR.remove(hexNumR.length()-1, 1);
        }
    }
    if (hexNumR.isEmpty())
        return 0;
    int num = 0;
    for(int i = 0; i < hexNumR.length(); ++i) {
        QChar charA(hexNumR.at(i));
        if (charA.isDigit())
            num += charA.digitValue() * pow(16, i);
        else num += (charA.toLatin1() - 87) * pow(16, i);
    }
    return num;
}

QAudioFormat noteplayer::getWaveFormat(QString fileName)
{
    QAudioFormat out;

    QFile file(fileName);
    if (!file.open(QIODevice::ReadOnly))
        return out;

    //check the 'RIFF' head
    QByteArray ba = file.read(4);
    if (ba != "RIFF") {
        qDebug("No RIFF in header, not wave?");
        return out;
    }
    //get the length of file (ignoring this for now)
    file.read(4);
    //check the 'WAVE' head
    ba = file.read(4);
    if (ba != "WAVE") {
        qDebug("No WAVE in header, not wave?");
        return out;
    }
    //skip the 'fmt_' chunk
    file.read(4);
    //skip 'length' of format header
    file.read(4);
    //skip the '0x01'
    file.read(2);
    //get the channel numbers
    ba = file.read(2);
    ba = ba.toHex();
    ba = reverseHexByteArray(ba);
    int numChan = hexToUDec(ba, true);
    //figure out what the chan number is
    if (numChan <= 0) {
        qDebug("Bad number of channels");
        return out;
    }
    qDebug("num of chan = " + QString::number(numChan).toLatin1());
    out.setChannelCount(numChan);

    //get the sample rate
    ba = file.read(4);
    ba = ba.toHex();
    ba = reverseHexByteArray(ba);
    int sampleRate = hexToUDec(ba, true);
    qDebug("Sample rate (dec) = " + QString::number(hexToUDec(ba, true)).toLatin1());
    out.setSampleRate(sampleRate);

    //get the bytes per second
    ba = file.read(4);
    ba = ba.toHex();
    ba = reverseHexByteArray(ba);
    int bytesPerSec = hexToUDec(ba, true) / sampleRate;
    qDebug("Bytes per sec mult = " + QString::number(bytesPerSec).toLatin1());

    //get the bytes per sample
    ba = file.read(2);
    ba = ba.toHex();
    ba = reverseHexByteArray(ba);
    int bytesPerSample = hexToUDec(ba, true);
    qDebug("Bytes per sample = " + QString::number(bytesPerSample).toLatin1());

    //get the bits per sample
    ba = file.read(2);
    ba = ba.toHex();
    ba = reverseHexByteArray(ba);
    int bitsPerSample = hexToUDec(ba, true);
    qDebug("Bits per sample = " + QString::number(bitsPerSample).toLatin1());
    out.setSampleSize(bitsPerSample);

    out.setCodec("audio/pcm");
    out.setByteOrder(QAudioFormat::LittleEndian);
    out.setSampleType(QAudioFormat::UnSignedInt);

    qDebug()<<file.pos();
    file.close();
    return out;
}


void printbla(QString string, bool nextLine = true)
{
    QTextStream out(stdout);
    out << string;
    if (nextLine)
    {
        out << Qt::endl;
    }

}

void noteplayer::hold(){
    /*QAudioFormat format = getWaveFormat(path);
    QAudioOutput *audio = new QAudioOutput(format);
    printbla("dos");
    QFile file;
    file.setFileName(path);
    file.open(QIODevice::ReadOnly);
    audio->start(&file);
    //audio->setNotifyInterval(100);
    //connect(audio, SIGNAL(notify()), this, SLOT(hold()));
    exec();*/
}

void noteplayer::handleStateChanged(QMediaPlayer::MediaStatus status)
{
    if (status==QMediaPlayer::EndOfMedia){
        player->stop();
        exit();
    }
}
void noteplayer::run()
{
    printbla("started");
    player = new QMediaPlayer;
    player->setMedia(QUrl(path));
    player->setVolume(volume);
    //player->setMuted(true);
    player->play();
    connect(player, SIGNAL(mediaStatusChanged(QMediaPlayer::MediaStatus)), this, SLOT(handleStateChanged(QMediaPlayer::MediaStatus)));

    /*QAudioFormat format = getWaveFormat(path);
    QAudioOutput *audio = new QAudioOutput(format);
    printbla("dos");
    QFile file;
    file.setFileName(path);
    file.open(QIODevice::ReadOnly);
    printbla(QString::number(qreal((double)volume/100)));
    audio->setVolume(qreal((double)volume/100));
    audio->start(&file);
    audio->setNotifyInterval(100);
    connect(audio, SIGNAL(notify()), this, SLOT(hold()));
    connect(audio, SIGNAL(stateChanged(QAudio::State)), this, SLOT(handleStateChanged(QAudio::State)));
    //QThread::msleep(2000);
    exec();*/


    printbla("tres");
    quit();

    /*QSound sound(path,this);
    printbla("fudge");
    sound.play(path);
    printbla("fudge");
    QThread::msleep(duration);
    sound.stop();*/
    //int initTime = (int)QDateTime::currentMSecsSinceEpoch();

}
