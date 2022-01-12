#ifndef MAINWINDOW_H
#define MAINWINDOW_H

#include <QMainWindow>
#include <QMouseEvent>
#include <selectioncontroldelegate.h>
#include <QTableWidgetItem>
#include <progressbarupdate.h>
#include <noteplayer.h>
#include <qmediaplayer.h>

QT_BEGIN_NAMESPACE
namespace Ui { class MainWindow; }
QT_END_NAMESPACE

class MainWindow : public QMainWindow
{
    Q_OBJECT

public:
    MainWindow(QWidget *parent = nullptr);
    ~MainWindow();
    ProgressBarUpdate *progressBarUpdating;
    noteplayer *bluf;
    //SelectionControlDelegate *delegate;


private slots:
    void on_tableWidget_currentCellChanged(int currentRow, int currentColumn, int previousRow, int previousColumn);

    void on_pushButton_clicked();

    void updateDrawings(bool mode);

    void resetAll();

    void onLoadingFailed();


    void onProgressPercentUpdated(int);

    void onProgressLabelUpdated(QString);

    void onLockWindow(bool);

    void import_images(QString);

    void play();


    void moveCursor(int);

    void onNextStep();


    void on_tableWidget_itemSelectionChanged();


    void on_tableWidget_itemPressed(QTableWidgetItem * item);

    void on_spinBox_valueChanged(int arg1);

    void on_spinBox_2_valueChanged(int arg1);

    void on_checkBox_toggled(bool checked);

    void on_spinBox_3_valueChanged(int arg1);

    void on_spinBox_editingFinished();


    void on_spinBox_3_editingFinished();

    void on_spinBox_4_valueChanged(int arg1);

    void on_renderButton_clicked();


    void on_actionViolin_Solo_Bach_Partita_1_mvt_8_triggered();


    void on_comboBox_5_currentIndexChanged(int index);

    void on_actionMIDI_mid_triggered();

    void on_saveAsButton_clicked();


    void on_renderButton_2_clicked();

    void on_playbutton_pressed();

    void on_playbutton_released();

    void on_begbutton_pressed();

    void on_begbutton_released();

    void on_endbutton_pressed();

    void on_endbutton_released();

    void finished();

    void on_begbutton_clicked();

    void on_endbutton_clicked();

    void update_cursor();

    void on_rendered_play_button_clicked();

    void update_slider(qint64);

    void update_eta();

    void fileDuration(qint64);

    void on_horizontalSlider_2_valueChanged(int value);

    void on_horizontalSlider_valueChanged(qint64 value);

    void endplayer(QMediaPlayer::MediaStatus status);


    void on_actionViolin_Solo_Bach_Partita_1_mvt_8_short_triggered();

    void on_horizontalSlider_valueChanged(int value);

    void on_actionViolin_Solo_Bach_Sonata_2_mvt_4_short_triggered();

    void on_actionViolin_Solo_Bach_Sonata_2_mvt_4_triggered();

    void on_checkBox_2_toggled(bool checked);

private:
    QFile file;
    Ui::MainWindow *ui;
};


#endif // MAINWINDOW_H
