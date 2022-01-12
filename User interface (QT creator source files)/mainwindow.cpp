#include "mainwindow.h"
#include "ui_mainwindow.h"
#include "QTextStream"
#include "tuple"
#include "mainwindow.h"
#include "ui_mainwindow.h"
#include "QFileDialog"
#include "QFile"
#include "QTextStream"
#include "QDir"
#include "QProcess"
#include "iostream"
#include "string"
#include "qmessagebox.h"
#include "QErrorMessage"
#include "QTextStream"
#include "QStringRef"
#include "QDragEnterEvent"
#include "QDragLeaveEvent"
#include "QDragMoveEvent"
#include "QMouseEvent"
#include "QDropEvent"
#include "QMimeData"
#include "QShortcut"
#include "QDateTime"
#include "tablewidgetheaderpainting.h"
#include "qstylefactory.h"
#include "progressbarupdate.h"
#include "qtimer.h"
#include "qscrollbar.h"
#include "qmediaplayer.h"
#include "qsound.h"
#include "qaudiooutput.h"
#include "qsoundeffect.h"
#include "noteplayer.h"
#include "QtMath"
#include "QtCore"
#include "noteduration.h"
#include "qaudio.h"
#include "renderedplayer.h"


int drag_init_row = 0;
int drag_init_col = 0;

QString reverb = "true";

int g_current_column = 0;

int column_add = 20;

QColor background_color = QColor(88,88,88);
QColor foreground_color = QColor(140, 181, 212);

int bpm = 120;

QList<std::tuple<int,int>> selected_array = {};
QList<std::tuple<int>> selected_columns = {};
QList<std::tuple<int,int>> nonconfirmed_array = {};
QList<std::tuple<int,int>> old_non_confirmed_array = {};

QList<int> current_pitches = {};

bool new_selection = false;

bool init_mode = true;

int resolution = 2;

int max_res = 12;
bool playing = false;
int vol = 0;
int dir = 0;


int current_pitch = 2000;

QTimer *timer;
QTimer *playingtimer;
QTimer *loadingtimer;

QDate bla(2000,4,20);
QDateTime created(bla);

bool first_render = true;

QMediaPlayer *player;

int view_row = 0;

int playpos = 0;

bool started_playing = false;

int globalVolume = 100;

int render_duration = 0;
int loading_duration = 0;
int secs_elapsed = 0;



bool hasEnding (QString const &fullString, QStringList valid_endings) {
    std::vector<int> includes;

    foreach(QString ending,valid_endings)
    {
        QStringRef stringEnding(&fullString, fullString.length() - ending.length(), ending.length());
        includes.push_back(QStringRef::compare(stringEnding, ending, Qt::CaseInsensitive));
    }
    if (std::find(includes.begin(), includes.end(), 0) != includes.end())
    {
        return true;
    }
    else
    {
        return false;
    }
}
void print(QString string, bool nextLine = true)
{
    QTextStream out(stdout);
    out << string;
    if (nextLine)
    {
        out << Qt::endl;
    }
}
void print(int string, bool nextLine = true)
{
    QTextStream out(stdout);
    out << string;
    if (nextLine)
    {
        out << Qt::endl;
    }
}
void print(double string, bool nextLine = true)
{
    QTextStream out(stdout);
    out << string;
    if (nextLine)
    {
        out << Qt::endl;
    }
}

void print(qint64 string, bool nextLine = true)
{
    QTextStream out(stdout);
    out << string;
    if (nextLine)
    {
        out << Qt::endl;
    }
}

void print(quint64 string, bool nextLine = true)
{
    QTextStream out(stdout);
    out << string;
    if (nextLine)
    {
        out << Qt::endl;
    }
}



bool contains(int x1, int y1,QList<std::tuple<int,int>> target){
    int grid_y = (int)(y1*max_res/resolution);
    int grid_x = x1;
    int duration = (int)(max_res/resolution);
    for (int i = grid_y;i<grid_y+duration;i++){
        if (target.contains({grid_x,i})) return true;
    }
    return false;
}


QList<std::tuple<int,int>> edit_array(int x1, int y1,QList<std::tuple<int,int>> target,int val = 1,bool grid_index = true){
    int grid_x,grid_y,duration;
    if (grid_index){
        grid_y = (int)(y1*max_res/resolution);
        grid_x = x1;
        duration = (int)(max_res/resolution);
    }
    else{
        grid_y = y1;
        grid_x = x1;
        duration = 1;
    }

    for (int i=grid_y;i<grid_y+duration;i++){
        if  (val==1){
            if (!target.contains({grid_x,i})) target.append({grid_x,i});

        }
        else{
            if (target.contains({grid_x,i})){
                int index = target.indexOf({grid_x,i});
                target.removeAt(index);
            }
        }
    }

    return target;
}


QList<std::tuple<int,int>> update_notes(int x,int y,QList<std::tuple<int,int>> target,bool mode=true, bool grid_index = true){
    if (mode==true){
        target = edit_array(x,y,target,1,grid_index);
    }
    else if (mode==false){
        target = edit_array(x,y,target,0,grid_index);
    }
    return target;
}




MainWindow::MainWindow(QWidget *parent)
    : QMainWindow(parent)
    , ui(new Ui::MainWindow)
{
    ui->setupUi(this);
    QApplication::setStyle(QStyleFactory::create("Fusion"));
    resetAll();
    ui->tableWidget->setMouseTracking(true);
    ui->centralwidget->setMouseTracking(true);
    ui->tableWidget->setEditTriggers(QAbstractItemView::NoEditTriggers);

    ui->tableWidget->setSelectionMode(QAbstractItemView::ExtendedSelection);
    ui->tableWidget->setSelectionBehavior(QAbstractItemView::SelectItems);
    ui->tableWidget->clearSelection();
    //delegate = ;
    ui->tableWidget->setItemDelegate(new SelectionControlDelegate(this));
    ui->tableWidget->setItemDelegateForColumn(0,new SelectionControlDelegate(this));
    print(ui->tableWidget->columnCount());
    selected_array = {};//make_array(ui->tableWidget->rowCount(),ui->tableWidget->columnCount()*(max_res/resolution));
    nonconfirmed_array = {};//make_array(ui->tableWidget->rowCount(),ui->tableWidget->columnCount()*(max_res/resolution));
    old_non_confirmed_array = {};//make_array(ui->tableWidget->rowCount(),ui->tableWidget->columnCount()*(max_res/resolution));
    new QShortcut(QKeySequence(Qt::CTRL + Qt::Key_Q), this, SLOT(close()));
    new QShortcut(QKeySequence(Qt::Key_Space), this, SLOT(on_playbutton_released()));
    progressBarUpdating = new ProgressBarUpdate(this);
    connect(progressBarUpdating,SIGNAL(progressPercentUpdated(int)),this,SLOT(onProgressPercentUpdated(int)));
    connect(progressBarUpdating,SIGNAL(progressLabelUpdated(QString)),this,SLOT(onProgressLabelUpdated(QString)));
    connect(progressBarUpdating,SIGNAL(nextStep()),this,SLOT(onNextStep()));
    connect(progressBarUpdating,SIGNAL(loadingFailed()),this,SLOT(onLoadingFailed()));
    connect(progressBarUpdating,SIGNAL(lockWindow(bool)),this,SLOT(onLockWindow(bool)));
    timer =  new QTimer(this);
    connect(timer, SIGNAL(timeout()), this, SLOT(play()));
    loadingtimer =  new QTimer(this);
    connect(loadingtimer, SIGNAL(timeout()), this, SLOT(update_eta()));

    connect(ui->tableWidget->horizontalHeader(), SIGNAL(sectionPressed(int)),this, SLOT(moveCursor(int)));

    updateDrawings(init_mode);
    player = new QMediaPlayer;
    ui->tableWidget->verticalScrollBar()->setValue(13);
    QDir working_dir("C:/ProgramData/Virtuoso/metadata");
    if (!working_dir.exists()){
        working_dir.mkpath(working_dir.path());
    }
    //QAudioFormat format = getWaveFormat("./notes/80.wav");
    //QAudioOutput *audio = new QAudioOutput(format,this);
    print("dos");
}

void MainWindow::update_eta(){
    secs_elapsed+=1;
    int secs_till = loading_duration-secs_elapsed;
    if (secs_till>=0){
        int min = secs_till/60;
        int sec = secs_till%60;
        ui->label_31->setText("ETA: "+ QString::number(min) + "m " + QString::number(sec)+"s");
    }
}

void MainWindow::finished(){
    print("blublublub123");
}



void MainWindow::moveCursor(int index){
    nonconfirmed_array = old_non_confirmed_array;
    view_row = index;
    current_pitch = -1;
    updateDrawings(init_mode);
    print("cursor");

}

void play_note(QString path,int duration){
    print("note");
    //QSound sound(path);
    //sound.play();
    //QThread::msleep(duration);
    //sound.stop();
}

void MainWindow::update_cursor(){
    for (int i = 0;i<ui->tableWidget->rowCount();i++){
        if (contains(i,view_row-1,selected_array) || contains(i,view_row-1,nonconfirmed_array)){
            ui->tableWidget->item(i,view_row-1)->setBackground(foreground_color);
        }
        else{
            ui->tableWidget->item(i,view_row-1)->setBackground(background_color);
        }
        if (contains(i,view_row,selected_array) || contains(i,view_row,nonconfirmed_array)){
            ui->tableWidget->item(i,view_row)->setBackground(QColor(foreground_color.red()*0.8,foreground_color.green()*0.8,foreground_color.blue()*0.8));
        }
        else{
            ui->tableWidget->item(i,view_row)->setBackground(QColor(background_color.red()*0.8,background_color.green()*0.8,background_color.blue()*0.8));
        }
    }
}

void MainWindow::play(){
    int view_width = ui->tableWidget->viewport()->size().width();
    int view_indexes = view_width/ui->tableWidget->columnWidth(0);

    print(ui->tableWidget->horizontalScrollBar()->value());
    ui->tableWidget->horizontalScrollBar()->setValue(view_row-(view_indexes/2));

    int pitch;
    bool found = false;
    QList<int> found_pitches = {};
    for (int i = 0;i<ui->tableWidget->rowCount();i++){
        int grid_y = (int)(view_row*max_res/resolution);
        int grid_x = i;
        if (selected_array.contains({grid_x,grid_y})||nonconfirmed_array.contains({grid_x,grid_y})){
            pitch = 96-i;
            found = true;
            found_pitches.append(pitch);
            if (!current_pitches.contains(pitch)){
                QMediaPlayer *bluf = new QMediaPlayer;
                bluf->setMedia(QUrl("qrc:/notes/notes/"+QString::number(pitch)+".wav"));
                bluf->setVolume(globalVolume);
                bluf->play();
            }
        }
    }
    current_pitches = found_pitches;

    if (view_row==ui->tableWidget->columnCount()-1){
        on_playbutton_released();
    }
    else{
        view_row+=1;
        update_cursor();
    }
}

void MainWindow::onProgressPercentUpdated(int i)
{
    if (ui->midiEditorStackedWidget->currentIndex()==1) ui->progressBar_2->setValue(i);
    else if (ui->midiEditorStackedWidget->currentIndex()==0){
        ui->progressBar->setValue(i);
        ui->progressBar->repaint();
    }
}

void MainWindow::onNextStep(){
    if (ui->midiEditorStackedWidget->currentIndex()==1){
        onLockWindow(false);
        int startTime;
        int endTime;

        QFile loadStart("C:/ProgramData/Virtuoso/metadata/last_start.txt");
        if (loadStart.exists())
        {
            while (true)
            {
                if (loadStart.open(QIODevice::ReadWrite))
                {
                    QTextStream stream(&loadStart);
                    startTime = stream.readLine().toFloat();
                    loadStart.close();
                    break;
                }
            }
        }

        //get the finish time of finishing the exe
        QFile runEnd("C:/ProgramData/Virtuoso/metadata/last_end.txt");
        if (runEnd.exists())
        {
            while (true)
            {
                if (runEnd.open(QIODevice::ReadWrite))
                {
                    QTextStream stream(&runEnd);
                    endTime = stream.readLine().toFloat();
                    runEnd.close();
                    break;
                }
            }
        }

        QFile loadStartOut("C:/ProgramData/Virtuoso/metadata/last_load_start.txt");
        if (loadStartOut.exists()) loadStartOut.remove();
        while (true)
        {
            if (loadStartOut.open(QIODevice::ReadWrite))
            {
                QTextStream stream(&loadStartOut);
                stream<<startTime;
                loadStartOut.close();
                break;
            }
        }
        //write the end time out
        QFile runEndOut("C:/ProgramData/Virtuoso/metadata/last_load_end.txt");
        if (runEndOut.exists()) runEndOut.remove();
        while (true)
        {
            if (runEndOut.open(QIODevice::ReadWrite))
            {
                QTextStream stream(&runEndOut);
                stream<<endTime;
                runEndOut.close();
                break;
            }
        }
        QFile bpmfile("C:/ProgramData/Virtuoso/metadata/bpm.txt");
        while (true)
        {
            if (bpmfile.open(QIODevice::ReadWrite))
            {
                QTextStream stream(&bpmfile);
                int bpm_ = stream.readLine().toInt();
                ui->spinBox_4->setValue(bpm_);
                bpm = bpm_;
                bpmfile.close();
                break;
            }
        }
        QFile resfile("C:/ProgramData/Virtuoso/metadata/minres.txt");
        while (true)
        {
            if (resfile.open(QIODevice::ReadWrite))
            {
                QTextStream stream(&resfile);
                int res = stream.readLine().toInt();
                print("res:");
                print(res);
                ui->spinBox_2->setValue(res);
                on_spinBox_2_valueChanged(res);
                resfile.close();
                break;
            }
        }
        int length = 0;
        QFile lengthfile("C:/ProgramData/Virtuoso/metadata/midi_length.txt");
        while (true)
        {
            if (lengthfile.open(QIODevice::ReadWrite))
            {
                QTextStream stream(&lengthfile);
                length = stream.readLine().toInt();
                lengthfile.close();
                break;
            }
        }
        column_add = (int)((double)(length*resolution)/max_res)+1-ui->tableWidget->columnCount();
        on_pushButton_clicked();
        column_add = 20;


        QFile file("C:/ProgramData/Virtuoso/metadata/midi_array.h5");
        QString array;
        while (true)
        {
            if (file.open(QIODevice::ReadWrite))
            {
                QTextStream stream(&file);
                array = stream.readAll();
                file.close();
                break;
            }
        }
        QStringList coods = array.split("_");
        coods.removeLast();
        QList<std::tuple<int,int>> coordinates = {};
        selected_array = {};
        nonconfirmed_array = {};
        foreach(QString item, coods){
            QApplication::processEvents();
            QStringList cood = item.split(" ");
            if (cood[0].toInt()>=0) selected_array = update_notes(cood[0].toInt(),cood[1].toInt(),selected_array,true,false);
        }
        ui->label->setText("Import Complete.");
        ui->midiEditorStackedWidget->setCurrentIndex(0);
        updateDrawings(true);
        onLockWindow(true);
        onProgressPercentUpdated(100);
    }
    else if (ui->midiEditorStackedWidget->currentIndex()==0){
        onLockWindow(false);
        print("well it got here...");

        ui->label->setText("Render Complete.");
        ui->stackedWidget->setCurrentIndex(2);
        onLockWindow(true);
        ui->label_31->setText("");
        first_render = true;
        print("why?");
    }
}


void MainWindow::onProgressLabelUpdated(QString text)
{
    if (ui->midiEditorStackedWidget->currentIndex()==1) ui->label->setText(text);
    else if (ui->midiEditorStackedWidget->currentIndex()==0) ui->label_28->setText(text);
}

void MainWindow::onLockWindow(bool truth){
    for(auto&& child:ui->centralwidget->findChildren<QWidget *>()){
        child->setEnabled(truth);
    }
    ui->menubar->setEnabled(truth);
    if (!truth){

        ui->playbutton->setStyleSheet({"border-radius:35px;"
                                         "image: url(:/images/playplaybuttonpressed.png);"
                                          "max-width:70px;"
                                          "max-height:70px;"
                                          "min-width:70px;"
                                          "min-height:70px;"});



        ui->begbutton->setStyleSheet({"border-radius:35px;"
                                         "image: url(:/images/beginningbuttonpressed.png);"
                                          "max-width:70px;"
                                          "max-height:70px;"
                                          "min-width:70px;"
                                          "min-height:70px;"});

        ui->endbutton->setStyleSheet({"border-radius:35px;"
                                         "image: url(:/images/endbuttonpressed.png);"
                                          "max-width:70px;"
                                          "max-height:70px;"
                                          "min-width:70px;"
                                          "min-height:70px;"});
    }
    else{
        ui->playbutton->setStyleSheet({"border-radius:35px;"
                                         "image: url(:/images/playbutton.png);"
                                          "max-width:70px;"
                                          "max-height:70px;"
                                          "min-width:70px;"
                                          "min-height:70px;"});



        ui->begbutton->setStyleSheet({"border-radius:35px;"
                                         "image: url(:/images/beginningbutton.png);"
                                          "max-width:70px;"
                                          "max-height:70px;"
                                          "min-width:70px;"
                                          "min-height:70px;"});

        ui->endbutton->setStyleSheet({"border-radius:35px;"
                                         "image: url(:/images/endbutton.png);"
                                          "max-width:70px;"
                                          "max-height:70px;"
                                          "min-width:70px;"
                                          "min-height:70px;"});
    }

}

void MainWindow::onLoadingFailed()
{
    if (ui->midiEditorStackedWidget->currentIndex()==1){
        ui->label->setText("Data loading failed.");
        ui->stackedWidget->setCurrentIndex(0);
        setAcceptDrops(true);
        QMessageBox dataLoadFailed(QMessageBox::Warning, "Data Invalid", "Failed to load the selected data. The data was either corrupt or the format was invalid.", QMessageBox::Ok, this);
        //dataLoadFailed.setStyleSheet("QLabel{min-width: 300px;}");
        dataLoadFailed.setWindowFlags(Qt::Dialog | Qt::CustomizeWindowHint | Qt::WindowTitleHint | Qt::WindowCloseButtonHint);
        dataLoadFailed.exec();
        ui->midiEditorStackedWidget->setCurrentIndex(0);
        onLockWindow(true);
    }
    else{
        ui->label->setText("Instrument rendering failed.");
        ui->stackedWidget->setCurrentIndex(0);
        setAcceptDrops(true);
        QMessageBox dataLoadFailed(QMessageBox::Warning, "Rendering Failed", "Failed to render the requested midi. Make sure the midi isn't empty and try again.", QMessageBox::Ok, this);
        //dataLoadFailed.setStyleSheet("QLabel{min-width: 300px;}");
        dataLoadFailed.setWindowFlags(Qt::Dialog | Qt::CustomizeWindowHint | Qt::WindowTitleHint | Qt::WindowCloseButtonHint);
        dataLoadFailed.exec();
        ui->midiEditorStackedWidget->setCurrentIndex(0);
        onLockWindow(true);
    }
}


MainWindow::~MainWindow()
{
    progressBarUpdating->wait();
    delete ui;
}

void MainWindow::resetAll(){
    for (int i=0;i<ui->tableWidget->rowCount();i++){
        for (int j=0;j<ui->tableWidget->columnCount();j++){
            ui->tableWidget->setItem(i,j, new QTableWidgetItem);
            ui->tableWidget->item(i,j)->setFlags(ui->tableWidget->item(i,j)->flags() & ~Qt::ItemIsEditable);
            //ui->tableWidget->item(i,j)->setFlags(ui->tableWidget->item(i,j)->flags() & ~Qt::ItemIsSelectable);
        }
    }
    QStringList labels = {};
    for (int i=0;i<ui->tableWidget->columnCount();i++){
        ui->tableWidget->setHorizontalHeaderItem(i,new QTableWidgetItem);
        ui->tableWidget->horizontalHeaderItem(i)->setBackground(QColor(59,59,59));
        ui->tableWidget->horizontalHeaderItem(i)->setTextColor(QColor(180,180,180));
        if (i%resolution==0){
            labels.append(QString::number((int)(i/resolution)));

        }
        else{
            labels.append("");
        }
    }
    on_spinBox_editingFinished();

    ui->tableWidget->setHorizontalHeaderLabels(labels);

    for (int i=0;i<ui->tableWidget->rowCount();i++){
        if(ui->tableWidget->verticalHeaderItem(i)->text().contains("#")){
            ui->tableWidget->verticalHeaderItem(i)->setBackground(QColor(0,0,0));
            ui->tableWidget->verticalHeaderItem(i)->setTextColor(QColor(255,255,255));
        }
        else ui->tableWidget->verticalHeaderItem(i)->setBackground(QColor(255,255,255));

    }



}


void MainWindow::updateDrawings(bool mode){
    for (int i=0;i<ui->tableWidget->rowCount();i++){
        for (int j = 0;j<ui->tableWidget->columnCount();j++){
            ui->tableWidget->item(i,j)->setBackground(background_color);
        }
    }

    QList<int> taken_columns = {};
    for (int i=nonconfirmed_array.length()-1;i>=0;i--){
        int x = std::get<0>(nonconfirmed_array[i]);
        int actual_y = std::get<1>(nonconfirmed_array[i]);
        if (taken_columns.contains(actual_y)){
            int index = nonconfirmed_array.indexOf({x,actual_y});
            nonconfirmed_array.removeAt(index);
        }
        else if (mode) taken_columns.append(actual_y);
    }
    for (int i=selected_array.length()-1;i>=0;i--){
        int x = std::get<0>(selected_array[i]);
        int actual_y = std::get<1>(selected_array[i]);
        if (taken_columns.contains(actual_y)){
            int index = selected_array.indexOf({x,actual_y});
            selected_array.removeAt(index);
        }
        else taken_columns.append(actual_y);

    }
    for (int i=selected_array.length()-1;i>=0;i--){
        int x = std::get<0>(selected_array[i]);
        int actual_y = std::get<1>(selected_array[i]);
        int y = (int)((double)std::get<1>(selected_array[i])*resolution/max_res);
        if(y<ui->tableWidget->columnCount()){
            ui->tableWidget->item(x,y)->setBackground(foreground_color);
        }
    }
    for (int i=nonconfirmed_array.length()-1;i>=0;i--){
        int x = std::get<0>(nonconfirmed_array[i]);
        int actual_y = std::get<1>(nonconfirmed_array[i]);
        int y = (int)((double)std::get<1>(nonconfirmed_array[i])*resolution/max_res);
        if(y<ui->tableWidget->columnCount()){
            if (mode){
                ui->tableWidget->item(x,y)->setBackground(foreground_color);
            }
            else{
                ui->tableWidget->item(x,y)->setBackground(background_color);
            }
        }
    }


    for (int i=0;i<ui->tableWidget->rowCount();i++){
        int red = ui->tableWidget->item(i,view_row)->backgroundColor().red();
        int green = ui->tableWidget->item(i,view_row)->backgroundColor().green();
        int blue = ui->tableWidget->item(i,view_row)->backgroundColor().blue();
        ui->tableWidget->item(i,view_row)->setBackground(QColor(red*0.8,green*0.8,blue*0.8));
    }

}

void MainWindow::on_tableWidget_currentCellChanged(int currentRow, int currentColumn, int previousRow, int previousColumn)
{
    drag_init_row = currentRow;

    old_non_confirmed_array = nonconfirmed_array;
    g_current_column = currentColumn;
    new_selection = false;
}

void MainWindow::on_tableWidget_itemSelectionChanged()
{
    print("thingy");
    QList<QTableWidgetItem *> items=ui->tableWidget->selectedItems();
    //items.append(ui->tableWidget->item(drag_init_row,g_current_column));
    nonconfirmed_array = {};
    for(int i=0;i<items.length();i++){
        QTableWidgetItem * item = items[i];
        if (init_mode==true){
            if (item->column()<=g_current_column && item->column()>=drag_init_col){
                nonconfirmed_array = update_notes(drag_init_row,item->column(),nonconfirmed_array);
            }
        }
        else{
             nonconfirmed_array = update_notes(item->row(),item->column(),nonconfirmed_array);
        }

    }
    print("thingy2");
    updateDrawings(init_mode);
}





void MainWindow::on_tableWidget_itemPressed(QTableWidgetItem *item)
{
    drag_init_col = item->column();

    for(int i =0;i<old_non_confirmed_array.length();i++){
        selected_array = update_notes(std::get<0>(old_non_confirmed_array[i]),std::get<1>(old_non_confirmed_array[i]),selected_array,init_mode,false);
    }



    if(contains(item->row(),item->column(),selected_array)){
        init_mode = false;
        print("nogucc");
    }
    else{
        print("gucc");
        init_mode = true;
    }
    MainWindow::on_tableWidget_currentCellChanged(item->row(),item->column(),1,1);
    MainWindow::on_tableWidget_itemSelectionChanged();
    print("fuhh");
    new_selection = true;
}

void MainWindow::on_spinBox_valueChanged(int arg1)
{
    if (abs(arg1-ui->tableWidget->columnWidth(0))==1){
        if (arg1 <50){
            ui->spinBox->setValue(50);
            arg1 = 50;
        }
        else if (arg1>300){
            ui->spinBox->setValue(300);
            arg1 = 300;
        }
        for (int j=0;j<ui->tableWidget->columnCount();j++){
            ui->tableWidget->setColumnWidth(j,arg1);
        }
    }
}

void MainWindow::on_spinBox_editingFinished()
{
    int arg1 = ui->spinBox->value();
    if (arg1 <50){
        ui->spinBox->setValue(50);
        arg1 = 50;
    }
    else if (arg1>300){
        ui->spinBox->setValue(300);
        arg1 = 300;
    }
    for (int j=0;j<ui->tableWidget->columnCount();j++){
        ui->tableWidget->setColumnWidth(j,arg1);
    }
}



void MainWindow::on_spinBox_2_valueChanged(int arg1)
{
    QList<int> valid_reses = {};
    for(int i=1;i<max_res;i++){
        if (max_res%i==0){
            print(i);
            valid_reses.append(i);
        }
    }
    print("bruvbruv");
    if (valid_reses.contains(arg1)){
        double rescale_factor = (double)arg1/resolution;
        resolution = arg1;
        int new_columnCount = (int)(ui->tableWidget->columnCount()*rescale_factor);
        int add_columns = new_columnCount-ui->tableWidget->columnCount();
        column_add = add_columns+1;
        on_pushButton_clicked();
        column_add = 20;

        print("so ig it died here then");


        updateDrawings(init_mode);
        QList<QString> labels = {};
        for (int i=0;i<ui->tableWidget->columnCount();i++){
            if (i%resolution==0){
                labels.append(QString::number((int)(i/resolution)));
            }
            else{
                labels.append("");
            }
        }
        ui->tableWidget->setHorizontalHeaderLabels(labels);
    }
    else{
        print("not supposed to be here...");
        if (arg1<=valid_reses[valid_reses.length()-1]){
            for (int i=0; i<valid_reses.length();i++){
                if (valid_reses[i]>resolution){
                    if (arg1>resolution) ui->spinBox_2->setValue(valid_reses[i]);
                    else ui->spinBox_2->setValue(valid_reses[i-1]);
                }
            }
        }
        else ui->spinBox_2->setValue(valid_reses[valid_reses.length()-1]);
    }
}

void MainWindow::on_pushButton_clicked()
{
    print(ui->tableWidget->columnCount()*(max_res/resolution)+1);
    ui->tableWidget->setColumnCount(ui->tableWidget->columnCount()+column_add);
    resetAll();
    updateDrawings(init_mode);
}

void MainWindow::on_checkBox_toggled(bool checked)
{
    ui->tableWidget->setShowGrid(checked);
}
void MainWindow::on_spinBox_3_valueChanged(int arg1)
{
    if (abs(arg1-ui->tableWidget->rowHeight(0))==1){
        if (arg1 <50){
            ui->spinBox_3->setValue(50);
            arg1 = 50;
        }
        else if (arg1>300){
            ui->spinBox_3->setValue(300);
            arg1 = 300;
        }
        for (int j=0;j<ui->tableWidget->rowCount();j++){
            ui->tableWidget->setRowHeight(j,arg1);
        }
    }
}

void MainWindow::on_spinBox_3_editingFinished()
{
    int arg1 = ui->spinBox_3->value();
    if (arg1 <50){
        ui->spinBox_3->setValue(50);
        arg1 = 50;
    }
    else if (arg1>300){
        ui->spinBox_3->setValue(300);
        arg1 = 300;
    }
    for (int j=0;j<ui->tableWidget->rowCount();j++){
        ui->tableWidget->setRowHeight(j,arg1);
    }
}


void MainWindow::on_spinBox_4_valueChanged(int arg1)
{
    if (arg1>200){
        ui->spinBox_4->setValue(200);
    }
    else if( arg1<1){
        ui->spinBox_4->setValue(1);
    }
    else bpm = arg1;
}

void MainWindow::on_renderButton_clicked()
{
    ui->stackedWidget->setCurrentIndex(1);
    onLockWindow(false);

    QProcess *process = new QProcess(this);
    QString exe("./scripts/render_midi.exe");
    process->startDetached(exe);


    QFile progressLabel("C:/ProgramData/Virtuoso/metadata/progress_label.txt");
    progressLabel.remove();
    while (true)
    {
        if (progressLabel.open(QIODevice::ReadWrite))
        {
            QTextStream stream(&progressLabel);
            stream << "Packing midi data...";
            progressLabel.close();
            break;
        }
    }

    QFile progressAmt("C:/ProgramData/Virtuoso/metadata/progress_percent.txt");
    while (true)
        if (progressAmt.open(QIODevice::ReadWrite))
        {
            QTextStream stream2(&progressAmt);
            stream2<< 0;
            progressAmt.close();
            break;
        }
    QFile bpm_file("C:/ProgramData/Virtuoso/metadata/bpm.txt");
    bpm_file.remove();
    while (true)
    {
        if (bpm_file.open(QIODevice::ReadWrite))
        {
            QTextStream stream(&bpm_file);
            stream << bpm;
            bpm_file.close();
            break;
        }
    }
    QFile reverbfile("C:/ProgramData/Virtuoso/metadata/reverb.txt");
    reverbfile.remove();
    while (true)
    {
        if (reverbfile.open(QIODevice::ReadWrite))
        {
            QTextStream stream(&reverbfile);
            stream << reverb;
            reverbfile.close();
            break;
        }
    }
    QFile midi_file("C:/ProgramData/Virtuoso/metadata/midi_array.h5");
    midi_file.remove();
    int max = 0;
    int first = 999999999;
    while (true)
    {
        if (midi_file.open(QIODevice::ReadWrite))
        {
            QTextStream stream(&midi_file);
            for (int i =0;i<selected_array.length();i++){
                stream << std::get<0>(selected_array[i]);
                stream << " ";
                stream << std::get<1>(selected_array[i]);
                if (std::get<1>(selected_array[i])>max){
                    max = std::get<1>(selected_array[i]);
                }
                else if (std::get<1>(selected_array[i])<first){
                    first = std::get<1>(selected_array[i]);
                }
                stream << "_";
            }
            for (int i =0;i<nonconfirmed_array.length();i++){
                stream << std::get<0>(nonconfirmed_array[i]);
                stream << " ";
                stream << std::get<1>(nonconfirmed_array[i]);
                if (std::get<1>(nonconfirmed_array[i])>max){
                    max = std::get<1>(nonconfirmed_array[i]);
                }
                else if (std::get<1>(nonconfirmed_array[i])<first){
                    first = std::get<1>(nonconfirmed_array[i]);
                }
                stream << "_";
            }
            midi_file.close();
            break;
        }
    }
    quint64 song_duration = ((quint64)max*1323000)/((quint64)max_res*(quint64)bpm);
    quint64 duration = (quint64)((double)(song_duration*0.0865931)+5732.37)+18000;
    print("duration:");
    print(max);
    print(duration);
    print(song_duration);
    progressBarUpdating->expectedDuration = duration;
    progressBarUpdating->start();
    loadingtimer->start(1000);
    secs_elapsed = 0;
    loading_duration = duration/1000;
    onLockWindow(true);
    ui->menubar->setEnabled(false);
    progressLabel.remove();
    while (true)
    {
        if (progressLabel.open(QIODevice::ReadWrite))
        {
            QTextStream stream(&progressLabel);
            stream << "Starting python rendering script...";
            progressLabel.close();
            break;
        }
    }
}


void MainWindow::import_images(QString path)
{
    onLockWindow(false);
    print("blub");
    QFile file("C:/ProgramData/Virtuoso/metadata/import_file_name.txt");
    QDir dir("C:/ProgramData/Virtuoso/metadata");
    if (!dir.exists())
        dir.mkpath(".");
    if (file.exists())
        file.remove();

    if (file.open(QIODevice::ReadWrite))
    {
        QTextStream stream(&file);
        stream << path;
        file.close();
    }
    QProcess *process = new QProcess(this);
    QString exe("./scripts/import_midi.exe");
    //if ()
    process->startDetached(exe);

    QFile progressLabel("C:/ProgramData/Virtuoso/metadata/progress_label.txt");
    progressLabel.remove();
    while (true)
    {
        if (progressLabel.open(QIODevice::ReadWrite))
        {
            QTextStream stream(&progressLabel);
            stream << "Reading midi file...";
            progressLabel.close();
            break;
        }
    }
    QFile progressAmt("C:/ProgramData/Virtuoso/metadata/progress_percent.txt");
    progressAmt.remove();
    while (true)
    {
        if (progressAmt.open(QIODevice::ReadWrite))
        {
            QTextStream stream(&progressAmt);
            stream << "0";
            progressAmt.close();
            break;
        }
    }
    int startTime;
    int endTime;
    //get the start time of starting the exe
    QFile loadStart("C:/ProgramData/Virtuoso/metadata/last_load_start.txt");
    if (loadStart.exists())
    {
        while (true)
        {
            if (loadStart.open(QIODevice::ReadWrite))
            {
                QTextStream stream(&loadStart);
                startTime = stream.readLine().toFloat();
                loadStart.close();
                break;
            }
        }
    }
    else
    {
        startTime = 0;
    }
    //get the finish time of finishing the exe
    QFile runEnd("C:/ProgramData/Virtuoso/metadata/last_load_end.txt");
    if (runEnd.exists())
    {
        while (true)
        {
            if (runEnd.open(QIODevice::ReadWrite))
            {
                QTextStream stream(&runEnd);
                endTime = stream.readLine().toFloat();
                runEnd.close();
                break;
            }
        }
    }
    else
    {
        endTime = 10000;
    }
    int duration = endTime-startTime;
    if (playing) on_playbutton_released();
    progressBarUpdating->expectedDuration = duration;
    progressBarUpdating->start();
}

void MainWindow::on_actionViolin_Solo_Bach_Partita_1_mvt_8_triggered()
{
    try
        {
            //checks to see if the C:/ProgramData/Virtuoso/metadata folder exists, and if not makes it
            QString import_path = "./demos/vp1-8tbd.mid";
            QDir dir("C:/ProgramData/Virtuoso/metadata");
            if (!dir.exists())
                dir.mkpath(".");

            //checks to see if the selected file is valid
            bool file_valid = false;
            QList<QString> list = {".mid"};
            if (hasEnding(import_path, list)){
                file_valid = true;
            }

            //If the file is valid, checks if info file exists, and if so delete it, then remake it (to only have a single path), then write the path out
            if (file_valid == true)
            {
                ui->midiEditorStackedWidget->setCurrentIndex(1);
                import_images(import_path);
            }
            else if (!import_path.isNull())
            {
                QMessageBox folderinvalid(QMessageBox::Warning, "File type unsupported", "Please select a file that is a supported data type.", QMessageBox::Ok, this);
                folderinvalid.setWindowFlags(Qt::Dialog | Qt::CustomizeWindowHint | Qt::WindowTitleHint | Qt::WindowCloseButtonHint);
                folderinvalid.exec();
            }
        }
        catch(...)
        {
            ;
        }
}


void MainWindow::on_comboBox_5_currentIndexChanged(int index)
{
    ui->comboBox_5->setCurrentIndex(0);
}

void MainWindow::on_actionMIDI_mid_triggered()
{
    try{
        //checks to see if the C:/ProgramData/Virtuoso/metadata folder exists, and if not makes it
        QString import_path = QFileDialog::getOpenFileName(this,tr("Select File"),"C:/Documents//","Midi Audio files (*.mid);;All files (*.*)");
        QDir dir("C:/ProgramData/Virtuoso/metadata");
        if (!dir.exists())
            dir.mkpath(".");

        //checks to see if the selected file is valid
        bool file_valid = false;
        QList<QString> list = {".mid"};
        if (hasEnding(import_path, list)){
            file_valid = true;
        }

        //If the file is valid, checks if info file exists, and if so delete it, then remake it (to only have a single path), then write the path out
        if (file_valid == true)
        {
            ui->midiEditorStackedWidget->setCurrentIndex(1);
            import_images(import_path);
        }
        else if (!import_path.isNull())
        {
            QMessageBox folderinvalid(QMessageBox::Warning, "File type unsupported", "Please select a file that is a supported data type.", QMessageBox::Ok, this);
            folderinvalid.setWindowFlags(Qt::Dialog | Qt::CustomizeWindowHint | Qt::WindowTitleHint | Qt::WindowCloseButtonHint);
            folderinvalid.exec();
        }
    }
    catch(...){
        ;
    }

}

void MainWindow::on_saveAsButton_clicked()
{
    QString save_path = QFileDialog::getSaveFileName(this,tr("Save As"),"C://","Wave Audio files (*.wav);;All files (*.*)");
    try{       
        QDir dest_dir(save_path);


        QString name = QFileInfo(save_path).fileName();
        if (!hasEnding(name,{".wav"})) name = name+".wav";
        print(dest_dir.absolutePath());
        QFile destfile(dest_dir.absolutePath());
        if (destfile.exists()) destfile.remove();
        QFile::copy("C:/ProgramData/Virtuoso/metadata/rendered.wav", dest_dir.absolutePath());
    }

    catch(...)
    {
        QMessageBox savingFailed(QMessageBox::Warning, "Saving Location Invalid", "Failed to save into " + save_path + ". Please check that the destination file is not currently in use and try again.", QMessageBox::Ok, this);
        savingFailed.setWindowFlags(Qt::Dialog | Qt::CustomizeWindowHint | Qt::WindowTitleHint | Qt::WindowCloseButtonHint);
        savingFailed.exec();
    }
}

void MainWindow::on_renderButton_2_clicked()
{
    on_renderButton_clicked();
}


void MainWindow::on_playbutton_pressed()
{
    ui->playbutton->setStyleSheet({"border-radius:35px;"
                                     "image: url(:/images/playbuttonpressed.png);"
                                      "max-width:70px;"
                                      "max-height:70px;"
                                      "min-width:70px;"
                                      "min-height:70px;"});

}

void MainWindow::on_playbutton_released()
{    
    if (playing){
        ui->playbutton->setStyleSheet({"border-radius:35px;"
                                         "image: url(:/images/playbutton.png);"
                                          "max-width:70px;"
                                          "max-height:70px;"
                                          "min-width:70px;"
                                          "min-height:70px;"});
        playing = false;
        timer->stop();
        if (started_playing){
            started_playing = false;
        }
    }
    else{

        playing = true;
        print(view_row);
        print("playback tick speed");
        print((int)(60000/(bpm*resolution)));
        timer->start((int)(60000/(bpm*resolution)));

        ui->playbutton->setStyleSheet({"border-radius:35px;"
                                         "image: url(:/images/pausebutton.png);"
                                          "max-width:70px;"
                                          "max-height:70px;"
                                          "min-width:70px;"
                                          "min-height:70px;"});
    }
}

void MainWindow::on_begbutton_pressed()
{
    ui->begbutton->setStyleSheet({"border-radius:35px;"
                                     "image: url(:/images/beginningbuttonpressed.png);"
                                      "max-width:70px;"
                                      "max-height:70px;"
                                      "min-width:70px;"
                                      "min-height:70px;"});
}
void MainWindow::on_begbutton_released()
{
    ui->begbutton->setStyleSheet({"border-radius:35px;"
                                     "image: url(:/images/beginningbutton.png);"
                                      "max-width:70px;"
                                      "max-height:70px;"
                                      "min-width:70px;"
                                      "min-height:70px;"});
}

void MainWindow::on_endbutton_pressed()
{
    ui->endbutton->setStyleSheet({"border-radius:35px;"
                                     "image: url(:/images/endbuttonpressed.png);"
                                      "max-width:70px;"
                                      "max-height:70px;"
                                      "min-width:70px;"
                                      "min-height:70px;"});
}

void MainWindow::on_endbutton_released()
{
    ui->endbutton->setStyleSheet({"border-radius:35px;"
                                     "image: url(:/images/endbutton.png);"
                                      "max-width:70px;"
                                      "max-height:70px;"
                                      "min-width:70px;"
                                      "min-height:70px;"});
}



void MainWindow::on_begbutton_clicked()
{
    ui->tableWidget->horizontalScrollBar()->setValue(0);
}

void MainWindow::on_endbutton_clicked()
{
    ui->tableWidget->horizontalScrollBar()->setValue(ui->tableWidget->width());
}

void MainWindow::fileDuration(qint64 duration) {
    print("duration change");
    disconnect(player,SIGNAL(durationChanged(qint64)),this,SLOT(fileDuration(qint64)));
    render_duration = duration;
    ui->horizontalSlider->setMaximum(duration);
}

void MainWindow::endplayer(QMediaPlayer::MediaStatus status){
    if (status==QMediaPlayer::EndOfMedia){
        player->stop();
        player->setPosition(0);
        ui->rendered_play_button->setText("Play");
    }
}
void MainWindow::on_rendered_play_button_clicked()
{
    if (ui->rendered_play_button->text()=="Play"){
        ui->rendered_play_button->setText("Pause");

        if (first_render){
            print("ok?1");
            first_render = false;

            player = new QMediaPlayer;
            connect(player,SIGNAL(durationChanged(qint64)),this,SLOT(fileDuration(qint64)));
            player->setMedia(QUrl("C:/ProgramData/Virtuoso/metadata/rendered.wav"));
            player->setVolume(globalVolume);
            //player->setMuted(true);
            player->play();
            ui->horizontalSlider->setValue(0);

            connect(player, SIGNAL(positionChanged(qint64)), this, SLOT(update_slider(qint64)));
            connect(player, SIGNAL(mediaStatusChanged(QMediaPlayer::MediaStatus)), this, SLOT(endplayer(QMediaPlayer::MediaStatus)));
            print("ok?");
        }

        else{
            print("resume");
            player->play();
        }
    }
    else{
        print("pause");
        ui->rendered_play_button->setText("Play");
        player->pause();
    }
}

void MainWindow::update_slider(qint64 pos){
    ui->horizontalSlider->setTracking(false);
    ui->horizontalSlider->setSliderPosition(pos);
}

void MainWindow::on_horizontalSlider_2_valueChanged(int value)
{
    globalVolume = value;
    player->setVolume(value);
}

void MainWindow::on_horizontalSlider_valueChanged(qint64 value)
{
    print(value);
    ui->horizontalSlider->setTracking(true);
    player->setPosition(value);
}

void MainWindow::on_horizontalSlider_valueChanged(int value)
{
    print(value);
    player->setPosition(value);
}


void MainWindow::on_actionViolin_Solo_Bach_Partita_1_mvt_8_short_triggered()
{
    try
        {
            //checks to see if the C:/ProgramData/Virtuoso/metadata folder exists, and if not makes it
            QString import_path = "./demos/short.mid";
            QDir dir("C:/ProgramData/Virtuoso/metadata");
            if (!dir.exists())
                dir.mkpath(".");

            //checks to see if the selected file is valid
            bool file_valid = false;
            QList<QString> list = {".mid"};
            if (hasEnding(import_path, list)){
                file_valid = true;
            }

            //If the file is valid, checks if info file exists, and if so delete it, then remake it (to only have a single path), then write the path out
            if (file_valid == true)
            {
                ui->midiEditorStackedWidget->setCurrentIndex(1);
                import_images(import_path);
            }
            else if (!import_path.isNull())
            {
                QMessageBox folderinvalid(QMessageBox::Warning, "File type unsupported", "Please select a file that is a supported data type.", QMessageBox::Ok, this);
                folderinvalid.setWindowFlags(Qt::Dialog | Qt::CustomizeWindowHint | Qt::WindowTitleHint | Qt::WindowCloseButtonHint);
                folderinvalid.exec();
            }
        }
        catch(...)
        {
            ;
        }
}



void MainWindow::on_actionViolin_Solo_Bach_Sonata_2_mvt_4_short_triggered()
{
    try
        {
            //checks to see if the C:/ProgramData/Virtuoso/metadata folder exists, and if not makes it
            QString import_path ="./demos/short_gig.mid";
            QDir dir("C:/ProgramData/Virtuoso/metadata");
            if (!dir.exists())
                dir.mkpath(".");

            //checks to see if the selected file is valid
            bool file_valid = false;
            QList<QString> list = {".mid"};
            if (hasEnding(import_path, list)){
                file_valid = true;
            }

            //If the file is valid, checks if info file exists, and if so delete it, then remake it (to only have a single path), then write the path out
            if (file_valid == true)
            {
                ui->midiEditorStackedWidget->setCurrentIndex(1);
                import_images(import_path);
            }
            else if (!import_path.isNull())
            {
                QMessageBox folderinvalid(QMessageBox::Warning, "File type unsupported", "Please select a file that is a supported data type.", QMessageBox::Ok, this);
                folderinvalid.setWindowFlags(Qt::Dialog | Qt::CustomizeWindowHint | Qt::WindowTitleHint | Qt::WindowCloseButtonHint);
                folderinvalid.exec();
            }
        }
        catch(...)
        {
            ;
        }
}

void MainWindow::on_actionViolin_Solo_Bach_Sonata_2_mvt_4_triggered()
{
    try
        {
            //checks to see if the C:/ProgramData/Virtuoso/metadata folder exists, and if not makes it
            QString import_path = "./demos/vp2-4gig.mid";
            QDir dir("C:/ProgramData/Virtuoso/metadata");
            if (!dir.exists())
                dir.mkpath(".");

            //checks to see if the selected file is valid
            bool file_valid = false;
            QList<QString> list = {".mid"};
            if (hasEnding(import_path, list)){
                file_valid = true;
            }

            //If the file is valid, checks if info file exists, and if so delete it, then remake it (to only have a single path), then write the path out
            if (file_valid == true)
            {
                ui->midiEditorStackedWidget->setCurrentIndex(1);
                import_images(import_path);
            }
            else if (!import_path.isNull())
            {
                QMessageBox folderinvalid(QMessageBox::Warning, "File type unsupported", "Please select a file that is a supported data type.", QMessageBox::Ok, this);
                folderinvalid.setWindowFlags(Qt::Dialog | Qt::CustomizeWindowHint | Qt::WindowTitleHint | Qt::WindowCloseButtonHint);
                folderinvalid.exec();
            }
        }
        catch(...)
        {
            ;
        }
}


void MainWindow::on_checkBox_2_toggled(bool checked)
{
    if (checked)reverb = "true";
    else reverb = "false";
}
