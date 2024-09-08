# -*- coding: utf-8 -*-
import sys
import os
import pandas as pd
import numpy as np
from enum import Enum
import json

import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from PyQt5.QtWidgets import QApplication, QStyle, QWidget, QVBoxLayout, QHBoxLayout, QFileDialog, QSizePolicy, QLabel, \
        QPushButton, QRubberBand, QFileSystemModel, QTreeView, QSlider, QComboBox, QMessageBox, QTableWidget, QTableWidgetItem, \
        QGroupBox, QGridLayout, QCheckBox
from PyQt5.QtMultimedia import QMediaPlayer, QMediaContent
from PyQt5.QtMultimediaWidgets import QVideoWidget
from PyQt5.QtCore import Qt, QUrl, pyqtSignal, QTimer, QSize, QRect, QDir, QModelIndex
from PyQt5.QtGui import QFont
from moviepy.editor import AudioFileClip


class State(Enum):
    IDLE = 1
    RUNNING = 2
    PAUSED = 3
    STOPPED = 4


def show_message(message, closeFlg=True):
    msg = QMessageBox()
    msg.setIcon(QMessageBox.Icon.Information)
    msg.setText(message)
    msg.setWindowTitle("提示")
    if closeFlg:
        QTimer.singleShot(1000, msg.close)
    msg.exec_()


class Console(QWidget):
    '''
    控制台
    '''

    def __init__(self, parent=None):
        super().__init__(parent)
        self.treetogle = QPushButton(self, text="显示目录")
        self.treetogle.setIcon(QApplication.style().standardIcon(QStyle.StandardPixmap.SP_ArrowLeft))

        self.openButton = QPushButton(self, text="打开文件")
        self.openButton.setIcon(QApplication.style().standardIcon(QStyle.StandardPixmap.SP_DirHomeIcon))

        self.stopbutton = QPushButton(self, text="停止")
        self.stopbutton.setIcon(QApplication.style().standardIcon(QStyle.StandardPixmap.SP_MediaStop))

        self.playButton = QPushButton(self, text="播放")
        self.playButton.setIcon(QApplication.style().standardIcon(QStyle.StandardPixmap.SP_MediaPlay))

        self.preButton = QPushButton(self, text="上一个")
        self.preButton.setIcon(QApplication.style().standardIcon(QStyle.StandardPixmap.SP_MediaSkipBackward))

        self.nextButton = QPushButton(self, text="下一个")
        self.nextButton.setIcon(QApplication.style().standardIcon(QStyle.StandardPixmap.SP_MediaSkipForward))

        self.backButton = QPushButton(self, text="后退1秒(30帧)")
        self.quickButton = QPushButton(self, text="快进10帧")

        self.checkbox = QCheckBox("删除模式")

        horizontalLayout = QHBoxLayout()
        horizontalLayout.addWidget(self.treetogle)
        horizontalLayout.addWidget(self.openButton)
        horizontalLayout.addWidget(self.stopbutton)
        horizontalLayout.addWidget(self.playButton)
        horizontalLayout.addWidget(self.preButton)
        horizontalLayout.addWidget(self.nextButton)
        horizontalLayout.addWidget(self.backButton)
        horizontalLayout.addWidget(self.quickButton)
        horizontalLayout.addWidget(self.checkbox)

        self.contrlSlider = QSlider(self)
        self.contrlSlider.setOrientation(Qt.Orientation.Horizontal)
        self.contrlSlider.setValue(0)

        self.time_label = QLabel()
        self.time_label.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Maximum)
        self.time_label.setText("00:00:00:000 ms")

        slider_layout = QHBoxLayout()
        slider_layout.addWidget(self.contrlSlider)
        slider_layout.addWidget(self.time_label)

        self.get_positon_button = QPushButton(self, text="设置当前时间")
        self.swallow_record = QPushButton(self, text="吞咽")
        self.detail_swallow1 = QPushButton(self, text="口腔期")
        self.detail_swallow2 = QPushButton(self, text="咽期")
        self.detail_swallow3 = QPushButton(self, text="食管期")
        self.detail_swallow4 = QPushButton(self, text="吞咽前")
        self.detail_swallow5 = QPushButton(self, text="吞咽暂停")
        self.detail_swallow6 = QPushButton(self, text="吞咽后")

        self.segment_label = QLabel()
        self.segment_label.setText("00:00:00:000 - 00:00:00:000 ms")
        self.segment_pre_time = 0
        self.segment_aft_time = 0

        self.segment_groupBox = QGroupBox("吞咽分割")
        segment_layout = QGridLayout()
        segment_layout.addWidget(self.swallow_record, 0, 0)
        segment_layout.addWidget(self.get_positon_button, 0, 1)
        segment_layout.addWidget(self.segment_label, 0, 2)
        segment_layout.addWidget(self.detail_swallow1, 1, 0)
        segment_layout.addWidget(self.detail_swallow2, 1, 1)
        segment_layout.addWidget(self.detail_swallow3, 1, 2)
        segment_layout.addWidget(self.detail_swallow4, 2, 0)
        segment_layout.addWidget(self.detail_swallow5, 2, 1)
        segment_layout.addWidget(self.detail_swallow6, 2, 2)
        self.segment_groupBox.setLayout(segment_layout)

        self.selected_model_label = QLabel("选择模型")
        self.model_comboBox = QComboBox()
        self.model_comboBox.addItem("CFSCNet")
        self.model_comboBox.addItem("Detail Model")
        self.model_segment_button = QPushButton(self, text="模型吞咽分割")
        self.detail_segment_button = QPushButton(self, text="模型细分割")
        self.model_groupBox = QGroupBox("模型分割")

        model_layout = QVBoxLayout()
        h_layout = QHBoxLayout()
        h_layout.addWidget(self.selected_model_label, 1)
        h_layout.addWidget(self.model_comboBox, 4)
        model_layout.addLayout(h_layout)
        model_layout.addWidget(self.model_segment_button)
        model_layout.addWidget(self.detail_segment_button)
        self.model_groupBox.setLayout(model_layout)

        verti_layout = QVBoxLayout()
        verti_layout.addLayout(slider_layout)
        verti_layout.addLayout(horizontalLayout)
        h_layout = QHBoxLayout()
        h_layout.addWidget(self.model_groupBox, 1)
        h_layout.addWidget(self.segment_groupBox, 4)
        verti_layout.addLayout(h_layout)
        self.setLayout(verti_layout)

    def set_icon(self, icon_type):
        self.playButton.setIcon(QApplication.style().standardIcon(icon_type))

    def set_slider_duration(self, duration):
        self.contrlSlider.setRange(0, duration)

    def set_cur_time(self, postion):
        self.segment_pre_time = self.segment_aft_time
        self.segment_aft_time = postion
        self.segment_label.setText(f"{format_time(self.segment_pre_time)} - {format_time(self.segment_aft_time)} ms")


class TreeWidget(QWidget):
    '''
    目录控件
    '''
    select_file = pyqtSignal(str)

    def __init__(self, parent=None):
        super().__init__(parent)

        self.treeview = QTreeView(self)
        self.model = QFileSystemModel(self)
        self.model.setRootPath(QDir.currentPath() + "/..")
        self.model.setNameFilterDisables(False)
        self.model.setNameFilters(["*.csv", "*.mp4", "*.wav", "*.avi", "*.json"])
        self.treeview.setModel(self.model)
        self.treeview.setColumnHidden(1, True)
        self.treeview.setColumnHidden(2, True)
        self.treeview.setColumnHidden(3, True)
        self.treeview.setRootIndex(self.model.index(self.model.rootPath()))
        self.treeview.clicked.connect(self.treeview_clicked)
        layout = QVBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.treeview)
        self.setLayout(layout)

    def treeview_clicked(self, index: QModelIndex):
        '''
        选择文件
        '''
        file_path = self.model.filePath(index)
        self.select_file.emit(file_path)

    def tree_open(self):
        '''
        打开文件夹
        '''
        fileroot = QFileDialog.getExistingDirectory(self, 'open')
        self.model.setRootPath(fileroot)
        self.treeview.setRootIndex(self.model.index(fileroot))
        self.folders = []
        for foldername, subfolders, filenames in os.walk(fileroot):
            if "Annotated.json" in filenames:
                self.folders.append(foldername)
        self.current_fold_idx = -1
        self.next()

    def next(self):
        if self.current_fold_idx < len(self.folders) - 1:
            try:
                self.current_fold_idx += 1
                dir_path = self.folders[self.current_fold_idx]
                entries = os.listdir(dir_path)
                for entry in entries:
                    self.select_file.emit(os.path.join(dir_path, entry))
                show_message("已切换下一个病人")
            except Exception as e:
                show_message(f"切换失败: {e}", closeFlg=False)

    def pre(self):
        if self.current_fold_idx > 0:
            try:
                self.current_fold_idx -= 1
                dir_path = self.folders[self.current_fold_idx]
                entries = os.listdir(dir_path)
                for entry in entries:
                    self.select_file.emit(os.path.join(dir_path, entry))
                show_message("已切换上一个病人")
            except Exception as e:
                show_message(f"切换失败: {e}", closeFlg=False)


def extract_process_folder(filepath):
    process_index = filepath.find("data")

    if process_index != -1:
        process_path = filepath[process_index + 4:]
        return process_path
    else:
        return None


class MplCanvas(FigureCanvas):
    '''
    基础绘图类
    '''
    signal_select = pyqtSignal(tuple)

    def __init__(self, length, parent=None, width=5, height=4, dpi=100, zero_line=False, color='green', lines=1):
        fig = Figure(figsize=(width, height), dpi=dpi)
        self.axes = fig.add_subplot(111)
        self.axes.set_facecolor('white')
        super(MplCanvas, self).__init__(fig)
        fig.tight_layout()
        self.axes.xaxis.set_visible(False)
        self.axes.set_position([0, 0, 1, 1])
        self.axes.set_xlim(0, length)
        self.axes.tick_params(axis='both', which='both', bottom=False, top=False, left=False, right=False)
        fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
        self.axes.yaxis.grid(True, linestyle='--', color='gray')
        if zero_line:
            self.axes.yaxis.set_ticks([0])
            self.axes.yaxis.grid(True, linestyle='-', color='black')
        if lines == 1:
            plot_refs = self.axes.plot(np.zeros(length), color=color)[0]
        else:
            colors = ['blue', 'green', 'red']
            plot_refs = [self.axes.plot(np.zeros(length), color=colors[i])[0] for i in range(lines)]
        self.plot = plot_refs

        # 设置rcParams参数
        plt.rcParams['agg.path.chunksize'] = 1000
        plt.rcParams['figure.dpi'] = 80
        plt.rcParams['figure.figsize'] = [6, 4]
        plt.rcParams['savefig.dpi'] = 100

        self.rubberBand = QRubberBand(QRubberBand.Shape.Rectangle, self)
        self.rubberBand.hide()
        self.origin = None

    '''
    鼠标框选函数
    '''
    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self.rubberBand.raise_()
            self.origin = event.pos()
            self.rubberBand.setGeometry(QRect(self.origin, QSize()))
            self.rubberBand.show()

    def mouseMoveEvent(self, event):
        if self.origin:
            self.rubberBand.setGeometry(QRect(self.origin, event.pos()).normalized())

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            selected_area = self.rubberBand.geometry()
            parent_width = self.width()
            left_distance = parent_width - selected_area.left()
            right_distance = parent_width - selected_area.right()
            left_percent = min(100, (left_distance / parent_width) * 100)
            right_percent = max(0, (right_distance / parent_width) * 100)
            self.signal_select.emit((left_percent, right_percent))
            self.rubberBand.hide()
            self.origin = None


class DynamicPlot_Audio(QWidget):
    '''
    音频类
    '''

    def __init__(self):
        super().__init__()
        self.audio_player = QMediaPlayer()
        self.buffer = []
        self.interval = 15
        self.samplerate = 16000
        self.window_length = 3000
        self.max_value, self.min_value = 0, 0
        self.length = int(self.window_length * self.samplerate / 1000)
        self.data = None
        self.dir_path, self.file_name = None, None
        self.plot_data = None
        self.window_sample = int(self.interval / 1000 * self.samplerate)
        self.raise_()
        layout = QVBoxLayout()
        self.canvas = MplCanvas(parent=self, length=self.length, color='blue', width=5, height=4, dpi=30)
        self.setMouseTracking(True)
        layout.addWidget(self.canvas)
        self.setLayout(layout)
        self.canvas.signal_select.connect(self.update_win)

    def setData(self, filepath):
        '''
        设置数据路径
        '''
        self.buffer = []
        self.stop()
        self.dir_path, self.file_name = os.path.split(filepath)
        self.root_path = extract_process_folder(self.dir_path)
        audio_clip = AudioFileClip(filepath, fps=self.samplerate)
        self.audio_duration = audio_clip.duration
        self.data = audio_clip.to_soundarray()[:, 0]
        media_content = QMediaContent(QUrl.fromLocalFile(filepath))
        self.audio_player.setMedia(media_content)
        self.dataGenerator()

    def dataGenerator(self):
        '''
        抽取绘图数据
        '''
        gen_plotdata = np.zeros(self.length)
        padding_length = (self.window_sample - len(self.data) % self.window_sample) % self.window_sample
        padded_data = np.pad(self.data, (0, padding_length), 'constant', constant_values=0)
        segments = np.array_split(padded_data, len(padded_data) // self.window_sample)
        buffer = []
        shift = self.window_sample
        self.totalIndex = len(segments) - 1
        for segment in segments:
            gen_plotdata = np.roll(gen_plotdata, -shift, axis=0)
            gen_plotdata[-shift:] = segment
            min_value = np.min(gen_plotdata)
            max_value = np.max(gen_plotdata)
            segment_with_minmax = np.concatenate(([min_value, max_value], gen_plotdata))
            buffer.append(segment_with_minmax)
        self.buffer = np.array(buffer)

    def play(self):
        if self.audio_player.state() == QMediaPlayer.State.PlayingState:
            self.audio_player.pause()
        else:
            self.audio_player.play()

    def stop(self):
        self.audio_player.stop()
        self.min_value, self.max_value = 0, 0

    def update_win(self, events):
        '''
        重新绘制窗口
        '''
        if self.plot_data is not None:
            for patch in self.canvas.axes.patches:
                patch.remove()
            data_lenth = len(self.plot_data)
            left = data_lenth * (1 - events[0] / 100)
            right = data_lenth * (1 - events[1] / 100)
            self.canvas.axes.axvspan(left, right, color='lightcoral', alpha=0.5)
            self.canvas.plot.set_ydata(self.plot_data)
            self.canvas.axes.set_ylim(self.plot_ymin, self.plot_ymax)
            self.canvas.draw()

    def update(self, position):
        '''
        更新展示数据
        '''
        current_index = position // self.interval
        if current_index > self.totalIndex:
            current_index = self.totalIndex
        self.plot_data = self.buffer[current_index][2:]
        self.plot_ymin, self.plot_ymax = self.buffer[current_index][0], self.buffer[current_index][1]
        self.canvas.plot.set_ydata(self.plot_data)
        self.canvas.axes.set_ylim(self.plot_ymin, self.plot_ymax)
        self.canvas.draw()

    def setPosition(self, position):
        self.audio_player.setPosition(position)


class DynamicPlot_Gas(QWidget):
    '''
    气流量类
    '''
    def __init__(self):
        super().__init__()
        self.setMouseTracking(True)
        self.buffer = []
        self.interval = 15
        self.samplerate = 100
        self.window_length = 3000
        self.max_value, self.min_value = 0, 0
        self.length = int(self.window_length * self.samplerate / 1000)
        self.canvas = MplCanvas(length=self.length,
                                parent=self, width=5,
                                height=4,
                                dpi=80,
                                zero_line=True,
                                color='green')
        self.data = None
        self.plot_data = None
        self.window_sample = int(self.interval / 1000 * self.samplerate)
        layout = QVBoxLayout()
        layout.addWidget(self.canvas)
        self.setLayout(layout)

    def update_win(self, events):
        '''
        重绘框选区域
        '''
        if self.plot_data is not None:
            for patch in self.canvas.axes.patches:
                patch.remove()
            data_lenth = len(self.plot_data)
            left = data_lenth * (1 - events[0] / 100)
            right = data_lenth * (1 - events[1] / 100)
            self.canvas.axes.axvspan(left, right, color='lightcoral', alpha=0.5)
            self.canvas.plot.set_ydata(self.plot_data)
            self.canvas.axes.set_ylim(self.plot_ymin, self.plot_ymax)
            self.canvas.draw()

    def setData(self, filepath):
        self.buffer = []
        self.stop()
        self.filepath = filepath
        self.data = pd.read_csv(filepath)
        self.dataGenerator()

    def dataGenerator(self):
        '''
        生成展示数据
        '''
        gen_plotdata = np.zeros(self.length)
        time_start = self.data['time'].iloc[0]  # 起始时间值

        def split_data(group):
            group_values = group['value'].values
            return group_values

        segments = self.data.groupby((self.data['time'] - time_start) // self.interval).apply(split_data)
        buffer = []
        self.totalIndex = len(segments) - 1
        for segment in segments:
            shift = len(segment)
            gen_plotdata = np.roll(gen_plotdata, -shift, axis=0)
            gen_plotdata[-shift:] = segment
            min_value = np.min(gen_plotdata)
            max_value = np.max(gen_plotdata)
            segment_with_minmax = np.concatenate(([min_value, max_value], gen_plotdata))
            buffer.append(segment_with_minmax)
        self.buffer = np.array(buffer)

    def stop(self):
        self.min_value, self.max_value = 0, 0

    def update(self, position):
        current_index = position // self.interval
        if current_index > self.totalIndex:
            current_index = self.totalIndex
        self.plot_data = self.buffer[current_index][2:]
        self.plot_ymin, self.plot_ymax = self.buffer[current_index][0], self.buffer[current_index][1]
        self.canvas.plot.set_ydata(self.plot_data)
        self.canvas.axes.set_ylim(self.plot_ymin, self.plot_ymax)
        self.canvas.draw()


class DynamicPlot_Imu(QWidget):
    '''
    三轴数据类
    '''
    def __init__(self):
        super().__init__()
        self.setMouseTracking(True)
        self.buffer = []
        self.interval = 15
        self.samplerate = 1000
        self.window_length = 3000
        self.max_value, self.min_value = 0, 0
        self.length = int(self.window_length * self.samplerate / 1000)
        self.canvas = MplCanvas(length=self.length, parent=self, width=5, height=4, dpi=20, color='red', lines=3)
        self.data = None
        self.plot_data = None
        self.window_sample = int(self.interval / 1000 * self.samplerate)
        layout = QVBoxLayout()
        layout.addWidget(self.canvas)
        self.setLayout(layout)

    def update_win(self, events):
        if self.plot_data is not None:
            min_value, max_value, samplex, sampley, samplez = self.plot_data
            for patch in self.canvas.axes.patches:
                patch.remove()
            data_lenth = len(samplex)
            left = data_lenth * (1 - events[0] / 100)
            right = data_lenth * (1 - events[1] / 100)
            self.canvas.axes.axvspan(left, right, color='lightcoral', alpha=0.5)
            self.canvas.plot[0].set_ydata(samplex)
            self.canvas.plot[1].set_ydata(sampley)
            self.canvas.plot[2].set_ydata(samplez)
            self.canvas.axes.set_ylim(min_value, max_value)
            self.canvas.draw()

    def setData(self, filepath):
        self.stop()
        self.buffer = []
        self.filepath = filepath
        self.data = pd.read_csv(filepath)
        self.dataGenerator()

    def dataGenerator(self):
        gen_plotdatax = np.zeros(self.length)
        gen_plotdatay = np.zeros(self.length)
        gen_plotdataz = np.zeros(self.length)
        time_start = self.data['time'].iloc[0]

        def split_data(group):
            group_values_x = group['X'].values
            group_values_y = group['Y'].values
            group_values_z = group['Z'].values
            return (group_values_x, group_values_y, group_values_z)

        segments = self.data.groupby((self.data['time'] - time_start) // self.interval).apply(split_data)
        self.totalIndex = len(segments) - 1
        for segment_x, segment_y, segment_z in segments:
            shift = len(segment_x)
            gen_plotdatax = np.roll(gen_plotdatax, -shift, axis=0)
            gen_plotdatay = np.roll(gen_plotdatay, -shift, axis=0)
            gen_plotdataz = np.roll(gen_plotdataz, -shift, axis=0)
            gen_plotdatax[-shift:] = segment_x
            gen_plotdatay[-shift:] = segment_y
            gen_plotdataz[-shift:] = segment_z
            min_value = min(np.min(gen_plotdatax), np.min(gen_plotdatay), np.min(gen_plotdataz))
            max_value = max(np.max(gen_plotdatax), np.max(gen_plotdatay), np.max(gen_plotdataz))
            self.buffer.append((min_value, max_value, gen_plotdatax, gen_plotdatay, gen_plotdataz))

    def stop(self):
        self.min_value, self.max_value = 0, 0

    def update(self, position):
        current_index = position // self.interval
        if current_index > self.totalIndex:
            current_index = self.totalIndex
        self.plot_data = self.buffer[current_index]
        min_value, max_value, samplex, sampley, samplez = self.plot_data
        self.canvas.plot[0].set_ydata(samplex)
        self.canvas.plot[1].set_ydata(sampley)
        self.canvas.plot[2].set_ydata(samplez)
        self.canvas.axes.set_ylim(min_value, max_value)
        self.canvas.draw()


class DynamicPlot(QWidget):
    '''
    多种数据绘制窗口
    '''
    def __init__(self):
        super().__init__()
        self.interval = 15
        self.imuPlot = DynamicPlot_Imu()
        self.gasPlot = DynamicPlot_Gas()
        self.audioPlot = DynamicPlot_Audio()
        self.label1 = QLabel()
        font = QFont()
        font.setPointSize(12)  # 设置字体大小
        self.label1.setFont(font)
        self.label1.setSizePolicy(QSizePolicy.Policy.Preferred,
                                  QSizePolicy.Policy.Maximum)
        self.label1.setStyleSheet("color: green;")
        self.label1.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.label1.setText('Time: s')
        self.label2 = QLabel()
        font = QFont()
        font.setPointSize(16)  # 设置字体大小
        self.label2.setFont(font)
        self.label2.setSizePolicy(QSizePolicy.Policy.Preferred,
                                  QSizePolicy.Policy.Maximum)
        self.label2.setStyleSheet("color: green;")
        self.label2.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.label2.setText(' - ')
        sub_layout1 = QHBoxLayout()
        sub_layout1.setContentsMargins(0, 0, 0, 0)
        sub_layout1.addWidget(self.label1)
        sub_layout1.addWidget(self.label2)
        self.label3 = QLabel()
        font = QFont()
        font.setPointSize(12)  # 设置字体大小
        self.label3.setFont(font)
        self.label3.setStyleSheet("color: black;")
        self.label3.setAlignment(Qt.AlignmentFlag.AlignTrailing)
        self.label3.setText('PAS:')
        self.combo_box = QComboBox(self)
        font_size = 25
        self.combo_box.setStyleSheet(f"font-size: {font_size}px;")
        self.combo_box.addItem('1')
        self.combo_box.addItem('2')
        self.combo_box.addItem('3')
        self.combo_box.addItem('4')
        self.combo_box.addItem('5')
        self.combo_box.addItem('6')
        self.combo_box.addItem('7')
        self.combo_box.addItem('8')
        self.select_button = QPushButton('Seg')
        self.select_button.setStyleSheet(
            "QPushButton:pressed { background-color: red; border-style: inset; }")
        self.delet_button = QPushButton('Del')
        self.delet_button.setStyleSheet(
            "QPushButton:pressed { background-color: red; border-style: inset; }")
        sub_layout2 = QHBoxLayout()
        sub_layout2.setContentsMargins(0, 0, 0, 0)
        sub_layout2.addWidget(self.label3)
        sub_layout2.addWidget(self.combo_box)
        sub_layout2.addWidget(self.select_button)
        sub_layout2.addWidget(self.delet_button)
        # 布局
        layout = QVBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.imuPlot, 1)
        layout.addWidget(self.gasPlot, 1)
        layout.addWidget(self.audioPlot, 1)
        layout.addLayout(sub_layout1)
        layout.addLayout(sub_layout2)
        self.setLayout(layout)

        self.timer = QTimer()
        self.timer.timeout.connect(self.update)
        self.audioPlot.stopSignal.connect(self.stop)
        self.audioPlot.canvas.signal_select.connect(self.audioPlot.update_win)
        self.audioPlot.canvas.signal_select.connect(self.imuPlot.update_win)
        self.audioPlot.canvas.signal_select.connect(self.gasPlot.update_win)

    def setPosition(self, position):
        self.audioPlot.setPosition(position)

    def play(self, state):
        self.audioPlot.play()
        self.gasPlot.play()
        self.imuPlot.play()
        if state:
            self.timer.start(self.interval)
        else:
            self.timer.stop()

    def update(self):
        positon = self.audioPlot.audio_player.position()
        self.audioPlot.update(positon)
        self.imuPlot.update(positon)
        self.gasPlot.update(positon)

    def stop(self):
        self.timer.stop()
        self.audioPlot.stop()
        self.imuPlot.stop()
        self.gasPlot.stop()

    '''
    设置文件路径
    '''
    def setCsv(self, file_path):
        if 'wav' in file_path:
            self.audioPlot.setData(file_path)
        elif 'gas' in file_path:
            self.gasPlot.setData(file_path)
        elif 'imu' in file_path:
            self.imuPlot.setData(file_path)


class VideoWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.mediaPlayer = QMediaPlayer(None, QMediaPlayer.Flag.VideoSurface)
        self.videoWidget = QVideoWidget()
        self.videoWidget.setStyleSheet("background-color: gray;")
        sizePolicy_v = self.videoWidget.sizePolicy()
        sizePolicy_v.setHorizontalStretch(0)
        sizePolicy_v.setVerticalStretch(0)
        sizePolicy_v.setHeightForWidth(self.videoWidget.sizePolicy().hasHeightForWidth())
        self.videoWidget.setSizePolicy(sizePolicy_v)
        self.mediaPlayer.setVideoOutput(self.videoWidget)
        layout = QVBoxLayout()
        layout.addWidget(self.videoWidget)
        self.setLayout(layout)

    def setMedia(self, filepath):
        if self.mediaPlayer.state() == QMediaPlayer.State.PlayingState:
            self.mediaPlayer.stop()
        media = QMediaContent(QUrl.fromLocalFile(filepath))
        self.mediaPlayer.setMedia(media)


class DataTableWidget(QWidget):
    def __init__(self):
        super().__init__()
        self.initUI()

    def initUI(self):
        layout = QVBoxLayout()

        # 创建表格控件
        self.table_widget = QTableWidget()
        self.table_widget.setRowCount(0)  # 设置行数
        self.table_widget.setColumnCount(7)  # 设置列数
        self.column_name = ['吞咽', '口腔期', '咽期', '食管期', "吞咽前", "吞咽暂停", "吞咽后"]
        self.table_widget.setHorizontalHeaderLabels(self.column_name)

        layout.addWidget(self.table_widget)
        self.setLayout(layout)
        self.json_path = None

    def refresh_data(self):
        max_row_count = 0
        for swallow_type in self.json_dict:
            length = len(self.json_dict[swallow_type])
            if length > max_row_count:
                max_row_count = length
        self.table_widget.setRowCount(max_row_count)

        for row in range(self.table_widget.rowCount()):
            for column, name in enumerate(self.column_name):
                if row < len(self.json_dict[name]):
                    item = QTableWidgetItem(f'{self.json_dict[name][row]["start"]}s - {self.json_dict[name][row]["end"]}s')
                    item.setTextAlignment(Qt.AlignmentFlag.AlignHCenter)
                    self.table_widget.setItem(row, column, item)
                else:
                    self.table_widget.setItem(row, column, QTableWidgetItem(''))

    def down_load_json_table(self):
        with open(self.json_path, 'w', encoding='utf-8') as json_file:
            json_file.write(json.dumps(self.json_dict, indent=4, ensure_ascii=False))

    def set_json_table(self, file_path):
        with open(file_path, 'r', encoding='utf-8') as json_file:
            self.json_dict = json.load(json_file)
        self.json_path = file_path
        self.refresh_data()

    def add_content(self, column, start, end, checked):
        if checked:
            self.json_dict[self.column_name[column]].pop()
        else:
            start = start / 1000
            end = end / 1000
            self.json_dict[self.column_name[column]].append({'start': round(start, 3), 'end': round(end, 3)})
        self.refresh_data()
        self.down_load_json_table()


class Mainwindows(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("吞咽分割检测系统")
        self.setWindowIcon(QApplication.style().standardIcon(QStyle.StandardPixmap.SP_MediaPlay))
        self.init_ui()
        self.showMaximized()

    def init_ui(self):
        '''
        初始化控件
        '''
        self.current_state = State.IDLE
        self.video_groupBox = QGroupBox("视频造影")
        self.data_groupBox = QGroupBox("数据可视化")
        self.console_groupBox = QGroupBox("控制台")
        self.table_groupBox = QGroupBox("分割结果")
        self.videoCT = VideoWidget()
        self.videoWin = VideoWidget()
        self.imu_win = DynamicPlot_Imu()
        self.gas_win = DynamicPlot_Gas()
        self.audio_win = DynamicPlot_Audio()
        self.console = Console()
        self.treeConsole = TreeWidget()
        self.treeConsole.setVisible(False)
        self.json_table = DataTableWidget()
        self.data_show_timer = QTimer(self)
        self.data_show_timer.setInterval(30)
        self.ui_timer = QTimer(self)
        self.ui_timer.setInterval(150)
        self.forward_timer = QTimer()
        self.forward_timer.setSingleShot(True)  # 设置为只触发一次

        vedio_layout = QHBoxLayout()
        vedio_layout.addWidget(self.videoWin, 2)
        vedio_layout.addWidget(self.videoCT, 2)
        self.video_groupBox.setLayout(vedio_layout)

        data_layout = QVBoxLayout()
        data_layout.addWidget(self.imu_win, 1)
        data_layout.addWidget(self.gas_win, 1)
        data_layout.addWidget(self.audio_win, 1)
        self.data_groupBox.setLayout(data_layout)

        view_layout = QHBoxLayout()
        view_layout.addWidget(self.video_groupBox, 4)
        view_layout.addWidget(self.data_groupBox, 3)

        self.console_groupBox.setLayout(self.console.layout())
        self.table_groupBox.setLayout(self.json_table.layout())
        console_layout = QHBoxLayout()
        console_layout.addWidget(self.console_groupBox, 4)
        console_layout.addWidget(self.table_groupBox, 3)

        main_layout = QVBoxLayout()
        main_layout.addLayout(view_layout, 2)
        main_layout.addLayout(console_layout, 1)

        tree_layout = QHBoxLayout()
        tree_layout.addWidget(self.treeConsole, 2)
        tree_layout.addLayout(main_layout, 9)
        self.setLayout(tree_layout)

        # 播放
        self.console.playButton.clicked.connect(self.play)

        # 播放结束
        self.console.stopbutton.clicked.connect(self.stop)

        # 拖动位置
        self.console.contrlSlider.sliderMoved.connect(self.sliderPosition)

        # 播放时间改变
        self.audio_win.audio_player.durationChanged.connect(self.console.set_slider_duration)

        # 播放结束
        self.audio_win.audio_player.stateChanged.connect(self.check_status)

        # 打开目录
        self.console.openButton.clicked.connect(self.treeConsole.tree_open)
        self.console.nextButton.clicked.connect(self.treeConsole.next)
        self.console.preButton.clicked.connect(self.treeConsole.pre)

        # 快进10帧
        self.console.quickButton.clicked.connect(self.fast_forward)
        self.forward_timer.timeout.connect(self.forward_timerout)

        # 后退30帧
        self.console.backButton.clicked.connect(self.backward)

        # 滑块改变
        self.console.contrlSlider.sliderMoved.connect(self.sliderPosition)

        # 树目录设置选择文件
        self.console.treetogle.clicked.connect(self.treetogle)
        self.treeConsole.select_file.connect(self.set_source)

        # 设置当前时间
        self.console.get_positon_button.clicked.connect(self.get_cur_postion)

        # 定时更新
        self.data_show_timer.timeout.connect(self.update)
        self.ui_timer.timeout.connect(self.updata_ui)

        # 更新表格内容
        self.console.swallow_record.clicked.connect(
            lambda: self.json_table.add_content(0,
                                                self.console.segment_pre_time,
                                                self.console.segment_aft_time,
                                                self.console.checkbox.isChecked()))
        self.console.detail_swallow1.clicked.connect(
            lambda: self.json_table.add_content(1,
                                                self.console.segment_pre_time,
                                                self.console.segment_aft_time,
                                                self.console.checkbox.isChecked()))
        self.console.detail_swallow2.clicked.connect(
            lambda: self.json_table.add_content(2,
                                                self.console.segment_pre_time,
                                                self.console.segment_aft_time,
                                                self.console.checkbox.isChecked()))
        self.console.detail_swallow3.clicked.connect(
            lambda: self.json_table.add_content(3,
                                                self.console.segment_pre_time,
                                                self.console.segment_aft_time,
                                                self.console.checkbox.isChecked()))
        self.console.detail_swallow4.clicked.connect(
            lambda: self.json_table.add_content(4,
                                                self.console.segment_pre_time,
                                                self.console.segment_aft_time,
                                                self.console.checkbox.isChecked()))
        self.console.detail_swallow5.clicked.connect(
            lambda: self.json_table.add_content(5,
                                                self.console.segment_pre_time,
                                                self.console.segment_aft_time,
                                                self.console.checkbox.isChecked()))
        self.console.detail_swallow6.clicked.connect(
            lambda: self.json_table.add_content(6,
                                                self.console.segment_pre_time,
                                                self.console.segment_aft_time,
                                                self.console.checkbox.isChecked()))

    def get_cur_postion(self):
        positon = self.audio_win.audio_player.position()
        self.console.set_cur_time(positon)

    def check_status(self, state):
        if state == QMediaPlayer.State.StoppedState:
            self.stop()

    def update(self):
        positon = self.audio_win.audio_player.position()
        self.audio_win.update(positon)
        self.imu_win.update(positon)
        self.gas_win.update(positon)
        self.console.time_label.setText(f"{format_time(positon)} ms")

    def updata_ui(self):
        position = self.audio_win.audio_player.position()
        self.console.contrlSlider.setValue(position)

    def set_source(self, file_path):
        file_name, file_extension = os.path.splitext(os.path.basename(file_path))
        if file_extension == '.wav':
            self.audio_win.setData(file_path)
        elif file_extension == '.mp4' or file_extension == '.avi':
            if 'ct' in file_name:
                self.videoCT.setMedia(file_path)
            else:
                self.videoWin.setMedia(file_path)
        elif file_extension == '.csv':
            if 'imu' in file_name:
                self.imu_win.setData(file_path)
            else:
                self.gas_win.setData(file_path)
        elif file_extension == '.json':
            self.json_table.set_json_table(file_path)
        else:
            raise Exception("type error")

    def sliderPosition(self, position):
        self.videoCT.mediaPlayer.setPosition(position)
        self.videoWin.mediaPlayer.setPosition(position)
        self.audio_win.audio_player.setPosition(position)

    def treetogle(self):
        self.treeConsole.setVisible(not self.treeConsole.isVisible())

    def fast_forward(self):
        if self.current_state == State.RUNNING:
            self.play()  # 先暂停
        self.play()
        self.forward_timer.start(33)  # 33ms后触发

    def backward(self):
        position = self.audio_win.audio_player.position()
        self.console.contrlSlider.setValue(position - 1000)
        self.sliderPosition(position - 1000)

    def forward_timerout(self):
        self.play()  # 先暂停

    def play(self):
        if self.current_state == State.RUNNING:
            self.videoCT.mediaPlayer.pause()
            self.videoWin.mediaPlayer.pause()
            self.audio_win.audio_player.pause()
            self.data_show_timer.stop()
            self.ui_timer.stop()
            self.current_state = State.PAUSED
            self.console.set_icon(QStyle.StandardPixmap.SP_MediaPause)
        else:
            self.videoCT.mediaPlayer.play()
            self.videoWin.mediaPlayer.play()
            self.audio_win.audio_player.play()
            self.data_show_timer.start()
            self.ui_timer.start()
            self.current_state = State.RUNNING
            self.console.set_icon(QStyle.StandardPixmap.SP_MediaPlay)

    def stop(self):
        self.videoCT.mediaPlayer.stop()
        self.videoWin.mediaPlayer.stop()
        self.imu_win.stop()
        self.gas_win.stop()
        self.audio_win.stop()
        self.data_show_timer.stop()
        self.ui_timer.stop()
        self.current_state = State.STOPPED


def format_time(pos):
    total_seconds, ms = divmod(pos, 1000)  # 将时间转换为总秒数
    minutes, seconds = divmod(total_seconds, 60)  # 分钟和秒钟的整除和余数
    hours, minutes = divmod(minutes, 60)  # 小时和分钟的整除和余数

    # 格式化为字符串
    time_string = "{:02d}:{:02d}:{:02d}:{:03d}".format(hours, minutes, seconds, ms)
    return time_string


def run_application():
    '''
    主程序入口
    '''
    app = QApplication(sys.argv)
    window = Mainwindows()
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    run_application()
