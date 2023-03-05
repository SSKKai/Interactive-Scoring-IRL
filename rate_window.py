from rate_trajectory import Ui_MainWindow
import sys
from PyQt5.QtWidgets import QApplication,QMainWindow
from PyQt5.QtMultimedia import QMediaContent, QMediaPlayer, QMediaPlaylist
from PyQt5.QtCore import QDir, QUrl
from PyQt5.QtTest import QTest
import os

class RatingWindow(QMainWindow, Ui_MainWindow):
    def __init__(self, parent=None):
        super(RatingWindow,self).__init__(parent)
        self.setupUi(self)

        self.fileRoot = os.path.dirname(__file__)+'/videos/'
        self.fileType = '.avi'
        self.video_num = 0
        self.referenceVideoWidgets = [self.rv_widget_1, self.rv_widget_2, self.rv_widget_3, self.rv_widget_4]
        self.referenceLineEdits = [self.rv_lineEdit_1,self.rv_lineEdit_2,self.rv_lineEdit_3,self.rv_lineEdit_4]

        self.ratePlayer = QMediaPlayer(None, QMediaPlayer.VideoSurface)
        self.ratePlaylist = QMediaPlaylist()
        self.ratePlayer.setVideoOutput(self.rate_widget)
        self.ratePlayer.setPlaylist(self.ratePlaylist)
        self.ratePlaylist.setPlaybackMode(self.ratePlaylist.Loop)

        self.referencePlayers = [QMediaPlayer(None, QMediaPlayer.VideoSurface) for _ in range(4)]
        self.referencePlaylists = [QMediaPlaylist() for _ in range(4)]

        for i in range(4):
            self.referencePlayers[i].setVideoOutput(self.referenceVideoWidgets[i])
            self.referencePlayers[i].setPlaylist(self.referencePlaylists[i])
            self.referencePlaylists[i].setPlaybackMode(self.referencePlaylists[i].Loop)

        self.rate_lineEdit.setPlaceholderText('rate your trajectory here')
        self.lineEdit_postfix.setPlaceholderText('name the postfix of the saving models')

        self.rate_pushButton.clicked.connect(self.change_wait)
        self.skip_pushButton.clicked.connect(self.change_skip)

        self.is_wait = True
        self.is_skip = False

    def rate_trajectory(self, rank_references):
        rank_label_ls = []
        rank_label_true_ls = []
        ref_label_change = [[], []]
        skip_index = []
        for index,rank_reference in enumerate(rank_references):
            rank_traj_number, ref_traj_number_ls, ref_labels_ls, rank_label_true = rank_reference
            print(rank_traj_number,ref_traj_number_ls,ref_labels_ls,rank_label_true)

            rateFile = self.fileRoot + str(rank_traj_number) + self.fileType
            self.ratePlaylist.addMedia(QMediaContent(QUrl.fromLocalFile(rateFile)))
            self.ratePlayer.play()

            for i,ref_traj_number in enumerate(ref_traj_number_ls):
                fileName = self.fileRoot + str(ref_traj_number) + self.fileType
                self.referencePlaylists[i].addMedia(QMediaContent(QUrl.fromLocalFile(fileName)))
                self.referenceLineEdits[i].setText(str(ref_labels_ls[i]))
            for referencePlayer in self.referencePlayers:
                referencePlayer.play()

            while self.is_wait:
                QTest.qWait(100)

            if self.is_skip:
                skip_index.append(index)
            else:
                rank_label_ls.append(float(self.rate_lineEdit.text()))
                rank_label_true_ls.append(rank_label_true)

            for i in range(len(ref_labels_ls)):
                if self.referenceLineEdits[i].text() != str(ref_labels_ls[i]):
                    changed_label = float(self.referenceLineEdits[i].text())
                    changed_number = ref_traj_number_ls[i]
                    if changed_number in ref_label_change[0]:
                        ref_label_change[1][ref_label_change[0].index(changed_number)] = changed_label
                    else:
                        ref_label_change[0].append(changed_number)
                        ref_label_change[1].append(changed_label)

            self.rate_lineEdit.clear()
            self.ratePlaylist.clear()
            for i in range(4):
                self.referencePlaylists[i].clear()
                self.referenceLineEdits[i].clear()
            self.is_wait = True
            self.is_skip = False
        return rank_label_ls, rank_label_true_ls, ref_label_change, skip_index

    def change_wait(self):
        self.is_wait = False

    def change_skip(self):
        self.is_wait = False
        self.is_skip = True

if __name__ == '__main__':
    app = QApplication(sys.argv)
    ui = RatingWindow()
    ui.show()
    sys.exit(app.exec())