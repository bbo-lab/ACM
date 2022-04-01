import sys, os
from PyQt5.QtWidgets import QApplication, QMainWindow, QFileDialog, QFrame, QGridLayout, \
                            QSlider, QComboBox, QLineEdit, QCheckBox,QScrollArea, QAction
from PyQt5.QtCore import Qt

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
from matplotlib.figure import Figure

import imageio
from ccvtools import rawio

import numpy as np
import torch

from ..model import map_m
from ..helper import get_calibration

from pprint import pprint

class Viewer(QMainWindow):
    app = []
    config = dict()
    vidreader = []
    results = []
    calib = []
    frames = []
    
    labels_dlc = None
    labels_manual = None

    camidx = 0
    frameidx = 0
    resultidx = 0

    frame_main = []

    frame_slider = []
    frame_lineedit = []
    cam_combobox = []
    result_combobox = []
    marker_checkboxes = []

    menu = dict()
    
    plot_components = {'dlc': [], 'manual': []}
    
    frame_cache = [-1, -1, []]

    clear = 4

    fig = []
    canvas = []
    ax = []

    globalnotify = True

    def __init__(self,app,config):
        super().__init__()

        self.app=app

        self.config = config

        if not 'videos' in self.config:
            self.config['videos'] = self.read_videos()

        for f in self.config['videos']:
            self.vidreader.append(imageio.get_reader(f))

        self.calib = get_calibration(self.get_config()['file_origin_coord'], self.get_config()['file_calibration'], self.get_config()['scale_factor'])

        self.results = os.listdir(os.path.join(self.get_config()['folder_project'], 'results'))
        self.results.append('')

        self.frames = np.asarray(list(range(self.get_config()['index_frame_start'],self.get_config()['index_frame_end'],self.get_config()['dt'])))

        self.make_menu()
        self.make_gui()

        print("Setting defaults")
        self.set_camidx(0, showframe=False)
        self.set_frameidx(self.get_config()['index_frame_start'], showframe=False)
        self.set_resultidx(0, showframe=False) #len(self.results)-1
        print("Done")
        self.show_frame()

    def get_config(self):
        return self.config

    def get_vidreader(self,vididx):
        return self.vidreader[vididx]

    def make_menu(self):
        self.menu['menu'] = self.menuBar()

        self.menu['menu_file'] = self.menu['menu'].addMenu("&Export")

        saveframeAction = QAction("&Save frame", self)
        saveframeAction.triggered.connect(self.save_frame)
        self.menu['menu_file'].addAction(saveframeAction)


    def make_gui(self):
        self.setGeometry(0, 0, 1280, 900)
        self.setWindowTitle('ACM viewer')

        self.frame_main = QFrame()
        layoutGrid = QGridLayout()
        layoutGrid.setSpacing(0)
        layoutGrid.setRowStretch(0, 9)
        layoutGrid.setRowStretch(1, 1)
        layoutGrid.setColumnStretch(0, 5)
        layoutGrid.setColumnStretch(1, 1)
        self.frame_main.setLayout(layoutGrid)

        frame_plot = QFrame()
        layoutGrid.addWidget(frame_plot, 0, 0)
        layoutGrid_plot = QGridLayout()
        layoutGrid_plot.setSpacing(0)
        frame_plot.setLayout(layoutGrid_plot)
        
        frame_labels = QScrollArea() #TODO: Make this properly scollable
        layoutGrid.addWidget(frame_labels, 0, 1)
        layoutGrid_labels = QGridLayout()
        layoutGrid_labels.setSpacing(0)
        frame_labels.setLayout(layoutGrid_labels)

        frame_controls = QFrame()
        layoutGrid.addWidget(frame_controls, 1, 0, 1, 2)
        layoutGrid_control = QGridLayout()
        layoutGrid_control.setSpacing(10)
        layoutGrid_control.setRowStretch(0, 1)
        layoutGrid_control.setRowStretch(1, 1)
        layoutGrid_control.setColumnStretch(0, 1)
        layoutGrid_control.setColumnStretch(1, 4)
        layoutGrid_control.setColumnStretch(1, 1)
        frame_controls.setLayout(layoutGrid_control)
        
        model = np.load(self.get_config()['file_model'],allow_pickle=True).item()
        for i, mlabel in enumerate(model['joint_marker_order']):
            self.marker_checkboxes.append(QCheckBox())
            self.marker_checkboxes[-1].setChecked(True)
            self.marker_checkboxes[-1].setText(mlabel)
            self.marker_checkboxes[-1].toggled.connect(lambda:self.show_frame(actions=["dlc","draw"]))
            layoutGrid_labels.addWidget(self.marker_checkboxes[-1], i, 0)
            
        self.frame_slider = QSlider(Qt.Horizontal)
        self.frame_slider.setMinimum(self.get_config()['index_frame_start']+1)
        self.frame_slider.setMaximum(self.get_config()['index_frame_end'])
        self.frame_slider.setTickInterval(1)
        self.frame_slider.sliderReleased.connect(lambda: self.set_frameidx(self.frame_slider.value()-1))
        layoutGrid_control.addWidget(self.frame_slider, 0, 0, 1, 2)

        self.frame_lineedit = QLineEdit()
        self.frame_lineedit.editingFinished.connect(lambda: self.set_frameidx(int(self.frame_lineedit.text())-1)) # TODO add QtValidator
        layoutGrid_control.addWidget(self.frame_lineedit, 0, 2, 1, 1)

        self.cam_combobox = QComboBox()
        self.cam_combobox.addItems([ f'Camera {n}' for n in range(len(self.vidreader)) ])
        self.cam_combobox.currentIndexChanged.connect(lambda n: self.set_camidx(n))
        layoutGrid_control.addWidget(self.cam_combobox, 1, 0)

        print(self.get_config()['folder_save'])
        self.result_combobox = QComboBox()
        self.result_combobox.addItems(self.results)
        self.result_combobox.currentIndexChanged.connect(lambda n: self.set_resultidx(n))
        layoutGrid_control.addWidget(self.result_combobox, 1, 1, 1, 2)

        self.fig = Figure(tight_layout=True)
        self.canvas = FigureCanvasQTAgg(self.fig)
        self.canvas.setParent(frame_plot)
        self.ax = self.fig.add_subplot(1,1,1)
        layoutGrid_plot.addWidget(self.canvas, 0, 0)

        self.frame_main.setLayout(layoutGrid)
        self.setCentralWidget(self.frame_main)
        self.setFocus()

        self.show()

    def set_camidx(self,camidx,notify=True,showframe=True):
        if notify and self.globalnotify:
            self.globalnotify = False
            self.cam_combobox.setCurrentIndex(camidx)
            self.globalnotify = True

        self.camidx = camidx

        if showframe and self.globalnotify:
            self.show_frame()

    def set_frameidx(self,frameidx,notify=True,showframe=True):
        frameidx = max(frameidx,self.get_config()['index_frame_start'])
        frameidx = min(frameidx,self.get_config()['index_frame_end']-1)

        if notify and self.globalnotify:
            self.globalnotify = False
            print(f"setting slider to {frameidx+1}")
            self.frame_slider.setValue(frameidx+1)
            self.frame_lineedit.setText(str(frameidx+1))
            self.globalnotify = True

        self.frameidx = frameidx

        if showframe and self.globalnotify:
            self.show_frame()

    def set_resultidx(self,resultidx,notify=True,showframe=True):
        if notify and self.globalnotify:
            self.globalnotify = False
            self.result_combobox.setCurrentIndex(resultidx)
            self.globalnotify = True

        self.resultidx = resultidx

        if showframe and self.globalnotify:
            self.show_frame()
    
    
    def get_dlc_labels(self):
        if self.labels_dlc is None:
            self.labels_dlc = np.load(self.get_config()['file_labelsDLC'],allow_pickle=True).item()
            
        return self.labels_dlc
    
    
    def get_manual_labels(self):
        if self.labels_manual is None:
            self.labels_manual = np.load(self.get_config()['file_labelsManual'],allow_pickle=True)['arr_0'].item()
            
        return self.labels_manual
    

    def show_frame(self,camidx=None,frameidx=None,resultidx=None,actions=["clear","imshow","dlc","manual","joints","draw"]):
        # TODO implement clearing of respective parts in each section
        if camidx is None:
            camidx = self.camidx
        if frameidx is None:
            frameidx = self.frameidx
        if resultidx is None:
            resultidx = self.resultidx

        if self.frame_cache[0] == camidx and self.frame_cache[1] == frameidx:
            frame = self.frame_cache[2]
        else:
            frame = self.get_vidreader(camidx).get_data(frameidx)
            if len(frame.shape)>2:
                frame=frame[:,:,0] # self.get_config()['folder_project']
            self.frame_cache = [camidx, frameidx, frame]
            
        calib = np.load(self.get_config()['file_calibration'],allow_pickle=True).item()

        if "clear" in actions:
            self.ax.clear()
            self.ax.set_xticklabels('')
            self.ax.set_yticklabels('')
            self.ax.set_xticks([])
            self.ax.set_yticks([])
            
            self.ax.imshow(frame,
                        aspect=1,
                        cmap='gray',
                        vmin=0,
                        vmax=255)
            
        
        # Plot dlc labels if applicable
        if "dlc" in actions:
            labels_dlc = self.get_dlc_labels()
            for c in self.plot_components["dlc"]:
                c.remove()
            checked_markers = np.asarray([mc.isChecked() for mc in self.marker_checkboxes]);
            used_markers = np.squeeze(labels_dlc['labels_all'][labels_dlc['frame_list']==frameidx,camidx,:,2]>=0.9)
            show_mask = used_markers & checked_markers
            print(f'DLC used/unused labels: {np.sum(used_markers)}/{np.sum(~used_markers)}')
            self.plot_components["dlc"] = self.ax.plot(
                labels_dlc['labels_all'][labels_dlc['frame_list']==frameidx,camidx,show_mask,0],
                labels_dlc['labels_all'][labels_dlc['frame_list']==frameidx,camidx,show_mask,1],
                'bo')

        # Plot manual labels if applicable
        if "manual" in actions:
            labels_man = self.get_manual_labels()
            for c in self.plot_components["manual"]:
                c.remove()
            self.plot_components["manual"] = []
            if frameidx in labels_man:
                for k in labels_man[frameidx]:
                    self.plot_components["manual"].append(self.ax.plot(labels_man[frameidx][k][camidx,0],labels_man[frameidx][k][camidx,1],'g+')[0])
        
        if "joints" in actions:
            # Calculate and plot joint positions
            if resultidx+1 < len(self.results):
                posefile = os.path.join(self.get_config()['folder_project'], 'results', self.results[resultidx], 'pose.npy')
                if os.path.isfile(posefile):
                    model = np.load(os.path.join(self.get_config()['folder_project'],"model.npy"),allow_pickle=True).item()
                    pose = np.load(posefile,allow_pickle=True).item()
                    closestposeidx = np.argmin(np.abs(self.frames-frameidx))
                    joint_positions_3d = pose['joint_positions_3d'][[closestposeidx],:,:]
                    joint_positions_2d = np.asarray(map_m(self.calib['RX1_fit'],
                                                self.calib['tX1_fit'],
                                                self.calib['A_fit'],
                                                self.calib['k_fit'],
                                                torch.from_numpy(joint_positions_3d)))
                    
                    for edge in model['skeleton_edges']:
                        self.ax.plot(joint_positions_2d[:,camidx,edge,0][0],joint_positions_2d[:,camidx,edge,1][0],'r')
                        

                    self.ax.plot(joint_positions_2d[:,camidx,:,0],joint_positions_2d[:,camidx,:,1],'r+')
                else:
                    print(f"{posefile} not found")
                    
        if "draw" in actions:
            self.canvas.draw()


    def read_videos(self):
        vidDlg = QFileDialog()
        dialogOptions = vidDlg.Options()
        #dialogOptions |= vidDlg.DontUseNativeDialog
        videoFiles, _ = QFileDialog.getOpenFileNames(vidDlg,
            "Choose video files",
            self.config['folder_project'],
            "video files (*.*)",
            options=dialogOptions)

        if len(videoFiles)==0: # TODO check match with calibration
            print("Select files matching your multicam calibration!", file=sys.stderr)
            #self.app.quit() # TODO find proper way here
            sys.exit(1)

        videoFiles = sorted(videoFiles)

        return videoFiles

    def save_frame(self):
        figpath = os.path.join(self.get_config()['folder_project'], 'results', self.results[self.resultidx], 'exports')
        print(f"Saving frame {self.frameidx} to {figpath}... ",end='')
        os.makedirs(figpath,exist_ok=True)
        self.fig.savefig(os.path.join(figpath, f'frame_{self.frameidx}_{self.results[self.resultidx]}.svg'),
                    bbox_inches='tight',
                    dpi=600,
                    transparent=True,
                    format='svg',
                    pad_inches=0)
        print('Success')


def start(config):
    app = QApplication([])
    viewer = Viewer(app,config)
    sys.exit(app.exec_())
