import os
import shutil
from PySide6.QtCore import Qt, QUrl, QThread, Signal
from PySide6.QtGui import QKeySequence, QShortcut
from PySide6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QGridLayout,
    QLabel, QPushButton, QProgressBar, QFrame, QFileDialog,
    QTableWidget, QTableWidgetItem, QHeaderView, QLineEdit,
    QComboBox, QDialog, QCheckBox, QSlider, QMessageBox, QSplitter
)
from PySide6.QtMultimedia import QMediaPlayer, QAudioOutput
from PySide6.QtMultimediaWidgets import QVideoWidget

from studio.core.config import SettingsManager, USER_DATA_DIR
from studio.core.project_db import ProjectStore
from studio.pipeline.one_click_worker import OneClickDubbingWorker
from studio.pipeline.media_audio import sec_to_tc, export_srt_file, render_final_dubbed_video, probe_video_duration
from studio.pipeline.ai_engines import PersistentVoxCPM2Singleton


class ExportWorker(QThread):
    completed = Signal(str, str)
    failed = Signal(str)

    def __init__(self, video_path, lines, output_path, settings):
        super().__init__()
        self.video_path = video_path
        self.lines = lines
        self.output_path = output_path
        self.settings = settings

    def run(self):
        try:
            duration = probe_video_duration(self.video_path)
            video, subtitle = render_final_dubbed_video(
                self.video_path, self.lines, duration, self.output_path,
                bg_vol=float(self.settings.get("bg_volume", 0.18)),
                burn_subs=bool(self.settings.get("burn_subtitles", True))
            )
            self.completed.emit(video, subtitle)
        except Exception as exc:
            self.failed.emit(str(exc))

STUDIO_STYLE = """
QMainWindow, QWidget#mainRoot {
    background-color: #090e14;
    color: #f1f5f9;
    font-family: 'Segoe UI', 'Khmer OS Siemreap', sans-serif;
    font-size: 13.5px;
}
QFrame#panel {
    background-color: #0f1722;
    border: 1px solid #1e2d3d;
    border-radius: 10px;
}
QLabel#titleLbl {
    font-size: 18px;
    font-weight: 800;
    color: #ffffff;
}
QLabel#stageLbl {
    font-size: 13px;
    font-weight: 600;
    color: #38bdf8;
}
QPushButton#heroImportBtn {
    background-color: #162333;
    color: #ffffff;
    border: 1px solid #2d435c;
    border-radius: 8px;
    padding: 12px 22px;
    font-size: 14px;
    font-weight: bold;
}
QPushButton#heroImportBtn:hover {
    border-color: #38bdf8;
    background-color: #1d2e42;
}
QPushButton#heroStartBtn {
    background-color: #0ea5e9;
    color: #041019;
    border: none;
    border-radius: 8px;
    padding: 12px 28px;
    font-size: 14px;
    font-weight: 900;
}
QPushButton#heroStartBtn:hover {
    background-color: #38bdf8;
}
QPushButton#heroStartBtn:disabled {
    background-color: #1e293b;
    color: #64748b;
}
QPushButton#secondaryBtn {
    background-color: #162333;
    color: #e2e8f0;
    border: 1px solid #26384d;
    border-radius: 6px;
    padding: 7px 14px;
    font-weight: 600;
}
QPushButton#secondaryBtn:hover {
    border-color: #38bdf8;
}
QLineEdit, QComboBox {
    background-color: #090e14;
    color: #f8fafc;
    border: 1px solid #26384d;
    border-radius: 6px;
    padding: 7px 10px;
}
QTableWidget {
    background-color: #090e14;
    color: #f1f5f9;
    gridline-color: #1a2636;
    border: 1px solid #1e2d3d;
    border-radius: 6px;
}
QHeaderView::section {
    background-color: #131e2c;
    color: #94a3b8;
    padding: 7px;
    border: none;
    border-bottom: 1px solid #1e2d3d;
    font-weight: bold;
}
QProgressBar {
    background-color: #090e14;
    border: 1px solid #1e2d3d;
    border-radius: 6px;
    text-align: center;
    color: #ffffff;
    font-weight: bold;
    min-height: 22px;
}
QProgressBar::chunk {
    background-color: #0ea5e9;
    border-radius: 5px;
}
"""

class AdvancedSettingsDialog(QDialog):
    def __init__(self, settings, parent=None):
        super().__init__(parent)
        self.settings = settings
        self.setWindowTitle("Advanced Settings — KHMERDUB AI Studio")
        self.resize(560, 440)
        self.setStyleSheet(STUDIO_STYLE)

        layout = QGridLayout(self)
        layout.setSpacing(12)
        layout.setContentsMargins(20, 20, 20, 20)

        layout.addWidget(QLabel("Translation Engine:"), 0, 0)
        self.cmb_prov = QComboBox()
        self.cmb_prov.addItems([
            "Gemini (Context-Aware Story Mode)",
            "Local Ollama (qwen2.5 - 100% Offline)",
            "HuggingFace Free (NLLB-200)"
        ])
        self.cmb_prov.setCurrentText(self.settings.get("translation_provider"))
        layout.addWidget(self.cmb_prov, 0, 1)

        layout.addWidget(QLabel("Gemini API Key:"), 1, 0)
        self.inp_gemini = QLineEdit(self.settings.get("gemini_api_key", ""))
        self.inp_gemini.setEchoMode(QLineEdit.Password)
        self.inp_gemini.setPlaceholderText("AIzaSy...")
        layout.addWidget(self.inp_gemini, 1, 1)

        layout.addWidget(QLabel("HuggingFace Token (Optional):"), 2, 0)
        self.inp_hf = QLineEdit(self.settings.get("hf_token", ""))
        self.inp_hf.setEchoMode(QLineEdit.Password)
        layout.addWidget(self.inp_hf, 2, 1)

        layout.addWidget(QLabel("Local Ollama URL:"), 3, 0)
        self.inp_ollama = QLineEdit(self.settings.get("ollama_url", "http://localhost:11434/v1"))
        layout.addWidget(self.inp_ollama, 3, 1)

        layout.addWidget(QLabel("Whisper Speech Model:"), 4, 0)
        self.cmb_whisper = QComboBox()
        self.cmb_whisper.addItems(["medium", "small", "large-v3"])
        self.cmb_whisper.setCurrentText(self.settings.get("whisper_model", "medium"))
        layout.addWidget(self.cmb_whisper, 4, 1)

        layout.addWidget(QLabel("VoxCPM2 Local Model Folder:"), 5, 0)
        self.inp_vox_dir = QLineEdit(self.settings.get("voxcpm2_local_dir", ""))
        layout.addWidget(self.inp_vox_dir, 5, 1)

        layout.addWidget(QLabel("VoxCPM2 Sample Rate:"), 6, 0)
        self.cmb_sr = QComboBox()
        self.cmb_sr.addItems(["24000", "16000", "48000"])
        self.cmb_sr.setCurrentText(str(self.settings.get("voxcpm2_sample_rate", 24000)))
        layout.addWidget(self.cmb_sr, 6, 1)

        self.chk_burn = QCheckBox("Burn Khmer Subtitles into Final Video")
        self.chk_burn.setChecked(bool(self.settings.get("burn_subtitles", True)))
        layout.addWidget(self.chk_burn, 7, 0, 1, 2)

        btn_save = QPushButton("💾 Save Settings")
        btn_save.setObjectName("heroStartBtn")
        btn_save.clicked.connect(self._save_and_close)
        layout.addWidget(btn_save, 8, 0, 1, 2)

    def _save_and_close(self):
        self.settings.set("translation_provider", self.cmb_prov.currentText())
        self.settings.set("gemini_api_key", self.inp_gemini.text().strip())
        self.settings.set("hf_token", self.inp_hf.text().strip())
        self.settings.set("ollama_url", self.inp_ollama.text().strip())
        self.settings.set("whisper_model", self.cmb_whisper.currentText())
        self.settings.set("voxcpm2_local_dir", self.inp_vox_dir.text().strip())
        self.settings.set("voxcpm2_sample_rate", int(self.cmb_sr.currentText()))
        self.settings.set("burn_subtitles", self.chk_burn.isChecked())
        self.accept()


class KhmerDubProMainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.settings = SettingsManager()
        self.store = None
        self.worker = None
        self.export_worker = None
        self.video_path = ""
        self.final_video_path = ""

        self.player = QMediaPlayer(self)
        self.audio_out = QAudioOutput(self)
        self.player.setAudioOutput(self.audio_out)

        self.line_player = QMediaPlayer(self)
        self.line_audio = QAudioOutput(self)
        self.line_player.setAudioOutput(self.line_audio)

        self.setWindowTitle("KHMERDUB AI STUDIO — Professional Edition")
        self.resize(1440, 860)
        self.setStyleSheet(STUDIO_STYLE)
        self._build_ui()
        self._bind_keyboard_shortcuts()
        self._try_restore_last_session()

    def _build_ui(self):
        root = QWidget()
        root.setObjectName("mainRoot")
        main_lay = QVBoxLayout(root)
        main_lay.setContentsMargins(16, 14, 16, 14)
        main_lay.setSpacing(12)

        # --- TOP HEADER: TWO PRIMARY HERO ACTIONS + SETTINGS ---
        header = QFrame()
        header.setObjectName("panel")
        hl = QHBoxLayout(header)
        hl.setContentsMargins(18, 12, 18, 12)

        title_box = QVBoxLayout()
        lbl_app = QLabel("KHMERDUB AI STUDIO <span style='color:#38bdf8; font-size:13px;'>PROFESSIONAL EDITION</span>")
        lbl_app.setObjectName("titleLbl")
        self.lbl_file_info = QLabel("Step 1: Click 'Import Video' to begin (Supports MP4, MKV, MOV)")
        self.lbl_file_info.setStyleSheet("color: #94a3b8; font-size: 12.5px;")
        title_box.addWidget(lbl_app)
        title_box.addWidget(self.lbl_file_info)
        hl.addLayout(title_box)

        hl.addStretch()

        self.btn_import = QPushButton("📁 1. Import Video")
        self.btn_import.setObjectName("heroImportBtn")
        self.btn_import.clicked.connect(self.import_video)
        hl.addWidget(self.btn_import)

        self.btn_start = QPushButton("🚀 2. Start Dubbing")
        self.btn_start.setObjectName("heroStartBtn")
        self.btn_start.setEnabled(False)
        self.btn_start.clicked.connect(self.start_one_click_dubbing)
        hl.addWidget(self.btn_start)

        self.btn_export = QPushButton("💾 Export Video / SRT")
        self.btn_export.setObjectName("secondaryBtn")
        self.btn_export.setEnabled(False)
        self.btn_export.clicked.connect(self.export_final_deliverables)
        hl.addWidget(self.btn_export)

        btn_adv = QPushButton("⚙️ Advanced Settings")
        btn_adv.setObjectName("secondaryBtn")
        btn_adv.clicked.connect(self.open_advanced_settings)
        hl.addWidget(btn_adv)

        main_lay.addWidget(header)

        # --- REAL-TIME PROGRESS BAR ---
        prog_panel = QFrame()
        prog_panel.setObjectName("panel")
        pl = QHBoxLayout(prog_panel)
        pl.setContentsMargins(16, 10, 16, 10)
        self.lbl_stage = QLabel("Ready — Auto-Save & Resume Protection Active")
        self.lbl_stage.setObjectName("stageLbl")
        pl.addWidget(self.lbl_stage, 3)

        self.progress = QProgressBar()
        self.progress.setValue(0)
        pl.addWidget(self.progress, 4)

        self.btn_stop = QPushButton("⏹ Stop")
        self.btn_stop.setObjectName("secondaryBtn")
        self.btn_stop.setEnabled(False)
        self.btn_stop.clicked.connect(self.stop_processing)
        pl.addWidget(self.btn_stop)
        main_lay.addWidget(prog_panel)

        # --- CENTER SPLITTER: VIDEO MONITOR (LEFT) + SUBTITLE & DUBBING EDITOR (RIGHT) ---
        splitter = QSplitter(Qt.Horizontal)

        # Left Video Preview Card
        vid_card = QFrame()
        vid_card.setObjectName("panel")
        vl = QVBoxLayout(vid_card)
        vl.setContentsMargins(12, 12, 12, 12)

        self.video_widget = QVideoWidget()
        self.video_widget.setMinimumSize(500, 320)
        self.video_widget.setStyleSheet("background-color: #03060a; border-radius: 6px;")
        self.player.setVideoOutput(self.video_widget)
        vl.addWidget(self.video_widget, 1)

        self.seek_slider = QSlider(Qt.Horizontal)
        self.seek_slider.sliderMoved.connect(self.player.setPosition)
        self.player.positionChanged.connect(self._on_player_pos_changed)
        self.player.durationChanged.connect(self.seek_slider.setMaximum)
        vl.addWidget(self.seek_slider)

        ctrl_bar = QHBoxLayout()
        self.btn_play = QPushButton("▶ Play / Pause (Space)")
        self.btn_play.setObjectName("secondaryBtn")
        self.btn_play.clicked.connect(self.toggle_playback)
        ctrl_bar.addWidget(self.btn_play)

        btn_back = QPushButton("⏪ -3s (Left)")
        btn_back.setObjectName("secondaryBtn")
        btn_back.clicked.connect(lambda: self.seek_relative(-3000))
        ctrl_bar.addWidget(btn_back)

        btn_fwd = QPushButton("⏩ +3s (Right)")
        btn_fwd.setObjectName("secondaryBtn")
        btn_fwd.clicked.connect(lambda: self.seek_relative(3000))
        ctrl_bar.addWidget(btn_fwd)

        self.lbl_time = QLabel("00:00 / 00:00")
        self.lbl_time.setStyleSheet("color: #94a3b8;")
        ctrl_bar.addStretch()
        ctrl_bar.addWidget(self.lbl_time)
        vl.addLayout(ctrl_bar)
        splitter.addWidget(vid_card)

        # Right Subtitle & Voice Editor Card
        ed_card = QFrame()
        ed_card.setObjectName("panel")
        el = QVBoxLayout(ed_card)
        el.setContentsMargins(12, 12, 12, 12)

        self.table = QTableWidget(0, 5)
        self.table.setHorizontalHeaderLabels(["#", "Time", "Speaker", "Original Dialogue", "Khmer Translation & Dub"])
        self.table.horizontalHeader().setSectionResizeMode(3, QHeaderView.Stretch)
        self.table.horizontalHeader().setSectionResizeMode(4, QHeaderView.Stretch)
        self.table.cellClicked.connect(self._on_table_click)
        el.addWidget(self.table, 1)

        edit_row = QHBoxLayout()
        self.inp_khmer = QLineEdit()
        self.inp_khmer.setPlaceholderText("Click any subtitle row to edit Khmer text, then press Ctrl+Enter or click Re-Generate...")
        edit_row.addWidget(self.inp_khmer, 4)

        btn_regen = QPushButton("🔄 Update & Dub Line")
        btn_regen.setObjectName("secondaryBtn")
        btn_regen.clicked.connect(self.regenerate_current_line)
        edit_row.addWidget(btn_regen, 1)

        btn_hear = QPushButton("🔊 Hear Line")
        btn_hear.setObjectName("secondaryBtn")
        btn_hear.clicked.connect(self.preview_current_line_audio)
        edit_row.addWidget(btn_hear, 1)
        el.addLayout(edit_row)

        splitter.addWidget(ed_card)
        splitter.setSizes([580, 820])
        main_lay.addWidget(splitter, 1)

        self.setCentralWidget(root)

    def _bind_keyboard_shortcuts(self):
        # Global application shortcuts so Space / Left / Right / Ctrl+Enter always work
        sc_space = QShortcut(QKeySequence(Qt.Key_Space), self)
        sc_space.setContext(Qt.ApplicationShortcut)
        sc_space.activated.connect(self._shortcut_space)

        sc_left = QShortcut(QKeySequence(Qt.Key_Left), self)
        sc_left.setContext(Qt.ApplicationShortcut)
        sc_left.activated.connect(lambda: self.seek_relative(-3000))

        sc_right = QShortcut(QKeySequence(Qt.Key_Right), self)
        sc_right.setContext(Qt.ApplicationShortcut)
        sc_right.activated.connect(lambda: self.seek_relative(3000))

        sc_regen = QShortcut(QKeySequence("Ctrl+Return"), self)
        sc_regen.setContext(Qt.ApplicationShortcut)
        sc_regen.activated.connect(self.regenerate_current_line)

    def _shortcut_space(self):
        # Allow typing spaces normally if editing text box
        if self.inp_khmer.hasFocus():
            self.inp_khmer.insert(" ")
            return
        self.toggle_playback()

    def toggle_playback(self):
        if self.player.playbackState() == QMediaPlayer.PlayingState:
            self.player.pause()
        else:
            self.player.play()

    def seek_relative(self, delta_ms):
        if self.inp_khmer.hasFocus():
            return
        new_pos = max(0, min(self.player.duration(), self.player.position() + delta_ms))
        self.player.setPosition(new_pos)

    def _on_player_pos_changed(self, pos_ms):
        self.seek_slider.setValue(pos_ms)
        cur_s = pos_ms // 1000
        tot_s = self.player.duration() // 1000
        self.lbl_time.setText(f"{cur_s // 60:02d}:{cur_s % 60:02d} / {tot_s // 60:02d}:{tot_s % 60:02d}")

    def _try_restore_last_session(self):
        last_dir = self.settings.get("last_project_dir", "")
        if last_dir and os.path.exists(os.path.join(last_dir, "project_state.sqlite")):
            self.store = ProjectStore(last_dir)
            saved_vid = self.store.get_meta("video_path", "")
            if saved_vid and os.path.exists(saved_vid):
                self._load_video_into_session(saved_vid, existing_store=True)
                self.lbl_stage.setText("Restored previous session automatically — Click 'Start Dubbing' to resume or re-render.")

    def import_video(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Import Video", "", "Video Files (*.mp4 *.mkv *.mov *.avi *.webm)"
        )
        if not path:
            return
        self._load_video_into_session(path, existing_store=False)

    def _load_video_into_session(self, path, existing_store=False):
        self.video_path = path
        safe_slug = "".join([c if c.isalnum() else "_" for c in os.path.splitext(os.path.basename(path))[0]])[:40]
        proj_dir = os.path.join(USER_DATA_DIR, f"proj_{safe_slug}")
        if not existing_store:
            if self.store:
                self.store.close()
            self.store = ProjectStore(proj_dir)
            self.store.set_meta("video_path", path)
            self.settings.set("last_project_dir", proj_dir)

        self.lbl_file_info.setText(f"Loaded: <b>{os.path.basename(path)}</b>  |  Auto-Saved in SQLite")
        self.player.setSource(QUrl.fromLocalFile(path))
        self.player.pause()
        self.btn_start.setEnabled(True)
        self.btn_export.setEnabled(True)
        self._refresh_table()

    def start_one_click_dubbing(self):
        if not self.video_path or not os.path.exists(self.video_path):
            QMessageBox.warning(self, "No Video", "Please click '1. Import Video' first.")
            return
        if self.worker and self.worker.isRunning():
            return

        self.btn_start.setEnabled(False)
        self.btn_import.setEnabled(False)
        self.btn_stop.setEnabled(True)
        self.progress.setValue(2)

        self.worker = OneClickDubbingWorker(self.video_path, self.store, self.settings)
        self.worker.stage_changed.connect(self._on_stage_changed)
        self.worker.line_updated.connect(lambda _: self._refresh_table())
        self.worker.pipeline_completed.connect(self._on_pipeline_completed)
        self.worker.pipeline_failed.connect(self._on_pipeline_failed)
        self.worker.start()

    def stop_processing(self):
        if self.worker and self.worker.isRunning():
            self.worker.stop()
            self.lbl_stage.setText("Stopping safely... Progress is auto-saved in SQLite.")

    def _on_stage_changed(self, pct, label):
        self.progress.setValue(pct)
        self.lbl_stage.setText(label)

    def _on_pipeline_completed(self, final_mp4, final_srt):
        self.final_video_path = final_mp4
        self.btn_start.setEnabled(True)
        self.btn_import.setEnabled(True)
        self.btn_export.setEnabled(True)
        self.btn_stop.setEnabled(False)
        self._refresh_table()
        if os.path.exists(final_mp4):
            self.player.setSource(QUrl.fromLocalFile(final_mp4))
            self.player.play()
        QMessageBox.information(self, "Dubbing Completed", f"Your Khmer dubbed video is ready!\n\nSaved at:\n{final_mp4}")

    def _on_pipeline_failed(self, friendly_err):
        self.btn_start.setEnabled(True)
        self.btn_import.setEnabled(True)
        self.btn_stop.setEnabled(False)
        self.lbl_stage.setText("Paused / Stopped — Click 'Start Dubbing' anytime to resume from last checkpoint.")
        QMessageBox.warning(self, "KHMERDUB Studio Notice", friendly_err)

    def _refresh_table(self):
        if not self.store:
            return
        lines = self.store.get_lines()
        self.table.setRowCount(len(lines))
        for r, ln in enumerate(lines):
            self.table.setItem(r, 0, QTableWidgetItem(str(ln["idx"])))
            self.table.setItem(r, 1, QTableWidgetItem(f"{sec_to_tc(ln['start_sec'])} --> {sec_to_tc(ln['end_sec'])}"))
            self.table.setItem(r, 2, QTableWidgetItem(ln["speaker_id"]))
            self.table.setItem(r, 3, QTableWidgetItem(ln["source_text"]))
            self.table.setItem(r, 4, QTableWidgetItem(ln.get("khmer_text", "")))

    def _on_table_click(self, row, col):
        if not self.store:
            return
        lines = self.store.get_lines()
        if 0 <= row < len(lines):
            ln = lines[row]
            self.inp_khmer.setText(ln.get("khmer_text", ""))
            self.player.setPosition(int(ln["start_sec"] * 1000))

    def regenerate_current_line(self):
        row = self.table.currentRow()
        if row < 0 or not self.store:
            return
        if (self.worker and self.worker.isRunning()) or (self.export_worker and self.export_worker.isRunning()):
            QMessageBox.information(self, "Task Running", "Wait for the current task to finish before regenerating a line.")
            return
        lines = self.store.get_lines()
        ln = lines[row]
        new_kh = self.inp_khmer.text().strip()
        self.store.update_line(ln["idx"], khmer_text=new_kh)
        self._refresh_table()

        self.worker = OneClickDubbingWorker(self.video_path, self.store, self.settings, single_line_idx=ln["idx"])
        self.worker.stage_changed.connect(self._on_stage_changed)
        self.worker.line_updated.connect(lambda _: self._refresh_table())
        self.worker.start()

    def preview_current_line_audio(self):
        row = self.table.currentRow()
        if row < 0 or not self.store:
            return
        lines = self.store.get_lines()
        wpath = lines[row].get("dub_wav_path", "")
        if wpath and os.path.exists(wpath):
            self.line_player.setSource(QUrl.fromLocalFile(wpath))
            self.line_player.play()

    def export_final_deliverables(self):
        if not self.store or not self.video_path:
            return
        out_path, selected_filter = QFileDialog.getSaveFileName(
            self, "Export Final Dubbed Deliverable", "Khmer_Dubbed_Final.mp4",
            "MP4 Video (*.mp4);;MKV Video (*.mkv);;Khmer Subtitle (*.srt)"
        )
        if not out_path:
            return
        lines = self.store.get_lines()
        if out_path.lower().endswith(".srt"):
            export_srt_file(out_path, lines, use_khmer=True)
            QMessageBox.information(self, "Exported", f"Subtitle exported:\n{out_path}")
            return

        if (self.worker and self.worker.isRunning()) or (self.export_worker and self.export_worker.isRunning()):
            QMessageBox.information(self, "Task Running", "Wait for the current task to finish before exporting.")
            return
        self.btn_export.setEnabled(False)
        self.lbl_stage.setText("Exporting in background. The interface remains available...")
        self.export_worker = ExportWorker(self.video_path, lines, out_path, self.settings)
        self.export_worker.completed.connect(self._on_export_completed)
        self.export_worker.failed.connect(self._on_export_failed)
        self.export_worker.start()

    def _on_export_completed(self, video_path, subtitle_path):
        self.btn_export.setEnabled(True)
        self.lbl_stage.setText("Export completed.")
        QMessageBox.information(self, "Exported", f"Final Video & SRT exported successfully:\n{video_path}\n{subtitle_path}")

    def _on_export_failed(self, message):
        self.btn_export.setEnabled(True)
        self.lbl_stage.setText("Export failed.")
        QMessageBox.warning(self, "Export Failed", message)

    def open_advanced_settings(self):
        dlg = AdvancedSettingsDialog(self.settings, self)
        dlg.exec()

    def closeEvent(self, event):
        running = [w for w in (self.worker, self.export_worker) if w and w.isRunning()]
        if running:
            QMessageBox.warning(self, "Task Still Running", "Stop or wait for the active task before closing the application. This protects the project database and output files.")
            event.ignore()
            return
        PersistentVoxCPM2Singleton.unload()
        if self.store:
            self.store.close()
        event.accept()
