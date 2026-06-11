# BSP: Bead Pairing Algorithm
# Advanced bead-by-bead chromatic aberration shift analysis tool

import sys
import json
import numpy as np
import tifffile
from pathlib import Path
import time
import traceback
from scipy.ndimage import gaussian_filter
from scipy.spatial import KDTree
from skimage import measure, morphology, feature
from skimage.feature import blob_log
import csv
from functools import partial
import multiprocessing

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
    QLabel, QPushButton, QFileDialog, QMessageBox, QProgressBar,
    QTextEdit, QGroupBox, QGridLayout, QDoubleSpinBox, 
    QCheckBox, QTableWidget, QTableWidgetItem, 
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal, pyqtSlot, QTimer
from PyQt5.QtGui import QFont

try:
    import cupy as cp
    import cucim.skimage
    import cucim.scipy.ndimage
    GPU_ENABLED = True
except ImportError:
    GPU_ENABLED = False


def _run_detection_task(args, gpu_available):
    """
    Wrapper function to run bead detection for a single channel in a separate process.
    """
    # Unpack arguments
    channel_index, channel_name, channel_image, settings, log_queue = args
    
    # This is a dummy class to capture log messages from the subprocess
    class SubprocessLogger:
        def emit(self, message, level):
            log_queue.put((f"({channel_name}) {message}", level))

    thread_instance = AnalysisThread(None, None, None, None, None, None, None)
    thread_instance.log_message = SubprocessLogger() # Redirect logging
    thread_instance.gpu_available = gpu_available
    
    if gpu_available:
        beads = thread_instance.detect_beads_advanced_gpu(channel_image, settings, channel_name)
    else:
        beads = thread_instance.detect_beads_advanced(channel_image, settings, channel_name)
        
    return (channel_name, beads)


def _run_matching_task(args):
    """
    Wrapper function to run bead matching for a single channel in a separate process.
    """
    # Unpack arguments
    channel_name, target_beads, reference_beads, similarity_params, ransac_params, log_queue = args

    # Dummy logger for subprocess
    class SubprocessLogger:
        def emit(self, message, level):
            # We can choose to log from subprocesses or keep it clean
            # For now, let's suppress most of it to avoid spamming the log
            if level in ["ERROR", "WARNING"]:
                log_queue.put((f"({channel_name}) {message}", level))

    thread_instance = AnalysisThread(None, None, None, None, similarity_params, None, ransac_params)
    thread_instance.log_message = SubprocessLogger() # Redirect logging

    if ransac_params['enabled']:
        matches = thread_instance.find_bead_matches_with_ransac(
            reference_beads, 
            target_beads, 
            channel_name,
            ransac_params['threshold']
        )
    else:
        matches = thread_instance.find_bead_matches(reference_beads, target_beads, channel_name)
        
    return (channel_name, matches)


class BeadShiftAnalyzer(QMainWindow):
    """
    Main application for bead-by-bead chromatic aberration shift analysis.
    Complete Parts 1-4: Infrastructure, Detection, Matching, and Shift Calculation.
    """
    
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Bead-by-Bead Shift Analyzer v1.1 - Batch Enabled")
        self.setGeometry(100, 100, 1400, 900)
        
        # Core data structures
        self.image_stack = None
        self.image_file_paths = []
        self.current_processing_file_path = None
        self.channel_names = ["DAPI", "Opal 570", "Opal 690", "Opal 620", "Opal 780", "Opal 520", "Sample AF"]
        self.detection_settings = {}
        self.default_settings = None
        self.af_settings = None
        
        # Batch processing state
        self.is_batch_processing = False
        self.current_file_index = 0
        
        # Analysis results
        self.detected_beads = {}
        self.bead_matches = {}
        self.shift_analysis = {}
        
        # Setup UI
        self.setup_ui()
        self.log_message("🔬 Bead-by-Bead Shift Analyzer initialized (Batch Enabled)")
        self.log_message("📋 Select single or multiple files, load settings, and start analysis.")
        
    def setup_ui(self):
        """Setup the main user interface."""
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # Main layout
        main_layout = QHBoxLayout(central_widget)
        
        # Left panel: Controls and settings
        left_panel = self.create_left_panel()
        main_layout.addWidget(left_panel, 1)
        
        # Right panel: Log and progress
        right_panel = self.create_right_panel()  
        main_layout.addWidget(right_panel, 2)

    def create_left_panel(self):
        """Create the left control panel."""
        panel = QWidget()
        layout = QVBoxLayout(panel)
        
        # File Input Section
        file_group = QGroupBox("📁 Input Files")
        file_layout = QVBoxLayout(file_group)
        
        self.batch_mode_check = QCheckBox("Process Multiple Files (Batch Mode)")
        self.batch_mode_check.setToolTip("Check this to select and process multiple image files in a batch.")
        self.batch_mode_check.toggled.connect(self.on_batch_mode_toggled)
        file_layout.addWidget(self.batch_mode_check)

        self.image_path_label = QLabel("No image selected")
        self.image_path_label.setStyleSheet("QLabel { background-color: #f0f0f0; padding: 5px; border: 1px solid #ccc; }")
        btn_select_image = QPushButton("Select Image Stack(s)")
        btn_select_image.clicked.connect(self.select_image_file)
        file_layout.addWidget(QLabel("Image Stack(s):"))
        file_layout.addWidget(self.image_path_label)
        file_layout.addWidget(btn_select_image)
        
        self.dapi_settings_label = QLabel("No DAPI settings loaded")
        self.dapi_settings_label.setStyleSheet("QLabel { background-color: #f0f0f0; padding: 5px; border: 1px solid #ccc; }")
        btn_load_dapi = QPushButton("Load DAPI Detection Settings")
        btn_load_dapi.clicked.connect(self.load_dapi_settings)
        file_layout.addWidget(QLabel("DAPI Settings (Default):"))
        file_layout.addWidget(self.dapi_settings_label)
        file_layout.addWidget(btn_load_dapi)
        
        self.af_settings_label = QLabel("No AF-specific settings loaded (Optional)")
        self.af_settings_label.setStyleSheet("QLabel { background-color: #f0f0f0; padding: 5px; border: 1px solid #ccc; }")
        btn_load_af = QPushButton("Load AF-Specific Settings (Optional)")
        btn_load_af.clicked.connect(self.load_af_settings)
        file_layout.addWidget(QLabel("Sample AF Settings (Override):"))
        file_layout.addWidget(self.af_settings_label)
        file_layout.addWidget(btn_load_af)
        layout.addWidget(file_group)
        
        # Preprocessing Options
        preprocess_group = QGroupBox("🔬 Preprocessing Options")
        preprocess_layout = QVBoxLayout(preprocess_group)
        self.apply_illum_corr_check = QCheckBox("Apply Illumination Correction")
        self.apply_illum_corr_check.setChecked(True)
        self.apply_illum_corr_check.setToolTip("Corrects for large-scale uneven illumination using the mean of all channels.")
        preprocess_layout.addWidget(self.apply_illum_corr_check)
        layout.addWidget(preprocess_group)

        # General Analysis Parameters
        params_group = QGroupBox("🔧 Analysis Parameters (Default)")
        params_layout = QGridLayout(params_group)
        params_layout.addWidget(QLabel("Max Distance (px):"), 0, 0)
        self.max_distance_spin = QDoubleSpinBox()
        self.max_distance_spin.setValue(5.0)
        self.max_distance_spin.setSingleStep(1.0)
        self.max_distance_spin.setToolTip("Maximum distance to search for a matching bead.")
        params_layout.addWidget(self.max_distance_spin, 0, 1)
        params_layout.addWidget(QLabel("Area Similarity:"), 1, 0)
        self.area_similarity_spin = QDoubleSpinBox()
        self.area_similarity_spin.setValue(0.5)
        self.area_similarity_spin.setSingleStep(0.05)
        self.area_similarity_spin.setRange(0.0, 1.0)
        self.area_similarity_spin.setToolTip("Minimum required ratio of bead areas (0.0 to 1.0).")
        params_layout.addWidget(self.area_similarity_spin, 1, 1)
        params_layout.addWidget(QLabel("Intensity Correlation:"), 2, 0)
        self.intensity_correlation_spin = QDoubleSpinBox()
        self.intensity_correlation_spin.setValue(0.3)
        self.intensity_correlation_spin.setSingleStep(0.05)
        self.intensity_correlation_spin.setRange(0.0, 1.0)
        self.intensity_correlation_spin.setToolTip("Minimum required ratio of bead intensities (0.0 to 1.0).")
        params_layout.addWidget(self.intensity_correlation_spin, 2, 1)
        self.enable_ransac_check = QCheckBox("Enable RANSAC Outlier Rejection")
        self.enable_ransac_check.setChecked(True)
        self.enable_ransac_check.setToolTip("Use RANSAC to remove outlier bead matches.")
        params_layout.addWidget(self.enable_ransac_check, 3, 0, 1, 2)
        params_layout.addWidget(QLabel("RANSAC Threshold (px):"), 4, 0)
        self.ransac_threshold_spin = QDoubleSpinBox()
        self.ransac_threshold_spin.setValue(2.0)
        self.ransac_threshold_spin.setSingleStep(0.5)
        self.ransac_threshold_spin.setToolTip("Maximum allowed distance from the model for a match to be considered an inlier.")
        params_layout.addWidget(self.ransac_threshold_spin, 4, 1)
        layout.addWidget(params_group)
        
        # AF-Specific Analysis Parameters
        self.af_params_group = QGroupBox("🔬 Sample AF Specific Analysis Parameters")
        self.af_params_group.setCheckable(True)
        self.af_params_group.setChecked(False)
        self.af_params_group.setToolTip("Check to enable and use specific analysis parameters for the Sample AF channel.")
        af_params_layout = QGridLayout(self.af_params_group)
        af_params_layout.addWidget(QLabel("Max Distance (px):"), 0, 0)
        self.af_max_distance_spin = QDoubleSpinBox()
        self.af_max_distance_spin.setValue(10.0)
        self.af_max_distance_spin.setSingleStep(1.0)
        af_params_layout.addWidget(self.af_max_distance_spin, 0, 1)
        af_params_layout.addWidget(QLabel("Area Similarity:"), 1, 0)
        self.af_area_similarity_spin = QDoubleSpinBox()
        self.af_area_similarity_spin.setValue(0.2)
        self.af_area_similarity_spin.setSingleStep(0.05)
        self.af_area_similarity_spin.setRange(0.0, 1.0)
        af_params_layout.addWidget(self.af_area_similarity_spin, 1, 1)
        af_params_layout.addWidget(QLabel("Intensity Correlation:"), 2, 0)
        self.af_intensity_correlation_spin = QDoubleSpinBox()
        self.af_intensity_correlation_spin.setValue(0.1)
        self.af_intensity_correlation_spin.setSingleStep(0.05)
        self.af_intensity_correlation_spin.setRange(0.0, 1.0)
        af_params_layout.addWidget(self.af_intensity_correlation_spin, 2, 1)
        self.af_enable_ransac_check = QCheckBox("Enable RANSAC Outlier Rejection")
        self.af_enable_ransac_check.setChecked(True)
        af_params_layout.addWidget(self.af_enable_ransac_check, 3, 0, 1, 2)
        af_params_layout.addWidget(QLabel("RANSAC Threshold (px):"), 4, 0)
        self.af_ransac_threshold_spin = QDoubleSpinBox()
        self.af_ransac_threshold_spin.setValue(3.0)
        self.af_ransac_threshold_spin.setSingleStep(0.5)
        af_params_layout.addWidget(self.af_ransac_threshold_spin, 4, 1)
        layout.addWidget(self.af_params_group)

        # Settings Preview Section
        preview_group = QGroupBox("⚙️ Channel Settings Preview")
        preview_layout = QVBoxLayout(preview_group)
        self.settings_table = QTableWidget()
        self.settings_table.setColumnCount(3)
        self.settings_table.setHorizontalHeaderLabels(["Channel", "Source", "Key Parameters"])
        self.settings_table.setRowCount(len(self.channel_names))
        for i, channel_name in enumerate(self.channel_names):
            self.settings_table.setItem(i, 0, QTableWidgetItem(channel_name))
            self.settings_table.setItem(i, 1, QTableWidgetItem("Not loaded"))
            self.settings_table.setItem(i, 2, QTableWidgetItem("--"))
        preview_layout.addWidget(self.settings_table)
        layout.addWidget(preview_group)
        
        # Action buttons
        button_layout = QVBoxLayout()
        self.btn_analyze = QPushButton("🔍 Start Analysis")
        self.btn_analyze.clicked.connect(self.start_analysis)
        self.btn_analyze.setEnabled(False)
        self.btn_analyze.setStyleSheet("QPushButton { background-color: #4CAF50; color: white; font-weight: bold; padding: 10px; }")
        self.btn_export = QPushButton("📊 Export Last Result")
        self.btn_export.clicked.connect(self.export_results)
        self.btn_export.setEnabled(False)
        button_layout.addWidget(self.btn_analyze)
        button_layout.addWidget(self.btn_export)
        layout.addLayout(button_layout)
        
        layout.addStretch()
        return panel
    
    def create_right_panel(self):
            """Create the right panel with log and progress."""
            panel = QWidget()
            layout = QVBoxLayout(panel)
            
            # Progress section
            progress_group = QGroupBox("📊 Analysis Progress")
            progress_layout = QVBoxLayout(progress_group)
            
            self.progress_label = QLabel("Ready to start analysis")
            self.progress_bar = QProgressBar()
            self.progress_bar.setRange(0, 100)
            
            progress_layout.addWidget(self.progress_label)
            progress_layout.addWidget(self.progress_bar)
            layout.addWidget(progress_group)
            
            # Log section
            log_group = QGroupBox("📝 Analysis Log")
            log_layout = QVBoxLayout(log_group)
            
            self.log_text = QTextEdit()
            self.log_text.setFont(QFont("Consolas", 9))
            self.log_text.setReadOnly(True)
            self.log_text.setStyleSheet("QTextEdit { background-color: #1e1e1e; color: #ffffff; }")
            
            log_layout.addWidget(self.log_text)
            layout.addWidget(log_group)
            
            return panel
    
    def on_batch_mode_toggled(self, checked):
        """Resets file selections when batch mode is toggled."""
        self.log_message(f"Batch mode {'enabled' if checked else 'disabled'}. Resetting file selection.", "INFO")
        self.image_file_paths = []
        self.image_stack = None
        self.image_path_label.setText("No image selected")
        self.image_path_label.setStyleSheet("QLabel { background-color: #f0f0f0; padding: 5px; border: 1px solid #ccc; }")
        self.check_ready_state()

    def _load_image(self, file_path):
        """Loads a single image stack and updates UI. Returns True on success."""
        try:
            self.log_message(f"📁 Loading image stack: {Path(file_path).name}")
            self.image_stack = tifffile.imread(file_path)
            
            if self.image_stack.ndim == 2:
                self.image_stack = self.image_stack[np.newaxis, ...]
            
            num_channels = self.image_stack.shape[0]
            if num_channels != 7:
                self.log_message(f"⚠️ Warning: Expected 7 channels, found {num_channels}", "WARNING")
                if num_channels < 7:
                    self.channel_names = self.channel_names[:num_channels]
                else: # If more than 7, use generic names
                    self.channel_names = [f"Channel {i+1}" for i in range(num_channels)]
                self.settings_table.setRowCount(num_channels) # Adjust table size
                self.update_settings_preview()
                self.log_message(f"📋 Adjusted to {num_channels} channels.")
            
            self.image_path_label.setText(f"✅ {Path(file_path).name} ({self.image_stack.shape})")
            self.image_path_label.setStyleSheet("QLabel { background-color: #e8f5e8; padding: 5px; border: 1px solid #4CAF50; }")
            
            self.log_message(f"✅ Image loaded: {self.image_stack.shape}, {self.image_stack.nbytes / 1e6:.1f} MB", "SUCCESS")
            self.check_ready_state()
            return True
            
        except Exception as e:
            self.log_message(f"❌ Failed to load image {Path(file_path).name}: {str(e)}", "ERROR")
            self.image_stack = None
            self.check_ready_state()
            QMessageBox.critical(self, "Error", f"Failed to load image:\n{Path(file_path).name}\n\n{str(e)}")
            return False


    def log_message(self, message, level="INFO"):
            """Add a timestamped message to the log."""
            timestamp = time.strftime("%H:%M:%S")
            
            # Color coding based on level
            colors = {
                "INFO": "#ffffff",
                "SUCCESS": "#4CAF50", 
                "WARNING": "#FFC107",
                "ERROR": "#f44336",
                "DEBUG": "#2196F3"
            }
            color = colors.get(level, "#ffffff")
            
            formatted_message = f'<span style="color: {color};">[{timestamp}] {message}</span>'
            self.log_text.append(formatted_message) # append() is thread-safe and auto-scrolls

    def select_image_file(self):
        """Select single or multiple image stack files based on batch mode."""
        self.image_file_paths = [] # Clear previous selection
        
        if self.batch_mode_check.isChecked():
            file_paths, _ = QFileDialog.getOpenFileNames(
                self, "Select Image Stacks for Batch Processing", "",
                "TIFF Files (*.tiff *.ome.tiff);;All Files (*.*)"
            )
            if not file_paths:
                self.check_ready_state()
                return
            
            self.image_file_paths = file_paths
            self.log_message(f"📁 Selected {len(self.image_file_paths)} files for batch processing.")
            self.image_path_label.setText(f"✅ {len(self.image_file_paths)} files selected")
            self.image_path_label.setStyleSheet("QLabel { background-color: #e8f5e8; padding: 5px; border: 1px solid #4CAF50; }")
            self.image_stack = None # In batch mode, we load one by one later
            self.check_ready_state()

        else: # Single file mode
            file_path, _ = QFileDialog.getOpenFileName(
                self, "Select Image Stack", "",
                "TIFF Files (*.tiff *.ome.tiff);;All Files (*.*)"
            )
            if not file_path:
                self.check_ready_state()
                return
                
            self.image_file_paths = [file_path]
            self._load_image(file_path)

    def load_dapi_settings(self):
        """Load DAPI detection settings (default for all channels)."""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Select DAPI Detection Settings", "",
            "JSON Files (*.json);;All Files (*.*)"
        )
        
        if not file_path:
            return
            
        try:
            self.log_message(f"📄 Loading DAPI settings: {Path(file_path).name}")
            
            with open(file_path, 'r') as f:
                settings_data = json.load(f)
            
            # Extract detection parameters
            self.default_settings = settings_data.get('parameters', {})
            
            # Update UI
            self.dapi_settings_label.setText(f"✅ {Path(file_path).name}")
            self.dapi_settings_label.setStyleSheet("QLabel { background-color: #e8f5e8; padding: 5px; border: 1px solid #4CAF50; }")
            
            self.log_message("✅ DAPI settings loaded successfully", "SUCCESS")
            self.log_message(f"📋 Key parameters: method={self.default_settings.get('detection_method', 'N/A')}, "
                        f"sensitivity={self.default_settings.get('sensitivity', 'N/A')}")
            
            self.update_settings_preview()
            self.check_ready_state()
            
        except Exception as e:
            self.log_message(f"❌ Failed to load DAPI settings: {str(e)}", "ERROR")
            QMessageBox.critical(self, "Error", f"Failed to load DAPI settings:\n{str(e)}")

    def load_af_settings(self):
        """Load AF-specific detection settings (optional override)."""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Select AF Detection Settings", "",
            "JSON Files (*.json);;All Files (*.*)"
        )
        
        if not file_path:
            self.log_message("ℹ️ No AF settings selected - will use DAPI defaults for Sample AF", "INFO")
            return
            
        try:
            self.log_message(f"📄 Loading AF-specific settings: {Path(file_path).name}")
            
            with open(file_path, 'r') as f:
                settings_data = json.load(f)
            
            # Extract detection parameters
            self.af_settings = settings_data.get('parameters', {})
            
            # Update UI
            self.af_settings_label.setText(f"✅ {Path(file_path).name}")
            self.af_settings_label.setStyleSheet("QLabel { background-color: #e8f5e8; padding: 5px; border: 1px solid #4CAF50; }")
            
            self.log_message("✅ AF-specific settings loaded successfully", "SUCCESS")
            self.log_message(f"📋 AF parameters: method={self.af_settings.get('detection_method', 'N/A')}, "
                        f"sensitivity={self.af_settings.get('sensitivity', 'N/A')}")
            
            self.update_settings_preview()
            
        except Exception as e:
            self.log_message(f"❌ Failed to load AF settings: {str(e)}", "ERROR")
            QMessageBox.critical(self, "Error", f"Failed to load AF settings:\n{str(e)}")



    def get_channel_settings(self, channel_name):
            """Get detection settings for a specific channel."""
            # Use AF-specific settings for Sample AF if available
            if channel_name == "Sample AF" and self.af_settings is not None:
                self.log_message(f"   🎯 Using AF-specific settings for {channel_name}", "DEBUG")
                return self.af_settings
            
            # Use default DAPI settings for all other channels
            if self.default_settings is not None:
                self.log_message(f"   📋 Using DAPI default settings for {channel_name}", "DEBUG")
                return self.default_settings
            
            self.log_message(f"   ⚠️ No settings available for {channel_name}", "WARNING")
            return {}
        
    def update_settings_preview(self):
            """Update the settings preview table."""
            if not self.default_settings:
                return
                
            for i, channel_name in enumerate(self.channel_names):
                if i >= self.settings_table.rowCount():
                    continue
                    
                settings = self.get_channel_settings(channel_name)
                
                # Determine source
                if channel_name == "Sample AF" and self.af_settings is not None:
                    source = "AF-specific"
                elif channel_name == "Opal 780":
                    source = "None (skipped)"
                else:
                    source = "DAPI default"
                
                # Key parameters summary
                if settings:
                    key_params = (f"Method: {settings.get('detection_method', 'N/A')}, "
                                f"Sens: {settings.get('sensitivity', 'N/A')}, "
                                f"Area: {settings.get('min_area', 'N/A')}-{settings.get('max_area', 'N/A')}")
                else:
                    key_params = "No parameters"
                
                # Update table items
                source_item = QTableWidgetItem(source)
                source_item.setBackground(Qt.lightGray if channel_name == "Opal 780" else Qt.white)
                
                self.settings_table.setItem(i, 1, source_item)
                self.settings_table.setItem(i, 2, QTableWidgetItem(key_params))
            
            self.settings_table.resizeColumnsToContents()

    def check_ready_state(self):
        """Check if all required inputs are loaded and enable analysis button."""
        image_ready = bool(self.image_file_paths)
        settings_ready = self.default_settings is not None
        
        ready = image_ready and settings_ready
        self.btn_analyze.setEnabled(ready)
        
        if ready:
            self.log_message("🎯 Ready to start analysis!", "SUCCESS")
        else:
            missing = []
            if not image_ready:
                missing.append("Image stack(s)")
            if not settings_ready:
                missing.append("DAPI settings")
            if missing:
                self.log_message(f"⏳ Waiting for: {', '.join(missing)}")


    def start_analysis(self):
        """Start the analysis for a single file or a batch of files."""
        if not self.image_file_paths:
            self.log_message("❌ No image files selected.", "ERROR")
            return

        self.is_batch_processing = self.batch_mode_check.isChecked() and len(self.image_file_paths) > 1
        self.current_file_index = 0
        self.btn_analyze.setEnabled(False)
        self.btn_export.setEnabled(False)
        
        if self.is_batch_processing:
            self.log_message(f"🚀 Starting batch analysis for {len(self.image_file_paths)} file(s)...", "INFO")
        else:
            self.log_message("🚀 Starting analysis...", "INFO")
            
        self.process_next_file()

    def process_next_file(self):
        """Process the next file in the list. This is the core of the batch loop."""
        if self.current_file_index >= len(self.image_file_paths):
            self.log_message("🎉 Analysis complete for all files.", "SUCCESS")
            self.is_batch_processing = False
            self.btn_analyze.setEnabled(True)
            self.progress_label.setText("Batch complete!")
            # The last result is still in memory, so enable manual export
            if self.shift_analysis:
                self.btn_export.setEnabled(True)
            return

        self.current_processing_file_path = self.image_file_paths[self.current_file_index]
        
        self.log_message("-" * 50, "INFO")
        self.log_message(f"Processing file {self.current_file_index + 1}/{len(self.image_file_paths)}: {Path(self.current_processing_file_path).name}", "INFO")
        
        # Load the image for the current file; _load_image handles logging and UI updates
        if not self._load_image(self.current_processing_file_path):
            self.log_message(f"Skipping file due to loading error: {Path(self.current_processing_file_path).name}", "WARNING")
            self.current_file_index += 1
            QTimer.singleShot(100, self.process_next_file) # Process next file after a short delay
            return
            
        self.progress_bar.setValue(0)
        
        # Gather parameters for the thread
        similarity_params = {
            'max_distance': self.max_distance_spin.value(),
            'area_similarity': self.area_similarity_spin.value(),
            'intensity_correlation': self.intensity_correlation_spin.value()
        }
        apply_illum_corr = self.apply_illum_corr_check.isChecked()
        ransac_params = {
            'enabled': self.enable_ransac_check.isChecked(),
            'threshold': self.ransac_threshold_spin.value()
        }
        af_similarity_params, af_ransac_params = None, None
        if self.af_params_group.isChecked():
            af_similarity_params = {'max_distance': self.af_max_distance_spin.value(), 'area_similarity': self.af_area_similarity_spin.value(), 'intensity_correlation': self.af_intensity_correlation_spin.value()}
            af_ransac_params = {'enabled': self.af_enable_ransac_check.isChecked(), 'threshold': self.af_ransac_threshold_spin.value()}

        # Create and start the analysis thread for the current image
        self.analysis_thread = AnalysisThread(
            self.image_stack, self.channel_names, self.default_settings, self.af_settings,
            similarity_params, apply_illum_corr, ransac_params, af_similarity_params, af_ransac_params
        )
        self.analysis_thread.progress_update.connect(self.update_progress)
        self.analysis_thread.log_message.connect(self.append_log_from_thread)
        self.analysis_thread.analysis_complete.connect(self.analysis_finished)
        self.analysis_thread.start()


    @pyqtSlot(str, str)
    def append_log_from_thread(self, message, level):
            """A dedicated, thread-safe slot to receive log messages."""
            self.log_message(message, level)
        
    def update_progress(self, value, message):
            """Update progress bar and label."""
            self.progress_bar.setValue(value)
            self.progress_label.setText(message)
            
    def analysis_finished(self, success, message, results):
        """Handle analysis completion for one file and trigger the next if in a batch."""
        if success:
            self.log_message(message, "SUCCESS")
            self.detected_beads = results.get('detected_beads', {})
            self.bead_matches = results.get('bead_matches', {})
            self.shift_analysis = results.get('shift_analysis', {})
            
            # In batch mode, auto-save results
            if self.is_batch_processing:
                self._auto_export_results()
            else: # In single file mode, enable manual export
                self.btn_export.setEnabled(True)
                self.btn_analyze.setEnabled(True)
        else:
            self.log_message(message, "ERROR")
            if not self.is_batch_processing:
                QMessageBox.critical(self, "Analysis Error", message)
                self.btn_analyze.setEnabled(True)

        # If in a batch, move to the next file
        if self.is_batch_processing or (not success and self.current_file_index < len(self.image_file_paths) - 1):
            self.current_file_index += 1
            QTimer.singleShot(100, self.process_next_file)

    def _get_output_base_path(self, input_path_str):
        """Generates the base output path from an input file path."""
        p = Path(input_path_str)
        base_name = p.name
        # Handle .ome.tiff first to avoid incorrect stem
        if base_name.lower().endswith('.ome.tiff'):
            stem = base_name[:-9]
        else:
            stem = p.stem
        return p.parent / f"{stem}_bead_shift_analysis"

    def _auto_export_results(self):
        """Automatically exports results for the currently processed file."""
        if not self.shift_analysis:
            self.log_message(f"No analysis results to auto-export for {Path(self.current_processing_file_path).name}", "WARNING")
            return
        
        try:
            output_base_path = self._get_output_base_path(self.current_processing_file_path)
            # Don't use with_suffix as it truncates at the first dot!
            json_path = f"{output_base_path}.json"
            
            self._perform_export(json_path)
            
        except Exception as e:
            error_msg = f"Failed to auto-export results: {str(e)}"
            self.log_message(f"❌ {error_msg}", "ERROR")
            self.log_message(f"Full traceback:\n{traceback.format_exc()}", "DEBUG")

        # ========================================
        # PART 6: RESULTS EXPORT & REPORTING
        # ========================================
        
    def _prepare_for_json(self, data):
        """Recursively converts numpy arrays and types to JSON-serializable lists and floats."""
        if isinstance(data, dict):
            return {key: self._prepare_for_json(value) for key, value in data.items()}
        if isinstance(data, list):
            return [self._prepare_for_json(item) for item in data]
        if isinstance(data, np.ndarray):
            return data.tolist()
        if isinstance(data, (np.float32, np.float64)):
            return float(data)
        if isinstance(data, (np.int32, np.int64)):
            return int(data)
        return data

    def export_results(self):
        """Manually export the most recent analysis results."""
        if not self.shift_analysis:
            QMessageBox.warning(self, "No Results", "No analysis results to export. Run analysis first.")
            return
        
        # Suggest a default filename based on the last processed file
        default_path = "bead_shift_analysis.json"
        if self.current_processing_file_path:
            output_base = self._get_output_base_path(self.current_processing_file_path)
            # Don't use with_suffix as it truncates at the first dot!
            default_path = f"{output_base}.json"

        file_path, _ = QFileDialog.getSaveFileName(
            self, "Export Analysis Results", default_path,
            "JSON Files (*.json);;All Files (*.*)"
        )
        if not file_path:
            return
            
        self._perform_export(file_path)
        
    def _perform_export(self, json_file_path):
        """Core logic to save results to a specified file path."""
        try:
            self.log_message(f"📤 Saving results to: {Path(json_file_path).name}")
            
            # Prepare data for export
            detection_export = {}
            for channel, beads_data in self.detected_beads.items():
                if beads_data is None:
                    detection_export[channel] = []
                    continue
                num_beads = beads_data['centroids'].shape[0]
                detection_export[channel] = [{
                    'centroid': beads_data['centroids'][i].tolist(),
                    'area': float(beads_data['areas'][i]),
                    'intensity': float(beads_data['intensities'][i])
                } for i in range(num_beads)]
            
            matching_export = {}
            ref_beads_all = self.detected_beads.get("DAPI")
            for channel, matches_data in self.bead_matches.items():
                if not matches_data or ref_beads_all is None:
                    matching_export[channel] = []
                    continue
                target_beads_all = self.detected_beads[channel]
                num_matches = len(matches_data['reference_indices'])
                matching_export[channel] = [{
                    'reference_centroid': ref_beads_all['centroids'][matches_data['reference_indices'][i]].tolist(),
                    'target_centroid': target_beads_all['centroids'][matches_data['target_indices'][i]].tolist(),
                    'distance': float(matches_data['distances'][i]),
                    'similarity_score': float(matches_data['similarities'][i])
                } for i in range(num_matches)]

            export_data = {
                'metadata': { 'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'), 'analyzer_version': '1.1-batch', 'image_path': self.current_processing_file_path, 'image_shape': list(self.image_stack.shape) if self.image_stack is not None else None, 'channel_names': self.channel_names },
                'matching_parameters': { 'default': self._prepare_for_json({ 'similarity': getattr(self.analysis_thread, 'similarity_params', {}), 'ransac': getattr(self.analysis_thread, 'ransac_params', {}) }), 'overrides': self._prepare_for_json({ 'Sample AF': { 'similarity': getattr(self.analysis_thread, 'af_similarity_params', None), 'ransac': getattr(self.analysis_thread, 'af_ransac_params', None) } } if getattr(self.analysis_thread, 'af_similarity_params', None) else {}) },
                'detection_settings': { 'default_settings': self.default_settings, 'af_settings': self.af_settings },
                'detection_results': detection_export,
                'matching_results': matching_export,
                'shift_analysis': self._prepare_for_json(self.shift_analysis)
            }
            
            with open(json_file_path, 'w') as f:
                json.dump(export_data, f, indent=2)
            
            csv_path = str(Path(json_file_path).with_suffix('.csv'))
            self.export_csv_summary(csv_path)
            
            self.log_message(f"✅ Results successfully saved to {Path(json_file_path).name} and {Path(csv_path).name}!", "SUCCESS")
            
            # Show a confirmation dialog only for manual exports
            if not self.is_batch_processing:
                QMessageBox.information(self, "Export Complete", f"Results exported to:\n{json_file_path}\nand\n{csv_path}")
                
        except Exception as e:
            error_msg = f"Failed to export results: {str(e)}"
            self.log_message(f"❌ {error_msg}", "ERROR")
            self.log_message(f"Full traceback:\n{traceback.format_exc()}", "DEBUG")
            if not self.is_batch_processing:
                QMessageBox.critical(self, "Export Error", error_msg)

    def export_csv_summary(self, csv_path):
        """Export a CSV summary of shift analysis, now robust to missing data."""
        try:
            with open(csv_path, 'w', newline='') as f:
                writer = csv.writer(f)
                
                writer.writerow([
                    'Channel', 'Bead_Count', 'Median_DY', 'Median_DX', 'Mean_DY', 'Mean_DX',
                    'STD_DY', 'STD_DX', 'Mean_Magnitude', 'STD_Magnitude', 'Quality_Score',
                    'Confidence', 'Warnings'
                ])
                
                for channel_name in self.channel_names:
                    if channel_name == "DAPI" or channel_name not in self.shift_analysis:
                        continue
                    
                    data = self.shift_analysis.get(channel_name)
                    if not data: continue

                    stats = data['statistics']
                    quality = data['quality_metrics']
                    
                    writer.writerow([
                        channel_name,
                        data['bead_count'],
                        f"{stats.get('median_dy', 0):.4f}",
                        f"{stats.get('median_dx', 0):.4f}",
                        f"{stats.get('mean_dy', 0):.4f}",
                        f"{stats.get('mean_dx', 0):.4f}",
                        f"{stats.get('std_dy', 0):.4f}",
                        f"{stats.get('std_dx', 0):.4f}",
                        f"{stats.get('mean_magnitude', 0):.4f}",
                        f"{stats.get('std_magnitude', 0):.4f}",
                        f"{quality.get('overall_quality', 0):.4f}",
                        quality.get('confidence', 'N/A'),
                        '; '.join(quality.get('warnings', [])) or 'None'
                    ])
        except Exception as e:
            self.log_message(f"❌ Error exporting CSV: {str(e)}", "ERROR")
            """Export a CSV summary of shift analysis."""
            try:
                import csv
                
                with open(csv_path, 'w', newline='') as f:
                    writer = csv.writer(f)
                    
                    # Write header
                    writer.writerow([
                        'Channel', 'Bead_Count', 'Median_DY', 'Median_DX', 'Mean_DY', 'Mean_DX',
                        'STD_DY', 'STD_DX', 'Mean_Magnitude', 'STD_Magnitude', 'Quality_Score',
                        'Confidence', 'Warnings'
                    ])
                    
                    # Write data for each channel
                    for channel_name, data in self.shift_analysis.items():
                        try:
                            stats = data['statistics']
                            quality = data['quality_metrics']
                            
                            writer.writerow([
                                channel_name,
                                data['bead_count'],
                                f"{stats['median_dy']:.4f}",
                                f"{stats['median_dx']:.4f}",
                                f"{stats['mean_dy']:.4f}",
                                f"{stats['mean_dx']:.4f}",
                                f"{stats['std_dy']:.4f}",
                                f"{stats['std_dx']:.4f}",
                                f"{stats['mean_magnitude']:.4f}",
                                f"{stats['std_magnitude']:.4f}",
                                f"{quality['overall_quality']:.4f}",
                                quality['confidence'],
                                '; '.join(quality['warnings']) if quality['warnings'] else 'None'
                            ])
                        except Exception as e:
                            self.log_message(f"⚠️ Error writing CSV row for {channel_name}: {str(e)}", "WARNING")
                            
            except Exception as e:
                self.log_message(f"❌ Error exporting CSV: {str(e)}", "ERROR")

class AnalysisThread(QThread):
    """Thread for running analysis without freezing the UI. Includes GPU acceleration."""
    progress_update = pyqtSignal(int, str)
    log_message = pyqtSignal(str, str)
    analysis_complete = pyqtSignal(bool, str, dict)

    def __init__(self, image_stack, channel_names, default_settings, af_settings, similarity_params, apply_illum_corr, ransac_params, af_similarity_params=None, af_ransac_params=None):
        super().__init__()
        # Data passed from main thread
        self.image_stack = image_stack
        self.channel_names = channel_names
        self.default_settings = default_settings
        self.af_settings = af_settings
        self.similarity_params = similarity_params
        self.apply_illum_corr = apply_illum_corr
        self.ransac_params = ransac_params
        self.af_similarity_params = af_similarity_params
        self.af_ransac_params = af_ransac_params

        # Analysis results stored in thread
        self.detected_beads = {}
        self.bead_matches = {}
        self.shift_analysis = {}

        # Check for GPU support
        self.gpu_available = GPU_ENABLED
        if self.gpu_available and not hasattr(self, '_already_logged_gpu'):
            self.log_message.emit("✅ NVIDIA GPU support detected (CuPy/cuCIM found). Analysis will be accelerated.", "SUCCESS")
            AnalysisThread._already_logged_gpu = True
        elif not hasattr(self, '_already_logged_gpu'):
            self.log_message.emit("⚠️ GPU support not found. Falling back to CPU-only analysis.", "WARNING")
            AnalysisThread._already_logged_gpu = True


    def run(self):
        """Run the complete analysis within the thread."""
        try:
            if self.apply_illum_corr:
                # Illumination correction still runs on CPU as it's fast enough
                self.image_stack = self.correct_illumination(self.image_stack)

            # Part 2: Bead Detection (will use GPU if available and multiprocessing)
            self.log_message.emit("🔍 Part 2: Starting bead detection (using parallel processing)...", "INFO")
            self.detect_beads_all_channels()

            total_beads = sum(data['centroids'].shape[0] for data in self.detected_beads.values() if data and 'centroids' in data)
            if total_beads == 0:
                raise Exception("No beads detected in any channel. Check detection settings.")

            # Part 3: Bead Matching (using parallel processing)
            self.log_message.emit("🎯 Part 3: Starting bead matching (using parallel processing)...", "INFO")
            self.match_beads_to_reference()

            total_matches = sum(len(m['reference_indices']) for m in self.bead_matches.values() if m and 'reference_indices' in m)
            if total_matches == 0:
                raise Exception("No bead matches found. Try adjusting similarity parameters.")

            # Part 4: Shift Calculation
            self.log_message.emit("📊 Part 4: Starting shift calculation...", "INFO")
            self.calculate_shift_analysis()

            results = {
                'detected_beads': self.detected_beads,
                'bead_matches': self.bead_matches,
                'shift_analysis': self.shift_analysis
            }
            self.progress_update.emit(100, "Analysis complete!")
            self.analysis_complete.emit(True, "🎉 Complete analysis finished successfully!", results)

        except Exception as e:
            error_msg = f"❌ Analysis failed: {str(e)}"
            self.log_message.emit(error_msg, "ERROR")
            self.log_message.emit(f"Full traceback:\n{traceback.format_exc()}", "DEBUG")
            self.analysis_complete.emit(False, error_msg, {})
            
    def _log_from_queue(self, log_queue):
        """Helper to pull log messages from a multiprocessing queue."""
        while not log_queue.empty():
            message, level = log_queue.get()
            self.log_message.emit(message, level)


    # ===================================================================
    # VECTORIZED & PARALLELIZED CORE LOGIC
    # ===================================================================


    def detect_beads_all_channels(self):
            """
            Detects beads in all channels using a multiprocessing pool for parallelism.
            """
            self.detected_beads = {}
            tasks = []
            
            # Use a manager queue for logging from subprocesses
            manager = multiprocessing.Manager()
            log_queue = manager.Queue()

            for i, channel_name in enumerate(self.channel_names):
                if channel_name == "Opal 780":
                    self.log_message.emit(f"⏭️ Skipping {channel_name} (as specified)", "INFO")
                    self.detected_beads[channel_name] = None
                    continue

                settings = self.get_channel_settings(channel_name)
                if not settings:
                    self.log_message.emit(f"⚠️ No settings for {channel_name}, skipping", "WARNING")
                    self.detected_beads[channel_name] = None
                    continue
                
                tasks.append((i, channel_name, self.image_stack[i], settings, log_queue))

            # Limit the number of worker processes to prevent memory exhaustion
            max_workers = min(4, multiprocessing.cpu_count())
            self.log_message.emit(f"🔬 Distributing bead detection for {len(tasks)} channels across up to {max_workers} CPU cores...", "INFO")
            self.progress_update.emit(5, "Detecting beads in parallel...")

            # Run detection in parallel
            try:
                # Use partial to pass the 'gpu_available' flag to the worker function
                worker_func = partial(_run_detection_task, gpu_available=self.gpu_available)
                with multiprocessing.Pool(processes=max_workers) as pool:
                    results = pool.map(worker_func, tasks)
                
                self._log_from_queue(log_queue) # Process logs from the queue
                for channel_name, beads_data in results:
                    self.detected_beads[channel_name] = beads_data
                    count = beads_data['centroids'].shape[0] if beads_data else 0
                    self.log_message.emit(f"✅ {channel_name}: {count} beads detected", "INFO")

            except Exception as e:
                self._log_from_queue(log_queue)
                self.log_message.emit(f"❌ Multiprocessing for detection failed: {e}", "ERROR")
                raise

            self.progress_update.emit(30, "Bead detection complete.")
            total_beads = sum(data['centroids'].shape[0] for data in self.detected_beads.values() if data)
            self.log_message.emit(f"🎯 Detection complete: {total_beads} total beads found", "SUCCESS")


    def match_beads_to_reference(self):
        """
        Matches beads to the DAPI reference channel using a multiprocessing pool.
        """
        self.log_message.emit("🎯 Part 3: Matching beads to DAPI reference...", "INFO")
        reference_channel = "DAPI"
        reference_beads = self.detected_beads.get(reference_channel)

        if not reference_beads or reference_beads['centroids'].shape[0] == 0:
            raise Exception(f"No beads detected in reference channel '{reference_channel}'")

        ref_count = reference_beads['centroids'].shape[0]
        self.log_message.emit(f"📍 Using {ref_count} beads from {reference_channel} as reference", "INFO")
        self.bead_matches = {}
        
        manager = multiprocessing.Manager()
        log_queue = manager.Queue()
        tasks = []
        
        channels_to_match = [ch for ch in self.channel_names if ch not in [reference_channel, "Opal 780"]]
        for channel_name in channels_to_match:
            target_beads = self.detected_beads.get(channel_name)
            if not target_beads or target_beads['centroids'].shape[0] == 0:
                self.log_message.emit(f"⚠️ No beads in {channel_name}, skipping matching", "WARNING")
                self.bead_matches[channel_name] = None
                continue
            current_sim_params = self.similarity_params
            current_ransac_params = self.ransac_params
            if channel_name == "Sample AF":
                if self.af_similarity_params:
                    self.log_message.emit(f"   🎯 Using specific similarity parameters for {channel_name}", "DEBUG")
                    current_sim_params = self.af_similarity_params
                if self.af_ransac_params:
                    self.log_message.emit(f"   🎯 Using specific RANSAC parameters for {channel_name}", "DEBUG")
                    current_ransac_params = self.af_ransac_params

            tasks.append((channel_name, target_beads, reference_beads, current_sim_params, current_ransac_params, log_queue))


        if not tasks:
            self.log_message.emit("🤷 No channels with beads to match.", "WARNING")
            return
            
        # Limit the number of worker processes
        max_workers = min(4, multiprocessing.cpu_count())
        self.log_message.emit(f"🔗 Distributing bead matching for {len(tasks)} channels across up to {max_workers} CPU cores...", "INFO")
        self.progress_update.emit(35, "Matching beads in parallel...")

        try:
            with multiprocessing.Pool(processes=max_workers) as pool:
                results = pool.map(_run_matching_task, tasks)
                
            self._log_from_queue(log_queue)
            for channel_name, matches_data in results:
                self.bead_matches[channel_name] = matches_data
                if matches_data:
                    match_count = len(matches_data['reference_indices'])
                    target_count = self.detected_beads[channel_name]['centroids'].shape[0]
                    match_rate = match_count / target_count * 100 if target_count > 0 else 0
                    self.log_message.emit(f"✅ {channel_name}: {match_count} matches ({match_rate:.1f}% match rate)", "SUCCESS")
                else:
                    self.log_message.emit(f"✅ {channel_name}: 0 matches found.", "INFO")

        except Exception as e:
            self._log_from_queue(log_queue)
            self.log_message.emit(f"❌ Multiprocessing for matching failed: {e}", "ERROR")
            raise

        self.progress_update.emit(60, "Bead matching complete.")
        total_matches = sum(len(m['reference_indices']) for m in self.bead_matches.values() if m)
        self.log_message.emit(f"🎯 Matching complete: {total_matches} total bead pairs matched", "SUCCESS")


    # ===================================================================
    # DETECTION FUNCTIONS (MODIFIED TO RETURN NUMPY DATA)
    # ===================================================================
    def _beads_list_to_numpy(self, beads_list):
        """Converts a list of bead dictionaries to a dictionary of numpy arrays."""
        if not beads_list:
            return None # Return None if no beads were found
        
        # Use float64 for precision in centroids
        centroids = np.array([b['centroid'] for b in beads_list], dtype=np.float64)
        areas = np.array([b['area'] for b in beads_list], dtype=np.float32)
        intensities = np.array([b['intensity'] for b in beads_list], dtype=np.float32)
        
        return {
            'centroids': centroids,
            'areas': areas,
            'intensities': intensities
        }

    def detect_beads_advanced_gpu(self, image_cpu, settings, channel_name):
        try:
            image = cp.asarray(image_cpu, dtype=cp.float32)
            min_area = int(settings.get('min_area', 4))
            max_area = int(settings.get('max_area', 500))
            sensitivity = float(settings.get('sensitivity', 85.0))
            subpixel_refinement = bool(settings.get('subpixel_refinement', True))
            noise_size = float(settings.get('noise_size', 3.0))

            # Integrated statistical denoising (GPU version)
            if noise_size > 0:
                img_mean = cp.mean(image)
                img_std = cp.std(image)
                threshold = img_mean + noise_size * img_std
                denoised = cp.maximum(image - threshold, 0)
            else:
                denoised = image

            # Blob detection on the denoised image
            img_norm = denoised / (denoised.max() + 1e-10)
            min_sigma = max(0.5, cp.sqrt(min_area / cp.pi) * 0.5)
            max_sigma = cp.sqrt(max_area / cp.pi) * 2
            if max_sigma <= min_sigma: max_sigma = min_sigma + 1.0
            threshold = max(0.001, 0.05 * ((100.0 - sensitivity) / 100.0))

            blobs = cucim.skimage.feature.blob_log(img_norm, min_sigma=min_sigma, max_sigma=max_sigma, num_sigma=15, threshold=threshold)
            
            beads_list = []
            h, w = image.shape
            
            for blob in blobs:
                y, x, sigma = float(blob[0]), float(blob[1]), float(blob[2])
                
                if subpixel_refinement:
                    y, x = self._refine_centroid_radial_symmetry(image_cpu, y, x)
                
                if not (0 <= y < h and 0 <= x < w): continue
                
                area = float(cp.pi * (cp.sqrt(2) * sigma)**2)
                
                if min_area <= area <= max_area:
                    beads_list.append({
                        'centroid': (y, x),
                        'area': area,
                        'intensity': float(image_cpu[int(y), int(x)])
                    })
            return self._beads_list_to_numpy(beads_list)

        except Exception as e:
            self.log_message.emit(f"❌ GPU detection failed for {channel_name}: {e}. Falling back to CPU.", "ERROR")
            return self.detect_beads_advanced(image_cpu, settings, channel_name)


    def detect_beads_advanced(self, image, settings, channel_name):
        if image is None or image.size == 0: return None
        try:
            detection_method = settings.get('detection_method', 'hybrid')
            noise_size = float(settings.get('noise_size', 3.0))
            background_size = int(settings.get('background_size', 30))
            merge_distance = float(settings.get('merge_distance', 10.0))
            include_borders = bool(settings.get('include_borders', False))
            auto_merge = bool(settings.get('auto_merge', True))

            working_image = image.astype(np.float32)
            
            # Integrated statistical denoising (CPU version)
            if noise_size > 0:
                img_mean = np.mean(working_image)
                img_std = np.std(working_image)
                threshold = img_mean + noise_size * img_std
                denoised = np.maximum(working_image - threshold, 0)
            else:
                denoised = working_image
            
            # Background subtraction on the denoised image
            if background_size > 0:
                selem = morphology.disk(max(1, int(background_size / 2)))
                background = morphology.opening(denoised, selem)
                foreground = denoised - background
                detection_image = np.maximum(foreground, 0)
            else:
                detection_image = denoised
            
            exclude_border = not include_borders
            if detection_method == 'local_max':
                beads_list = self._detect_local_maxima(detection_image, exclude_border, settings)
            elif detection_method == 'blob':
                beads_list = self._detect_blob_log(detection_image, settings)
            else: # hybrid
                beads_list = self._detect_hybrid(detection_image, exclude_border, settings)
            
            if auto_merge and beads_list and len(beads_list) > 1:
                beads_list = self._merge_nearby_beads(beads_list, merge_distance)

            return self._beads_list_to_numpy(beads_list)
        except Exception as e:
            self.log_message.emit(f"❌ Error in bead detection for {channel_name}: {e}", "ERROR")
            return None

    def _detect_local_maxima(self, image, exclude_border, settings):
        sensitivity = float(settings.get('sensitivity', 85.0))
        min_distance = int(settings.get('min_distance', 5))
        min_area, max_area = int(settings.get('min_area', 4)), int(settings.get('max_area', 500))
        subpixel = bool(settings.get('subpixel_refinement', True))
        if np.max(image) == 0: return []
        threshold = np.percentile(image[image > 0], sensitivity) if np.any(image > 0) else 0
        peaks = feature.peak_local_max(image, min_distance=max(1, min_distance), threshold_abs=threshold, exclude_border=exclude_border)
        beads_list = []
        h, w = image.shape
        for y_int, x_int in peaks:
            y, x = (y_int, x_int)
            if subpixel: y, x = self._refine_centroid_radial_symmetry(image, y, x)
            if not (0 <= y < h and 0 <= x < w): continue
            local_region = self._get_local_region(image, y, x, radius=max(1, min_distance))
            if local_region.size == 0: continue
            local_max_intensity = image[int(y), int(x)]
            area = np.sum(local_region > (local_max_intensity / 2)) if local_max_intensity > 0 else 0
            if min_area <= area <= max_area:
                beads_list.append({'centroid': (y, x), 'area': float(area), 'intensity': float(local_max_intensity)})
        return beads_list

    def _detect_hybrid(self, image, exclude_border, settings):
        min_area, max_area = int(settings.get('min_area', 4)), int(settings.get('max_area', 500))
        min_distance = int(settings.get('min_distance', 5))
        subpixel = bool(settings.get('subpixel_refinement', True))
        
        maxima_beads = self._detect_local_maxima(image, exclude_border, settings)
        if np.max(image) == 0: return maxima_beads
        
        threshold = np.percentile(image[image > 0], float(settings.get('sensitivity', 85.0))) if np.any(image > 0) else 0
        binary = image > threshold
        binary = morphology.remove_small_objects(binary, min_size=max(1, min_area))
        labeled = measure.label(binary)
        regions = measure.regionprops(labeled, intensity_image=image)
        
        component_beads = []
        maxima_centroids = np.array([b['centroid'] for b in maxima_beads]) if maxima_beads else np.empty((0,2))
        
        for region in regions:
            if min_area <= region.area <= max_area:
                y, x = region.weighted_centroid
                if subpixel: y, x = self._refine_centroid_radial_symmetry(image, y, x)
                
                # Check if this bead is too close to an existing one from local maxima
                if maxima_centroids.shape[0] > 0:
                    dist = np.min(np.sqrt(np.sum((maxima_centroids - (y, x))**2, axis=1)))
                    if dist < max(1, min_distance):
                        continue
                        
                component_beads.append({'centroid': (y, x), 'area': float(region.area), 'intensity': float(region.mean_intensity)})
                
        return maxima_beads + component_beads

    def _detect_blob_log(self, image, settings):
        sensitivity = float(settings.get('sensitivity', 85.0))
        min_area, max_area = int(settings.get('min_area', 4)), int(settings.get('max_area', 500))
        subpixel = bool(settings.get('subpixel_refinement', True))
        if np.max(image) == 0: return []
        
        img_norm = image / (np.max(image) + 1e-10)
        min_sigma, max_sigma = max(0.5, np.sqrt(min_area / np.pi) * 0.5), np.sqrt(max_area / np.pi) * 2
        if max_sigma <= min_sigma: max_sigma = min_sigma + 1.0
        threshold = max(0.001, 0.05 * ((100.0 - sensitivity) / 100.0))
        
        blobs = blob_log(img_norm, min_sigma=min_sigma, max_sigma=max_sigma, num_sigma=15, threshold=threshold)
        beads_list = []
        h, w = image.shape
        for y_blob, x_blob, sigma in blobs:
            y, x = (y_blob, x_blob)
            if subpixel: y, x = self._refine_centroid_radial_symmetry(image, y, x)
            if not (0 <= y < h and 0 <= x < w): continue
            
            area = np.pi * (np.sqrt(2) * sigma)**2
            if min_area <= area <= max_area:
                beads_list.append({'centroid': (y, x), 'area': float(area), 'intensity': float(image[int(y), int(x)])})
        return beads_list

    def _merge_nearby_beads(self, beads, merge_distance):
        # This function is now much faster with numpy
        if len(beads) < 2: return beads
        
        centroids = np.array([b['centroid'] for b in beads])
        intensities = np.array([b['intensity'] for b in beads])
        areas = np.array([b['area'] for b in beads])
        
        merged_mask = np.zeros(len(beads), dtype=bool)
        merged_beads = []
        
        # Sort by intensity to merge smaller beads into brighter ones
        for i in np.argsort(intensities)[::-1]:
            if merged_mask[i]: continue
            
            distances = np.sqrt(np.sum((centroids - centroids[i])**2, axis=1))
            nearby_indices = np.where((distances < merge_distance) & ~merged_mask)[0]
            
            if len(nearby_indices) > 1:
                group_centroids = centroids[nearby_indices]
                group_intensities = intensities[nearby_indices]
                group_areas = areas[nearby_indices]

                # Weighted average for centroid, sum for area, max for intensity
                weights = np.maximum(1e-10, group_intensities)
                merged_centroid = np.average(group_centroids, axis=0, weights=weights)
                merged_area = np.sum(group_areas)
                merged_intensity = np.max(group_intensities)
                
                merged_beads.append({'centroid': tuple(merged_centroid), 'area': merged_area, 'intensity': merged_intensity})
                merged_mask[nearby_indices] = True
            else:
                merged_beads.append(beads[i])
                merged_mask[i] = True
                
        return merged_beads

    # ===================================================================
    # MATCHING FUNCTIONS (MODIFIED FOR NUMPY DATA)
    # ===================================================================
    
    def find_bead_matches(self, reference_beads, target_beads, channel_name):
        max_dist = self.similarity_params['max_distance']
        area_sim_thresh = self.similarity_params['area_similarity']
        intens_sim_thresh = self.similarity_params['intensity_correlation']
        
        if not all([reference_beads, target_beads]): return None
        
        ref_centroids = reference_beads['centroids']
        target_centroids = target_beads['centroids']
        
        if ref_centroids.shape[0] == 0 or target_centroids.shape[0] == 0: return None
        
        ref_tree = KDTree(ref_centroids)
        
        # Query all target points at once
        distances, ref_indices = ref_tree.query(target_centroids, k=5, distance_upper_bound=max_dist)
        
        matches_ref_idx = []
        matches_target_idx = []
        matches_dist = []
        matches_sim = []
        
        used_ref_indices = set()
        
        # Iterate through each target bead and its potential matches
        for target_idx in range(target_centroids.shape[0]):
            
            # Get properties of the current target bead
            target_area = target_beads['areas'][target_idx]
            target_intensity = target_beads['intensities'][target_idx]
            
            best_match_idx = -1
            best_similarity = -1.0
            best_dist = -1.0
            
            # Find the best valid match among the k-nearest neighbors
            for i, ref_idx in enumerate(ref_indices[target_idx]):
                if ref_idx >= ref_centroids.shape[0] or ref_idx in used_ref_indices:
                    continue # Skip invalid or already used reference beads
                
                dist = distances[target_idx][i]
                
                # Calculate similarity
                ref_area = reference_beads['areas'][ref_idx]
                area_ratio = min(ref_area, target_area) / max(ref_area, target_area)
                
                ref_intensity = reference_beads['intensities'][ref_idx]
                intensity_ratio = min(ref_intensity, target_intensity) / max(ref_intensity, target_intensity)

                if area_ratio < area_sim_thresh or intensity_ratio < intens_sim_thresh:
                    continue

                spatial_sim = max(0.0, 1.0 - (dist / max_dist))
                area_sim = area_ratio
                intensity_sim = intensity_ratio
                
                # Weighted similarity score
                similarity = (0.4 * spatial_sim + 0.4 * area_sim + 0.2 * intensity_sim)

                if similarity > best_similarity:
                    best_similarity = similarity
                    best_match_idx = ref_idx
                    best_dist = dist
            
            # If a good enough match was found, add it
            if best_similarity >= 0.6:
                matches_ref_idx.append(best_match_idx)
                matches_target_idx.append(target_idx)
                matches_dist.append(best_dist)
                matches_sim.append(best_similarity)
                used_ref_indices.add(best_match_idx)

        if not matches_ref_idx:
            return None
            
        return {
            'reference_indices': np.array(matches_ref_idx),
            'target_indices': np.array(matches_target_idx),
            'distances': np.array(matches_dist),
            'similarities': np.array(matches_sim)
        }


    def find_bead_matches_with_ransac(self, reference_beads, target_beads, channel_name, residual_threshold=2.0):
        try:
            from sklearn.linear_model import RANSACRegressor
            
            initial_matches = self.find_bead_matches(reference_beads, target_beads, channel_name)
            
            if not initial_matches or len(initial_matches['reference_indices']) < 10:
                self.log_message.emit(f"   ℹ️ Skipping RANSAC for {channel_name}: too few matches.", "DEBUG")
                return initial_matches

            self.log_message.emit(f"   🔍 Running RANSAC to refine matches for {channel_name}...", "DEBUG")

            # Prepare data for RANSAC
            ref_points = reference_beads['centroids'][initial_matches['reference_indices']]
            target_points = target_beads['centroids'][initial_matches['target_indices']]

            ransac = RANSACRegressor(min_samples=5, residual_threshold=residual_threshold, max_trials=100)
            ransac.fit(ref_points, target_points)
            inlier_mask = ransac.inlier_mask_
            outlier_count = np.sum(~inlier_mask)

            # Filter the matches to keep only inliers
            if outlier_count > 0:
                self.log_message.emit(f"   ✅ RANSAC complete: Removed {outlier_count} outlier(s) for {channel_name}.", "INFO")
                # Filter all result arrays using the mask
                for key in initial_matches:
                    initial_matches[key] = initial_matches[key][inlier_mask]
            else:
                self.log_message.emit(f"   ✅ RANSAC complete: No outliers found for {channel_name}.", "DEBUG")

            return initial_matches

        except ImportError:
            self.log_message.emit("❌ RANSAC requires scikit-learn. Please 'pip install scikit-learn'.", "ERROR")
            return self.find_bead_matches(reference_beads, target_beads, channel_name)
        except Exception as e:
            self.log_message.emit(f"❌ RANSAC failed for {channel_name}: {e}", "ERROR")
            return self.find_bead_matches(reference_beads, target_beads, channel_name)
    
    # ===================================================================
    # ANALYSIS & UTILITY FUNCTIONS (ADAPTED FOR NUMPY)
    # ===================================================================

    def calculate_shift_analysis(self):
        self.log_message.emit("📊 Part 4: Starting shift calculation and quality assessment...", "INFO")
        self.shift_analysis = {}
        ref_beads_data = self.detected_beads['DAPI']
        
        channels_to_analyze = [ch for ch in self.channel_names if ch in self.bead_matches and self.bead_matches[ch]]

        for i, channel_name in enumerate(channels_to_analyze):
            try:
                matches = self.bead_matches[channel_name]
                target_beads_data = self.detected_beads[channel_name]
                
                self.log_message.emit(f"📐 Calculating shifts for {len(matches['reference_indices'])} matched beads in {channel_name}...", "INFO")
                
                # Get centroids of matched beads using indices
                ref_centroids = ref_beads_data['centroids'][matches['reference_indices']]
                target_centroids = target_beads_data['centroids'][matches['target_indices']]

                # Vectorized shift calculation
                shifts_array = target_centroids - ref_centroids # Shape: (n_matches, 2)
                dy, dx = shifts_array[:, 0], shifts_array[:, 1]
                
                magnitudes = np.sqrt(dy**2 + dx**2)
                angles = np.arctan2(dy, dx) * 180 / np.pi
                
                # Create detailed shift data for potential export
                bead_shifts_data = {
                    'dy': dy, 'dx': dx, 'magnitude': magnitudes, 'angle': angles,
                    'ref_centroids': ref_centroids, 'target_centroids': target_centroids,
                    'similarities': matches['similarities']
                }

                # Calculate statistics
                stats = {
                    'median_dy': float(np.median(dy)), 'median_dx': float(np.median(dx)),
                    'mean_dy': float(np.mean(dy)), 'mean_dx': float(np.mean(dx)),
                    'std_dy': float(np.std(dy)), 'std_dx': float(np.std(dx)),
                    'mean_magnitude': float(np.mean(magnitudes)),
                    'std_magnitude': float(np.std(magnitudes)),
                    'median_magnitude': float(np.median(magnitudes)),
                }
                
                quality_metrics = self.calculate_shift_quality_metrics(bead_shifts_data)
                
                self.shift_analysis[channel_name] = {
                    'bead_count': len(dy),
                    'individual_shifts': bead_shifts_data, # Storing numpy arrays now
                    'statistics': stats,
                    'quality_metrics': quality_metrics
                }
                
                self.log_message.emit(f"✅ {channel_name}: Median shift dy={stats['median_dy']:.3f}, dx={stats['median_dx']:.3f} | Quality: {quality_metrics['overall_quality']:.3f}", "SUCCESS")
                
            except Exception as e:
                self.log_message.emit(f"❌ Error in shift analysis for {channel_name}: {e}", "ERROR")
            
            progress = 60 + int((i + 1) / len(channels_to_analyze) * 40)
            self.progress_update.emit(progress, f"Calculating shifts: {channel_name}")
        
        self.log_message.emit("📊 Shift analysis complete for all channels", "SUCCESS")
        self.generate_analysis_summary()

    def calculate_shift_quality_metrics(self, bead_shifts_data):
        count = len(bead_shifts_data['dy'])
        if count < 3:
            return {'overall_quality': 0.0, 'confidence': 'low', 'warnings': ['Too few beads']}

        magnitudes = bead_shifts_data['magnitude']
        similarities = bead_shifts_data['similarities']
        centroids = bead_shifts_data['ref_centroids']
        
        magnitude_cv = np.std(magnitudes) / np.mean(magnitudes) if np.mean(magnitudes) > 0 else 0.0
        angle_consistency = self.calculate_angle_consistency(bead_shifts_data['angle'])
        spatial_coverage = self.calculate_spatial_coverage(centroids)
        mean_similarity = np.mean(similarities)
        
        # Using numpy arrays for these calculations
        shifts_array = np.column_stack([bead_shifts_data['dy'], bead_shifts_data['dx']])
        outlier_fraction = self.detect_shift_outliers(shifts_array, magnitudes)
        edge_bias = self.calculate_edge_bias(centroids)
        
        quality = {
            'consistency': max(0, 1.0 - magnitude_cv),
            'angle_consistency': angle_consistency,
            'spatial_coverage': spatial_coverage,
            'similarity_quality': mean_similarity,
            'outlier_penalty': max(0, 1.0 - outlier_fraction * 2),
            'edge_bias_penalty': max(0, 1.0 - edge_bias)
        }
        weights = {'consistency': 0.25, 'angle_consistency': 0.15, 'spatial_coverage': 0.2, 'similarity_quality': 0.2, 'outlier_penalty': 0.1, 'edge_bias_penalty': 0.1}
        overall_quality = sum(weights[k] * v for k, v in quality.items())

        warnings = []
        if count < 10: warnings.append(f"Low bead count ({count})")
        if magnitude_cv > 0.5: warnings.append("High magnitude variability")
        if mean_similarity < 0.7: warnings.append("Low average similarity")
        if outlier_fraction > 0.2: warnings.append(f"High outlier fraction ({outlier_fraction:.1%})")
        if spatial_coverage < 0.3: warnings.append("Poor spatial coverage")
        if edge_bias > 0.5: warnings.append("High edge bias")
        
        if overall_quality > 0.8 and count >= 20: confidence = 'high'
        elif overall_quality > 0.6 and count >= 10: confidence = 'medium'
        else: confidence = 'low'
            
        return {'overall_quality': float(overall_quality), 'confidence': confidence, 'warnings': warnings}

    # --- Other utility and calculation methods (mostly unchanged, but benefit from numpy) ---

    def get_channel_settings(self, channel_name):
        if channel_name == "Sample AF" and self.af_settings is not None:
            self.log_message.emit(f"   🎯 Using AF-specific settings for {channel_name}", "DEBUG")
            return self.af_settings
        if self.default_settings is not None:
            self.log_message.emit(f"   📋 Using DAPI default settings for {channel_name}", "DEBUG")
            return self.default_settings
        self.log_message.emit(f"   ⚠️ No settings available for {channel_name}", "WARNING")
        return {}

    def correct_illumination(self, image_stack, sigma=50.0):
        try:
            self.log_message.emit("💡 Applying retrospective illumination correction...", "INFO")
            stack_float = image_stack.astype(np.float32)
            icf = np.mean(stack_float, axis=0)
            icf = gaussian_filter(icf, sigma=sigma)
            mean_icf = np.mean(icf)
            if mean_icf > 1e-6:
                icf /= mean_icf
            else:
                self.log_message.emit("⚠️ Illumination correction field is near-zero. Skipping correction.", "WARNING")
                return image_stack
            corrected_stack = stack_float / icf[np.newaxis, :, :]
            corrected_stack = np.clip(corrected_stack, 0, np.iinfo(image_stack.dtype).max if np.issubdtype(image_stack.dtype, np.integer) else np.finfo(np.float32).max)
            self.log_message.emit("✅ Illumination correction applied successfully.", "SUCCESS")
            return corrected_stack.astype(image_stack.dtype)
        except Exception as e:
            self.log_message.emit(f"❌ Illumination correction failed: {str(e)}", "ERROR")
            return image_stack

    def _refine_centroid_radial_symmetry(self, image, y, x, window_size=9):
        try:
            y_int, x_int = int(round(y)), int(round(x))
            half_win = window_size // 2
            y_min, y_max = max(0, y_int - half_win), min(image.shape[0], y_int + half_win + 1)
            x_min, x_max = max(0, x_int - half_win), min(image.shape[1], x_int + half_win + 1)
            window = image[y_min:y_max, x_min:x_max].astype(np.float64)
            if window.shape[0] < 3 or window.shape[1] < 3: return float(y), float(x)
            gy, gx = np.gradient(window)
            y_coords_win, x_coords_win = np.mgrid[0:window.shape[0], 0:window.shape[1]]
            gxgx, gygy, gxgy = np.sum(gx * gx), np.sum(gy * gy), np.sum(gx * gy)
            b1, b2 = np.sum(gx * (x_coords_win * gx + y_coords_win * gy)), np.sum(gy * (x_coords_win * gx + y_coords_win * gy))
            A = np.array([[gxgx, gxgy], [gxgy, gygy]])
            try:
                offset = np.linalg.solve(A, [b1, b2])
                cx_win, cy_win = offset[0], offset[1]
            except np.linalg.LinAlgError: return float(y), float(x)
            y_refined, x_refined = y_min + cy_win, x_min + cx_win
            if abs(y_refined - y) > half_win or abs(x_refined - x) > half_win: return float(y), float(x)
            return float(y_refined), float(x_refined)
        except Exception: return float(y), float(x)

    def _get_local_region(self, image, y, x, radius=5):
        h, w = image.shape
        y1, y2 = max(0, int(y - radius)), min(h, int(y + radius + 1))
        x1, x2 = max(0, int(x - radius)), min(w, int(x + radius + 1))
        return image[y1:y2, x1:x2]

    def calculate_angle_consistency(self, angles_deg):
        if len(angles_deg) < 2: return 1.0
        angles_rad = np.deg2rad(angles_deg)
        mean_vector = np.mean(np.column_stack([np.cos(angles_rad), np.sin(angles_rad)]), axis=0)
        return float(np.linalg.norm(mean_vector))

    def calculate_spatial_coverage(self, centroids):
        if len(centroids) < 3: return 0.0
        y_range, x_range = np.ptp(centroids[:, 0]), np.ptp(centroids[:, 1])
        img_size = max(self.image_stack.shape[-2:]) if self.image_stack is not None else 1000
        return float(min(1.0, (y_range * x_range) / (img_size**2)))

    def detect_shift_outliers(self, shifts_array, magnitudes):
        if len(shifts_array) < 5: return 0.0
        q75, q25 = np.percentile(magnitudes, [75, 25])
        iqr = q75 - q25
        return float(np.sum(magnitudes > (q75 + 1.5 * iqr)) / len(magnitudes)) if iqr > 0 else 0.0

    def calculate_edge_bias(self, centroids):
        if len(centroids) < 5: return 0.0
        y_min, y_max = np.min(centroids[:, 0]), np.max(centroids[:, 0])
        x_min, x_max = np.min(centroids[:, 1]), np.max(centroids[:, 1])
        y_margin, x_margin = (y_max - y_min) * 0.1, (x_max - x_min) * 0.1
        edge_beads = np.sum((centroids[:, 0] < y_min + y_margin) | (centroids[:, 0] > y_max - y_margin) | (centroids[:, 1] < x_min + x_margin) | (centroids[:, 1] > x_max - x_margin))
        return float(edge_beads / len(centroids))

    def generate_analysis_summary(self):
        self.log_message.emit("📋 Generating analysis summary...", "INFO")
        total_analyzed = len(self.shift_analysis)
        total_pairs = sum(data['bead_count'] for data in self.shift_analysis.values())
        self.log_message.emit(f"🎯 ANALYSIS SUMMARY: {total_analyzed} channels analyzed, {total_pairs} total bead pairs.", "SUCCESS")
        for channel_name, data in self.shift_analysis.items():
            stats, quality = data['statistics'], data['quality_metrics']
            self.log_message.emit(f"📊 {channel_name}: Beads: {data['bead_count']}, Quality: {quality['confidence']} ({quality['overall_quality']:.3f}), Median shift: ({stats['median_dy']:.3f}, {stats['median_dx']:.3f}) px", "INFO")
            if quality['warnings']: self.log_message.emit(f"   Warnings: {', '.join(quality['warnings'])}", "WARNING")

def main():
    """Main entry point."""
    # This is crucial for multiprocessing to work correctly on all platforms
    multiprocessing.freeze_support()
    
    app = QApplication(sys.argv)
    
    # Set application style
    app.setStyle('Fusion')
    
    analyzer = BeadShiftAnalyzer()
    analyzer.show()
    
    # Show startup message
    QMessageBox.information(analyzer, "Bead Shift Analyzer - Fixed Version", 
                          "🔬 Bead-by-Bead Shift Analyzer v1.0 (FIXED)\n\n"
                          "Key fixes in this version:\n"
                          "• Robust error handling in bead matching\n"
                          "• Safe division and array operations\n"
                          "• Better validation of input data\n"
                          "• AF settings are now truly optional\n"
                          "• Improved KDTree query handling\n\n"
                          "Load your image stack and DAPI settings to get started!\n"
                          "AF settings are optional - DAPI defaults will be used if not provided.")
    
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()


