# beadtunerv4_fixed.py
#
# Advanced bead detection tuner optimized for microscopy images.
# Handles PSF effects, chromatic aberration, imperfect bead shapes, and border detection.
# Version: 4.0 - Fixed UI initialization issues

import sys
import numpy as np
import cv2
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.patches import Rectangle, Circle, FancyArrow
import tkinter as tk
from tkinter import filedialog
import tifffile
from skimage import measure, morphology, feature
from skimage.registration import phase_cross_correlation
from scipy import ndimage
from scipy.spatial import KDTree
from scipy.optimize import curve_fit
import json
from collections import defaultdict
import time
import matplotlib as mpl

from PyQt5.QtWidgets import (
    QApplication, QWidget, QHBoxLayout, QVBoxLayout, QSlider, QPushButton,
    QCheckBox, QGroupBox, QRadioButton, QLabel, QFileDialog, QMessageBox,
    QSizePolicy
)
from PyQt5.QtCore import Qt, QTimer

# Matplotlib rcParams for performance
mpl.rcParams['path.simplify'] = True
mpl.rcParams['path.simplify_threshold'] = 0.5
mpl.rcParams['agg.path.chunksize'] = 10000


class MplCanvas(FigureCanvas):
    """Matplotlib canvas widget to embed in PyQt."""
    def __init__(self, parent=None, width=5, height=4, dpi=100):
        self.fig = Figure(figsize=(width, height), dpi=dpi, tight_layout=True)
        super().__init__(self.fig)
        self.setMinimumSize(300, 300)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)


class AdvancedBeadTuner(QWidget):
    def __init__(self, image_stack, channel_names, parent=None):
        super().__init__(parent)
        self.image_stack = image_stack
        self.channel_names = channel_names
        self.current_channel = 0
        self.crop_region = None
        self.cropped_image = None

        # Parameters
        self.detection_method = 'hybrid'
        self.noise_size = 3.0
        self.background_size = 30
        self.sensitivity = 85.0
        self.min_area = 4
        self.max_area = 500
        self.min_distance = 5
        self.merge_distance = 10.0
        self.selection_padding = 0.0
        self.search_radius = 5.0
        self.include_borders = False
        self.subpixel_refinement = True
        self.show_rejected = False
        self.overlay_ref = False
        self.show_shifts = False
        self.auto_merge = True

        # Results
        self.current_beads = []
        self.rejected_beads = []
        self.detection_image = None
        self.preview_shifts = {}
        self.all_channel_results = {}
        self.chromatic_corrections = {}
        self.analysis_parameters = None
        
        # UI state
        self.selection_mode = False
        self.rect_start = None
        self.pan_start = None
        self.zoom_factor = 1.0
        self.pan_x = 0
        self.pan_y = 0

        # Initialize UI components to None first
        self.ax_main = None
        self.ax_crop = None
        self.ax_process = None
        self.ax_hist = None
        self.canvas_main = None
        self.canvas_crop = None
        self.canvas_process = None
        self.canvas_hist = None

        # Setup UI first, then update display
        self.setup_ui()
        # Only call update_main_display after UI is fully set up
        if self.ax_main is not None:
            self.update_main_display()

    def setup_ui(self):
        """Setup the main UI layout."""
        self.setWindowTitle("Advanced Bead Tuner v4.0")
        self.setGeometry(100, 100, 1600, 900)
        
        main_layout = QHBoxLayout()
        self.setLayout(main_layout)
        
        # Left panel for controls
        left_panel = self.create_left_panel()
        main_layout.addLayout(left_panel, 1)
        
        # Right panel for plots
        right_panel = self.create_right_panel()
        main_layout.addLayout(right_panel, 4)

    def create_left_panel(self):
        """Create the left control panel."""
        layout = QVBoxLayout()
        layout.setAlignment(Qt.AlignTop)

        # Channel selection
        channel_box = QGroupBox("Channels")
        channel_layout = QVBoxLayout()
        self.channel_radios = {}
        for i, name in enumerate(self.channel_names):
            rb = QRadioButton(name)
            rb.toggled.connect(lambda checked, ch=i: self.change_channel(checked, ch) if self.ax_main is not None else None)
            channel_layout.addWidget(rb)
            self.channel_radios[i] = rb
        self.channel_radios[0].setChecked(True)
        channel_box.setLayout(channel_layout)
        layout.addWidget(channel_box)

        # Method selection
        method_box = QGroupBox("Detection Method")
        method_layout = QVBoxLayout()
        self.method_radios = {}
        methods = {'Hybrid': 'hybrid', 'Local Max': 'local_max', 'Blob Log': 'blob'}
        for text, key in methods.items():
            rb = QRadioButton(text)
            rb.toggled.connect(lambda checked, m=key: self.change_method(checked, m))
            method_layout.addWidget(rb)
            self.method_radios[key] = rb
        self.method_radios['hybrid'].setChecked(True)
        method_box.setLayout(method_layout)
        layout.addWidget(method_box)
        
        # Sliders
        sliders_box = QGroupBox("Detection Parameters")
        sliders_layout = QVBoxLayout()
        self.sliders = {}
        slider_params = {
            'noise_size': ('Statistical Filter (σ×10)', 0, 100, 20, 10),  # Default 2.0σ
            'background_size': ('Background (px)', 10, 100, self.background_size, 1),
            'sensitivity': ('Sensitivity (%)', 500, 995, int(self.sensitivity*10), 10),
            'min_area': ('Min Area (px²)', 4, 15, self.min_area, 1),
            'max_area': ('Max Area (px²)', 16, 1000, self.max_area, 1),
            'min_distance': ('Min Distance (px)', 0, 20, self.min_distance, 1),
            'selection_padding': ('Selection Padding (px)', 0, 100, int(self.selection_padding*10), 10),
            'search_radius': ('Search Radius (px)', 10, 200, int(self.search_radius*10), 10)
        }
        for key, (label, min_v, max_v, init_v, scale) in slider_params.items():
            slider_label = QLabel(label)
            sliders_layout.addWidget(slider_label)
            slider = QSlider(Qt.Horizontal)
            slider.setRange(min_v, max_v)
            slider.setValue(init_v)
            slider.valueChanged.connect(self.on_slider_value_changed) 
            slider.sliderReleased.connect(self.update_crop_display)   
            sliders_layout.addWidget(slider)
            self.sliders[key] = (slider, scale, slider_label)
        sliders_box.setLayout(sliders_layout)
        layout.addWidget(sliders_box)

        # Options
        options_box = QGroupBox("Display Options")
        options_layout = QVBoxLayout()
        self.check_boxes = {}
        options = {
            'show_rejected': ('Show Rejected', self.show_rejected),
            'auto_merge': ('Auto Merge', self.auto_merge),
            'include_borders': ('Include Borders', self.include_borders),
            'subpixel_refinement': ('Subpixel Refine', self.subpixel_refinement),
            'overlay_ref': ('Overlay Ref', self.overlay_ref),
            'show_shifts': ('Show Shifts', self.show_shifts)
        }
        for key, (text, checked) in options.items():
            cb = QCheckBox(text)
            cb.setChecked(checked)
            cb.toggled.connect(self.toggle_options)
            options_layout.addWidget(cb)
            self.check_boxes[key] = cb
        options_box.setLayout(options_layout)
        layout.addWidget(options_box)
        
        # Presets
        preset_box = QGroupBox("Presets")
        preset_layout = QVBoxLayout()
        self.preset_radios = {}
        presets = ['Default', 'AF-Specific', 'High Sens']
        for p_name in presets:
            rb = QRadioButton(p_name)
            rb.toggled.connect(lambda checked, label=p_name: self.load_preset(checked, label))
            preset_layout.addWidget(rb)
            self.preset_radios[p_name] = rb
        preset_box.setLayout(preset_layout)
        layout.addWidget(preset_box)

        layout.addStretch()
        return layout

    def create_right_panel(self):
        """Create the right panel with plots and action buttons."""
        layout = QVBoxLayout()
        
        # Top plots area
        plots_layout = QHBoxLayout()
        
        # Create matplotlib canvases and axes
        self.canvas_main = MplCanvas(self, width=5, height=4)
        self.ax_main = self.canvas_main.fig.add_subplot(111)
        self.ax_main.set_title('Click and drag to select region')
        self.ax_main.axis('off')
        
        self.canvas_crop = MplCanvas(self, width=5, height=4)
        self.ax_crop = self.canvas_crop.fig.add_subplot(111)
        self.ax_crop.set_title('Selected Region')
        self.ax_crop.axis('off')

        self.canvas_process = MplCanvas(self, width=5, height=4)
        self.ax_process = self.canvas_process.fig.add_subplot(111)
        self.ax_process.set_title('Detection Map')
        self.ax_process.axis('off')
        
        plots_layout.addWidget(self.canvas_main)
        plots_layout.addWidget(self.canvas_crop)
        plots_layout.addWidget(self.canvas_process)
        layout.addLayout(plots_layout, 4)

        # Bottom stats and histogram area
        bottom_layout = QHBoxLayout()
        self.stats_label = QLabel("Statistics will be shown here.")
        self.stats_label.setStyleSheet("background-color: #f0f0f0; border: 1px solid #ccc; padding: 5px; border-radius: 4px;")
        self.stats_label.setAlignment(Qt.AlignTop)
        self.stats_label.setWordWrap(True)
        
        self.canvas_hist = MplCanvas(self, width=6, height=3)
        self.ax_hist = self.canvas_hist.fig.add_subplot(111)
        self.ax_hist.set_title('Bead Area Histogram')

        bottom_layout.addWidget(self.stats_label, 1)
        bottom_layout.addWidget(self.canvas_hist, 2)
        layout.addLayout(bottom_layout, 2)

        # Buttons area
        buttons_layout = QHBoxLayout()
        buttons_layout.setAlignment(Qt.AlignRight)
        
        btn_full = QPushButton("Process Full Image")
        btn_full.clicked.connect(self.process_full_image)
        buttons_layout.addWidget(btn_full)

        btn_apply = QPushButton("Apply to All Channels")
        btn_apply.clicked.connect(self.apply_to_all_channels)
        buttons_layout.addWidget(btn_apply)

        btn_save = QPushButton("Save Results")
        btn_save.clicked.connect(self.save_results)
        buttons_layout.addWidget(btn_save)

        btn_reset = QPushButton("Reset View")
        btn_reset.clicked.connect(self.reset_view)
        buttons_layout.addWidget(btn_reset)

        btn_preset = QPushButton("Save Preset")
        btn_export_settings = QPushButton("Export Detection Settings")
        btn_export_settings.clicked.connect(self.export_detection_settings)
        buttons_layout.addWidget(btn_export_settings)
        btn_preset.clicked.connect(self.save_preset)
        buttons_layout.addWidget(btn_preset)
        
        btn_help = QPushButton("Help")
        btn_help.clicked.connect(self.show_help)
        buttons_layout.addWidget(btn_help)

        layout.addLayout(buttons_layout)

        # Connect mouse events after canvases are created
        self.canvas_main.mpl_connect('button_press_event', self.on_mouse_press)
        self.canvas_main.mpl_connect('button_release_event', self.on_mouse_release)
        self.canvas_main.mpl_connect('motion_notify_event', self.on_mouse_move)
        self.canvas_main.mpl_connect('scroll_event', self.on_scroll)
        self.canvas_crop.mpl_connect('scroll_event', self.on_scroll)
        
        return layout

    #
    # UI EVENT HANDLERS
    #
    def on_slider_value_changed(self):
            """Applies slider values and updates labels in real-time without redrawing the plot."""
            self._apply_slider_values()
        
    def _apply_slider_values(self):
        self.noise_size = self.sliders['noise_size'][0].value() / self.sliders['noise_size'][1]
        self.background_size = self.sliders['background_size'][0].value()
        self.sensitivity = self.sliders['sensitivity'][0].value() / self.sliders['sensitivity'][1]
        self.min_area = self.sliders['min_area'][0].value()
        self.max_area = self.sliders['max_area'][0].value()
        self.min_distance = self.sliders['min_distance'][0].value()
        self.selection_padding = self.sliders['selection_padding'][0].value() / self.sliders['selection_padding'][1]
        self.search_radius = self.sliders['search_radius'][0].value() / self.sliders['search_radius'][1]
        
        # Update labels in real-time
        self.sliders['noise_size'][2].setText(f"Statistical Filter ({self.noise_size:.1f} σ)")
        self.sliders['background_size'][2].setText(f"Background ({self.background_size} px)")
        self.sliders['sensitivity'][2].setText(f"Sensitivity ({self.sensitivity:.1f} %)")
        self.sliders['min_area'][2].setText(f"Min Area ({self.min_area} px²)")
        self.sliders['max_area'][2].setText(f"Max Area ({self.max_area} px²)")
        self.sliders['min_distance'][2].setText(f"Min Distance ({self.min_distance} px)")
        self.sliders['selection_padding'][2].setText(f"Selection Padding ({self.selection_padding:.1f} px)")
        self.sliders['search_radius'][2].setText(f"Search Radius ({self.search_radius:.1f} px)")

    def toggle_options(self):
        self.show_rejected = self.check_boxes['show_rejected'].isChecked()
        self.auto_merge = self.check_boxes['auto_merge'].isChecked()
        self.include_borders = self.check_boxes['include_borders'].isChecked()
        self.subpixel_refinement = self.check_boxes['subpixel_refinement'].isChecked()
        self.overlay_ref = self.check_boxes['overlay_ref'].isChecked()
        self.show_shifts = self.check_boxes['show_shifts'].isChecked()
        if self.ax_crop is not None:
            self.update_crop_display()
    
    def change_channel(self, checked, channel_index):
        if checked and self.ax_main is not None:
            self.current_channel = channel_index
            self.update_main_display()
            if self.crop_region is not None:
                self.extract_crop_region()
                self.update_crop_display()

    def change_method(self, checked, method_key):
        if checked:
            self.detection_method = method_key
            if self.ax_crop is not None:
                self.update_crop_display()
            
    def show_help(self):
        help_text = """
        <b>INSTRUCTIONS:</b>
        <ol>
        <li>Click and drag on the main image (left) to select a region of interest.</li>
        <li>Use the controls in the left panel to adjust bead detection parameters. Changes will update live in the cropped views.</li>
        <li>The middle panel shows the detected beads on the cropped image.</li>
        <li>The right panel shows the processed "detection map" used to find beads.</li>
        <li>When satisfied, click <b>'Apply to All Channels'</b> to calculate chromatic shifts based on the current settings.</li>
        <li>Click <b>'Save Results'</b> to save the calculated shifts and parameters to a JSON file.</li>
        </ol>
        <b>TIPS:</b>
        <ul>
        <li>Use the <b>middle mouse button to pan</b> and the <b>scroll wheel to zoom</b> in the main image.</li>
        <li><b>'Subpixel Refine'</b> provides more precise centroid locations by fitting a 2D Gaussian.</li>
        <li><b>'Overlay Ref'</b> shows the DAPI channel in green on top of the current channel's crop to visually inspect misalignment.</li>
        <li>For best results, select a crop region that contains a good number of well-distributed beads.</li>
        </ul>
        """
        QMessageBox.information(self, "Help", help_text)

    #
    # PRESET HANDLING
    #
    def load_preset(self, checked, label):
        if not checked:
            return
        if label == 'Default':
            params = {'noise_size': 3.0, 'background_size': 30, 'sensitivity': 85.0, 'min_area': 4, 'max_area': 500, 'min_distance': 5, 'selection_padding': 0.0, 'search_radius': 5.0}
        elif label == 'AF-Specific':
            params = {'noise_size': 2.0, 'background_size': 120, 'sensitivity': 80.0, 'min_area': 6, 'max_area': 800, 'min_distance': 10, 'selection_padding': 0.0, 'search_radius': 10.0}
        elif label == 'High Sens':
            params = {'noise_size': 1.0, 'background_size': 50, 'sensitivity': 70.0, 'min_area': 4, 'max_area': 500, 'min_distance': 5, 'selection_padding': 1.0, 'search_radius': 7.0}
        self.set_params(params)

    def set_params(self, params):
        for key, value in params.items():
            if key in self.sliders:
                slider, scale, _ = self.sliders[key]
                slider.setValue(int(value * scale))
        QApplication.processEvents() # Ensure UI updates before re-calculating
        if self.ax_crop is not None:
            self.update_crop_display()
        
    def save_preset(self, event=None):
        params = {
            'noise_size': self.noise_size, 'background_size': self.background_size,
            'sensitivity': self.sensitivity, 'min_area': self.min_area, 'max_area': self.max_area,
            'min_distance': self.min_distance, 'selection_padding': self.selection_padding,
            'search_radius': self.search_radius
        }
        file_path, _ = QFileDialog.getSaveFileName(self, "Save Preset", "", "JSON Files (*.json)")
        if file_path:
            with open(file_path, 'w') as f:
                json.dump(params, f, indent=4)
            print(f"Preset saved to {file_path}")

    #
    # DISPLAY UPDATE METHODS
    #
    def update_main_display(self):
        if self.ax_main is None:
            return
            
        self.ax_main.clear()
        if self.current_channel >= len(self.image_stack): 
            return

        img = self.image_stack[self.current_channel]
        h, w = img.shape
        
        y1_view, y2_view, x1_view, x2_view = 0, h, 0, w
        if self.zoom_factor > 1:
            crop_h, crop_w = int(h / self.zoom_factor), int(w / self.zoom_factor)
            center_y, center_x = h//2 + self.pan_y, w//2 + self.pan_x
            y1_view = max(0, center_y - crop_h//2)
            y2_view = min(h, y1_view + crop_h)
            x1_view = max(0, center_x - crop_w//2)
            x2_view = min(w, x1_view + crop_w)
        
        img_display = img[y1_view:y2_view, x1_view:x2_view]
        vmin, vmax = np.percentile(img_display, [1, 99.5]) if img_display.size > 0 else (0, 255)
        
        self.ax_main.imshow(img_display, cmap='gray', vmin=vmin, vmax=vmax, extent=(x1_view, x2_view, y2_view, y1_view))
        
        if self.crop_region is not None:
            y1, y2, x1, x2 = self.crop_region
            rect = Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=1, edgecolor='r', facecolor='none')
            self.ax_main.add_patch(rect)

        self.ax_main.set_title(f'{self.channel_names[self.current_channel]} (Zoom: {self.zoom_factor:.1f}x)')
        self.ax_main.axis('off')
        self.canvas_main.draw()

    def update_crop_display(self):
        if self.ax_crop is None or self.ax_process is None or self.ax_hist is None:
            return
            
        if self.cropped_image is None:
            for ax in [self.ax_crop, self.ax_process, self.ax_hist]:
                ax.clear()
                ax.axis('off')
            self.ax_crop.set_title('Selected Region')
            self.ax_process.set_title('Detection Map')
            self.ax_hist.set_title('Bead Area Histogram')
            for canvas in [self.canvas_crop, self.canvas_process, self.canvas_hist]:
                canvas.draw()
            return
        
        self.rejected_beads = []
        self.current_beads = self.detect_beads_advanced(self.cropped_image)
        
        # Display cropped image
        self.ax_crop.clear()
        vmin, vmax = np.percentile(self.cropped_image, [1, 99.5])
        self.ax_crop.imshow(self.cropped_image, cmap='gray', vmin=vmin, vmax=vmax)
        
        if self.overlay_ref:
            ref_idx = self.channel_names.index('DAPI') if 'DAPI' in self.channel_names else 0
            if self.crop_region:
                y1, y2, x1, x2 = self.crop_region
                ref_crop = self.image_stack[ref_idx][y1:y2, x1:x2]
                self.ax_crop.imshow(ref_crop, cmap='viridis', alpha=0.3)

        # Draw beads
        for bead in self.current_beads:
            y, x = bead['centroid']
            radius = np.sqrt(bead.get('area', 0) / np.pi) + self.selection_padding
            self.ax_crop.add_patch(Circle((x, y), radius, fill=False, edgecolor='lime', lw=1.5, alpha=0.8))

        if self.show_rejected:
            for rej in self.rejected_beads:
                y, x = rej['centroid']
                self.ax_crop.plot(x, y, 'rx', markersize=5, alpha=0.6)
        
        self.ax_crop.set_title(f'Detected: {len(self.current_beads)} beads')
        self.ax_crop.axis('off')
        self.canvas_crop.draw()

        # Update other elements
        lca_result = self.preview_lca_shifts()
        if self.show_shifts and lca_result and 'dy' in lca_result and lca_result['matches'] > 0:
            dy, dx = lca_result['dy'], lca_result['dx']
            h, w = self.cropped_image.shape
            arrow = FancyArrow(w*0.1, h*0.1, dx*10, dy*10, width=max(h,w)*0.01, head_width=max(h,w)*0.05, color='red', alpha=0.7)
            self.ax_crop.add_patch(arrow)

        self.update_process_display()
        self.update_statistics(lca_result)
        self.update_histogram()

    def update_process_display(self):
        if self.ax_process is None:
            return
            
        self.ax_process.clear()
        if self.detection_image is not None and self.detection_image.size > 0:
            vmax = np.percentile(self.detection_image, 99.5) if np.any(self.detection_image) else 1
            self.ax_process.imshow(self.detection_image, cmap='hot', vmin=0, vmax=vmax)
        self.ax_process.set_title('Detection Map')
        self.ax_process.axis('off')
        self.canvas_process.draw()

    def update_histogram(self):
        if self.ax_hist is None:
            return
            
        self.ax_hist.clear()
        if self.current_beads:
            areas = [b['area'] for b in self.current_beads]
            self.ax_hist.hist(areas, bins=20, color='blue', alpha=0.7)
        self.ax_hist.set_title('Bead Area Histogram')
        self.ax_hist.set_xlabel('Area (px²)')
        self.ax_hist.set_ylabel('Count')
        self.canvas_hist.fig.tight_layout()
        self.canvas_hist.draw()
        
    def update_statistics(self, lca_result):
        if not hasattr(self, 'stats_label') or self.stats_label is None:
            return
            
        if not self.current_beads:
            self.stats_label.setText("No beads detected in selected region.")
            return

        areas = [b['area'] for b in self.current_beads]
        stats_lines = [f"<b>Total detected: {len(self.current_beads)} beads</b>"]
        stats_lines.append(f"Area range: {min(areas):.0f} - {max(areas):.0f} px (mean: {np.mean(areas):.1f})")

        if lca_result and lca_result.get('matches', 0) > 0 and 'dy' in lca_result and 'dx' in lca_result:
            stats_lines.append(f"<b>LCA Preview:</b> dy={lca_result['dy']:.2f}, dx={lca_result['dx']:.2f} px")
            stats_lines.append(f"Matches: {lca_result['matches']} ({lca_result.get('match_rate',0)*100:.1f}%) | Confidence: {lca_result.get('confidence',0):.2f}")
            if lca_result['matches'] < 5 or lca_result.get('confidence',0) < 0.5:
                stats_lines.append("<font color='red'>⚠️ Low Quality: Increase beads or adjust params</font>")
        else:
            stats_lines.append("LCA Preview: Insufficient matches for a robust fit.")
            
        self.stats_label.setText("<br>".join(stats_lines))
        
    #
    # MOUSE/KEYBOARD EVENT HANDLERS
    #
    def on_mouse_press(self, event):
        if event.inaxes != self.ax_main: return
        if event.button == 1:
            self.selection_mode = True
            self.rect_start = (event.xdata, event.ydata)
        elif event.button == 2: # Middle mouse for panning
            self.pan_start = (event.xdata, event.ydata)
            self.pan_start_offset = (self.pan_x, self.pan_y)

    def on_mouse_release(self, event):
        if event.button == 1 and self.selection_mode:
            self.selection_mode = False
            if self.rect_start and event.xdata is not None and event.ydata is not None:
                x1, y1 = self.rect_start
                x2, y2 = event.xdata, event.ydata
                
                # Handle zoom offset
                h, w = self.image_stack[self.current_channel].shape
                y1_view, _, x1_view, _ = 0, h, 0, w
                if self.zoom_factor > 1:
                    crop_h, crop_w = int(h / self.zoom_factor), int(w / self.zoom_factor)
                    center_y, center_x = h//2 + self.pan_y, w//2 + self.pan_x
                    y1_view = max(0, center_y - crop_h//2)
                    x1_view = max(0, center_x - crop_w//2)

                self.set_crop_region(min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2))
        elif event.button == 2:
            self.pan_start = None

    def on_mouse_move(self, event):
        if self.pan_start and event.button == 2 and event.inaxes == self.ax_main:
            if event.xdata is None or event.ydata is None: return
            dx = event.xdata - self.pan_start[0]
            dy = event.ydata - self.pan_start[1]
            self.pan_x = self.pan_start_offset[0] - int(dx)
            self.pan_y = self.pan_start_offset[1] - int(dy)
            self.update_main_display()

    def on_scroll(self, event):
        if event.inaxes not in [self.ax_main, self.ax_crop]: return
        zoom_speed = 1.2
        if event.button == 'up':
            self.zoom_factor = min(self.zoom_factor * zoom_speed, 10.0)
        elif event.button == 'down':
            self.zoom_factor = max(self.zoom_factor / zoom_speed, 1.0)
        
        if self.zoom_factor == 1.0:
            self.pan_x, self.pan_y = 0, 0

        self.update_main_display()
        if self.crop_region is not None:
             self.update_crop_display()

    
             
    def keyPressEvent(self, event):
        if event.key() == Qt.Key_R:
            self.reset_view()

    #
    # CORE PROCESSING AND LOGIC
    #
    def process_full_image(self, event=None):
        h, w = self.image_stack[self.current_channel].shape
        self.set_crop_region(0, 0, w, h)

    def export_detection_settings(self, event=None):
        """Export current detection parameters for use in whole-slide processing."""
        current_params = {
            'detection_method': self.detection_method,
            'noise_size': self.noise_size,
            'background_size': self.background_size,
            'sensitivity': self.sensitivity,
            'min_area': self.min_area,
            'max_area': self.max_area,
            'min_distance': self.min_distance,
            'merge_distance': self.merge_distance,
            'selection_padding': self.selection_padding,
            'search_radius': self.search_radius,
            'include_borders': self.include_borders,
            'subpixel_refinement': self.subpixel_refinement,
            'auto_merge': self.auto_merge
        }
        
        # Get current channel name for labeling
        current_channel_name = self.channel_names[self.current_channel]
        
        settings_data = {
            'export_info': {
                'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
                'tuned_on_channel': current_channel_name,
                'crop_region': self.crop_region
            },
            'parameters': current_params,
            'channel_names': self.channel_names
        }
        
        file_path, _ = QFileDialog.getSaveFileName(self, "Export Detection Settings", 
                                                f"detection_settings_{current_channel_name}.json", 
                                                "JSON Files (*.json)")
        if file_path:
            with open(file_path, 'w') as f:
                json.dump(settings_data, f, indent=4)
            print(f"Detection settings exported to {file_path}")
            QMessageBox.information(self, "Exported", f"Settings saved to:\n{file_path}")

    def extract_crop_region(self):
        if self.crop_region is None:
            self.cropped_image = None
            return
        y1, y2, x1, x2 = map(int, self.crop_region)
        img = self.image_stack[self.current_channel]
        h, w = img.shape
        y1, y2 = max(0, min(y1, h)), max(0, min(y2, h))
        x1, x2 = max(0, min(x1, w)), max(0, min(x2, w))
        if y2 > y1 and x2 > x1:
            self.cropped_image = img[y1:y2, x1:x2].copy()
        else:
            self.cropped_image = None
    
    def set_crop_region(self, x1, y1, x2, y2):
        self.crop_region = (int(y1), int(y2), int(x1), int(x2))
        self.extract_crop_region()
        self.update_main_display()
        self.update_crop_display()

    def reset_view(self, event=None):
        self.zoom_factor = 1.0
        self.pan_x, self.pan_y = 0, 0
        self.update_main_display()

    def detect_beads_advanced(self, image):
        if image is None: return []
        working_image = image.astype(np.float32)
        
        # Integrated statistical denoising
        if self.noise_size > 0:
            img_mean = np.mean(working_image)
            img_std = np.std(working_image)
            threshold = img_mean + self.noise_size * img_std
            working_image = np.maximum(working_image - threshold, 0)
        
        # Background subtraction on the denoised image
        if self.background_size > 0:
            selem = morphology.disk(max(1, int(self.background_size / 2)))
            background = morphology.opening(working_image, selem)
            foreground = working_image - background
            self.detection_image = np.maximum(foreground, 0)
        else:
            self.detection_image = working_image
        
        exclude_border = not self.include_borders
        if self.detection_method == 'local_max':
            beads = self._detect_local_maxima(self.detection_image, exclude_border)
        elif self.detection_method == 'blob':
            beads = self._detect_blob_log(self.detection_image)
        else: # hybrid
            beads = self._detect_hybrid(self.detection_image, exclude_border)
        
        if self.auto_merge and len(beads) > 1:
            beads = self._merge_nearby_beads(beads)
        
        return beads

    def _detect_local_maxima(self, image, exclude_border):
        threshold = np.percentile(image[image > 0], self.sensitivity) if np.any(image > 0) else 0
        peaks = feature.peak_local_max(image, min_distance=int(self.min_distance), threshold_abs=threshold, exclude_border=exclude_border)
        
        beads, rejected = [], []
        h, w = image.shape
        for peak in peaks:
            y, x = peak
            if self.subpixel_refinement:
                y, x = self.refine_centroid_subpixel(image, y, x)

            # Boundary check: Skip if the refined peak is outside the image
            if not (0 <= y < h and 0 <= x < w):
                rejected.append({'centroid': (y, x), 'area': 0, 'reason': 'out_of_bounds'})
                continue
            
            local_region = self._get_local_region(image, y, x, radius=5)
            if local_region.size == 0: continue
            
            local_max = image[int(y), int(x)]
            area = np.sum(local_region > (local_max / 2))
            padded_area = area + self.selection_padding * np.sqrt(area / np.pi) * 2 * np.pi if area > 0 else area
            
            if self.min_area <= padded_area <= self.max_area:
                beads.append({'centroid': (y, x), 'area': area, 'intensity': local_max, 'method': 'local_max'})
            else:
                rejected.append({'centroid': (y, x), 'area': area, 'reason': 'area_filter'})
        self.rejected_beads.extend(rejected)
        return beads
        
    def _detect_blob_log(self, image):
        from skimage.feature import blob_log
        img_norm = image / (np.max(image) + 1e-10)
        min_sigma = np.sqrt(self.min_area / np.pi) * 0.5
        max_sigma = np.sqrt(self.max_area / np.pi) * 2
        blobs = blob_log(img_norm, min_sigma=max(0.5, min_sigma), max_sigma=max_sigma, num_sigma=15, threshold=0.05 * (1 - self.sensitivity/100))
        
        beads, rejected = [], []
        h, w = image.shape
        for blob in blobs:
            y, x, sigma = blob
            radius = np.sqrt(2) * sigma
            area = np.pi * radius**2
            if self.subpixel_refinement:
                y, x = self.refine_centroid_subpixel(image, y, x)

            # Boundary check: Skip if the refined peak is outside the image
            if not (0 <= y < h and 0 <= x < w):
                rejected.append({'centroid': (y, x), 'area': area, 'reason': 'out_of_bounds'})
                continue

            padded_radius = radius + self.selection_padding
            padded_area = np.pi * padded_radius**2
            if self.min_area <= padded_area <= self.max_area:
                beads.append({'centroid': (y, x), 'area': area, 'intensity': image[int(y), int(x)], 'radius': radius, 'method': 'blob_log'})
            else:
                rejected.append({'centroid': (y, x), 'area': area, 'reason': 'area_filter'})
        self.rejected_beads.extend(rejected)
        return beads
        
    def _detect_hybrid(self, image, exclude_border):
        maxima_beads = self._detect_local_maxima(image, exclude_border)
        threshold = np.percentile(image[image > 0], self.sensitivity) if np.any(image > 0) else 0
        binary = image > threshold
        binary = morphology.remove_small_objects(binary, min_size=self.min_area)
        labeled = measure.label(binary)
        regions = measure.regionprops(labeled, intensity_image=image)
        
        component_beads = []
        for region in regions:
            area = region.area
            padded_area = area + self.selection_padding * np.sqrt(area / np.pi) * 2 * np.pi if area > 0 else area
            if self.min_area <= padded_area <= self.max_area:
                y, x = region.weighted_centroid
                if self.subpixel_refinement:
                    y, x = self.refine_centroid_subpixel(image, y, x)
                already_detected = any(np.sqrt((y-b['centroid'][0])**2 + (x-b['centroid'][1])**2) < self.min_distance for b in maxima_beads)
                if not already_detected:
                    component_beads.append({'centroid': (y, x), 'area': area, 'intensity': region.mean_intensity, 'method': 'component'})
        
        return maxima_beads + component_beads

    def _merge_nearby_beads(self, beads):
        if len(beads) < 2: return beads
        centroids = np.array([b['centroid'] for b in beads])
        merged_mask = np.zeros(len(beads), dtype=bool)
        merged_beads = []
        
        for i in range(len(beads)):
            if merged_mask[i]: continue
            
            distances = np.sqrt(np.sum((centroids - centroids[i])**2, axis=1))
            nearby_indices = np.where((distances < self.merge_distance) & ~merged_mask)[0]
            
            if len(nearby_indices) > 1:
                group_beads = [beads[j] for j in nearby_indices]
                weights = np.array([b['intensity'] for b in group_beads])
                
                # Check if weights sum to zero to prevent ZeroDivisionError
                if np.sum(weights) > 0:
                    weighted_centroid = np.average([b['centroid'] for b in group_beads], axis=0, weights=weights)
                else:
                    # Fallback to an unweighted average if all intensities are zero
                    weighted_centroid = np.mean([b['centroid'] for b in group_beads], axis=0)

                merged_beads.append({
                    'centroid': tuple(weighted_centroid), 'area': sum(b['area'] for b in group_beads),
                    'intensity': max(weights), 'merged_count': len(group_beads), 'method': 'merged'
                })
                merged_mask[nearby_indices] = True
            else:
                merged_beads.append(beads[i])
                merged_mask[i] = True
        
        return merged_beads

    def refine_centroid_subpixel(self, image, y, x, window_size=5):
        half_win = window_size // 2
        y_int, x_int = int(y), int(x)
        y_min, y_max = max(0, y_int - half_win), min(image.shape[0], y_int + half_win + 1)
        x_min, x_max = max(0, x_int - half_win), min(image.shape[1], x_int + half_win + 1)
        
        window = image[y_min:y_max, x_min:x_max].astype(np.float64)
        
        # Skip fitting on empty or flat windows to prevent errors and warnings
        if window.size < 9 or np.ptp(window) < 1e-6:
            return float(y), float(x)
        
        y_coords, x_coords = np.mgrid[0:window.shape[0], 0:window.shape[1]]
        
        def gaussian_2d(coords, amplitude, y0, x0, sigma, offset):
            y, x = coords
            exp_term = -((y - y0)**2 + (x - x0)**2) / (2 * sigma**2)
            return (amplitude * np.exp(exp_term) + offset).ravel()
        
        try:
            p0 = [np.max(window) - np.min(window), window.shape[0]/2, window.shape[1]/2, 1.5, np.min(window)]
            popt, _ = curve_fit(gaussian_2d, (y_coords.ravel(), x_coords.ravel()), window.ravel(), p0=p0, maxfev=1000)
            y_refined, x_refined = y_min + popt[1], x_min + popt[2]
            return (float(y_refined), float(x_refined)) if abs(y_refined - y) < 2 and abs(x_refined - x) < 2 else (float(y), float(x))
        except (RuntimeError, ValueError): # Catch fitting errors more broadly
            return float(y), float(x)
        
    def preview_lca_shifts(self):
        if not self.crop_region or not self.current_beads: return None
        ref_idx = self.channel_names.index('DAPI') if 'DAPI' in self.channel_names else 0
        y1, y2, x1, x2 = map(int, self.crop_region)
        ref_crop = self.image_stack[ref_idx][y1:y2, x1:x2]
        
        ref_beads = self.detect_beads_advanced(ref_crop)
        if len(ref_beads) < 5 or len(self.current_beads) < 5:
            return {'matches': 0, 'confidence': 0}

        ref_centroids = np.array([b['centroid'] for b in ref_beads])
        query_centroids = np.array([b['centroid'] for b in self.current_beads])
        
        ref_tree = KDTree(ref_centroids)
        distances, indices = ref_tree.query(query_centroids, k=1, distance_upper_bound=self.search_radius)
        valid_matches = np.isfinite(distances)
        matches = np.sum(valid_matches)
        
        if matches < 5: return {'matches': matches, 'confidence': 0}
        
        matched_ref = ref_centroids[indices[valid_matches]]
        matched_query = query_centroids[valid_matches]
        shift_result = self.calculate_robust_shift(matched_ref, matched_query)
        shift_result['matches'] = int(matches)
        shift_result['match_rate'] = matches / len(self.current_beads)
        
        return shift_result

    def calculate_robust_shift(self, matched_ref, matched_query):
        displacements = matched_query - matched_ref
        median_shift = np.median(displacements, axis=0)
        shift_std = np.std(displacements, axis=0)
        confidence = 1.0 / (1.0 + np.mean(shift_std)) if np.mean(shift_std) > 0 else 1.0
        return {
            'dy': median_shift[0], 'dx': median_shift[1], 
            'shift_std_y': shift_std[0], 'shift_std_x': shift_std[1],
            'method': 'median', 'confidence': confidence
        }

    def _get_local_region(self, image, y, x, radius=5):
        h, w = image.shape
        y1, y2 = max(0, int(y - radius)), min(h, int(y + radius + 1))
        x1, x2 = max(0, int(x - radius)), min(w, int(x + radius + 1))
        return image[y1:y2, x1:x2]
        
    def apply_to_all_channels(self, event=None):
        """
        Detects beads in each channel within the selected crop region and calculates
        chromatic shifts based on centroid positions relative to the reference channel.
        """
        if self.crop_region is None:
            QMessageBox.warning(self, "Warning", "Please select a region of interest first!")
            return
        
        # Capture parameters at the moment of analysis
        self.analysis_parameters = {
            'detection_method': self.detection_method, 'noise_size': self.noise_size,
            'background_size': self.background_size, 'sensitivity': self.sensitivity,
            'min_area': self.min_area, 'max_area': self.max_area,
            'min_distance': self.min_distance, 'merge_distance': self.merge_distance,
            'selection_padding': self.selection_padding, 'search_radius': self.search_radius
        }

        print("\n" + "=" * 50 + "\nDETECTING BEADS IN ALL CHANNELS (in selected region)\n" + "=" * 50)
        y1, y2, x1, x2 = self.crop_region
        
        all_channel_beads = {}
        for i, name in enumerate(self.channel_names):
            if i >= len(self.image_stack): 
                continue
            
            crop = self.image_stack[i][y1:y2, x1:x2].copy()
            detected_beads = self.detect_beads_advanced(crop)
            
            for bead in detected_beads:
                y_crop, x_crop = bead['centroid']
                bead['centroid'] = (y_crop + y1, x_crop + x1) 
            
            all_channel_beads[name] = detected_beads
            print(f"  ✅ Detected {len(detected_beads)} beads in {name}")

        self.calculate_chromatic_corrections_from_beads(all_channel_beads)
    def calculate_chromatic_corrections_from_beads(self, all_channel_beads):
        """
        Calculates robust chromatic shifts by matching bead centroids between
        each channel and the reference 'DAPI' channel.
        """
        print("\n" + "=" * 50 + "\nCHROMATIC ABERRATION ANALYSIS (from beads)\n" + "=" * 50)
        ref_channel_name = 'DAPI' if 'DAPI' in self.channel_names else self.channel_names[0]
        
        ref_beads = all_channel_beads.get(ref_channel_name)
        if not ref_beads:
            QMessageBox.warning(self, "Error", f"Reference channel '{ref_channel_name}' has no detected beads. Cannot calculate shifts.")
            return

        ref_centroids = np.array([b['centroid'] for b in ref_beads])
        self.chromatic_corrections = {}

        for name, target_beads in all_channel_beads.items():
            if name == ref_channel_name:
                continue
            
            if len(target_beads) < 5:
                print(f"\n{name}:\n  ⚠️ Not enough beads ({len(target_beads)}) to calculate shift. Skipping.")
                self.chromatic_corrections[name] = {'dy': 0, 'dx': 0, 'method': 'skipped_low_beads', 'matches': 0}
                continue

            query_centroids = np.array([b['centroid'] for b in target_beads])
            
            # Match beads between reference and target
            ref_tree = KDTree(ref_centroids)
            distances, indices = ref_tree.query(query_centroids, k=1, distance_upper_bound=self.search_radius)
            
            valid_matches_mask = np.isfinite(distances)
            matches_count = np.sum(valid_matches_mask)

            if matches_count < 5:
                print(f"\n{name}:\n  ⚠️ Insufficient matches ({matches_count}) for a robust fit. Skipping.")
                self.chromatic_corrections[name] = {'dy': 0, 'dx': 0, 'method': 'skipped_low_matches', 'matches': int(matches_count)}
                continue

            matched_ref_centroids = ref_centroids[indices[valid_matches_mask]]
            matched_query_centroids = query_centroids[valid_matches_mask]
            
            # Calculate robust shift from the matched pairs
            shift_result = self.calculate_robust_shift(matched_ref_centroids, matched_query_centroids)
            shift_result['matches'] = int(matches_count)
            shift_result['match_rate'] = matches_count / len(target_beads)
            
            self.chromatic_corrections[name] = shift_result
            print(f"\n{name}:\n  Shift (dy, dx): ({shift_result['dy']:+.3f}, {shift_result['dx']:+.3f}) px")
            print(f"  Found {shift_result['matches']} matches ({shift_result['match_rate']*100:.1f}%) with confidence {shift_result['confidence']:.3f}")
        
        self.chromatic_corrections[ref_channel_name] = {'dy': 0, 'dx': 0, 'method': 'reference', 'matches': len(ref_beads)}
        QMessageBox.information(self, "Success", "Chromatic aberration calculated for all channels using bead centroids. See console for details.")
        
    def save_results(self, event=None):
        if not self.chromatic_corrections or self.analysis_parameters is None:
            QMessageBox.warning(self, "No Results", "Please run 'Apply to All Channels' first to generate results.")
            return
        
        output_path, _ = QFileDialog.getSaveFileName(self, "Save Detection Results", "", "JSON Files (*.json);;All Files (*)")
        if not output_path: return
        
        save_data = {
            'parameters': self.analysis_parameters,
            'crop_region': self.crop_region,
            'chromatic_corrections': self.chromatic_corrections,
        }
        
        with open(output_path, 'w') as f:
            json.dump(save_data, f, indent=4)
        print(f"\n✅ Results saved successfully to: {output_path}")
        QMessageBox.information(self, "Saved", f"Results saved to:\n{output_path}")


def main():
    # Use a single Tk root for the initial file dialog to keep it simple
    root = tk.Tk()
    root.withdraw()
    
    print("ADVANCED BEAD TUNER V4.0 - Fixed Version")
    print("=" * 50)
    
    file_path = filedialog.askopenfilename(
        title="Select Bead Image",
        filetypes=[("OME-TIFF", "*.ome.tiff"), ("TIFF", "*.tiff"), ("All files", "*.*")]
    )
    if not file_path:
        print("No file selected. Exiting.")
        return
        
    try:
        image_stack = tifffile.imread(file_path)
    except Exception as e:
        print(f"Error loading image: {e}")
        return

    if len(image_stack.shape) == 2:
        image_stack = image_stack[np.newaxis, ...]
    
    default_channels = ['DAPI', 'Opal 570', 'Opal 690', 'Opal 620', 'Opal 780', 'Opal 520', 'Sample AF']
    channel_names = default_channels[:image_stack.shape[0]]

    app = QApplication(sys.argv)
    tuner = AdvancedBeadTuner(image_stack, channel_names)
    tuner.show()
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()