# Annotation Tools

This document describes the annotation-related tools available in the `extra` directory.

## `annotation_ui.py`

This script provides a user interface for annotating points in an image. It can be used to create, visualize, and edit point annotations, which can then be saved to a CSV file. The UI is built using `napari`.

### Usage

To start the annotation UI, run the following command:

```bash
python extra/annotation_ui.py <path_to_image> [options]
```

**Arguments:**

-   `<path_to_image>`: Path to the image file to be annotated.

**Options:**

-   `-p`, `--points <path_to_points_file>`: Path to an existing annotation file (either `.npy` or `.csv`). If not provided, it will default to a CSV file with the same name as the image.
-   `-t`, `--threshold <value>`: Initial threshold for the Laplacian of Gaussian (LoG) blob detector.

### Interactive Controls

The annotation widget provides several keyboard shortcuts for interacting with the annotations:

-   **`q`**: Toggle the visibility of the points.
-   **`s`**: Save the current annotations to the points file. You will be prompted to confirm before overwriting an existing file.
-   **`d`**: Detect points in the current field of view using the LoG detector. This will replace existing points within the view.
-   **`w`**: Increase the LoG detection threshold.
-   **`e`**: Decrease the LoG detection threshold.

The widget also includes sliders to adjust the `max_sigma` and `threshold` for the LoG detector, and a checkbox to enable/disable Gaussian fitting for refining point locations.

## `analyze_spot_clusters.py`

This script is used to detect and analyze clusters of spots in an image. It uses a `spotiflow` model to detect individual spots and then groups them into clusters based on a specified radius. The output is a CSV file containing statistics for each cluster, such as the mean coordinates and the number of spots.

### Usage

```bash
python extra/analyze_spot_clusters.py --input <path_to_image> --model <path_to_model> --output <path_to_output_dir> [options]
```

**Arguments:**

-   `--input <path_to_image>`: Path to the input image file.
-   `--model <path_to_model>`: Path to the `spotiflow` model directory.
-   `--output <path_to_output_dir>`: Path to the directory where the output CSV file will be saved.

**Options:**

-   `--max-distance <value>`: The maximum distance (in pixels) between two spots to be considered part of the same cluster. The default value is `11.0`.

## `run_starfish_spotiflow.py`

This script demonstrates how to integrate `spotiflow` into a `starfish` pipeline for spot detection. It is an end-to-end example that performs the following steps:

1.  Loads STARmap test data.
2.  Registers the image stack.
3.  Detects spots using the `SpotiflowDetector` class from `spotiflow.starfish`.
4.  Decodes the detected spots using the `starfish` codebook.
5.  Saves the decoded results to a CSV file named `decoded_starmap_spotiflow.csv`.

### Usage

To run the example, simply execute the script:

```bash
python extra/run_starfish_spotiflow.py
```

The script does not require any command-line arguments as it uses test data provided by `starfish`.