"""
    Retrieve basic statistics of spot clusters in an image by running Spotiflow to detect individual spots and then aggreagating them according to a radius search.
    
    Usage:
        python analyze_spot_clusters.py --input /PATH/TO/IMG --model SPOITFLOW_MODEL --output ./out
"""
import argparse
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd
from skimage import io
from sklearn.neighbors import radius_neighbors_graph
from spotiflow.model import Spotiflow
from spotiflow.utils import write_coords_csv


def analyze_clusters(spots: np.ndarray, max_distance: float = 11.0):
    """
    Get information of clusters by building an r-radius graph.
    """
    adj_matrix = radius_neighbors_graph(
        spots, radius=max_distance, mode="distance", metric="euclidean"
    )
    graph = nx.from_scipy_sparse_array(adj_matrix)
    conn_components = nx.connected_components(graph)
    columns = ["cluster_id", "mean_y", "mean_x", "num_spots"]
    if spots.shape[1] == 3:
        columns.insert(1, "mean_z")
    df = pd.DataFrame(columns=columns)
    for i, component in enumerate(conn_components):
        curr_spots = spots[list(component)]
        center = np.mean(curr_spots, axis=0)
        if center.shape[0] == 3:
            mean_z, mean_y, mean_x = center
        else:
            mean_y, mean_x = center

        component_data = {
            "cluster_id": i,
            "num_spots": len(component),
            "mean_y": mean_y,
            "mean_x": mean_x,
        }
        if center.shape[0] == 3:
            component_data["mean_z"] = mean_z
        curr_df = pd.DataFrame(component_data, index=[0])
        df = pd.concat([df, curr_df], ignore_index=True)
    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, help="Path to the image")
    parser.add_argument("--model", type=Path, help="Path to the model")
    parser.add_argument("--output", type=Path, help="Path to the output folder")
    parser.add_argument("--max-distance", type=float, default=11.0, help="Max distance to consider two spots as part of the same cluster")
    args = parser.parse_args()

    args.output.mkdir(exist_ok=True, parents=True)

    img = io.imread(args.input)  # load the image
    model = Spotiflow.from_folder(args.model)  # load the clusters model
    spots, _ = model.predict(img, normalizer="auto")
    print("Analyzing clusters...")
    clusters_df = analyze_clusters(spots, max_distance=args.max_distance)
    clusters_df.to_csv(args.output / f"{args.input.stem}_clusters.csv", index=False)
    print("Done!")
