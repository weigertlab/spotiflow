"""
    This script showcases how Spotiflow can be used to detect spots in an end-to-end starfish pipeline
    through the spotiflow.starfish.SpotiflowDetector class.
    
    Usage:
        python run_starfish_spotiflow.py
"""
from starfish import data
from starfish import FieldOfView
from starfish.util.plot import imshow_plane
from starfish.types import Axes
from spotiflow.starfish import SpotiflowDetector
import matplotlib.pyplot as plt
import tifffile
import starfish
from starfish.types import TraceBuildingStrategies
import numpy as np


if __name__ == "__main__":
    print("Loading data...")
    experiment = data.STARmap(use_test_data=True)
    stack = experiment['fov_000'].get_image('primary')
    print("Projecting...")
    projection = stack.reduce({Axes.CH, Axes.ZPLANE}, func="max")
    reference_image = projection.sel({Axes.ROUND: 0})
    print("Registering...")
    ltt = starfish.image.LearnTransform.Translation(
        reference_stack=reference_image,
        axes=Axes.ROUND,
        upsampling=1000,
    )
    transforms = ltt.run(projection)

    warp = starfish.image.ApplyTransform.Warp()
    stack = warp.run(
        stack=stack,
        transforms_list=transforms,
    )

    print("Detecting spots...")
    bd = SpotiflowDetector(
        model="smfish_3d",
        min_distance=1,
        is_volume=True,
        probability_threshold=.4,
    )

    spots_spotiflow = bd.run(stack, n_processes=1)

    print("Decoding...")
    decoder = starfish.spots.DecodeSpots.PerRoundMaxChannel(
        codebook=experiment.codebook,
        anchor_round=0,
        search_radius=10,
        trace_building_strategy=TraceBuildingStrategies.NEAREST_NEIGHBOR
)

    decoded_spotiflow = decoder.run(spots=spots_spotiflow)
    decoded_spotiflow_df = decoded_spotiflow.to_features_dataframe()[["z", "y", "x", "target"]]
    decoded_spotiflow_df_nonan = decoded_spotiflow_df[decoded_spotiflow_df["target"] != "nan"].reset_index(drop=True)
    decoded_spotiflow_df.to_csv("decoded_starmap_spotiflow.csv", index=False)
    print("Decoded results saved to decoded_starmap_spotiflow.csv")
