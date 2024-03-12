from spotiflow.utils.peaks import points_to_prob, points_to_flow, flow_to_vector, prob_to_points
import numpy as np
import tifffile


if __name__ == "__main__":
    shape = (32,32,32)
    print("orig pt:")
    pt = (0,14.2,28.34124)

    points = np.array([pt])
    print("flow...")
    flow = points_to_flow(points, shape, sigma=1.5)
    flow_reversed = flow_to_vector(flow, sigma=1.5)
    print("flow ok!")

    print("heatmap...")
    heatmap = points_to_prob(points, shape, sigma=1.5, mode="max")
    print("heatmap ok!")
    
    print("nms")
    pts_rec = prob_to_points(heatmap, mode="fast", exclude_border=False)
    print("nms ok!")

    print(pts_rec)