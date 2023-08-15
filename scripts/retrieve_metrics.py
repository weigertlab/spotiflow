from pathlib import Path
from scipy.integrate import trapezoid
from tqdm.auto import tqdm

import configargparse
import numpy as np
import pandas as pd

from spotipy_torch.utils import points_matching_dataset, points_matching

def read_coords_csv(fname: str): 
    """ parses a csv file and returns correctly ordered points array     
    """
    try:
        df = pd.read_csv(fname)
    except Exception as _:
        print(f"{fname} is empty. Continuing...")
        return 
    df = df.rename(columns = str.lower)
    cols = set(df.columns)
    col_candidates = (('axis-0', 'axis-1'), ('y','x'))
    points = None 
    for possible_columns in col_candidates:
        if cols.issuperset(set(possible_columns)):
            points = df[list(possible_columns)].to_numpy().astype(np.float32)
            break 

    if points is None: 
        raise ValueError(f'could not get points from csv file {fname}')

    return points

def load_data(gt_path, pred_path):
    gt_path = Path(gt_path)
    pred_path = Path(pred_path)
    gt_files = sorted(gt_path.glob("*.csv"))
    pred_files = sorted(pred_path.glob("*.csv"))

    assert len(gt_files)==len(pred_files), f"Number of files in GT ({len(gt_files)}) and predictions {len(pred_files)} do not match"

    if not all(gt_f.stem.startswith(pred_f.stem.split("_predict")[0]) for gt_f, pred_f in zip(gt_files, pred_files)):
        print('possible mitsmatching filenames?')
        print(pred_files[0], gt_files[0])

    # assert all(gt_f.stem.replace("_refined_r1", "")==pred_f.stem.replace("_prediction", "") for gt_f, pred_f in zip(gt_files, pred_files))

    gts = [read_coords_csv(gt_f) for gt_f in gt_files]
    preds = [read_coords_csv(pred_f) for pred_f in pred_files]
    return gts, preds

def get_metrics(gts, preds, cutoffs):
    cd_min, cd_max, dcd = cutoffs
    res = pd.DataFrame()
    for cd in tqdm(np.arange(cd_min, cd_max+dcd, dcd), desc="Sweeping cutoff distances"):
        metrics = points_matching_dataset(gts, preds, cutoff_distance=cd, by_image=False)
        res_tmp = pd.DataFrame(vars(metrics), index=[0])[["accuracy", "precision", "recall", "f1", "tp", "fp", "fn"]]
        res_tmp["cutoff_distance"] = cd
        res = pd.concat((res, res_tmp), ignore_index=True)
    f1_auc = trapezoid(res.f1, dx=dcd)/(res.cutoff_distance.max()-res.cutoff_distance.min())
    res["integral_f1"] = f1_auc
    return res

def get_metrics_single(gts, preds, cutoffs):
    cd_min, cd_max, dcd = cutoffs
    df = []

    for i, (gt, pred) in tqdm(enumerate(zip(gts, preds)), desc="getting single metrics", total=len(gts)):
        rows = []
        for cd in np.arange(cd_min, cd_max+dcd, dcd):
            stats = points_matching(gt, pred, cutoff_distance=cd)
            d = dict(image=i, cutoff_distance=cd)
            d.update(dict(((k,vars(stats)[k]) for k in ("accuracy", "precision", "recall", "f1", "tp", "fp", "fn"))))
            rows.append(d)
        res = pd.DataFrame.from_records(rows)
        res["integral_f1"] = trapezoid(res.f1, dx=dcd)/(res.cutoff_distance.max()-res.cutoff_distance.min())
        df.append(res)
    df = pd.concat(df) if len(df)>0 else pd.DataFrame()
    return df


def main():
    parser = configargparse.ArgumentParser(
        description="Script to retrieve metrics from prediction CSVs"
    )
    parser.add_argument("--ground-truth", type=str, required=True, help="Path containing ground truth CSVs")
    parser.add_argument("--predictions", type=str, required=True, help="Path containing predicted CSVs")
    parser.add_argument("-o", "--outfile", type=str, required=False, default=None, help="Path to write the predictions to")
    parser.add_argument("--cutoffs", type=float, nargs="+", default=(1, 5, .25), help="Cutoff distances to use (x_min, x_max, dx)")
    parser.add_argument("--display-cutoff", type=float, default=3, help="Cutoff distance to display certain metrics on. Must be in [x_min, x_max]")

    args = parser.parse_args()
    assert args.display_cutoff >= args.cutoffs[0] and args.display_cutoff <= args.cutoffs[1], "Display cutoff distance must be in [x_min, x_max]"
    gts, preds = load_data(args.ground_truth, args.predictions)
    df = get_metrics(gts, preds, args.cutoffs)
    print(args.ground_truth, args.predictions)
    df_single = get_metrics_single(gts, preds, args.cutoffs)

    print("Aggregated metrics:")
    print(f"--------------------------------------")
    print(f"Accuracy @ {args.display_cutoff}  is {df[df.cutoff_distance.between(args.display_cutoff-1e-6, args.display_cutoff+1e-6)].accuracy.iloc[0]:.4f}")
    print(f"Precision @ {args.display_cutoff} is {df[df.cutoff_distance.between(args.display_cutoff-1e-6, args.display_cutoff+1e-6)].precision.iloc[0]:.4f}")
    print(f"Recall @ {args.display_cutoff}    is {df[df.cutoff_distance.between(args.display_cutoff-1e-6, args.display_cutoff+1e-6)].recall.iloc[0]:.4f}")
    print(f"F1 @ {args.display_cutoff}        is {df[df.cutoff_distance.between(args.display_cutoff-1e-6, args.display_cutoff+1e-6)].f1.iloc[0]:.4f}")
    
    print(f"TP @ {args.display_cutoff}        is {df[df.cutoff_distance.between(args.display_cutoff-1e-6, args.display_cutoff+1e-6)].tp.iloc[0]}")
    print(f"FP @ {args.display_cutoff}        is {df[df.cutoff_distance.between(args.display_cutoff-1e-6, args.display_cutoff+1e-6)].fp.iloc[0]}")
    print(f"FN @ {args.display_cutoff}        is {df[df.cutoff_distance.between(args.display_cutoff-1e-6, args.display_cutoff+1e-6)].fn.iloc[0]}")

    print(f"Integral-F1 score is {df.integral_f1.iloc[0]:.4f}")
    print(f"--------------------------------------")
    print(f"F1 @ {args.display_cutoff}        is {df[df.cutoff_distance.between(args.display_cutoff-1e-6, args.display_cutoff+1e-6)].f1.iloc[0]:.4f}")
    print(f"--------------------------------------")

    print("Single metrics:")
    print(f"--------------------------------------")
    df_subset = df_single[df_single.cutoff_distance.between(args.display_cutoff-1e-6, args.display_cutoff+1e-6)]
    print(f"F1 @ {args.display_cutoff}          is {df_subset.f1.mean():.3f} ± {df_subset.f1.std():.3f}")
    print(f"Accuracy @ {args.display_cutoff}    is {df_subset.accuracy.mean():.3f} ± {df_subset.accuracy.std():.3f}")
    print(f"Integral F1 Score is {df_subset.integral_f1.mean():.3f} ± {df_subset.integral_f1.std():.3f}")

    # print(f"Precision @ {args.display_cutoff} is {df_subset.precision.mean():.3f} ± {df_subset.precision.std():.3f}")
    # print(f"Recall @ {args.display_cutoff}    is {df_subset.recall.mean():.3f} ± {df_subset.recall.std():.3f}")
    
    print(f"--------------------------------------")
    print(f"DATASET: {args.ground_truth.split('/')[-2]}")
    if args.outfile is not None: 
        outfile = Path(args.outfile)
        outfile.parent.mkdir(exist_ok=True, parents=True)
        df.to_csv(outfile, index=False)
        outfile_single = outfile.parent/f'{outfile.stem}_single.csv'
        df_single.to_csv(outfile_single, index=False)
    return df, df_single


if __name__ == "__main__":
    df, df_single = main()