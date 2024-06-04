import numpy as np
import pandas as pd
import laspy
import argparse
import os
from joblib import Parallel, delayed

def las_to_pandas(las_file_path, csv_file_path=None):
    file_content = laspy.read(las_file_path)

    basic_dimensions = list(file_content.point_format.dimension_names)

    # Filter only available dimensions
    available_dimensions = [dim for dim in basic_dimensions if hasattr(file_content, dim.lower())]

    # Put all basic dimensions into a numpy array
    basic_points = np.vstack([getattr(file_content, dim.lower()) for dim in available_dimensions]).T
    
    # Fetch any extra dimensions
    gt_extra_dimensions = list(file_content.point_format.extra_dimension_names)

    # get only the extra dimensions which are not already in the basic dimensions
    gt_extra_dimensions = list(set(gt_extra_dimensions) - set(available_dimensions))

    if gt_extra_dimensions:
        extra_points = np.vstack([getattr(file_content, dim) for dim in gt_extra_dimensions]).T
        # Combine basic and extra dimensions
        all_points = np.hstack((basic_points, extra_points))
        all_columns = available_dimensions + gt_extra_dimensions
    else:
        all_points = basic_points
        all_columns = available_dimensions

    # Create dataframe
    points_df = pd.DataFrame(all_points, columns=all_columns)

    # Save pandas dataframe to csv
    if csv_file_path is not None:
        points_df.to_csv(csv_file_path, index=False, header=True, sep=',')

    return points_df

def process_file(las_file_path, output_folder):
    filename = os.path.basename(las_file_path)
    csv_file_path = os.path.join(output_folder, f"{os.path.splitext(filename)[0]}.csv")
    las_to_pandas(las_file_path, csv_file_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Convert las or laz files in a folder to pandas dataframes.')
    parser.add_argument('-i', '--input_folder', type=str, required=True, help='Path to the input folder containing LAS files.')
    parser.add_argument('-o', '--output_folder', type=str, required=True, help='Path to the output folder where CSV files will be saved.')

    args = parser.parse_args()

    # Create output folder if it does not exist
    if not os.path.exists(args.output_folder):
        os.makedirs(args.output_folder)

    # Get all las files in the input folder
    las_files = [os.path.join(args.input_folder, f) for f in os.listdir(args.input_folder) if f.endswith('.las')]

    # Process each file in parallel
    Parallel(n_jobs=-1)(delayed(process_file)(las_file, args.output_folder) for las_file in las_files)


# python convert_las_to_csv.py -i path/to/input/folder -o path/to/output/folder
