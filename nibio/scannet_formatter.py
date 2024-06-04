import pandas as pd
import numpy as np
from tqdm import tqdm
from plyfile import PlyElement, PlyData
import argparse
import json
import os
from joblib import Parallel, delayed
from scipy.spatial import Delaunay


class ScanNetFormatter:
    def __init__(self, csv_files, output_dir, verbose=False):
        self.csv_files = csv_files
        self.output_dir = output_dir
        self.verbose = verbose

    def csv_to_ply_clean(self, df, ply_file):
        # Print column names for debugging
        # remove duplicated columns
        df = df.loc[:,~df.columns.duplicated()]

        # Replace spaces in column names with underscores
        df.columns = [col.replace(' ', '_') for col in df.columns]

        # Create a structured numpy array with dtype based on the columns of the DataFrame
        dtypes = [(col, 'f4') for col in df.columns]

        dtypes = [
            ('x', 'f4'),
            ('y', 'f4'),
            ('z', 'f4'),
            ('intensity', 'uint32')
        ]
        
        df_to_ply = df[['x', 'y', 'z', 'intensity']]

        data = np.array(list(map(tuple, df_to_ply.to_records(index=False))), dtype=dtypes)

        # Create a new PlyElement
        vertex = PlyElement.describe(data, 'vertex')
        
        faces = self.generate_faces(df[['x', 'y', 'z']])

        face_element = PlyElement.describe(faces, 'face')

        ply_data = PlyData([vertex, face_element], text=False)

        ply_data.write(ply_file)


    def csv_to_ply_labels(self, df, ply_file):
        # Print column names for debugging
        # remove duplicated columns
        df = df.loc[:,~df.columns.duplicated()]

        # Replace spaces in column names with underscores
        df.columns = [col.replace(' ', '_') for col in df.columns]

        # Create a structured numpy array with dtype based on the columns of the DataFrame
        dtypes = [(col, 'f4') for col in df.columns]

        # add just the columns we need x, y, z, intensity, label, treeid

        dtypes = [
            ('x', 'f4'),
            ('y', 'f4'),
            ('z', 'f4'),
            ('intensity', 'uint32'),
            ('label', 'uint32'),
            ('treeid', 'uint32')
        ]

        df_to_ply = df[['x', 'y', 'z', 'intensity', 'label', 'treeid']]

        data = np.array(list(map(tuple, df_to_ply.to_records(index=False))), dtype=dtypes)

        # Create a new PlyElement
        vertex = PlyElement.describe(data, 'vertex')

        faces = self.generate_faces(df[['x', 'y', 'z']])

        face_element = PlyElement.describe(faces, 'face')

        ply_data = PlyData([vertex, face_element], text=False)

        ply_data.write(ply_file)

    def generate_faces(self, points):
        tri = Delaunay(points[['x', 'y', 'z']])
        # Ensure only triangles are used
        faces = []
        for simplex in tri.simplices:
            for i in range(4):
                face = simplex[np.arange(len(simplex)) != i]
                faces.append((tuple(face),))
        # faces = np.array(faces, dtype=[('vertex_indices', 'i4', (3,))])

        # format like this: data = np.array(list(map(tuple, df_to_ply.to_records(index=False))), dtype=dtypes)
        faces = np.array(list(map(tuple, faces)), dtype=[('vertex_indices', 'i4', (3,))])

        return faces



    def generate_segs(self, df, scene_id, segs_file):
        seg_indices = df['treeid'].tolist()
        segs_content = {
            "params": {
                "kThresh": "0.0001",
                "segMinVerts": "20",
                "minPoints": "750",
                "maxPoints": "30000",
                "thinThresh": "0.05",
                "flatThresh": "0.001",
                "minLength": "0.02",
                "maxLength": "1"
            },
            "sceneId": scene_id,
            "segIndices": seg_indices
        }
        with open(segs_file, 'w') as f:
            json.dump(segs_content, f, indent=2)

    def generate_aggregation(self, df, scene_id, aggregation_file):
        aggregation_data = {
            "sceneId": scene_id,
            "appId": "Aggregator.v2",
            "segGroups": []
        }
        
        # Group the DataFrame by 'treeid' and 'label'
        grouped = df.groupby(['treeid', 'label'])

        if self.verbose:
            print("Number of groups:", len(grouped))        
            print("Grouped by treeid and label")
            # print the combingation of treeid and label
            for name, group in grouped:
                print(name)

        
        # Iterate over each group created by the 'treeid' and 'label' columns
        for idx, ((treeid, label), group) in enumerate(grouped):
            # Get the list of indices for the current group
            segments = group.index.tolist()

            # Map numeric labels to ScanNet category names TODO: change this to take from a file
            if label == 1:
                label = "ground"
            elif label == 2:
                label = "vegetation"
            elif label == 4:
                label = "trunk"

            # Append a new segment group to the 'segGroups' list in the aggregation data
            aggregation_data['segGroups'].append({
                "id": idx,
                "objectId": treeid,
                "segments": segments,
                "label": str(label)
            })
        
        # Write the aggregation data to the specified file in JSON format
        with open(aggregation_file, 'w') as f:
            json.dump(aggregation_data, f, indent=2)



    def process_plot(self, csv_file):
        plot_name = os.path.splitext(os.path.basename(csv_file))[0]
        scene_id = plot_name
        ply_file_clean = os.path.join(self.output_dir, f"{plot_name}_vh_clean_2.ply")
        ply_file_labels = os.path.join(self.output_dir, f"{plot_name}_vh_clean_2.labels.ply")
        segs_file = os.path.join(self.output_dir, f"{plot_name}_vh_clean_2.0.010000.segs.json")
        aggregation_file = os.path.join(self.output_dir, f"{plot_name}.aggregation.json")
        
        df = pd.read_csv(csv_file)
        # Convert all column headers to lower case
        df.columns = df.columns.str.lower()

        self.csv_to_ply_clean(df, ply_file_clean)
        self.csv_to_ply_labels(df, ply_file_labels)
        self.generate_segs(df, scene_id, segs_file)
        self.generate_aggregation(df, scene_id, aggregation_file)

        # put files generated for each csv to a folder with the same name as the csv
        if not os.path.exists(os.path.join(self.output_dir, plot_name)):
            os.makedirs(os.path.join(self.output_dir, plot_name))

        os.rename(ply_file_clean, os.path.join(self.output_dir, plot_name, f"{plot_name}_vh_clean_2.ply"))
        os.rename(ply_file_labels, os.path.join(self.output_dir, plot_name, f"{plot_name}_vh_clean_2.labels.ply"))
        os.rename(segs_file, os.path.join(self.output_dir, plot_name, f"{plot_name}_vh_clean_2.0.010000.segs.json"))
        os.rename(aggregation_file, os.path.join(self.output_dir, plot_name, f"{plot_name}.aggregation.json"))

    def process_all_plots(self):
        Parallel(n_jobs=-1)(delayed(self.process_plot)(csv_file) for csv_file in tqdm(self.csv_files))

def main():
    parser = argparse.ArgumentParser(description='Convert CSV point cloud data to ScanNet format.')
    parser.add_argument('-i', '--input_dir', required=True, help='Directory containing the CSV files')
    parser.add_argument('-o', '--output_dir', required=True, help='Directory to save the output files')
    parser.add_argument('-v', '--verbose', action='store_true', help='Print verbose output')
    
    args = parser.parse_args()
    # Check if the output directory exists
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    csv_files = [os.path.join(args.input_dir, file) for file in os.listdir(args.input_dir) if file.endswith('.csv')]
    formatter = ScanNetFormatter(csv_files, args.output_dir, args.verbose)
    formatter.process_all_plots()

if __name__ == '__main__':
    main()

# Example usage:
# python3 convert_to_scannet.py -i path/to/input_directory -o path/to/output_directory
