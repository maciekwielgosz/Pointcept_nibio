import pandas as pd
import numpy as np
from plyfile import PlyElement, PlyData
import argparse
import json
import os


class ScanNetFormatter:
    def __init__(self, csv_files, output_dir):
        self.csv_files = csv_files
        self.output_dir = output_dir

    def csv_to_ply(self, df, ply_file):
        # Print column names for debugging
        # remove duplicated columns
        df = df.loc[:,~df.columns.duplicated()]

        # Replace spaces in column names with underscores
        df.columns = [col.replace(' ', '_') for col in df.columns]

        # Create a structured numpy array with dtype based on the columns of the DataFrame
        dtypes = [(col, 'f4') for col in df.columns]
        data = np.array(list(map(tuple, df.to_records(index=False))), dtype=dtypes)

        # Create a new PlyElement
        vertex = PlyElement.describe(data, 'vertex')

        # get just, x, y, z, intensity, label, treeid
        vertex = PlyElement.describe(data, 'vertex', comments=['x', 'y', 'z', 'intensity', 'label', 'treeid'])

        ply_data = PlyData([vertex], text=False)

        ply_data.write(ply_file)


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
        
        # Create a set to track unique object IDs
        unique_object_ids = set()

        # Iterate over each group created by the 'treeid' and 'label' columns
        for idx, ((treeid, label), group) in enumerate(grouped):
            # Get the list of indices for the current group
            segments = group.index.tolist()
            
            # Ensure objectId is unique and corresponds to the 'treeid'
            if treeid not in unique_object_ids:
                unique_object_ids.add(treeid)
            
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
        ply_file = os.path.join(self.output_dir, f"{plot_name}.ply")
        segs_file = os.path.join(self.output_dir, f"{plot_name}_vh_clean_2.0.010000.segs.json")
        aggregation_file = os.path.join(self.output_dir, f"{plot_name}.aggregation.json")
        
        df = pd.read_csv(csv_file)
        # Convert all column headers to lower case
        df.columns = df.columns.str.lower()

        self.csv_to_ply(df, ply_file)
        self.generate_segs(df, scene_id, segs_file)
        self.generate_aggregation(df, scene_id, aggregation_file)

    def process_all_plots(self):
        for csv_file in self.csv_files:
            self.process_plot(csv_file)

def main():
    parser = argparse.ArgumentParser(description='Convert CSV point cloud data to ScanNet format.')
    parser.add_argument('-i', '--input_dir', required=True, help='Directory containing the CSV files')
    parser.add_argument('-o', '--output_dir', required=True, help='Directory to save the output files')
    
    args = parser.parse_args()
    # Check if the output directory exists
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    csv_files = [os.path.join(args.input_dir, file) for file in os.listdir(args.input_dir) if file.endswith('.csv')]
    formatter = ScanNetFormatter(csv_files, args.output_dir)
    formatter.process_all_plots()

if __name__ == '__main__':
    main()

# Example usage:
# python3 convert_to_scannet.py -i path/to/input_directory -o path/to/output_directory
