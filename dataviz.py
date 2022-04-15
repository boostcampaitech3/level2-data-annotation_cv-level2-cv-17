import glob
import fiftyone as fo
import json
from pathlib import Path
from utils import read_json
import argparse


def main(args):
    data = read_json(args.json_path)
    annotations = data.copy()['images']

    # Create samples for your data
    samples = []
    for filepath in glob.glob(args.images_path):
        sample = fo.Sample(filepath=filepath)

        # Convert detections to FiftyOne format
        detections = []

        # Add Polylines Information and Tag Information
        annotation = annotations[filepath.split('/')[-1]]
        for word in annotation['words']:
            points = annotation['words'][word]['points']
            points = [points] # Polyline points fields must contain a list of lists of (x, y) pairs

            attr = annotation['words'][word].copy()
            attr.pop('points')

            # illegebility fill color
            if annotation['words'][word]['illegibility'] :
                fill = True
            else:
                fill = False

            detections.append(
                    fo.Polyline(points=points,**attr,closed=True,filled=fill)
                )


        # Store polylines and annotation information
        sample["polylines"] = fo.Polylines(polylines=detections)
        sample['tags'] = ['None' if annotation['tags'] is None else annotation['tags']]
        samples.append(sample)

    # Create dataset
    dataset = fo.Dataset("my-detection-dataset")
    dataset.add_samples(samples) # https://voxel51.com/docs/fiftyone/user_guide/using_views.html#object-patches

    # Lauch App
    session = fo.launch_app(dataset, port =args.port, address ="0.0.0.0")
    session.wait()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--images_path', '-i', type=str, default="/opt/ml/input/data/ICDAR17_Korean/images/*",
                        help='imageData directory')
    parser.add_argument('--json_path', '-j', type=str, default="../input/data/ICDAR17_Korean/ufo/train_div.json",
                        help='annotation Data directory')
    parser.add_argument('--port', '-p', type=int, default=30001,
                        help='Port Number')
    args = parser.parse_args()
    main(args=args)
    