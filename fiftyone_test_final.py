
import os
import fiftyone as fo

images_patt = "WAID-main/WAID/images/train"
annotation_patt = "WAID-main/WAID/labels/train"
# # Ex: your custom label format
# annotations = {
#     "/path/to/images/000001.jpg": [
#         {"bbox": ..., "label": ...},
#         ...
#     ],
#     ...
# }

labels = {}
bboxes = {}

for file in os.listdir(annotation_patt):
    f = open(os.path.join(annotation_patt, file), "r")
    data = f.readlines()[0]
    parts = data.split(' ')
    labels[file.split('.')[0]] = parts[0]
    bboxes[file.split('.')[0]] = []
    for i in parts[1:]:
        bboxes[file.split('.')[0]].append(float(i.strip()))

# Create samples for your data
samples = []
for filepath in os.listdir(images_patt):
    sample = fo.Sample(filepath=os.path.join(images_patt, filepath))

    # Convert detections to FiftyOne format
    detections = []
    for obj in filepath:
        label = labels[filepath.split('.')[0]]

        # Bounding box coordinates should be relative values
        # in [0, 1] in the following format:
        # [top-left-x, top-left-y, width, height]
        bounding_box = bboxes[filepath.split('.')[0]]

        detections.append(
            fo.Detection(label=label, bounding_box=bounding_box)
        )

    # Store detections in a field name of your choice
    sample["ground_truth"] = fo.Detections(detections=detections)

    samples.append(sample)

# Create dataset
dataset = fo.Dataset("my-detection-dataset")
dataset.add_samples(samples)

# Launch the App
session = fo.launch_app(dataset)

# (Perform any additional operations here)

# Blocks execution until the App is closed
session.wait()
