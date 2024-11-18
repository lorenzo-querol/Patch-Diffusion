import os
import json


def inspect_dataset(dataset_path):
    """
    Inspect dataset directory to check image files and labels
    """
    print(f"Inspecting dataset at: {dataset_path}")

    # Check image files
    image_files = sorted([f for f in os.listdir(dataset_path) if f.endswith(".png")])
    print(f"\nFound {len(image_files)} image files")
    print("Sample image files:", image_files[:5])

    # Check dataset.json
    json_path = os.path.join(dataset_path, "dataset.json")
    if os.path.exists(json_path):
        with open(json_path) as f:
            data = json.load(f)
            labels = data.get("labels", {})
            print(f"\nFound {len(labels)} labels in dataset.json")
            if isinstance(labels, dict):
                print("Sample labels (dict):", dict(list(labels.items())[:5]))
            elif isinstance(labels, list):
                print("Sample labels (list):", labels[:5])

            # Check for specific file
            problem_file = "img00040000.png"
            if problem_file in image_files:
                print(f"\nProblem file {problem_file} exists in directory")
                if isinstance(labels, dict) and problem_file in labels:
                    print(f"Label for {problem_file}: {labels[problem_file]}")
                else:
                    print(f"No label found for {problem_file}")
    else:
        print("\nNo dataset.json found!")


# Run for both train and validation sets
base_path = "data/cifar100"  # Replace with your dataset path
for split in ["train", "valid", "test"]:
    split_path = os.path.join(base_path, split)
    if os.path.exists(split_path):
        print(f"\n=== Checking {split} set ===")
        inspect_dataset(split_path)
