import json, csv
from get_result_summary import TASK_TYPE
import argparse
# print json dict as csv
# so dict is like <dataset>: <acc>
# the csv should be dataset, acc

def save_json_as_csv(json_file, valuers_only=True):
    out_csv = json_file.replace(".json", ".csv")
    with open(json_file, 'r') as f:
        data = json.load(f)['metrics']

    with open(out_csv, 'w', newline='') as f:
        writer = csv.writer(f)
        if not valuers_only:
            writer.writerow(["Dataset", "Score"])  # header
    
        for modality, v in data.items():
            if modality in {"metadata", "all_scores"}:
                continue
            for dataset, acc in v.items():
                if dataset in TASK_TYPE.keys() or dataset == 'overall':
                    continue
                if not valuers_only:
                    writer.writerow([dataset, acc])
                else:
                    writer.writerow([acc])

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--json_file', "-j", type=str, required=True, help='Path to the JSON file')
    parser.add_argument('--values_only', '-v', action='store_true', default=True, help='Output only values without headers')
    args = parser.parse_args()
    save_json_as_csv(args.json_file)