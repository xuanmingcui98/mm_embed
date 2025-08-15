import json, os

all_tasks = ['classification', 'vqa', 'retrieval', 'grounding']

tasks = {
    "classification": {"ImageNet-1K", "N24News", "HatefulMemes", "VOC2007", "SUN397", "Place365", "ImageNet-A", "ImageNet-R", "ObjectNet", "Country211"},
    "vqa": {"OK-VQA", "A-OKVQA", "DocVQA", "InfographicsVQA", "ChartQA", "Visual7W", "ScienceQA", "VizWiz", "GQA", "TextVQA"},
    "retrieval": {"VisDial", "CIRR", "VisualNews_t2i", "VisualNews_i2t", "MSCOCO_t2i", "MSCOCO_i2t", "NIGHTS", "WebQA", "FashionIQ", "Wiki-SS-NQ", "OVEN", "EDIS"},
    "grounding": {"MSCOCO", "RefCOCO", "RefCOCO-Matching", "Visual7W-Pointing"}
}

ood = {
    "classification": {"Place365", "ImageNet-A", "ImageNet-R", "ObjectNet", "Country211"},
    "vqa": {"ScienceQA", "VizWiz", "GQA", "TextVQA"},
    "retrieval": {"FashionIQ", "Wiki-SS-NQ", "OVEN", "EDIS"},
    "grounding": {"RefCOCO", "RefCOCO-Matching", "Visual7W-Pointing"}
}


def get_summary(dir):

    files = os.listdir(dir)

    result_files = [f for f in files if f.endswith("score.json")]

    overall = []
    all_ind = []
    all_ood = []
    by_task = {}


    for f in result_files:
        subset = f.replace("_score.json", "").replace("_sftonly", "")
        task = None
        for t in all_tasks:
            if subset in tasks[t]:
                task = t
        assert task is not None, f"Task not found for subset {subset}"
        is_ood = "ood" if subset in ood[task] else "ind"

        with open(os.path.join(dir, f), "r") as file:
            res = json.load(file)['acc']

        overall.append(res)
        if task not in by_task:
            by_task[task] = {}
        
        if is_ood not in by_task[task]:
            by_task[task][is_ood] = {}

        by_task[task][is_ood][subset] = res


    for task,v in by_task.items():
        task_all = []

        if "ood" in v:
            vs = v["ood"].values()
            all_ood.extend(vs)
            task_all.extend(vs)
        if "ind" in v:
            vs = v["ind"].values()
            all_ind.extend(vs)
            task_all.extend(vs)

        by_task[task]['overall'] = round(sum(task_all) / len(task_all), 4)
        if "ind" in by_task[task]:
            by_task[task]['ind']['overall'] = round(sum(v["ind"].values()) / len(v["ind"]), 4)
        if "ood" in by_task[task]:
            by_task[task]['ood']['overall'] = round(sum(v["ood"].values()) / len(v["ood"]), 4)

    by_task['overall'] = round(sum(overall) / len(overall), 4)
    if len(all_ind) > 0:
        by_task['ind'] = round(sum(all_ind) / len(all_ind), 4)
    if len(all_ood) > 0:
        by_task['ood'] = round(sum(all_ood) / len(all_ood), 4)
    print("Overall Results:")
    print(json.dumps(by_task, indent=4))
    
    json.dump(by_task, open(os.path.join(dir, "summary.json"), "w"), indent=4)

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Get result summary from result directory")
    parser.add_argument("--dir", type=str, required=True, help="Directory containing result files")
    args = parser.parse_args()

    get_summary(args.dir)