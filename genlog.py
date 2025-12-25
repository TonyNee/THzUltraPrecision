import os
import yaml
import csv


def flatten_dict(d, parent_key="", sep="_"):
    """
    将嵌套 dict 扁平化
    """
    items = {}
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.update(flatten_dict(v, new_key, sep=sep))
        elif isinstance(v, list):
            # list 转成字符串，避免 CSV 结构破坏
            items[new_key] = "-".join(map(str, v))
        else:
            items[new_key] = v
    return items


def load_config_yaml(yaml_path):
    with open(yaml_path, "r") as f:
        cfg = yaml.safe_load(f)
    return flatten_dict(cfg)


def collect_yaml_from_root(root_dir, yaml_name="config.yaml"):
    """
    遍历 root_dir 下的所有子目录，读取 config.yaml
    """
    records = []

    for sub in sorted(os.listdir(root_dir)):
        sub_dir = os.path.join(root_dir, sub)
        if not os.path.isdir(sub_dir):
            continue

        yaml_path = os.path.join(sub_dir, yaml_name)
        if not os.path.exists(yaml_path):
            continue

        try:
            record = load_config_yaml(yaml_path)
            record["experiment_dir"] = sub  # 记录时间戳目录名
            records.append(record)
        except Exception as e:
            print(f"[WARN] Failed to read {yaml_path}: {e}")

    return records


def save_csv(records, output_csv):
    """
    自动对齐所有字段，写 CSV
    """
    if not records:
        print("No records found.")
        return

    # 收集所有可能的字段名
    all_keys = set()
    for r in records:
        all_keys.update(r.keys())

    fieldnames = sorted(all_keys)

    with open(output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in records:
            writer.writerow(r)

    print(f"[OK] CSV saved to: {output_csv}")


if __name__ == "__main__":
    # ====== 你只需要改这里 ======
    ROOT_DIR = "./output/resmlp"
    OUTPUT_CSV = "./output/resmlp/log.csv"

    records = collect_yaml_from_root(ROOT_DIR)
    save_csv(records, OUTPUT_CSV)
