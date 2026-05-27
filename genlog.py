"""
=============================================================================
模块名称: genlog.py
项目名称: THzUltraPrecision - 太赫兹超高精度频率校准系统
作　　者: TonyNee
创建日期: 2025-12-20
最后修改: 2026-05-27
=============================================================================

功能概述:
  实验日志生成器, 用于批量收集和整理多次实验的配置和结果。

工作流程:
  1. 遍历指定根目录下所有子目录
  2. 读取每个子目录中的 config.yaml 文件
  3. 将嵌套字典扁平化 (嵌套 key 用 _ 连接)
  4. 自动对齐所有实验中出现的字段
  5. 汇总写入一个 CSV 文件, 方便在 Excel/Pandas 中对比分析

典型用途:
  消融实验管理 — 同时运行多组不同超参数实验, 用此脚本一键汇总所有 config.yaml,
  生成统一的实验日志 CSV, 便于横向对比各组实验的配置和结果。

使用方式:
  修改 __main__ 中的 ROOT_DIR 和 OUTPUT_CSV 路径后运行:
  python genlog.py
=============================================================================
"""

import os
import yaml
import csv


def flatten_dict(d, parent_key="", sep="_"):
    """
    递归地将嵌套字典扁平化为单层字典

    例如:
      {'model': {'type': 'resmlp', 'arch': [128, 256]}}
      -> {'model_type': 'resmlp', 'model_arch': '128-256'}

    参数:
      d:          待扁平化的字典
      parent_key: 父级键名 (递归使用)
      sep:        键名分隔符, 默认 "_"

    返回:
      dict: 扁平化后的单层字典
    """
    items = {}
    for k, v in d.items():
        # 构造新键名: 父键 + 分隔符 + 当前键
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            # 递归处理嵌套字典
            items.update(flatten_dict(v, new_key, sep=sep))
        elif isinstance(v, list):
            # 列表转为字符串 (用 "-" 连接元素), 避免破坏 CSV 结构
            items[new_key] = "-".join(map(str, v))
        else:
            items[new_key] = v
    return items


def load_config_yaml(yaml_path):
    """
    加载并扁平化单个 config.yaml 文件

    参数:
      yaml_path: YAML 文件路径

    返回:
      dict: 扁平化后的配置字典
    """
    with open(yaml_path, "r") as f:
        cfg = yaml.safe_load(f)
    return flatten_dict(cfg)


def collect_yaml_from_root(root_dir, yaml_name="config.yaml"):
    """
    遍历根目录下所有子目录, 收集所有 config.yaml

    遍历策略: 只会检查 root_dir 的直接子目录, 不会递归深入

    参数:
      root_dir:   根目录路径 (如 ./output/resmlp/)
      yaml_name:  配置文件名, 默认 "config.yaml"

    返回:
      list[dict]: 扁平化后的配置记录列表
    """
    records = []

    for sub in sorted(os.listdir(root_dir)):
        sub_dir = os.path.join(root_dir, sub)
        if not os.path.isdir(sub_dir):
            continue                                               # 跳过非目录文件

        yaml_path = os.path.join(sub_dir, yaml_name)
        if not os.path.exists(yaml_path):
            continue                                               # 跳过无配置的子目录

        try:
            record = load_config_yaml(yaml_path)
            record["experiment_dir"] = sub                         # 记录时间戳目录名
            records.append(record)
        except Exception as e:
            print(f"[WARN] Failed to read {yaml_path}: {e}")

    return records


def save_csv(records, output_csv):
    """
    将所有实验记录写入 CSV 文件

    自动收集所有记录中出现的字段名, 统一对齐后写入,
    缺失字段留空。

    参数:
      records:    扁平化记录列表
      output_csv: 输出 CSV 文件路径
    """
    if not records:
        print("No records found.")
        return

    # 收集所有记录中出现过的字段名 (union)
    all_keys = set()
    for r in records:
        all_keys.update(r.keys())

    # 排序保证列顺序稳定可复现
    fieldnames = sorted(all_keys)

    with open(output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in records:
            writer.writerow(r)

    print(f"[OK] CSV saved to: {output_csv}")


if __name__ == "__main__":
    # ===== 配置区: 修改根目录和输出路径 =====
    ROOT_DIR = "./output/resmlp"                                # 实验输出根目录
    OUTPUT_CSV = "./output/resmlp/log.csv"                      # 汇总日志 CSV 路径

    records = collect_yaml_from_root(ROOT_DIR)
    save_csv(records, OUTPUT_CSV)
