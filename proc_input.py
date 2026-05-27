"""
=============================================================================
模块名称: proc_input.py
项目名称: THzUltraPrecision - 太赫兹超高精度频率校准系统
作　　者: TonyNee
创建日期: 2026-01-19
最后修改: 2026-05-27
=============================================================================

功能概述:
  输入数据预处理脚本, 用于批量处理 CSV 格式的频率数据文件。

当前处理操作:
  将输入目录中所有 CSV 文件的第一列数值统一减去 0.2 (GHz 偏移校正)。

适用场景:
  当测量设备的频率值存在系统性偏移时, 用此脚本进行批量校正,
  处理后的数据存入新目录, 原始数据保持不变。

使用方式:
  修改 input_dir 和 output_dir 路径后运行:
  python proc_input.py
=============================================================================
"""

import pandas as pd
import os
from pathlib import Path


def process_csv_files():
    """
    批量处理 CSV 文件: 第一列数值减去 0.2 GHz 偏移量

    处理流程:
      1. 检查输入目录是否存在
      2. 创建输出目录 (不存在则递归创建)
      3. 遍历所有 .csv 文件
      4. 读取第一列数值数据, 全部减去 0.2
      5. 保存到输出目录, 保持原文件名

    错误处理:
      - 无数据列的文件跳过
      - 非数值列的文件跳过
      - 单文件处理失败不影响其他文件
    """
    # 设置输入输出目录
    input_dir = Path("input/20260113/")
    output_dir = Path("input/20260119/")

    # 检查输入目录是否存在
    if not input_dir.exists():
        print(f"错误: 输入目录 {input_dir} 不存在!")
        return

    # 创建输出目录 (递归创建中间目录)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 获取所有 CSV 文件
    csv_files = list(input_dir.glob("*.csv"))

    if not csv_files:
        print(f"在 {input_dir} 中没有找到CSV文件!")
        return

    print(f"找到 {len(csv_files)} 个CSV文件")

    # 逐个处理 CSV 文件
    for csv_file in csv_files:
        try:
            # 读取 CSV 文件
            df = pd.read_csv(csv_file)

            # 检查是否有数据列
            if df.shape[1] < 1:
                print(f"警告: 文件 {csv_file.name} 没有数据列, 跳过处理")
                continue

            # 获取第一列的表头名
            first_column = df.columns[0]

            # 确保第一列是数值类型, 避免字符串/日期等非数值数据
            if not pd.api.types.is_numeric_dtype(df[first_column]):
                print(f"警告: 文件 {csv_file.name} 的第一列不是数值类型, 跳过处理")
                continue

            # 核心操作: 第一列所有数据减去 0.2 (GHz 偏移校正)
            df[first_column] = df[first_column] - 0.2

            # 构建输出路径 (保持原文件名)
            output_file = output_dir / csv_file.name

            # 保存处理后的 CSV (不含行索引)
            df.to_csv(output_file, index=False)

            print(f"已处理: {csv_file.name} -> {output_file}")

        except Exception as e:
            print(f"处理文件 {csv_file.name} 时出错: {e}")


if __name__ == "__main__":
    process_csv_files()
