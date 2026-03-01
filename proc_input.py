import pandas as pd
import os
from pathlib import Path

def process_csv_files():
    # 设置输入和输出目录
    input_dir = Path("input/20260113/")
    output_dir = Path("input/20260119/")
    
    # 检查输入目录是否存在
    if not input_dir.exists():
        print(f"错误：输入目录 {input_dir} 不存在！")
        return
    
    # 创建输出目录
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 获取所有csv文件
    csv_files = list(input_dir.glob("*.csv"))
    
    if not csv_files:
        print(f"在 {input_dir} 中没有找到CSV文件！")
        return
    
    print(f"找到 {len(csv_files)} 个CSV文件")
    
    # 处理每个CSV文件
    for csv_file in csv_files:
        try:
            # 读取CSV文件
            df = pd.read_csv(csv_file)
            
            # 检查是否有足够的数据列
            if df.shape[1] < 1:
                print(f"警告：文件 {csv_file.name} 没有数据列，跳过处理")
                continue
            
            # 获取第一列的表头名
            first_column = df.columns[0]
            
            # 检查第一列是否包含数值数据
            if not pd.api.types.is_numeric_dtype(df[first_column]):
                print(f"警告：文件 {csv_file.name} 的第一列不是数值类型，跳过处理")
                continue
            
            # 将第一列所有数据减去0.2
            df[first_column] = df[first_column] - 0.2
            
            # 构建输出文件路径
            output_file = output_dir / csv_file.name
            
            # 保存到输出目录
            df.to_csv(output_file, index=False)
            
            print(f"已处理：{csv_file.name} -> {output_file}")
            
        except Exception as e:
            print(f"处理文件 {csv_file.name} 时出错: {e}")

if __name__ == "__main__":
    process_csv_files()

