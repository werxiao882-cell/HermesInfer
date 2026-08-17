#!/usr/bin/env python3
"""
预处理脚本：将原始jsonl文件转换为简化格式
原始格式：{"conversations": [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}], "image": "..."}
目标格式：{"image": "...", "content": "..."}
"""

import json
import argparse
from pathlib import Path
from typing import Dict, Any, Optional


def extract_assistant_content(conversations: list) -> Optional[str]:
    """
    从conversations中提取assistant的content
    
    Args:
        conversations: 包含role和content的对话列表
        
    Returns:
        assistant的content内容，如果找不到则返回None
    """
    for conv in conversations:
        if conv.get("role") == "assistant":
            return conv.get("content")
    return None


def process_jsonl_line(line: str) -> Optional[Dict[str, str]]:
    """
    处理单行jsonl数据
    
    Args:
        line: 原始jsonl行
        
    Returns:
        转换后的字典，如果处理失败返回None
    """
    try:
        # 解析JSON
        data = json.loads(line.strip())
        
        # 提取图片名称
        image = data.get("image")
        if not image:
            print(f"警告：找不到image字段，跳过该行")
            return None
        
        # 提取assistant的回复内容
        conversations = data.get("conversations", [])
        content = extract_assistant_content(conversations)
        
        if not content:
            print(f"警告：找不到assistant的回复内容，跳过该行: {image}")
            return None
        
        # 返回新格式
        return {
            "image": image,
            "content": content
        }
        
    except json.JSONDecodeError as e:
        print(f"JSON解析错误：{e}")
        return None
    except Exception as e:
        print(f"处理行时发生错误：{e}")
        return None


def preprocess_jsonl(input_file: str, output_file: str) -> None:
    """
    预处理整个jsonl文件
    
    Args:
        input_file: 输入文件路径
        output_file: 输出文件路径
    """
    input_path = Path(input_file)
    output_path = Path(output_file)
    
    if not input_path.exists():
        raise FileNotFoundError(f"输入文件不存在: {input_file}")
    
    # 创建输出目录
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    processed_count = 0
    skipped_count = 0
    
    print(f"开始处理文件: {input_file}")
    print(f"输出文件: {output_file}")
    
    with open(input_path, 'r', encoding='utf-8') as infile, \
         open(output_path, 'w', encoding='utf-8') as outfile:
        
        for line_num, line in enumerate(infile, 1):
            if line.strip():  # 跳过空行
                processed_data = process_jsonl_line(line)
                
                if processed_data:
                    # 写入转换后的数据
                    json.dump(processed_data, outfile, ensure_ascii=False, separators=(',', ':'))
                    outfile.write('\n')
                    processed_count += 1
                else:
                    skipped_count += 1
            
            # 每处理1000行显示进度
            if line_num % 1000 == 0:
                print(f"已处理 {line_num} 行，成功转换 {processed_count} 行，跳过 {skipped_count} 行")
    
    print(f"处理完成！")
    print(f"总计处理: {line_num} 行")
    print(f"成功转换: {processed_count} 行")
    print(f"跳过行数: {skipped_count} 行")
    print(f"输出文件: {output_file}")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="预处理jsonl文件，提取图片和助手回复内容")
    parser.add_argument("--input_file", default="/home/lixin/workspace/personal_learning/siglip_from_scratch/dataset/pretrain_data.jsonl", help="输入的jsonl文件路径")
    parser.add_argument("--output_file", default="/home/lixin/workspace/personal_learning/siglip_from_scratch/dataset/processed_data.jsonl", help="输出的jsonl文件路径")    
    args = parser.parse_args()
    
    preprocess_jsonl(args.input_file, args.output_file)


if __name__ == "__main__":
    main()