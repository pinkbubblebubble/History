#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
合并多个输出文件中的正确答案

这个脚本用于合并多个JSONL格式的结果文件中标记为正确的回答，
并将结果保存为新的JSONL文件和Excel文件。
"""

import argparse
import json
import os
import pandas as pd
from pathlib import Path
from datetime import datetime
import logging
import shutil
import re

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("combine_results")

def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="合并多个结果文件中的正确答案")
    parser.add_argument(
        "input_files", 
        nargs="+", 
        help="输入的JSONL文件路径，可以指定多个文件"
    )
    parser.add_argument(
        "--output-dir", 
        "-o", 
        type=str, 
        default="output/combined", 
        help="输出目录，默认为 output/combined"
    )
    parser.add_argument(
        "--output-name", 
        type=str, 
        default=f"combined_{datetime.now().strftime('%Y%m%d_%H%M%S')}", 
        help="输出文件名称（不含扩展名），默认使用时间戳"
    )
    parser.add_argument(
        "--conflict-strategy", 
        type=str, 
        choices=["first", "latest", "model"], 
        default="first",
        help="冲突解决策略: first=保留第一个正确答案, latest=保留最新答案, model=保留指定模型的答案"
    )
    parser.add_argument(
        "--preferred-model", 
        type=str,
        help="当使用model冲突策略时，指定优先使用的模型ID"
    )
    parser.add_argument(
        "--formats",
        type=str,
        nargs="+",
        choices=["jsonl", "excel", "txt", "all"],
        default=["all"],
        help="指定输出格式，可选jsonl、excel、txt或all(全部)，默认为all"
    )
    parser.add_argument(
        "--add-readme",
        action="store_true",
        help="在输出目录中生成README.md文件，说明合并过程和结果"
    )
    parser.add_argument(
        "--level",
        type=str,
        choices=["level1", "level2", "level3", "all"],
        default="all",
        help="筛选特定级别的问题，可选level1、level2、level3或all(全部)，默认为all"
    )
    parser.add_argument(
        "--copy-images",
        action="store_true",
        help="复制与正确答案相关的图像文件到输出目录"
    )
    parser.add_argument(
        "--images-dir",
        type=str,
        default="dataset",
        help="图像文件所在的根目录，默认为dataset"
    )
    return parser.parse_args()

def read_jsonl_file(file_path):
    """
    读取JSONL文件并返回解析后的列表
    
    参数:
        file_path (str): JSONL文件路径
        
    返回:
        list: 包含文件中所有JSON对象的列表
    """
    logger.info(f"读取文件: {file_path}")
    results = []
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                try:
                    results.append(json.loads(line))
                except json.JSONDecodeError:
                    logger.warning(f"警告: 第{i+1}行不是有效的JSON，已跳过")
    except Exception as e:
        logger.error(f"读取文件 {file_path} 时出错: {e}")
    
    logger.info(f"成功读取 {len(results)} 条记录")
    return results

def extract_correct_answers(results, level=None):
    """
    从结果中提取正确答案
    
    参数:
        results (list): 结果列表
        level (str): 要筛选的问题级别，如果为None或'all'则不筛选
        
    返回:
        dict: 以task_id为键的正确答案字典
    """
    correct_answers = {}
    for item in results:
        try:
            # 如果需要筛选级别
            if level and level != "all":
                # 从task_id中提取级别信息
                task_id = item.get("task_id", "")
                # 尝试不同的任务ID格式
                level_in_id = None
                
                # 格式1: "level1/task123"
                if "/" in task_id:
                    level_in_id = task_id.split("/")[0]
                # 格式2: "level1_task123"
                elif "_" in task_id and task_id.split("_")[0] in ["level1", "level2", "level3"]:
                    level_in_id = task_id.split("_")[0]
                # 格式3: 通过其他字段识别
                elif "level" in item:
                    level_in_id = item.get("level")
                    
                # 如果没有匹配到所需级别，则跳过
                if level_in_id != level:
                    continue
            
            # 只保留正确答案
            if item.get("is_correct", False):
                task_id = item.get("task_id", "unknown")
                correct_answers[task_id] = item
                
        except (KeyError, TypeError) as e:
            logger.warning(f"处理结果时出错: {e}")
    
    return correct_answers

def combine_results(input_files, conflict_strategy="first", preferred_model=None, level=None):
    """
    合并多个文件中的正确答案
    
    参数:
        input_files (list): 输入文件路径列表
        conflict_strategy (str): 冲突解决策略 ("first", "latest", "model")
        preferred_model (str): 当策略为"model"时，指定优先使用的模型ID
        level (str): 要筛选的问题级别，如果为None或'all'则不筛选
        
    返回:
        dict: 合并后的正确答案字典
        dict: 每个文件贡献的正确答案数量统计
        int: 冲突数量
    """
    all_correct_answers = {}
    file_stats = {}
    conflict_count = 0
    
    for file_path in input_files:
        file_name = os.path.basename(file_path)
        file_stats[file_name] = {"total": 0, "added": 0, "conflicts": 0}
        
        # 读取文件
        results = read_jsonl_file(file_path)
        
        # 提取正确答案
        correct_answers = extract_correct_answers(results, level)
        file_stats[file_name]["total"] = len(correct_answers)
        
        # 合并到总结果中
        for task_id, answer in correct_answers.items():
            if task_id not in all_correct_answers:
                # 新答案，直接添加
                all_correct_answers[task_id] = answer
                file_stats[file_name]["added"] += 1
            else:
                # 发现冲突，根据策略处理
                conflict_count += 1
                file_stats[file_name]["conflicts"] += 1
                
                existing_answer = all_correct_answers[task_id]
                
                if conflict_strategy == "first":
                    # 保留第一个答案（默认行为）
                    logger.info(f"冲突: 问题 {task_id} 在多个文件中都有正确答案，保留第一个")
                    continue
                
                elif conflict_strategy == "latest":
                    # 比较时间戳，保留最新的
                    existing_timestamp = existing_answer.get("timestamp", 0)
                    new_timestamp = answer.get("timestamp", 0)
                    
                    if new_timestamp > existing_timestamp:
                        logger.info(f"冲突: 问题 {task_id} 保留更新的答案 ({file_name})")
                        all_correct_answers[task_id] = answer
                        file_stats[file_name]["added"] += 1
                    else:
                        logger.info(f"冲突: 问题 {task_id} 保留现有答案 (更新)")
                
                elif conflict_strategy == "model" and preferred_model:
                    # 根据模型ID选择
                    existing_model = existing_answer.get("model_id", "")
                    new_model = answer.get("model_id", "")
                    
                    if new_model == preferred_model and existing_model != preferred_model:
                        logger.info(f"冲突: 问题 {task_id} 保留首选模型 {preferred_model} 的答案")
                        all_correct_answers[task_id] = answer
                        file_stats[file_name]["added"] += 1
                    else:
                        logger.info(f"冲突: 问题 {task_id} 保留现有答案 (不是首选模型)")
    
    # 计算简化版的文件统计（只保留添加的条目数）
    simple_file_stats = {name: stats["added"] for name, stats in file_stats.items()}
    
    return all_correct_answers, simple_file_stats, conflict_count

def save_results(combined_answers, output_dir, output_name):
    """
    保存合并后的结果
    
    参数:
        combined_answers (dict): 合并后的正确答案字典
        output_dir (str): 输出目录
        output_name (str): 输出文件名（不含扩展名）
    
    返回:
        tuple: (jsonl_path, excel_path) 保存的JSONL和Excel文件路径
    """
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)
    
    # 准备输出文件路径
    jsonl_path = os.path.join(output_dir, f"{output_name}.jsonl")
    excel_path = os.path.join(output_dir, f"{output_name}.xlsx")
    
    # 转换为列表
    answers_list = list(combined_answers.values())
    
    # 保存为JSONL
    logger.info(f"保存JSONL文件: {jsonl_path}")
    with open(jsonl_path, 'w', encoding='utf-8') as f:
        for item in answers_list:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    
    # 保存为Excel
    logger.info(f"保存Excel文件: {excel_path}")
    df = pd.DataFrame(answers_list)
    df.to_excel(excel_path, index=False)
    
    return jsonl_path, excel_path

def generate_statistics(combined_answers, file_stats, conflict_count):
    """
    生成合并结果的统计信息
    
    参数:
        combined_answers (dict): 合并后的正确答案字典
        file_stats (dict): 每个文件贡献的正确答案数量
        conflict_count (int): 冲突的数量
        
    返回:
        str: 格式化的统计信息字符串
    """
    stats = []
    stats.append("=" * 50)
    stats.append("合并结果统计")
    stats.append("=" * 50)
    stats.append(f"合并后的正确答案总数: {len(combined_answers)}")
    stats.append(f"发生冲突的问题数: {conflict_count}")
    stats.append("\n各文件贡献统计:")
    
    for file_name, count in file_stats.items():
        stats.append(f" - {file_name}: {count} 条正确答案")
    
    if len(combined_answers) > 0:
        # 计算每个任务类型的数量
        task_types = {}
        for item in combined_answers.values():
            task = item.get("task", "未知")
            if task not in task_types:
                task_types[task] = 0
            task_types[task] += 1
        
        stats.append("\n按任务类型统计:")
        for task, count in sorted(task_types.items(), key=lambda x: x[1], reverse=True):
            percentage = count / len(combined_answers) * 100
            stats.append(f" - {task}: {count} 条 ({percentage:.2f}%)")
        
        # 按模型统计
        model_stats = {}
        for item in combined_answers.values():
            model = item.get("model_id", "未知")
            if model not in model_stats:
                model_stats[model] = 0
            model_stats[model] += 1
        
        if len(model_stats) > 1:  # 只有当有多个模型时才显示
            stats.append("\n按模型统计:")
            for model, count in sorted(model_stats.items(), key=lambda x: x[1], reverse=True):
                percentage = count / len(combined_answers) * 100
                stats.append(f" - {model}: {count} 条 ({percentage:.2f}%)")
    
    return "\n".join(stats)

def collect_image_files(combined_answers, images_dir):
    """
    收集与正确答案相关的图像文件
    
    参数:
        combined_answers (dict): 合并后的正确答案字典
        images_dir (str): 图像文件所在的根目录
        
    返回:
        dict: 以task_id为键，图像文件路径列表为值的字典
    """
    image_files = {}
    
    # 图像文件扩展名
    image_extensions = ['.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff']
    
    # 遍历每个正确答案
    for task_id, answer in combined_answers.items():
        task_images = []
        
        # 从问题中提取图像引用
        question = answer.get("question", "")
        
        # 查找图像路径的可能模式
        image_patterns = [
            # 从问题文本中提取图像文件路径
            r'(图片：\s*([^\s]+\.(?:jpg|jpeg|png|gif|bmp|tiff)))',
            r'(图像：\s*([^\s]+\.(?:jpg|jpeg|png|gif|bmp|tiff)))',
            r'(图\s*\d+：\s*([^\s]+\.(?:jpg|jpeg|png|gif|bmp|tiff)))',
            r'(图像\s*\d+：\s*([^\s]+\.(?:jpg|jpeg|png|gif|bmp|tiff)))',
            # 提取可能的相对路径
            r'((?:\/)?(?:dataset|data|images)\/[^\s]+\.(?:jpg|jpeg|png|gif|bmp|tiff))'
        ]
        
        # 从问题中提取图像路径
        image_paths = []
        for pattern in image_patterns:
            matches = re.findall(pattern, question, re.IGNORECASE)
            for match in matches:
                if isinstance(match, tuple):
                    # 如果匹配结果是元组，取最后一个元素（实际文件路径）
                    image_path = match[-1].strip()
                else:
                    # 如果匹配结果是字符串，直接使用
                    image_path = match.strip()
                
                # 去除引号和其他不需要的字符
                image_path = re.sub(r'^[\'"]|[\'"]$', '', image_path)
                
                # 添加到路径列表
                if image_path and image_path not in image_paths:
                    image_paths.append(image_path)
        
        # 从文件字段提取图像信息
        files = answer.get("files", [])
        if isinstance(files, list):
            for file_info in files:
                if isinstance(file_info, dict):
                    file_path = file_info.get("path", "")
                    if any(file_path.lower().endswith(ext) for ext in image_extensions):
                        if file_path and file_path not in image_paths:
                            image_paths.append(file_path)
        
        # 尝试使用task_id查找图像
        if "/" in task_id:
            # 处理形如 "level2/task123" 的task_id
            parts = task_id.split("/")
            level = parts[0]
            task_name = parts[-1]
            
            # 尝试查找对应的图像文件
            possible_image_locations = [
                os.path.join(images_dir, level, f"{task_name}.jpg"),
                os.path.join(images_dir, level, f"{task_name}.png"),
                os.path.join(images_dir, level, "images", f"{task_name}.jpg"),
                os.path.join(images_dir, level, "images", f"{task_name}.png")
            ]
            
            for img_path in possible_image_locations:
                if os.path.exists(img_path) and img_path not in image_paths:
                    image_paths.append(img_path)
        
        # 检查所有路径是否存在
        for path in image_paths:
            # 处理相对路径
            if not os.path.isabs(path):
                # 尝试不同的基础路径
                possible_paths = [
                    path,  # 原始路径
                    os.path.join(images_dir, path),  # 相对于图像目录
                    # 如果路径以 'dataset/'、'data/' 或 'images/' 开头，尝试去除这个前缀
                    re.sub(r'^(?:dataset|data|images)\/', '', path)
                ]
                
                for p in possible_paths:
                    if os.path.exists(p):
                        task_images.append(p)
                        break
            else:
                # 绝对路径
                if os.path.exists(path):
                    task_images.append(path)
        
        # 如果找到了图像文件，添加到结果中
        if task_images:
            image_files[task_id] = task_images
    
    return image_files

def copy_images_to_output(image_files, output_dir):
    """
    将图像文件复制到输出目录
    
    参数:
        image_files (dict): 以task_id为键，图像文件路径列表为值的字典
        output_dir (str): 输出目录
        
    返回:
        dict: 以task_id为键，复制后的图像文件路径列表为值的字典
    """
    # 创建图像输出目录
    images_output_dir = os.path.join(output_dir, "images")
    os.makedirs(images_output_dir, exist_ok=True)
    
    copied_images = {}
    
    for task_id, image_paths in image_files.items():
        task_copied_images = []
        
        for i, src_path in enumerate(image_paths):
            # 构造目标路径
            file_ext = os.path.splitext(src_path)[1]
            dest_filename = f"{task_id.replace('/', '_')}_{i+1}{file_ext}"
            dest_path = os.path.join(images_output_dir, dest_filename)
            
            # 复制文件
            try:
                shutil.copy2(src_path, dest_path)
                logger.info(f"复制图像文件: {src_path} -> {dest_path}")
                
                # 存储相对路径（用于在README和TXT中引用）
                rel_path = os.path.join("images", dest_filename)
                task_copied_images.append(rel_path)
            except Exception as e:
                logger.error(f"复制图像文件时出错: {e}")
        
        if task_copied_images:
            copied_images[task_id] = task_copied_images
    
    return copied_images

def save_to_txt(combined_answers, output_dir, output_name, copied_images=None):
    """
    将合并结果保存为TXT格式
    
    参数:
        combined_answers (dict): 合并后的正确答案字典
        output_dir (str): 输出目录
        output_name (str): 输出文件名称（不含扩展名）
        copied_images (dict, optional): 以task_id为键，图像文件路径列表为值的字典
        
    返回:
        str: 保存的TXT文件路径
    """
    output_path = os.path.join(output_dir, f"{output_name}.txt")
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write("=" * 80 + "\n")
        f.write(f"合并结果报告 - 生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"共处理 {len(combined_answers)} 个问题\n\n")
        
        # 按照task_id排序
        sorted_answers = sorted(combined_answers.items(), key=lambda x: x[0])
        
        for idx, (task_id, item) in enumerate(sorted_answers, 1):
            f.write("-" * 80 + "\n")
            f.write(f"问题 {idx}: {task_id}\n")
            f.write("-" * 80 + "\n\n")
            
            # 基本信息
            question = item.get("question", "无问题文本")
            answer_type = item.get("answer_type", "")
            data_request = item.get("data_request", "")
            our_answer = item.get("answer", "无答案")
            correct_answer = item.get("correct_answer", "")
            
            f.write(f"问题: {question}\n\n")
            
            if answer_type:
                f.write(f"答案类型: {answer_type}\n")
            if data_request:
                f.write(f"数据要求: {data_request}\n")
            
            f.write("\n")
            
            # 显示图像引用
            if copied_images and task_id in copied_images:
                f.write("相关图像:\n")
                for img_path in copied_images[task_id]:
                    f.write(f"- {img_path}\n")
                f.write("\n")
            
            # 问题答案
            f.write(f"我们的答案: {our_answer}\n\n")
            
            if correct_answer:
                f.write(f"正确答案: {correct_answer}\n\n")
            
            # 附加信息
            file_name = item.get("file_name", "")
            file_type = item.get("file_type", "")
            tool = item.get("tool", "")
            model_id = item.get("model_id", "")
            
            if file_name:
                f.write(f"文件名: {file_name}\n")
            if file_type:
                f.write(f"文件类型: {file_type}\n")
            if tool:
                f.write(f"使用工具: {tool}\n")
            if model_id:
                f.write(f"模型ID: {model_id}\n")
            
            # 摘要
            summary = item.get("summary", "")
            if summary:
                f.write("\n摘要:\n")
                f.write(summary + "\n")
            
            f.write("\n\n")
    
    return output_path

def create_readme(path, input_files, args, saved_files, statistics, answer_count, copied_images=None):
    """
    创建README.md文件，记录合并过程和结果
    
    参数:
        path (str): README文件路径
        input_files (list): 输入文件列表
        args (Namespace): 命令行参数
        saved_files (dict): 保存的文件路径
        statistics (str): 统计信息
        answer_count (int): 正确答案数量
        copied_images (dict, optional): 以task_id为键，图像文件路径列表为值的字典
    """
    with open(path, 'w', encoding='utf-8') as f:
        f.write("# 结果合并报告\n\n")
        
        # 时间信息
        now = datetime.now()
        f.write(f"**生成时间**: {now.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        # 合并信息
        f.write("## 合并信息\n\n")
        f.write(f"- 合并了 **{len(input_files)}** 个结果文件\n")
        f.write(f"- 共有 **{answer_count}** 个正确答案\n")
        
        if args.level != "all":
            f.write(f"- 筛选级别: **{args.level}**\n")
            
        f.write(f"- 冲突解决策略: **{args.conflict_strategy}**")
        if args.conflict_strategy == "model":
            f.write(f", 首选模型: **{args.preferred_model}**")
        f.write("\n\n")
        
        # 输入文件
        f.write("## 输入文件\n\n")
        for i, file_path in enumerate(input_files, 1):
            f.write(f"{i}. `{os.path.basename(file_path)}`\n")
        f.write("\n")
        
        # 输出文件
        f.write("## 输出文件\n\n")
        for format_name, file_path in saved_files.items():
            f.write(f"- **{format_name}**: [`{os.path.basename(file_path)}`]({os.path.basename(file_path)})\n")
        f.write("\n")
        
        # 相关图像
        if copied_images and len(copied_images) > 0:
            f.write("## 相关图像\n\n")
            f.write(f"共收集了 **{sum(len(imgs) for imgs in copied_images.values())}** 张与问题相关的图像\n\n")
            
            # 创建图像索引表格
            f.write("### 图像索引\n\n")
            f.write("| 问题ID | 图像 |\n")
            f.write("|--------|------|\n")
            
            for task_id, img_paths in sorted(copied_images.items()):
                image_cells = []
                for img_path in img_paths:
                    # 创建图像缩略图链接
                    image_cells.append(f"[![{task_id}]({img_path})]({img_path})")
                
                # 添加到表格
                f.write(f"| {task_id} | {' '.join(image_cells)} |\n")
            
            f.write("\n")
        
        # 统计信息
        f.write("## 统计信息\n\n")
        f.write("```\n")
        f.write(statistics)
        f.write("\n```\n\n")
        
        # 使用说明
        f.write("## 使用说明\n\n")
        f.write("### JSONL格式\n\n")
        f.write("JSONL格式文件中包含每条正确答案的完整信息，每行是一个JSON对象。可使用任何支持JSON的工具或语言处理。\n\n")
        
        f.write("### Excel格式\n\n")
        f.write("Excel文件包含结构化的答案信息，适合直观查看和筛选数据。\n\n")
        
        f.write("### TXT格式\n\n")
        f.write("TXT文件是人类可读的格式，包含问题、答案和相关信息的清晰呈现。\n\n")
        
        # 命令行参考
        f.write("## 命令行参考\n\n")
        f.write("以下是生成此结果的命令行：\n\n")
        
        cmd = ["python combine_results.py"]
        cmd.extend([f'"{file}"' for file in input_files])
        if args.output_dir != "output/combined":
            cmd.append(f'--output-dir "{args.output_dir}"')
        if not args.output_name.startswith("combined_"):
            cmd.append(f'--output-name "{args.output_name}"')
        if args.conflict_strategy != "first":
            cmd.append(f'--conflict-strategy {args.conflict_strategy}')
        if args.preferred_model:
            cmd.append(f'--preferred-model "{args.preferred_model}"')
        if args.formats != ["all"]:
            cmd.append(f'--formats {" ".join(args.formats)}')
        if args.level != "all":
            cmd.append(f'--level {args.level}')
        if args.copy_images:
            cmd.append(f'--copy-images')
            if args.images_dir != "dataset":
                cmd.append(f'--images-dir "{args.images_dir}"')
        if args.add_readme:
            cmd.append('--add-readme')
        
        f.write("```\n")
        f.write(" \\\n    ".join(cmd))
        f.write("\n```\n")

def main():
    """主函数"""
    args = parse_args()
    
    # 检查输入文件是否存在
    input_files = []
    for file_path in args.input_files:
        if os.path.exists(file_path):
            input_files.append(file_path)
        else:
            logger.error(f"错误: 文件 {file_path} 不存在")
    
    if not input_files:
        logger.error("错误: 没有有效的输入文件")
        return
    
    # 检查冲突策略
    if args.conflict_strategy == "model" and not args.preferred_model:
        logger.warning("警告: 冲突策略设置为'model'但未指定首选模型，将回退到'first'策略")
        args.conflict_strategy = "first"
    
    # 合并结果
    logger.info(f"开始合并 {len(input_files)} 个文件的结果")
    logger.info(f"冲突解决策略: {args.conflict_strategy}" + 
                (f", 首选模型: {args.preferred_model}" if args.conflict_strategy == "model" else ""))
    
    if args.level != "all":
        logger.info(f"筛选级别: {args.level}")
    
    combined_answers, file_stats, conflict_count = combine_results(
        input_files, 
        conflict_strategy=args.conflict_strategy,
        preferred_model=args.preferred_model,
        level=args.level
    )
    
    # 生成统计信息
    statistics = generate_statistics(combined_answers, file_stats, conflict_count)
    logger.info("\n" + statistics)
    
    # 保存结果
    if combined_answers:
        # 确定需要保存的格式
        output_formats = args.formats
        if "all" in output_formats:
            output_formats = ["jsonl", "excel", "txt"]
        
        # 确保输出目录存在
        os.makedirs(args.output_dir, exist_ok=True)
        
        # 构建输出名称，如果筛选了级别，则在输出名称中包含级别信息
        output_name = args.output_name
        if args.level != "all":
            output_name = f"{output_name}_{args.level}"
        
        # 收集和复制图像文件
        copied_images = None
        if args.copy_images:
            logger.info("开始收集与问题相关的图像文件...")
            image_files = collect_image_files(combined_answers, args.images_dir)
            logger.info(f"找到 {sum(len(imgs) for imgs in image_files.values())} 张图像，分布在 {len(image_files)} 个问题中")
            
            if image_files:
                logger.info("开始复制图像文件到输出目录...")
                copied_images = copy_images_to_output(image_files, args.output_dir)
                logger.info(f"复制了 {sum(len(imgs) for imgs in copied_images.values())} 张图像")
        
        # 根据选择的格式保存文件
        saved_files = {}
        
        if "jsonl" in output_formats:
            jsonl_path = os.path.join(args.output_dir, f"{output_name}.jsonl")
            answers_list = list(combined_answers.values())
            with open(jsonl_path, 'w', encoding='utf-8') as f:
                for item in answers_list:
                    f.write(json.dumps(item, ensure_ascii=False) + '\n')
            saved_files["JSONL"] = jsonl_path
            logger.info(f" - JSONL: {jsonl_path}")
        
        if "excel" in output_formats:
            excel_path = os.path.join(args.output_dir, f"{output_name}.xlsx")
            df = pd.DataFrame(list(combined_answers.values()))
            # 尝试优化列顺序
            columns_order = ["task_id", "question", "answer", "reasoning", "model_id", 
                            "timestamp", "is_correct", "summary"]
            # 筛选出存在的列
            columns = [col for col in columns_order if col in df.columns]
            # 添加剩余的列
            columns.extend([col for col in df.columns if col not in columns])
            df = df[columns]
            df.to_excel(excel_path, index=False)
            saved_files["Excel"] = excel_path
            logger.info(f" - Excel: {excel_path}")
        
        if "txt" in output_formats:
            txt_path = save_to_txt(combined_answers, args.output_dir, output_name, copied_images)
            saved_files["TXT"] = txt_path
            logger.info(f" - TXT: {txt_path}")
        
        # 保存统计信息
        stats_path = os.path.join(args.output_dir, f"{output_name}_stats.txt")
        with open(stats_path, 'w', encoding='utf-8') as f:
            f.write(statistics)
        saved_files["统计信息"] = stats_path
        logger.info(f" - 统计: {stats_path}")
        
        # 如果需要，生成README.md
        if args.add_readme:
            readme_path = os.path.join(args.output_dir, "README.md")
            create_readme(readme_path, input_files, args, saved_files, statistics, len(combined_answers), copied_images)
            logger.info(f" - README: {readme_path}")
    else:
        logger.warning("警告: 没有找到任何正确答案，未生成输出文件")

if __name__ == "__main__":
    main() 