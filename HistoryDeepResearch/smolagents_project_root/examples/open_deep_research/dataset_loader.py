import os
import pandas as pd
import json
from datasets import Dataset
from typing import Optional, List

def load_custom_dataset(
    excel_path: str,
    files_dir: Optional[str] = "Historical/Historical",
    sheet_name: Optional[str] = 'level 2',
    test_mode: bool = False,
    results_json_path: Optional[str] = None
) -> Dataset:
    """
    加载自定义Excel格式的数据集，处理各种类型的Data Requirements
    
    参数:
        excel_path: Excel文件的路径
        files_dir: 相关文件所在的目录
        sheet_name: 要加载的工作表名称或索引，None表示加载所有表单
        test_mode: 是否只加载前三条记录进行测试
        results_json_path: 先前结果的JSON文件路径，如果提供，将根据正确答案过滤数据集
        
    返回:
        Dataset: 兼容GAIA数据集格式的数据集对象
    """
    # 确保目录存在
    os.makedirs(files_dir, exist_ok=True)
    
    # 加载Excel文件
    print(f"正在加载Excel文件: {excel_path}")
    try:
        # 读取所有表单
        if sheet_name is None:
            # 先获取所有表单名称
            xls = pd.ExcelFile(excel_path)
            sheet_names = xls.sheet_names
            print(f"检测到{len(sheet_names)}个表单: {sheet_names}")
            
            # 读取所有表单数据
            all_data = []
            for sheet in sheet_names:
                print(f"正在读取表单: {sheet}")
                df = pd.read_excel(excel_path, sheet_name=sheet)
                
                # 检查表单是否为空
                if df.empty:
                    print(f"表单 {sheet} 为空，跳过")
                    continue
                
                # 检查表单是否有必要的列
                required_columns = ["ID", "Question", "Answer", "Data Requirements", "Answer Type"]
                missing_columns = [col for col in required_columns if col not in df.columns]
                if missing_columns:
                    print(f"表单 {sheet} 缺少必要的列: {missing_columns}，尝试使用其他列名")
                    
                    # 尝试映射列名
                    column_mapping = {
                        "ID": ["ID", "id", "编号", "序号"],
                        "Question": ["Question", "question", "问题"],
                        "Answer": ["Answer", "answer", "答案"],
                        "Data Requirements": ["Data Requirements", "data requirements", "文件", "附件"],
                        "Answer Type": ["Answer Type", "answer type", "答案类型", "类型"]
                    }
                    
                    for req_col, possible_names in column_mapping.items():
                        for col_name in df.columns:
                            if col_name in possible_names or any(name.lower() in col_name.lower() for name in possible_names):
                                df = df.rename(columns={col_name: req_col})
                                print(f"将列 '{col_name}' 映射为 '{req_col}'")
                                break
                
                # 再次检查必要的列
                missing_columns = [col for col in required_columns if col not in df.columns]
                if missing_columns:
                    print(f"表单 {sheet} 仍然缺少必要的列: {missing_columns}，跳过此表单")
                    continue
                
                # 添加表单名称作为任务名称
                df["task"] = sheet
                
                # 添加到总数据中
                all_data.append(df)
            
            if not all_data:
                print("所有表单都不符合要求，返回空数据集")
                return Dataset.from_pandas(pd.DataFrame({
                    "task_id": ["1"],
                    "question": ["示例问题"],
                    "true_answer": ["示例答案"],
                    "task": ["示例任务"],
                    "file_name": [""],
                    "file_type": [""],
                    "file_tool": [""],
                    "answer_type": [""]
                }))
            
            # 合并所有表单数据
            df = pd.concat(all_data, ignore_index=True)
        else:
            # 读取指定表单
            df = pd.read_excel(excel_path, sheet_name=sheet_name)
            df["task"] = str(sheet_name)  # 使用表单名称作为任务名称
    except Exception as e:
        print(f"加载Excel文件时出错: {e}")
        return Dataset.from_pandas(pd.DataFrame({
            "task_id": ["1"],
            "question": ["示例问题"],
            "true_answer": ["示例答案"],
            "task": ["示例任务"],
            "file_name": [""],
            "file_type": [""],
            "file_tool": [""],
            "answer_type": [""]
        }))
    
    # 定义列映射
    column_mapping = {
        "ID": "task_id",
        "Question": "question",
        "Answer": "true_answer",
        "Data Requirements": "data_requirement",  # 改名为data_requirement更准确
        "Answer Type": "answer_type"
    }
    
    # 应用列重命名
    for old_col, new_col in column_mapping.items():
        if old_col in df.columns:
            df = df.rename(columns={old_col: new_col})
    
    # 添加缺失的列
    for col in ["task_id", "question", "true_answer", "task", "data_requirement", "answer_type", 
                "file_name", "file_type", "file_tool", "data_type"]:  # 添加data_type字段
        if col not in df.columns:
            df[col] = ""
    
    # 添加文件类型列
    df["file_type"] = ""
    df["file_tool"] = ""
    
    # 处理Data Requirements
    def process_data_requirement(row):
        """处理Data Requirements字段，确定其类型和相应的处理方法"""
        # 获取data_requirement值，确保它是字符串
        data_req = row.get("data_requirement", "")
        
        # 检查是否为NaN或None，转换为空字符串
        if pd.isna(data_req) or data_req is None:
            data_req = ""
        else:
            # 确保是字符串类型
            data_req = str(data_req).strip()
        
        # 如果是空字符串，设置为none类型
        if not data_req:
            row["data_type"] = "none"
            return row
            
        # 检查是否包含多个文件路径（用分号分隔）
        if ";" in data_req:
            # 正确分割路径并过滤空字符串
            file_paths = [path.strip() for path in data_req.split(";") if path.strip()]
            print(f"检测到多个文件路径: {file_paths}")
            
            # 初始化文件列表
            valid_files = []
            missing_files = []
            
            # 处理所有文件路径
            for path in file_paths:
                full_path = os.path.join(files_dir, path)
                # 检查文件是否存在
                if os.path.exists(full_path):
                    valid_files.append(full_path)
                else:
                    # 如果文件不存在但扩展名合法，也认为是有效文件
                    if path.lower().endswith(('.pdf', '.docx', '.xlsx', '.jpg', '.jpeg', '.png', '.gif', '.mp3', '.wav', '.zip')):
                        # 确保使用绝对路径
                        valid_files.append(full_path)
                        print(f"文件 {path} 不存在但扩展名合法，将被标记为有效")
                    else:
                        missing_files.append(path)
            
            # 如果有有效文件，设置为file类型
            if valid_files:
                # 保存主文件（第一个有效文件）用于兼容旧代码
                row["file_name"] = valid_files[0]
                # 保存所有有效文件路径到file_names字段 - 这是一个列表
                row["file_names"] = valid_files
                row["data_type"] = "file"
                
                print(f"有效文件数量: {len(valid_files)}")
                print(f"主文件: {row['file_name']}")
                if len(valid_files) > 5:
                    print(f"全部有效文件: 太多文件，只显示前5个 {valid_files[:5]} ...")
                else:
                    print(f"全部有效文件: {valid_files}")
                    
                if missing_files:
                    print(f"警告: 以下文件不存在: {missing_files}")
            else:
                print(f"警告: 没有有效文件: {data_req}")
                row["data_type"] = "unknown"
                row["file_name"] = ""
                row["file_names"] = []
            
            return row
            
        # 检查是否是单个文件路径
        file_path = os.path.join(files_dir, data_req)
        if os.path.exists(file_path) or data_req.lower().endswith(('.pdf', '.docx', '.xlsx', '.jpg', '.jpeg', '.png', '.gif', '.mp3', '.wav', '.zip')):
            # 设置单个文件
            row["file_name"] = file_path
            # 同时设置file_names列表，便于统一处理
            row["file_names"] = [file_path]
            row["data_type"] = "file"
            print(f"检测到单个文件: {file_path}")
        
        # 检查是否是外语文本（包含非ASCII字符且不是文件路径）
        elif any(ord(c) > 127 for c in data_req) and not os.path.exists(file_path):
            row["data_type"] = "foreign_text"
            print(f"检测到外语文本: {data_req[:50]}...")
        
        # 检查是否是书名或需要搜索的信息
        elif any(keyword in data_req.lower() for keyword in ["book", "novel", "article", "paper", "search", "find", "look up"]):
            row["data_type"] = "search_query"
            print(f"检测到搜索查询: {data_req}")
        
        # 其他情况，可能是普通文本或指令
        else:
            row["data_type"] = "text"
            print(f"检测到普通文本: {data_req[:50]}...")
            
        return row
    
    # 应用数据需求处理
    print("正在处理Data Requirements...")
    df = df.apply(process_data_requirement, axis=1)

    
    # 根据先前的结果过滤数据集
    if results_json_path and os.path.exists(results_json_path):
        print(f"检测到结果JSON文件: {results_json_path}")
        try:
            # 读取结果JSON文件
            with open(results_json_path, 'r', encoding='utf-8') as f:
                results_data = []
                for line in f:
                    try:
                        results_data.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue
            
            # 提取已经正确回答的task_id
            correct_task_ids = [item['task_id'] for item in results_data if item.get('is_correct', "true")]
            
            if correct_task_ids:
                # 过滤掉已经正确回答的问题
                original_count = len(df)
                df = df[~df['task_id'].astype(str).isin([str(task_id) for task_id in correct_task_ids])]
                filtered_count = original_count - len(df)
                
                print(f"根据结果JSON过滤：已正确回答的问题数量: {len(correct_task_ids)}")
                print(f"从数据集中移除的问题数量: {filtered_count}")
                print(f"过滤后的数据集大小: {len(df)}")
        except Exception as e:
            print(f"处理结果JSON时出错: {e}")
            
            
    # 如果是测试模式，只保留前三条记录
    if test_mode and len(df) > 3:
        print(f"测试模式：只保留前3条记录（共{len(df)}条）")
        df = df.head(3)
    
    # 确保所有列的数据类型正确（转换为字符串）
    for col in df.columns:
        df[col] = df[col].fillna("").astype(str)
        print(f"列 '{col}' 转换为字符串类型")
    
    # 转换为Dataset对象
    dataset = Dataset.from_pandas(df)
    print(f"数据集加载完成，共{len(dataset)}条记录")
    
    return dataset

def load_json_dataset(
    json_path: str,
    files_dir: Optional[str] = "Historical_js/Historical_js"
) -> Dataset:
    """
    加载JSON格式的数据集，自动处理各种文件类型
    
    参数:
        json_path: JSON文件的路径
        files_dir: 相关文件所在的目录
        
    返回:
        Dataset: 兼容GAIA数据集格式的数据集对象
    """
    # 确保目录存在
    os.makedirs(files_dir, exist_ok=True)
    
    # 加载JSON文件
    print(f"正在加载JSON文件: {json_path}")
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except Exception as e:
        print(f"加载JSON文件时出错: {e}")
        return Dataset.from_pandas(pd.DataFrame({
            "task_id": ["1"],
            "question": ["示例问题"],
            "true_answer": ["示例答案"],
            "task": ["示例任务"],
            "file_name": [""],
            "file_type": [""],
            "file_tool": [""]
        }))
    
    # 提取所有问题
    all_questions = []
    
    # 处理JSON结构 - 假设结构类似于提供的示例
    for level, questions in data.items():
        if isinstance(questions, list):
            for q in questions:
                # 映射字段
                question_data = {
                    "task_id": q.get("ID", ""),
                    "question": q.get("Question", ""),
                    "true_answer": q.get("Answer", ""),
                    "task": level,  # 使用level作为task
                    "file_name": q.get("Data Requirements", ""),
                    "file_type": "",
                    "file_tool": ""
                }
                all_questions.append(question_data)
    
    # 如果没有找到任何问题，显示警告
    if not all_questions:
        print(f"警告: 在JSON文件中未找到任何问题。请检查JSON结构是否正确。")
        print(f"JSON结构: {list(data.keys())}")
        # 返回一个空的数据集
        return Dataset.from_pandas(pd.DataFrame({
            "task_id": ["1"],
            "question": ["示例问题"],
            "true_answer": ["示例答案"],
            "task": ["示例任务"],
            "file_name": [""],
            "file_type": [""],
            "file_tool": [""]
        }))
    
    # 创建DataFrame
    df = pd.DataFrame(all_questions)
    
    # 处理文件路径并添加文件类型列
    def process_file_info(row):
        """处理文件路径并检测文件类型"""
        if not row.get("file_name") or not isinstance(row["file_name"], str):
            row["file_name"] = ""
            return row
            
        file_path = row["file_name"].strip()
        
        # 移除可能存在的非法字符
        clean_path = file_path.replace("\n", "").replace("\r", "")
        
        # 避免重复添加目录前缀
        if clean_path and not clean_path.startswith(files_dir):
            row["file_name"] = os.path.join(files_dir, clean_path)
        else:
            row["file_name"] = clean_path  # 如果已经包含前缀，则不再添加
        
        # 检测文件类型
        if row["file_name"] and os.path.exists(row["file_name"]):
            file_type, tool_name = detect_file_type(row["file_name"])
            row["file_type"] = file_type
            if isinstance(tool_name, list):
                # 如果有多个可能的工具，默认使用第一个
                row["file_tool"] = tool_name[0]
            elif tool_name:
                row["file_tool"] = tool_name
            
            print(f"检测到文件类型: {row['file_name']} -> {file_type}, 工具: {row['file_tool']}")
        else:
            if row["file_name"]:
                print(f"警告: 文件不存在: {row['file_name']}")
            row["file_type"] = "unknown"
            row["file_tool"] = ""
            
        return row
    
    # 应用文件处理
    print("正在处理文件信息...")
    df = df.apply(process_file_info, axis=1)
    
    # 确保所有列的数据类型正确（转换为字符串）
    for col in df.columns:
        df[col] = df[col].fillna("").astype(str)
        print(f"列 '{col}' 转换为字符串类型")
    
    # 转换为Dataset对象
    try:
        dataset = Dataset.from_pandas(df)
        print(f"成功加载JSON数据集: {len(dataset)}条记录")
        
        # 打印数据集前几条记录
        print("数据集前3条记录:")
        for i in range(min(3, len(dataset))):
            print(f"记录 {i+1}:")
            try:
                record = dataset[i]
                if hasattr(record, 'items'):
                    for k, v in record.items():
                        if k == "file_name" and v:
                            try:
                                file_exists = os.path.exists(v)
                            except:
                                file_exists = False
                            print(f"  {k}: {v} (文件存在: {file_exists})")
                        else:
                            print(f"  {k}: {v}")
                else:
                    print(f"  记录类型错误: {type(record)}")
            except Exception as e:
                print(f"  无法显示记录 {i+1}: {e}")
        
        return dataset
    except Exception as e:
        print(f"转换为Dataset时出错: {e}")
        # 尝试诊断问题
        print("尝试诊断问题...")
        print(f"DataFrame信息:")
        print(f"- 行数: {len(df)}")
        print(f"- 列数: {len(df.columns)}")
        print(f"- 列名: {df.columns.tolist()}")
        print(f"- 数据类型: {df.dtypes}")
        
        # 返回一个最简单的数据集作为后备方案
        print("创建一个空的后备数据集...")
        empty_df = pd.DataFrame({
            "task_id": ["1"],
            "question": ["示例问题"],
            "true_answer": ["示例答案"],
            "task": ["示例任务"],
            "file_name": [""],
            "file_type": [""],
            "file_tool": [""]
        })
        return Dataset.from_pandas(empty_df)

def detect_file_type(file_path):
    """
    自动检测文件类型并推荐合适的处理工具
    
    Args:
        file_path (str): 文件路径
        
    Returns:
        tuple: (file_type, tool_name) 文件类型和推荐使用的工具
    """
    if not os.path.exists(file_path):
        return "unknown", None
        
    # 提取文件扩展名
    _, ext = os.path.splitext(file_path)
    ext = ext.lower()
    
    # 基于扩展名的类型映射
    image_exts = ['.png', '.jpg', '.jpeg', '.bmp', '.gif', '.webp']
    audio_exts = ['.wav', '.mp3', '.m4a', '.ogg', '.flac']
    video_exts = ['.mp4', '.avi', '.mov', '.wmv', '.mkv']
    document_exts = {
        '.pdf': ("pdf", "PDF_Tool"),
        '.docx': ("docx", "DOCX_Tool"),
        '.doc': ("doc", "DOCX_Tool"),  # 使用DOCX工具处理
        '.xlsx': ("xlsx", "XLSX_Tool"),
        '.xls': ("xls", "XLSX_Tool"),  # 使用XLSX工具处理
        '.pptx': ("pptx", "PPTX_Tool"),
        '.ppt': ("ppt", "PPTX_Tool"),  # 使用PPTX工具处理
        '.txt': ("text", "Text_Inspector_Tool"),
        '.csv': ("csv", "XLSX_Tool"),  # 使用XLSX工具处理
        '.json': ("json", "Text_Inspector_Tool"),
        '.xml': ("xml", "Text_Inspector_Tool"),
        '.html': ("html", "Text_Inspector_Tool")
    }
    
    # 图像文件处理 - 可以用OCR或图像分析
    if ext in image_exts:
        return "image", ["Image_Analysis_Tool", "Text_Detector_Tool"]
    
    # 音频文件处理
    elif ext in audio_exts:
        return "audio", "Speech_Recognition_Tool"
    
    # 视频文件处理 - 目前没有专门的视频工具，但可以提取关键帧
    elif ext in video_exts:
        return "video", None  # 暂不支持直接处理视频
    
    # 文档文件处理
    elif ext in document_exts:
        return document_exts[ext]
    
    # 尝试通过文件内容检测类型
    else:
        try:
            # 读取文件头部以检测文件类型
            with open(file_path, 'rb') as f:
                header = f.read(20)  # 读取前20个字节
                
                # PDF签名检测
                if header.startswith(b'%PDF'):
                    return "pdf", "PDF_Tool"
                
                # 图像签名检测
                if (header.startswith(b'\xff\xd8\xff') or  # JPEG
                    header.startswith(b'\x89PNG\r\n\x1a\n') or  # PNG
                    header.startswith(b'GIF8') or  # GIF
                    header.startswith(b'BM')):  # BMP
                    return "image", ["Image_Analysis_Tool", "Text_Detector_Tool"]
                
                # Office文档签名检测
                if header.startswith(b'PK\x03\x04'):  # ZIP格式，可能是Office文档
                    if ext == '.docx' or 'word' in file_path.lower():
                        return "docx", "DOCX_Tool"
                    elif ext == '.xlsx' or 'excel' in file_path.lower():
                        return "xlsx", "XLSX_Tool"
                    elif ext == '.pptx' or 'powerpoint' in file_path.lower():
                        return "pptx", "PPTX_Tool"
                    else:
                        return "archive", None  # ZIP或其他压缩文件
                
        except Exception as e:
            print(f"检测文件类型时出错: {e}")
        
        # 如果无法确定类型，可以尝试作为文本文件处理
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                f.read(100)  # 尝试读取前100个字符
                return "text", "Text_Inspector_Tool"  # 如果能以文本模式读取，则视为文本文件
        except UnicodeDecodeError:
            pass  # 不是文本文件
            
        return "binary", None  # 无法识别的二进制文件