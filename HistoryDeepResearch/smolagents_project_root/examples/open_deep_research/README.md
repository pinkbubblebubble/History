# Open Deep Research

Welcome to this open replication of [OpenAI's Deep Research](https://openai.com/index/introducing-deep-research/)!

Read more about this implementation's goal and methods [in our blog post](https://huggingface.co/blog/open-deep-research).

This agent achieves 55% pass@1 on GAIA validation set, vs 67% for Deep Research.

To install it, first run
```bash
pip install -r requirements.txt
```

And install smolagents dev version
```bash
pip install smolagents[dev]
```

Then you're good to go! Run the run.py script, as in:
```bash
python run.py --model-id "o1" "Your question here!"
```

## 汉语问答系统

本系统是针对中文历史问题的自动问答系统，利用大语言模型（如GPT-4）结合专门的代理层次结构来回答历史问题。

## 运行方法

确保您已经配置好环境变量：

```
export OPENAI_API_KEY="你的OpenAI API密钥"
export SERPAPI_API_KEY="你的SerpApi API密钥（可选）"
export IMGBB_API_KEY="你的ImgBB API密钥（可选）"
```

然后，使用以下命令运行程序：

```bash
cd HistoryDeepResearch/smolagents/examples/open_deep_research
python run_hle.py --run-name "测试运行" --use-image-agent --use-file-agent
```

## 命令行参数

- `--concurrency`: 并发任务数量（默认：8）
- `--model-id`: 模型ID（默认：gpt-4o）
- `--run-name`: 运行名称（必需）
- `--api-key`: OpenAI API密钥（默认使用环境变量）
- `--use-image-agent`: 启用图像信息代理，用于分析图像和进行反向图像搜索
- `--use-file-agent`: 启用文件处理代理，用于处理各种文件类型
- `--use-literature-agent`: 启用文献搜索代理，用于查找和分析学术文献
- `--baseline`: 使用基线代理而不是代理层次结构，结果存储在output_baseline/目录
- `--level`: 指定要测试的问题级别，可选值为"level1"、"level2"或"level3"（默认：level2）
- `--results-json-path`: 指定先前运行结果的JSON文件路径，用于过滤已回答正确的问题
- `--question-ids`: 指定要运行的特定问题ID，以逗号分隔（例如："16,24,35"）。可以使用数字ID（将自动添加level前缀）或完整ID
- `--start-id`: 指定要运行的问题ID范围的起始ID
- `--end-id`: 指定要运行的问题ID范围的结束ID

### 示例命令

使用标准代理层次结构（针对level2问题）：
```bash
python run_hle.py --run-name "test_all" --use-image-agent --use-file-agent
```

使用基线代理（针对level1问题）：
```bash
python run_hle.py --run-name "baseline_test" --baseline --level level1
```

使用全功能代理层次结构（包括文献搜索，针对level3问题）：
```bash
python run_hle.py --run-name "full_research" --use-image-agent --use-file-agent --use-literature-agent --level level3
```

运行特定问题ID（例如运行level2中的第16题，24题和35题）：
```bash
python run_hle.py --run-name "specific_questions" --level level2 --question-ids "16,24,35"
```

运行指定范围的问题（例如运行level1中的第10题到第30题）：
```bash
python run_hle.py --run-name "question_range" --level level1 --start-id 10 --end-id 30
```

结合使用多个特性（使用图像代理运行level3中的第5题和第8题）：
```bash
python run_hle.py --run-name "combined_example" --level level3 --use-image-agent --question-ids "5,8"
```

## 输出文件

程序运行后会在指定目录（默认为`output/level2_summary`或者`output_baseline/level2_summary`）生成以下文件：

- JSONL文件：包含详细的执行结果
- Excel文件：用于数据分析
- TXT文件：人类可读的输出
- 统计文件：记录准确率分析
- 日志文件：用于监控和调试

## 日志系统

程序使用Python的logging模块记录运行过程。日志文件存储在`output/{SET}/logs/`目录下：

- `main.log`：包含程序的主要流程和总体运行信息
- `task_{task_id}.log`：针对每个具体问题的日志，包含问题处理的详细信息
- `errors.log`：记录所有错误和异常

## 代理说明

系统支持以下代理：

- **图像信息代理**：处理图像文件，进行OCR识别，反向图像搜索等
- **文件处理代理**：处理各种类型的文件，如PDF、Word、Excel等
- **网络浏览代理**：执行在线搜索查询和信息检索
- **文献搜索代理**：专门用于搜索和分析学术文献的代理，包括以下工具：
  - `literature_searching_task`：在Google Scholar中搜索高影响力、最新的学术文章
  - `relevant_literature_finder`：分析和筛选最相关的文献资源
  - `general_browser_task`：执行通用网络搜索
- **基线代理**：使用`--baseline`参数时的简化代理，没有复杂的层次结构

## 学术文献搜索能力

系统现在增强了对历史学术研究的支持能力：

1. **高质量文献搜索**：从Google Scholar等学术数据库中查找高引用率、高影响力的学术文章
2. **智能文献筛选**：基于相关性和引用影响自动筛选和排序最相关的文献
3. **全文信息提取**：从筛选后的文献中提取关键信息，支持填空题的精确匹配

这些功能特别适合：
- 需要权威历史资料的问题
- 需要查找特定历史时期或事件的研究文献
- 需要多个来源佐证的复杂历史问题

## 结果合并工具

为了便于分析不同模型或策略产生的结果，我们提供了一个专用的结果合并工具 `combine_results.py`，它能够从多个结果文件中提取正确答案并合并到一个文件中，同时支持多种输出格式。

### 功能特点

- **多文件合并**：合并多个JSONL结果文件中的正确答案
- **冲突解决策略**：支持多种冲突解决策略（保留第一个、保留最新、根据模型选择）
- **多格式输出**：支持JSONL、Excel和TXT格式输出
- **统计分析**：生成详细的统计信息，包括不同文件的贡献、任务类型分布和模型表现
- **自动生成报告**：可选生成README.md报告文件，便于结果共享和分析

### 使用方法

```bash
python combine_results.py [input_files] [options]
```

### 命令行参数

| 参数 | 说明 |
|------|------|
| `input_files` | 输入的JSONL文件路径，可以指定多个文件 |
| `--output-dir`, `-o` | 输出目录，默认为 output/combined |
| `--output-name` | 输出文件名称（不含扩展名），默认使用时间戳 |
| `--conflict-strategy` | 冲突解决策略: first=保留第一个正确答案, latest=保留最新答案, model=保留指定模型的答案 |
| `--preferred-model` | 当使用model冲突策略时，指定优先使用的模型ID |
| `--formats` | 指定输出格式，可选jsonl、excel、txt或all(全部)，默认为all |
| `--add-readme` | 在输出目录中生成README.md文件，说明合并过程和结果 |
| `--level` | 筛选特定级别的问题，可选level1、level2、level3或all(全部)，默认为all |
| `--copy-images` | 复制与正确答案相关的图像文件到输出目录 |
| `--images-dir` | 图像文件所在的根目录，默认为dataset |

### 使用示例

1. 合并两个结果文件中的正确答案：

```bash
python combine_results.py output/results1.jsonl output/results2.jsonl
```

2. 合并多个文件并指定输出目录和名称：

```bash
python combine_results.py output/*.jsonl --output-dir analysis --output-name combined_results
```

3. 使用特定冲突解决策略：

```bash
python combine_results.py output/*.jsonl --conflict-strategy latest
```

4. 优先选择特定模型的结果：

```bash
python combine_results.py output/*.jsonl --conflict-strategy model --preferred-model "gpt-4"
```

5. 只输出Excel格式：

```bash
python combine_results.py output/*.jsonl --formats excel
```

6. 生成详细的README报告：

```bash
python combine_results.py output/*.jsonl --add-readme
```

7. 筛选特定级别的问题：

```bash
python combine_results.py output/*.jsonl --level level2
```

8. 复制相关的图像文件：

```bash
python combine_results.py output/*.jsonl --copy-images --images-dir dataset/images
```

9. 综合使用多个参数：

```bash
python combine_results.py output/*.jsonl --level level3 --conflict-strategy model --preferred-model "gpt-4-turbo" --formats jsonl txt --add-readme --copy-images
```

### 图像文件处理

当使用 `--copy-images` 参数时，合并工具会自动：

1. 从问题文本中提取图像文件引用
2. 根据任务ID尝试定位相关图像
3. 将找到的图像复制到输出目录的images子文件夹
4. 在TXT输出文件中为每个问题添加图像引用
5. 在README.md中创建图像索引表格，便于查看

这对处理包含图像分析的问题特别有用，使得最终合并报告能够同时展示问题、答案和相关图像。