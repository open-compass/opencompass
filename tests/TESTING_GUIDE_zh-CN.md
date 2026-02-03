# OpenCompass 数据集单元测试指引

本文档提供了为 `opencompass/datasets/` 下的数据集实现添加单元测试的指引和规范。

> **English Version**: See [TESTING_GUIDE_EN.md](./TESTING_GUIDE_EN.md) for the English version of this guide.

## 目录结构

测试文件应放在 `tests/dataset/` 目录下，命名规范为 `test_<dataset_name>.py`。

```
tests/
  dataset/
    test_<dataset_name>.py
    test_humaneval.py  # 示例
    test_beyondaime.py  # 示例
```

## 测试覆盖范围

数据集单元测试应覆盖以下方面：

### 1. Dataset Loader 测试

测试数据集的 `load` 静态方法，确保：
- 能够正确加载数据
- 数据格式符合预期
- 列名和数据结构正确
- 处理边界情况（空数据、异常路径等）

**示例测试点：**
- 测试 `load` 方法返回正确的 `Dataset` 或 `DatasetDict`
- 测试数据集的列名是否符合预期
- 测试数据集的样本数量
- 测试数据转换逻辑（如列重命名、数据清洗等）

### 2. Dataset Reader 测试

测试 `DatasetReader` 的初始化和使用：
- 测试 `input_columns` 和 `output_column` 配置
- 测试 reader 能够正确读取数据集
- 测试 train/test split 的处理

### 3. Postprocessor 测试（如果存在）

如果数据集有 `postprocess` 函数，应测试：
- 各种输入格式的处理
- 边界情况（空字符串、特殊字符等）
- 输出格式的正确性

参考 `test_humaneval.py` 中的 postprocessor 测试示例。

### 4. Evaluator 测试（如果存在自定义 Evaluator）

如果数据集有自定义的 `Evaluator`，应测试：
- `score` 方法的正确性
- 预测和参考值的匹配逻辑
- 边界情况处理
- **评估结果结构验证**：验证结果字典包含所有必需字段
- **指标计算**：验证准确率、精确率、召回率、F1 等指标计算正确
- **结果一致性**：验证结果中的统计信息一致（如准确率与正确数/总数匹配）
- **详细信息完整性**：验证 details 数组与样本数量匹配

对于使用 `CascadeEvaluator` 的数据集（如 AIME2025），应测试：
- Rule-based evaluator 结果（如 `MATHVerifyEvaluator`）
- LLM evaluator 结果（如 `GenericLLMEvaluator`）
- Cascade 模式 vs parallel 模式的行为
- 最终正确性判断逻辑
- 所有 cascade 统计信息（rule_accuracy、llm_accuracy、final_accuracy 等）

### 5. 集成测试

测试数据集与 reader、inferencer、evaluator 的集成：
- 测试完整的数据流（load -> read -> infer -> eval）
- 测试配置文件的正确性

## 测试编写规范

### 1. 使用 unittest 框架

```python
import unittest
from datasets import Dataset
from opencompass.datasets import YourDataset

class TestYourDataset(unittest.TestCase):
    def test_load(self):
        # 测试代码
        pass
```

### 2. 使用 Mock 处理外部依赖

对于需要从 HuggingFace 或其他外部源加载数据的数据集，使用 `unittest.mock` 来模拟：

```python
from unittest.mock import patch, MagicMock

@patch('datasets.load_dataset')
def test_load_with_mock(self, mock_load_dataset):
    # 创建模拟数据
    mock_dataset = MagicMock()
    mock_dataset.rename_column.return_value = mock_dataset
    mock_load_dataset.return_value = mock_dataset
    
    # 测试
    result = YourDataset.load(path='test_path')
    mock_load_dataset.assert_called_once()
```

### 3. 使用临时文件测试文件读取

对于从本地文件读取的数据集，使用临时文件：

```python
import tempfile
import json

def test_load_from_file(self):
    with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.jsonl') as f:
        # 写入测试数据
        json.dump({'test': 'data'}, f)
        f.write('\n')
        f.flush()
        
        # 测试加载
        result = YourDataset.load(path=f.name)
        # 断言
```

### 4. 测试数据验证

确保测试数据符合预期格式：

```python
def test_load_returns_dataset(self):
    dataset = YourDataset.load(path='test_path')
    self.assertIsInstance(dataset, Dataset)
    self.assertGreater(len(dataset), 0)
    self.assertIn('expected_column', dataset.column_names)
```

### 5. 测试异常处理

测试错误情况的处理：

```python
def test_load_invalid_path(self):
    with self.assertRaises(FileNotFoundError):
        YourDataset.load(path='non_existent_path')
```

## 测试示例

### 简单数据集测试（如 BeyondAIME）

```python
import unittest
from unittest.mock import patch, MagicMock
from datasets import Dataset
from opencompass.datasets import BeyondAIMEDataset

class TestBeyondAIMEDataset(unittest.TestCase):
    
    @patch('datasets.load_dataset')
    def test_load_renames_column(self, mock_load_dataset):
        # 创建模拟数据集
        mock_dataset = MagicMock()
        mock_dataset.rename_column.return_value = mock_dataset
        mock_load_dataset.return_value = mock_dataset
        
        # 执行加载
        result = BeyondAIMEDataset.load(path='test_path')
        
        # 验证
        mock_load_dataset.assert_called_once_with(path='test_path', split='test')
        mock_dataset.rename_column.assert_called_once_with('problem', 'question')
        self.assertEqual(result, mock_dataset)
    
    @patch('datasets.load_dataset')
    def test_load_returns_dataset(self, mock_load_dataset):
        from datasets import Dataset
        # 创建真实的 Dataset 对象用于测试
        test_data = [{'problem': 'test question', 'answer': 'test answer'}]
        mock_dataset = Dataset.from_list(test_data)
        mock_load_dataset.return_value = mock_dataset
        
        result = BeyondAIMEDataset.load(path='test_path')
        
        self.assertIsInstance(result, Dataset)
        self.assertIn('question', result.column_names)
        self.assertNotIn('problem', result.column_names)
```

### JSONL 文件数据集测试（如 AIME2025）

对于使用 `CustomDataset` 从 JSONL 文件加载的数据集：

```python
import json
import tempfile
import unittest
from unittest.mock import patch
from datasets import Dataset
from opencompass.datasets.custom import CustomDataset

class TestAime2025Dataset(unittest.TestCase):
    
    def setUp(self):
        """设置测试数据。"""
        self.test_data = [
            {
                'question': '计算 $\\sqrt{16}$ 的值',
                'answer': '4',
                'origin_prompt': '计算 $\\sqrt{16}$ 的值',
                'gold_answer': '4'
            }
        ]
    
    def _create_temp_jsonl_file(self, data):
        """创建包含测试数据的临时 JSONL 文件。"""
        temp_file = tempfile.NamedTemporaryFile(
            mode='w', suffix='.jsonl', delete=False, encoding='utf-8'
        )
        for item in data:
            temp_file.write(json.dumps(item, ensure_ascii=False) + '\n')
        temp_file.close()
        return temp_file.name
    
    @patch('opencompass.utils.get_data_path')
    def test_load_reads_jsonl_file(self, mock_get_data_path):
        """测试 load 方法正确读取 JSONL 文件。"""
        # 创建临时 JSONL 文件
        temp_file = self._create_temp_jsonl_file(self.test_data)
        mock_get_data_path.return_value = temp_file
        
        try:
            # 执行加载
            result = CustomDataset.load(path='test_path.jsonl')
            
            # 验证数据集加载正确
            self.assertIsInstance(result, Dataset)
            self.assertEqual(len(result), 1)
            self.assertIn('question', result.column_names)
            self.assertIn('answer', result.column_names)
            self.assertEqual(result[0]['question'], self.test_data[0]['question'])
        finally:
            import os
            if os.path.exists(temp_file):
                os.unlink(temp_file)
    
    @patch('opencompass.utils.get_data_path')
    def test_dataset_initialization(self, mock_get_data_path):
        """测试数据集可以使用正确的配置进行初始化。"""
        temp_file = self._create_temp_jsonl_file(self.test_data)
        mock_get_data_path.return_value = temp_file
        
        try:
            dataset = CustomDataset(
                path='test_path.jsonl',
                abbr='aime2025_test',
                reader_cfg=dict(input_columns=['question'], output_column='answer')
            )
            
            # 验证数据集已创建
            self.assertIsNotNone(dataset.dataset)
            self.assertIsNotNone(dataset.reader)
            self.assertEqual(dataset.reader.input_columns, ['question'])
            self.assertEqual(dataset.reader.output_column, 'answer')
        finally:
            import os
            if os.path.exists(temp_file):
                os.unlink(temp_file)
```

### 带 Postprocessor 的数据集测试（如 HumanEval）

参考 `test_humaneval.py`，测试各种输入格式的处理。

### 带自定义 Evaluator 的数据集测试

#### 简单 Evaluator 测试

```python
import unittest
from opencompass.openicl.icl_evaluator import BaseEvaluator
from opencompass.datasets import YourDataset, YourEvaluator

class TestYourEvaluator(unittest.TestCase):
    
    def test_score_correct_prediction(self):
        evaluator = YourEvaluator()
        predictions = ['A']
        references = ['A']
        result = evaluator.score(predictions, references)
        self.assertEqual(result['score'], 1.0)
    
    def test_score_incorrect_prediction(self):
        evaluator = YourEvaluator()
        predictions = ['A']
        references = ['B']
        result = evaluator.score(predictions, references)
        self.assertEqual(result['score'], 0.0)
```

#### Cascade Evaluator 测试（如 AIME2025）

对于使用 `CascadeEvaluator` 的数据集，测试评估结果结构和指标：

```python
import unittest

class TestAime2025EvalResultValidation(unittest.TestCase):
    """AIME2025 评估结果验证测试用例。"""
    
    def test_result_structure(self):
        """测试评估结果具有正确的结构。"""
        result = {
            'accuracy': 80.0,
            'cascade_stats': {
                'total_samples': 5,
                'rule_correct': 4,
                'rule_accuracy': 80.0,
                'llm_evaluated': 1,
                'llm_correct': 0,
                'llm_accuracy': 0.0,
                'final_correct': 4,
                'final_accuracy': 80.0,
                'parallel_mode': False,
            },
            'details': [{'rule_evaluation': {'correct': True}}] * 5
        }
        
        # 验证结构
        self.assertIn('accuracy', result)
        self.assertIn('cascade_stats', result)
        self.assertIn('details', result)
        
        # 验证 cascade_stats
        stats = result['cascade_stats']
        required_keys = [
            'total_samples', 'rule_correct', 'rule_accuracy',
            'llm_evaluated', 'llm_correct', 'llm_accuracy',
            'final_correct', 'final_accuracy', 'parallel_mode'
        ]
        for key in required_keys:
            self.assertIn(key, stats, f"缺少键: {key}")
    
    def test_accuracy_calculation(self):
        """测试准确率计算是否正确。"""
        result = {
            'accuracy': 80.0,
            'cascade_stats': {
                'total_samples': 5,
                'final_correct': 4,
                'final_accuracy': 80.0,
            }
        }
        
        expected_accuracy = (4 / 5) * 100
        self.assertAlmostEqual(result['accuracy'], expected_accuracy, places=1)
        self.assertAlmostEqual(
            result['cascade_stats']['final_accuracy'],
            expected_accuracy,
            places=1
        )
    
    def test_cascade_mode_result(self):
        """测试 cascade 模式评估结果。"""
        result = {
            'accuracy': 100.0,
            'cascade_stats': {
                'total_samples': 3,
                'rule_correct': 2,
                'rule_accuracy': 66.67,
                'llm_evaluated': 1,  # 仅失败的样本
                'llm_correct': 1,
                'llm_accuracy': 100.0,
                'final_correct': 3,
                'final_accuracy': 100.0,
                'parallel_mode': False,
            },
            'details': [
                {
                    'rule_evaluation': {'correct': True},
                    'llm_evaluation': None,
                    'cascade_correct': True
                },
                {
                    'rule_evaluation': {'correct': True},
                    'llm_evaluation': None,
                    'cascade_correct': True
                },
                {
                    'rule_evaluation': {'correct': False},
                    'llm_evaluation': {'llm_correct': True},
                    'cascade_correct': True
                }
            ]
        }
        
        # 验证 cascade 逻辑
        self.assertEqual(result['cascade_stats']['parallel_mode'], False)
        self.assertEqual(result['cascade_stats']['llm_evaluated'], 1)
        
        # 检查 cascade_correct 逻辑
        for detail in result['details']:
            rule_correct = detail['rule_evaluation'].get('correct', False)
            llm_correct = detail.get('llm_evaluation', {}).get('llm_correct', False) if detail.get('llm_evaluation') else False
            cascade_correct = detail['cascade_correct']
            self.assertEqual(cascade_correct, rule_correct or llm_correct)
    
    def test_result_statistics_consistency(self):
        """测试结果中的统计信息是否一致。"""
        result = {
            'accuracy': 60.0,
            'cascade_stats': {
                'total_samples': 5,
                'rule_correct': 3,
                'rule_accuracy': 60.0,
                'llm_evaluated': 2,
                'llm_correct': 0,
                'llm_accuracy': 0.0,
                'final_correct': 3,
                'final_accuracy': 60.0,
                'parallel_mode': False,
            },
            'details': [{'rule_evaluation': {'correct': i < 3}} for i in range(5)]
        }
        
        stats = result['cascade_stats']
        
        # 检查准确率计算
        self.assertAlmostEqual(
            stats['rule_accuracy'],
            (stats['rule_correct'] / stats['total_samples']) * 100,
            places=1
        )
        
        if stats['llm_evaluated'] > 0:
            self.assertAlmostEqual(
                stats['llm_accuracy'],
                (stats['llm_correct'] / stats['llm_evaluated']) * 100,
                places=1
            )
        
        # 检查 final_accuracy 与顶层 accuracy 匹配
        self.assertAlmostEqual(
            result['accuracy'],
            stats['final_accuracy'],
            places=1
        )
    
    def test_details_count_matches_total_samples(self):
        """测试 details 数量与 total_samples 匹配。"""
        total_samples = 10
        result = {
            'cascade_stats': {'total_samples': total_samples},
            'details': [{}] * total_samples
        }
        
        self.assertEqual(len(result['details']), result['cascade_stats']['total_samples'])
    
    def test_result_metrics_completeness(self):
        """测试结果中所有必需的指标都存在。"""
        required_metrics = [
            'accuracy',
            'cascade_stats.total_samples',
            'cascade_stats.rule_correct',
            'cascade_stats.rule_accuracy',
            'cascade_stats.llm_evaluated',
            'cascade_stats.llm_correct',
            'cascade_stats.llm_accuracy',
            'cascade_stats.final_correct',
            'cascade_stats.final_accuracy',
            'cascade_stats.parallel_mode',
            'details',
        ]
        
        result = {
            'accuracy': 80.0,
            'cascade_stats': {
                'total_samples': 5,
                'rule_correct': 4,
                'rule_accuracy': 80.0,
                'llm_evaluated': 1,
                'llm_correct': 0,
                'llm_accuracy': 0.0,
                'final_correct': 4,
                'final_accuracy': 80.0,
                'parallel_mode': False,
            },
            'details': [{}] * 5
        }
        
        # 检查顶层 accuracy
        self.assertIn('accuracy', result)
        self.assertIsInstance(result['accuracy'], (int, float))
        
        # 检查 cascade_stats
        self.assertIn('cascade_stats', result)
        stats = result['cascade_stats']
        for key in ['total_samples', 'rule_correct', 'rule_accuracy',
                   'llm_evaluated', 'llm_correct', 'llm_accuracy',
                   'final_correct', 'final_accuracy', 'parallel_mode']:
            self.assertIn(key, stats, f"缺少指标: {key}")
        
        # 检查 details
        self.assertIn('details', result)
        self.assertIsInstance(result['details'], list)
```

## 运行测试

### 前置要求

运行测试前，确保已安装所有必要的依赖：

```bash
pip install -r requirements.txt
```

某些测试可能需要额外的依赖，如 `evaluate`、`transformers` 等。

### 运行单个测试文件

```bash
python -m pytest tests/dataset/test_<dataset_name>.py -v
```

### 运行所有数据集测试

```bash
python -m pytest tests/dataset/ -v
```

### 运行特定测试方法

```bash
python -m pytest tests/dataset/test_<dataset_name>.py::TestYourDataset::test_load -v
```

### 使用真实数据集文件进行测试

可以通过设置 `COMPASS_DATA_CACHE` 环境变量来使用真实的数据集文件进行测试：

```bash
export COMPASS_DATA_CACHE=/path/to/data/cache
python -m pytest tests/dataset/test_aime2025.py::TestAime2025Dataset::test_load_with_real_data -v
```

如果 `COMPASS_DATA_CACHE` 未设置或数据集文件不存在，测试将自动跳过。

使用真实数据的测试示例：

```python
def test_load_with_real_data(self):
    """使用 COMPASS_DATA_CACHE 中的真实数据测试数据集加载。"""
    import pytest
    
    # 检查是否设置了 COMPASS_DATA_CACHE
    cache_dir = os.environ.get('COMPASS_DATA_CACHE', '')
    if not cache_dir:
        pytest.skip("COMPASS_DATA_CACHE 未设置，跳过真实数据测试")
    
    try:
        # 加载真实数据集
        dataset = CustomDataset.load(path='opencompass/aime2025')
        
        # 验证数据集已加载
        self.assertIsInstance(dataset, Dataset)
        self.assertGreater(len(dataset), 0)
        self.assertIn('question', dataset.column_names)
    except (FileNotFoundError, ValueError) as e:
        pytest.skip(f"真实数据集未找到: {e}")
```

### 处理导入错误

如果遇到导入错误（如 `ModuleNotFoundError: No module named 'evaluate'`），可以：

1. 安装缺失的依赖
2. 使用 mock 来避免导入完整的 opencompass 包
3. 在测试中使用 `pytest.skip()` 来跳过需要特定依赖的测试

## 最佳实践

1. **保持测试独立**：每个测试应该独立运行，不依赖其他测试的状态
2. **使用描述性的测试名称**：测试方法名应该清楚描述测试的内容
3. **测试边界情况**：包括空数据、无效输入、异常情况等
4. **使用 fixtures**：对于重复的测试数据，使用 fixtures 或 setUp 方法
5. **Mock 外部依赖**：避免在测试中实际下载或访问外部资源
6. **保持测试快速**：单元测试应该快速执行，避免长时间运行的操作
7. **测试覆盖率**：尽量覆盖所有代码路径，包括错误处理

## 注意事项

1. **数据路径**：测试时使用 mock 或临时文件，避免依赖实际的数据文件路径
2. **环境变量**：注意 `DATASET_SOURCE` 等环境变量可能影响数据集加载逻辑
3. **依赖项**：确保测试环境安装了所有必要的依赖
4. **版本兼容性**：测试应该在不同版本的依赖库下都能通过

## 参考示例

- `tests/dataset/test_humaneval.py` - Postprocessor 测试示例
- `tests/dataset/test_local_datasets.py` - 数据集加载测试示例
- `tests/dataset/test_beyondaime.py` - 简单数据集测试示例（HuggingFace 数据集）
- `tests/dataset/test_aime2025.py` - JSONL 文件数据集测试示例（CustomDataset）
- `tests/dataset/test_aime2025_eval.py` - 评估器结果验证测试示例（CascadeEvaluator）