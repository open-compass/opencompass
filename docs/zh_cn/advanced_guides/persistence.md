# 评测结果持久化

## 介绍

通常情况下，OpenCompass的评测结果将会保存到工作目录下。 但在某些情况下，可能会产生用户间的数据共享，以及快速查看已有的公共评测结果等需求。 因此，我们提供了一个能够将评测结果快速转存到外部公共数据站的接口，并且在此基础上提供了对数据站的上传、更新、读取等功能。

## 快速开始

### 向数据站存储数据

通过在CLI评测指令中添加`args`或在Eval脚本中添加配置，即可将本次评测结果存储到您所指定的路径，示例如下：

（方式1）在指令中添加`args`选项并指定你的公共路径地址。

```bash
opencompass  ...  -sp '/your_path'
```

（方式2）在Eval脚本中添加配置。

```pythonE
station_path = '/your_path'
```

### 向数据站更新数据

上述存储方法在上传数据前会首先根据模型和数据集配置中的`abbr`属性来判断数据站中是否已有相同任务结果。若已有结果，则取消本次存储。如果您需要更新这部分结果，请在指令中添加`station-overwrite`选项，示例如下：

```bash
opencompass  ...  -sp '/your_path' --station-overwrite
```

### 读取数据站中已有的结果

您可以直接从数据站中读取已有的结果，以避免重复进行评测任务。读取到的结果会直接参与到`summarize`步骤。采用该配置时，仅有数据站中未存储结果的任务会被启动。示例如下：

```bash
opencompass  ...  -sp '/your_path' --read-from-station
```

### 指令组合

1. 仅向数据站上传最新工作目录下结果，不补充运行缺失结果的任务：

```bash
opencompass  ...  -sp '/your_path' -r latest -m viz
```

## 数据站存储格式

在数据站中，评测结果按照每个`model-dataset`对的结果存储为`json`文件。具体的目录组织形式为`/your_path/dataset_name/model_name.json`。每个`json`文件都存储了对应结果的字典，包括`predictions`、`results`以及`cfg`三个子项，具体示例如下：

```pythonE
Result = {
    'predictions': List[Dict],
    'results': Dict,
    'cfg': Dict = {
        'models': Dict,
        'datasets': Dict,
        (Only subjective datasets)'judge_models': Dict
    }
}
```

其中，`predictions`记录了模型对数据集中每一条数据的prediction的结果，`results`记录了模型在该数据集上的评分，`cfg`记录了该评测任务中模型和数据集的详细配置。
