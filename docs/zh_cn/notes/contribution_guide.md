# 为 OpenCompass 做贡献

- [为OpenCompass做贡献](#为opencompass做贡献)
  - [工作流程](#工作流程)
  - [代码风格](#代码风格)
    - [Python](#python)
  - [预提交钩子 (Pre-commit Hook)](#预提交钩子-pre-commit-hook)

感谢你对于OpenCompass的贡献！我们欢迎各种形式的贡献，包括但不限于以下几点。

- 修改错别字或修复bug
- 添加文档或将文档翻译成其它语言
- 添加新功能和组件

## 工作流程

我们建议潜在的贡献者遵循以下的贡献工作流程。

1. Fork并拉取最新的OpenCompass仓库，按照[开始使用](https://OpenCompass.readthedocs.io/en/latest/get_started.html)来设置环境。
2. 检出一个新的分支（**不要使用master或dev分支来创建PR**）

```bash
git checkout -b xxxx # xxxx 是新分支的名称
```

3. 编辑相关文件，并且遵循下面提到的代码风格
4. 使用[预提交钩子](https://pre-commit.com/)来检查和格式化你的更改。
5. 提交你的更改
6. 创建一个带有相关信息的PR

## 代码风格

### Python

我们采用[PEP8](https://www.python.org/dev/peps/pep-0008/)作为首选的代码风格。

我们使用以下工具进行linting和格式化：

- [flake8](https://github.com/PyCQA/flake8): 一个围绕一些linter工具的封装器。
- [isort](https://github.com/timothycrosley/isort): 一个用于排序Python导入的实用程序。
- [yapf](https://github.com/google/yapf): 一个Python文件的格式化器。
- [codespell](https://github.com/codespell-project/codespell): 一个Python实用程序，用于修复文本文件中常见的拼写错误。
- [mdformat](https://github.com/executablebooks/mdformat): mdformat是一个有明确定义的Markdown格式化程序，可以用来在Markdown文件中强制执行一致的样式。
- [docformatter](https://github.com/myint/docformatter): 一个格式化docstring的工具。

yapf和isort的样式配置可以在[setup.cfg](https://github.com/OpenCompass/blob/main/setup.cfg)中找到。

## 预提交钩子 (Pre-commit Hook)

我们使用[预提交钩子](https://pre-commit.com/)用于在每次提交时自动检查与格式化`flake8`、`yapf`、`isort`、`trailing whitespaces`、`markdown files`，
修复`end-of-files`、`double-quoted-strings`、`python-encoding-pragma`、`mixed-line-ending`，并自动排序`requirements.txt`。预提交钩子的配置存储在[.pre-commit-config](<>)中。

在你克隆仓库后，你需要安装并初始化预提交钩子。

```shell
pip install -U pre-commit
```

从仓库文件夹运行

```shell
pre-commit install
```

之后，在每次提交时都会强制执行代码 linters 和格式化器。

> 在你创建PR前，确保你的代码通过了 lint 检查并被 yapf 格式化。
