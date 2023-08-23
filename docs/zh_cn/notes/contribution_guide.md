# 为 OpenCompass 做贡献

- [为 OpenCompass 做贡献](#为-opencompass-做贡献)
  - [什么是拉取请求？](#什么是拉取请求)
  - [基本的工作流：](#基本的工作流)
  - [具体步骤](#具体步骤)
    - [1. 获取最新的代码库](#1-获取最新的代码库)
    - [2. 从 `main` 分支创建一个新的开发分支](#2-从-main-分支创建一个新的开发分支)
    - [3. 提交你的修改](#3-提交你的修改)
    - [4. 推送你的修改到复刻的代码库，并创建一个拉取请求](#4-推送你的修改到复刻的代码库并创建一个拉取请求)
    - [5. 讨论并评审你的代码](#5-讨论并评审你的代码)
    - [6. `拉取请求`合并之后删除该分支](#6-拉取请求合并之后删除该分支)
  - [代码风格](#代码风格)
    - [Python](#python)

感谢你对于OpenCompass的贡献！我们欢迎各种形式的贡献，包括但不限于以下几点。

- 修改错别字或修复bug
- 添加文档或将文档翻译成其它语言
- 添加新功能和组件

## 什么是拉取请求？

`拉取请求` (Pull Request), [GitHub 官方文档](https://docs.github.com/en/github/collaborating-with-pull-requests/proposing-changes-to-your-work-with-pull-requests/about-pull-requests)定义如下。

```
拉取请求是一种通知机制。你修改了他人的代码，将你的修改通知原来作者，希望他合并你的修改。
```

## 基本的工作流：

1. 获取最新的代码库
2. 从最新的 `main` 分支创建分支进行开发
3. 提交修改 ([不要忘记使用 pre-commit hooks!](#3-提交你的修改))
4. 推送你的修改并创建一个 `拉取请求`
5. 讨论、审核代码
6. 将开发分支合并到 `main` 分支

## 具体步骤

### 1. 获取最新的代码库

- 当你第一次提 PR 时

  复刻 OpenCompass 原代码库，点击 GitHub 页面右上角的 **Fork** 按钮即可
  ![avatar](https://github.com/InternLM/opencompass/assets/22607038/851ed33d-02db-49c9-bf94-7c62eee89eb2)

  克隆复刻的代码库到本地

  ```bash
  git clone git@github.com:XXX/opencompass.git
  ```

  添加原代码库为上游代码库

  ```bash
  git remote add upstream git@github.com:InternLM/opencompass.git
  ```

- 从第二个 PR 起

  检出本地代码库的主分支，然后从最新的原代码库的主分支拉取更新。

  ```bash
  git checkout main
  git pull upstream main
  ```

### 2. 从 `main` 分支创建一个新的开发分支

```bash
git checkout main -b branchname
```

### 3. 提交你的修改

- 如果你是第一次尝试贡献，请在 OpenCompass 的目录下安装并初始化 pre-commit hooks。

  ```bash
  pip install -U pre-commit
  pre-commit install
  ```

  ````{tip}
  对于中国地区的用户，由于网络原因，安装 pre-commit hook 可能会失败。可以尝试以下命令切换为国内镜像源：
  ```bash
  pre-commit install -c .pre-commit-config-zh-cn.yaml
  pre-commit run –all-files -c .pre-commit-config-zh-cn.yaml
  ```
  ````

- 提交修改。在每次提交前，pre-commit hooks 都会被触发并规范化你的代码格式。

  ```bash
  # coding
  git add [files]
  git commit -m 'messages'
  ```

  ```{note}
  有时你的文件可能会在提交时被 pre-commit hooks 自动修改。这时请重新添加并提交修改后的文件。
  ```

### 4. 推送你的修改到复刻的代码库，并创建一个拉取请求

- 推送当前分支到远端复刻的代码库

  ```bash
  git push origin branchname
  ```

- 创建一个拉取请求

  ![avatar](https://github.com/InternLM/opencompass/assets/22607038/08feb221-b145-4ea8-8e20-05f143081604)

- 修改拉取请求信息模板，描述修改原因和修改内容。还可以在 PR 描述中，手动关联到相关的议题 (issue),（更多细节，请参考[官方文档](https://docs.github.com/en/issues/tracking-your-work-with-issues/linking-a-pull-request-to-an-issue)）。

- 你同样可以把 PR 关联给相关人员进行评审。

### 5. 讨论并评审你的代码

- 根据评审人员的意见修改代码，并推送修改

### 6. `拉取请求`合并之后删除该分支

- 在 PR 合并之后，你就可以删除该分支了。

  ```bash
  git branch -d branchname # 删除本地分支
  git push origin --delete branchname # 删除远程分支
  ```

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
