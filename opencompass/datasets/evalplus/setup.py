from setuptools import setup, find_packages

setup(
    name='evalplus',            # 包名 (安装时使用: pip install mypackage)
    version='0.1.0',             # 版本号
    author='default',          # 作者信息
    description='A sample Python package',  # 简短描述
    packages=find_packages(),    # 自动发现子包
    install_requires=[],         # (可选) 依赖项列表
    python_requires='>=3.10',     # (可选) 支持的Python版本
)

