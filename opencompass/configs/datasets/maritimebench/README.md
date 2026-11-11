## 📘 About MaritimeBench

**MaritimeBench** 是航运行业首个基于“学科（一级）- 子学科（二级）- 具体考点（三级）”分类体系构建的专业知识评测集。该数据集包含 **1888 道客观选择题**，覆盖以下核心领域：

- 航海
- 轮机
- 电子电气员
- GMDSS（全球海上遇险与安全系统）
- 船员培训

评测内容涵盖理论知识、操作技能及行业规范，旨在：

- 提升 AI 模型在航运领域的 **理解与推理能力**
- 确保其在关键知识点上的 **准确性与适应性**
- 支持航运专业考试、船员培训及资质认证的 **自动化测评**
- 优化船舶管理、导航操作、海上通信等场景下的 **智能问答与决策系统**

MaritimeBench 基于行业权威标准，构建了 **系统、科学的知识评测体系**，全面衡量模型在航运各专业领域的表现，助力其专业化发展。

---

## 🧪 示例

请回答单选题。要求只输出选项，不输出解释，将选项放在 `< >` 内，直接输出答案。  

**题目 1：**  
在船舶主推进动力装置中，传动轴系在运转中承受以下复杂的应力和负荷，但不包括______。  
选项：  
A. 电磁力  
B. 压拉应力  
C. 弯曲应力  
D. 扭应力  
**答：** `<A>`

**题目 2：**  
当船舶实行 PMS 检验时，应将 CCS 现行规范中规定的特别检验纳入在 PMS 计划表中，下列应包括______。  
① 每年应进行的确认性检查项目  
② 每年应进行的拆检项目  
③ 5 年内应拆检的项目  
④ 5 年内应进行的确认性检查项目  
选项：  
A. ①④  
B. ②④  
C. ①③  
D. ①②③④  
**答：** `<C>`

---

## 📂 Dataset Links

- [MaritimeBench on Hugging Face](https://huggingface.co/datasets/Hi-Dolphin/MaritimeBench)
- [MaritimeBench on ModelScope](https://modelscope.cn/datasets/HiDolphin/MaritimeBench/summary)

---

## 📊 模型测试结果

| dataset | version | metric | mode | Qwen2.5-32B |
|----- | ----- | ----- | ----- | -----|
| maritimebench | 6d56ec | accuracy | gen | 72.99 |
