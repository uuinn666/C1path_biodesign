# C1path_biodesign

本仓库归档多底物一碳途径研究的正式结果图、结构化补充材料和可复现绘图代码。

## 目录结构

```text
C1path_biodesign/
├── code/                       # 图2–图5绘图代码
├── supplementary_materials/    # 结构化源数据及图5网页截图
├── results/                    # 图1–图5正式PNG/PDF结果
├── requirements.txt            # 已验证的Python依赖版本
└── 笔记.txt                    # 文件来源、用途与删除说明
```

图1为静态研究流程图，目前没有可精确重建其图层结构的绘图脚本。图2–图5均可由仓库内代码和补充材料重新生成。

## 环境

已验证环境使用Python 3.9.21，并安装：

```bash
python -m pip install -r requirements.txt
```

图5标题使用`Droid Sans Bold`字体。运行图5代码的系统需要安装Droid Sans字体，以保证像素级复现。

## 小样本绘图测试

```bash
python code/figure2/06_plot_fig2_pathway_overview.py --smoke --output-stem validation/figure2/Fig2_pathway_overview
python code/figure3/07_plot_fig3_reaction_analysis.py --smoke --output-stem validation/figure3/Fig3_reaction_analysis
python code/figure4/08_plot_fig4_product_landscape.py --smoke --output-stem validation/figure4/Fig4_product_landscape
python code/figure5/10_prepare_static_figures.py --smoke
```

## 正式图复现

```bash
python code/figure2/06_plot_fig2_pathway_overview.py
python code/figure3/07_plot_fig3_reaction_analysis.py
python code/figure4/08_plot_fig4_product_landscape.py
python code/figure5/10_prepare_static_figures.py
```

正式输出将写入对应的`results/figure*`目录，运行日志写入`logs/`。图3还会生成版式质量检查JSON。

## 数据说明

- `supplementary_materials/figure2`：图2五个面板的结构化TSV源数据。
- `supplementary_materials/figure3`：路径步数、反应频率、反应映射、组样本量及源数据校验报告。
- `supplementary_materials/figure4`：底物–产物可达性、产品类别覆盖率和路径分布数据。
- `supplementary_materials/figure5/source_panels`：图5六个网页截图面板。

## 许可

当前仓库尚未声明开源许可证。公开可见不等于自动授权他人复制、修改或再分发；如需开放复用，应由研究者另行选择并添加许可证。
