import matplotlib
matplotlib.use('Agg')  # 使用非交互式后端，避免GUI显示
import matplotlib.pyplot as plt
import pandas as pd
import json
import os
import glob
from matplotlib import rcParams
import numpy as np

# 设置字体更大
rcParams.update({'font.size': 20})

def load_fine_threshold_evaluation_data():
    """
    从fine_threshold_evaluation_summary.json文件中加载阈值评估数据
    """
    base_dir = os.getcwd()
    eval_file = os.path.join(base_dir, 'fine_threshold_evaluation_summary.json')
    
    if not os.path.exists(eval_file):
        print(f"错误: 找不到文件 {eval_file}")
        return None

def load_result_curves(result_dir='result'):
    """
    从指定的 result 目录读取所有 JSON 文件，提取每个模型的 Sensitivity vs FP Rate 曲线。
    返回列表：[{'name': 模型名, 'fp': [...], 'sens': [...]}]
    """
    base_dir = os.getcwd()
    dir_path = os.path.join(base_dir, result_dir)
    if not os.path.isdir(dir_path):
        print(f"错误: 找不到结果目录 {dir_path}")
        return []

    curves = []
    for json_path in glob.glob(os.path.join(dir_path, '*.json')):
        name = os.path.splitext(os.path.basename(json_path))[0]
        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)

            fp_list = []
            sens_list = []
            if 'all_thresholds' in data:
                for _, threshold_info in data['all_thresholds'].items():
                    if ('full_result' in threshold_info and
                        'event_results' in threshold_info['full_result']):
                        er = threshold_info['full_result']['event_results']
                        sens = er.get('sensitivity', None)
                        fp = er.get('fpRate', None)
                        # 跳过NaN或None
                        if sens is None or fp is None:
                            continue
                        if sens != sens or fp != fp:  # NaN 检查
                            continue
                        sens_list.append(float(sens))
                        fp_list.append(float(fp))

            # 若收集到数据则排序按FP升序
            if fp_list and sens_list:
                pairs = sorted(zip(fp_list, sens_list), key=lambda x: x[0])
                fp_sorted, sens_sorted = zip(*pairs)
                curves.append({'name': name, 'fp': list(fp_sorted), 'sens': list(sens_sorted)})
        except Exception as e:
            print(f"读取 {json_path} 失败: {e}")
            continue

    print(f"从 {dir_path} 加载到 {len(curves)} 条模型曲线")
    return curves
    
    try:
        with open(eval_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # 提取阈值数据
        threshold_data = []
        if 'all_thresholds' in data:
            for threshold_key, threshold_info in data['all_thresholds'].items():
                threshold_val = float(threshold_key)
                
                # 检查是否有有效的event_results数据
                if 'full_result' in threshold_info and 'event_results' in threshold_info['full_result']:
                    event_results = threshold_info['full_result']['event_results']
                    
                    # 跳过NaN值
                    if (event_results['sensitivity'] != event_results['sensitivity'] or  # NaN检查
                        event_results['precision'] != event_results['precision'] or
                        event_results['f1'] != event_results['f1']):
                        continue
                    
                    threshold_data.append({
                        'threshold': threshold_val,
                        'sensitivity': event_results['sensitivity'],
                        'precision': event_results['precision'],
                        'f1': event_results['f1'],
                        'fpRate': event_results['fpRate']
                    })
        
        # 按阈值排序
        threshold_data.sort(key=lambda x: x['threshold'])
        
        print(f"成功读取fine_threshold_evaluation_summary.json: {len(threshold_data)} 个有效阈值点")
        return threshold_data
        
    except Exception as e:
        print(f"读取文件 {eval_file} 时出错: {e}")
        return None

def plot_single_threshold_comparison(threshold_data):
    """
    绘制单个模型的阈值比较图
    """
    if not threshold_data:
        print("没有有效的阈值数据")
        return
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 10))
    
    # 提取数据
    thresholds = [d['threshold'] for d in threshold_data]
    f1_scores = [d['f1'] for d in threshold_data]
    sensitivities = [d['sensitivity'] for d in threshold_data]
    fp_rates = [d['fpRate'] for d in threshold_data]
    
    # 第一个子图：F1 Score vs Threshold
    ax1.plot(thresholds, f1_scores, marker='o', label='F1 Score', 
             linewidth=2, color='#1f77b4', markersize=6)
    
    ax1.set_xlabel('Threshold', fontsize=20)
    ax1.set_ylabel('F1 Score', fontsize=20)
    ax1.set_title('F1 Score vs Threshold', fontsize=20)
    ax1.grid(True)
    ax1.tick_params(labelsize=20)
    ax1.legend(loc='upper right', fontsize=14, frameon=True, fancybox=True, shadow=True)
    
    # 第二个子图：Sensitivity vs FP Rate (ROC风格)
    ax2.plot(fp_rates, sensitivities, marker='o', label='Sensitivity vs FP Rate', 
             linewidth=2, color='#ff7f0e', markersize=6)
    
    ax2.set_xlabel('FP / 24h', fontsize=20)
    ax2.set_ylabel('Sensitivity', fontsize=20)
    ax2.set_title('Sensitivity vs FP Rate', fontsize=20)
    ax2.grid(True)
    ax2.tick_params(labelsize=20)
    ax2.legend(loc='upper right', fontsize=14, frameon=True, fancybox=True, shadow=True)
    
    plt.tight_layout()
    plt.savefig('fine_threshold_comparison.png', dpi=300, bbox_inches='tight')
    print("图表已保存为 fine_threshold_comparison.png")

def plot_single_roc_style_comparison(threshold_data):
    """
    绘制单个模型的ROC风格的Sensitivity vs FP Rate比较图，并添加其他方法的散点标识
    """
    if not threshold_data:
        print("没有有效的阈值数据")
        return
    
    fig, ax = plt.subplots(figsize=(14, 10))
    
    # 提取数据
    sensitivities = [d['sensitivity'] for d in threshold_data]
    fp_rates = [d['fpRate'] for d in threshold_data]
    
    # 绘制我们的方法曲线
    ax.plot(fp_rates, sensitivities, marker='o', label='Fine Threshold Evaluation', 
           linewidth=2, color='#1f77b4', markersize=6)
    
    # 其他方法的数据点 (只保留FP <= 100的方法)
    comparison_methods = [
        {'name': 'EventNet 2024', 'sensitivity': 59/100, 'fp': 9, 'color': '#ff7f0e', 'marker': 's'},
        {'name': 'Zhu Transformer 2023', 'sensitivity': 67/100, 'fp': 16, 'color': '#2ca02c', 'marker': '^'},
        {'name': 'DINO-EEG 200s (Ours)', 'sensitivity': 46.8/100, 'fp': 7.48, 'color': '#d62728', 'marker': 'D'},
        {'name': 'SeizUnet', 'sensitivity': 29/100, 'fp': 4, 'color': '#9467bd', 'marker': 'v'},
        {'name': 'DynaSD', 'sensitivity': 66/100, 'fp': 80, 'color': '#e377c2', 'marker': 'h'},
        {'name': 'He Random-Forest', 'sensitivity': 17/100, 'fp': 9, 'color': '#7f7f7f', 'marker': '*'}
    ]
    
    # 绘制其他方法的散点
    for method in comparison_methods:
        ax.scatter(method['fp'], method['sensitivity'], 
                  color=method['color'], marker=method['marker'], 
                  s=100, label=method['name'], alpha=0.8, edgecolors='black', linewidth=1)
    
    ax.set_xlabel("FP / 24h", fontsize=20)
    ax.set_ylabel("Sensitivity", fontsize=20)
    ax.set_title("Sensitivity vs FP Rate Comparison", fontsize=20)
    ax.grid(True, alpha=0.3)
    ax.tick_params(labelsize=18)
    ax.set_ylim(0.0, 1.05)
    ax.set_xlim(0, 100)
    
    # 添加图例 - 分成两列显示
    ax.legend(loc='center right', fontsize=12, frameon=True, fancybox=True, shadow=True, 
              ncol=1, bbox_to_anchor=(1.25, 0.5))
    
    plt.tight_layout()
    plt.savefig('fine_sensitivity_vs_fprate_with_comparison.png', dpi=300, bbox_inches='tight')
    print("ROC风格比较图表已保存为 fine_sensitivity_vs_fprate_with_comparison.png")

def plot_results_curves_with_scatter(result_curves):
    """
    绘制 result 目录中所有模型的 Sensitivity vs FP Rate 曲线，并叠加用户提供的散点。
    """
    if not result_curves:
        print("没有找到可绘制的结果曲线")
        return

    fig, ax = plt.subplots(figsize=(14, 10))

    # 颜色循环
    color_cycle = plt.rcParams['axes.prop_cycle'].by_key().get('color', ['#1f77b4','#ff7f0e','#2ca02c','#d62728','#9467bd','#8c564b','#e377c2','#7f7f7f','#bcbd22','#17becf'])

    max_fp_val = 0.0

    # 绘制每个文件的曲线
    for idx, curve in enumerate(result_curves):
        color = color_cycle[idx % len(color_cycle)]
        ax.plot(curve['fp'], curve['sens'], label=curve['name'], linewidth=2, color=color)
        if curve['fp']:
            max_fp_val = max(max_fp_val, max(curve['fp']))

    # 用户提供的散点（Sensitivity按百分比提供，这里转换为0-1）
    scatter_points = [
        {'name': 'Wu Transformer', 'sensitivity': 57/100, 'fp': 19, 'color': '#444444', 'marker': 's'},
        {'name': 'DeepSOZ-HEM', 'sensitivity': 62/100, 'fp': 37, 'color': '#9467bd', 'marker': '^'},
        {'name': 'EventNet 2024', 'sensitivity': 54/100, 'fp': 28, 'color': '#ff7f0e', 'marker': 'o'},
        {'name': 'Zhu Transformer 2023', 'sensitivity': 60/100, 'fp': 51, 'color': '#2ca02c', 'marker': 'D'},
        {'name': 'SeizUnet', 'sensitivity': 57/100, 'fp': 50, 'color': '#17becf', 'marker': 'v'},
        {'name': 'Self-GNN - Tang 2021', 'sensitivity': 62/100, 'fp': 106, 'color': '#8c564b', 'marker': 'p'},
        {'name': 'SD2025', 'sensitivity': 70/100, 'fp': 147, 'color': '#e377c2', 'marker': 'h'},
        {'name': 'DynaSD', 'sensitivity': 34/100, 'fp': 49, 'color': '#d62728', 'marker': 'X'},
        {'name': 'Gotman 1982', 'sensitivity': 70/100, 'fp': 249, 'color': '#aec7e8', 'marker': '*'},
        {'name': 'EEGWaveNet 2021', 'sensitivity': 49/100, 'fp': 138, 'color': '#bcbd22', 'marker': '1'},
        {'name': 'Velland_2025 v0.1', 'sensitivity': 60/100, 'fp': 190, 'color': '#7f7f7f', 'marker': '+'},
    ]

    for pt in scatter_points:
        ax.scatter(pt['fp'], pt['sensitivity'], color=pt['color'], marker=pt['marker'], s=100,
                   label=pt['name'], alpha=0.85, edgecolors='black', linewidth=1)
        max_fp_val = max(max_fp_val, pt['fp'])

    ax.set_xlabel('FP / 24h', fontsize=20)
    ax.set_ylabel('Sensitivity', fontsize=20)
    ax.set_title('Result Curves + Methods Scatter (Sensitivity vs FP)', fontsize=20)
    ax.grid(True, alpha=0.3)
    ax.tick_params(labelsize=18)
    ax.set_ylim(0.0, 1.05)
    ax.set_xlim(0, max(100, min(320, max_fp_val * 1.05)))

    # 图例：将曲线与散点一起显示，避免爆炸，采用两列
    ax.legend(loc='center right', fontsize=10, frameon=True, fancybox=True, shadow=True,
              ncol=1, bbox_to_anchor=(1.28, 0.5))

    plt.tight_layout()
    plt.savefig('result_sensitivity_vs_fp_with_scatter.png', dpi=300, bbox_inches='tight')
    print('图表已保存为 result_sensitivity_vs_fp_with_scatter.png')

if __name__ == '__main__':
    # 加载fine_threshold_evaluation_summary.json数据
    threshold_data = load_fine_threshold_evaluation_data()

    if threshold_data is None or len(threshold_data) == 0:
        print("没有找到有效的fine_threshold阈值数据")
    else:
        print(f"\n=== 开始绘制fine_threshold图表 ===")
        # 绘制阈值比较图
        plot_single_threshold_comparison(threshold_data)
        # 绘制ROC风格比较图
        plot_single_roc_style_comparison(threshold_data)
        # 显示统计信息
        print(f"\n=== 阈值评估统计 ===")
        # 找到最佳F1分数及其对应的阈值
        best_f1_data = max(threshold_data, key=lambda x: x['f1'])
        print(f"最佳阈值: {best_f1_data['threshold']:.2f}")
        print(f"最佳F1分数: {best_f1_data['f1']:.4f}")
        print(f"对应Sensitivity: {best_f1_data['sensitivity']:.4f}")
        print(f"对应Precision: {best_f1_data['precision']:.4f}")
        print(f"对应FP Rate: {best_f1_data['fpRate']:.2f}")

    # 无论是否有fine_threshold数据，都绘制result目录的曲线与散点
    result_curves = load_result_curves('result')
    if result_curves:
        print(f"\n=== 开始绘制result曲线+散点图 ===")
        plot_results_curves_with_scatter(result_curves)
    else:
        print("没有找到result目录中的可用曲线数据")

    print(f"\n所有图表已生成完成！")