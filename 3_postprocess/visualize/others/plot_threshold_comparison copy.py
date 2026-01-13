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

def load_all_evaluation_data():
    """
    从avgs文件夹中加载所有子文件夹的evaluation_summary.json数据
    """
    all_data = {}
    
    # 获取avgs文件夹路径
    base_dir = os.getcwd()
    avgs_dir = os.path.join(base_dir, 'avgs', 'avgs')
    
    if not os.path.exists(avgs_dir):
        print(f"错误: 找不到文件夹 {avgs_dir}")
        return {}
    
    # 获取所有子文件夹
    subdirs = [d for d in os.listdir(avgs_dir) 
               if os.path.isdir(os.path.join(avgs_dir, d)) and not d.startswith('.')]
    
    print(f"找到 {len(subdirs)} 个子文件夹: {subdirs}")
    
    for subdir in subdirs:
        eval_file = os.path.join(avgs_dir, subdir, 'quick_eval_results', 'evaluation_summary.json')
        
        if os.path.exists(eval_file):
            try:
                with open(eval_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                # 提取阈值数据
                threshold_data = []
                if 'all_thresholds' in data:
                    for threshold_key, threshold_info in data['all_thresholds'].items():
                        threshold_val = float(threshold_key)
                        event_results = threshold_info['full_result']['event_results']
                        
                        threshold_data.append({
                            'threshold': threshold_val,
                            'sensitivity': event_results['sensitivity'],
                            'precision': event_results['precision'],
                            'f1': event_results['f1'],
                            'fpRate': event_results['fpRate']
                        })
                
                # 按阈值排序
                threshold_data.sort(key=lambda x: x['threshold'])
                all_data[subdir] = threshold_data
                
                print(f"成功读取 {subdir}: {len(threshold_data)} 个阈值点")
                
            except Exception as e:
                print(f"读取文件 {eval_file} 时出错: {e}")
        else:
            print(f"警告: 找不到文件 {eval_file}")
    
    return all_data

def plot_threshold_comparison(data):
    """
    绘制阈值比较图
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 10))
    
    # 定义丰富的颜色配置 - 严格参考excel.py
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', 
              '#e377c2', '#7f7f7f', '#bcbd22', '#17becf', '#aec7e8', '#ffbb78',
              '#98df8a', '#ff9896', '#c5b0d5', '#c49c94', '#f7b6d3', '#c7c7c7',
              '#dbdb8d', '#9edae5', '#393b79', '#637939', '#8c6d31', '#843c39']
    
    # 第一个子图：F1 Score vs Threshold
    for i, (model_name, model_data) in enumerate(data.items()):
        color = colors[i % len(colors)]
        ax1.plot(model_data['thresholds'], model_data['f1_scores'], 
                marker='o', label=model_name, linewidth=2, color=color, markersize=6)
    
    ax1.set_xlabel('Threshold', fontsize=20)
    ax1.set_ylabel('F1 Score', fontsize=20)
    ax1.set_title('F1 Score vs Threshold', fontsize=20)
    ax1.grid(True)
    ax1.tick_params(labelsize=20)
    ax1.legend(loc='upper right', fontsize=14, frameon=True, fancybox=True, shadow=True, 
               ncol=2, columnspacing=1.0, handletextpad=0.5, handlelength=1.5)
    
    # 第二个子图：Sensitivity vs FP Rate (ROC风格)
    for i, (model_name, model_data) in enumerate(data.items()):
        color = colors[i % len(colors)]
        ax2.plot(model_data['fp_rates'], model_data['sensitivities'], 
                marker='o', label=model_name, linewidth=2, color=color, markersize=6)
    
    ax2.set_xlabel('FP / 24h', fontsize=20)
    ax2.set_ylabel('Sensitivity', fontsize=20)
    ax2.set_title('Sensitivity vs FP Rate', fontsize=20)
    ax2.grid(True)
    ax2.tick_params(labelsize=20)
    ax2.legend(loc='upper right', fontsize=14, frameon=True, fancybox=True, shadow=True, 
               ncol=2, columnspacing=1.0, handletextpad=0.5, handlelength=1.5)
    
    plt.tight_layout()
    plt.savefig('threshold_comparison_all_models.png', dpi=300, bbox_inches='tight')
    # plt.show()  # 注释掉避免GUI显示
    print("图表已保存为 threshold_comparison_all_models.png")

def plot_roc_style_comparison(data):
    """
    绘制ROC风格的Sensitivity vs FP Rate比较图
    """
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # 定义丰富的颜色配置 - 严格参考excel.py
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', 
              '#e377c2', '#7f7f7f', '#bcbd22', '#17becf', '#aec7e8', '#ffbb78',
              '#98df8a', '#ff9896', '#c5b0d5', '#c49c94', '#f7b6d3', '#c7c7c7',
              '#dbdb8d', '#9edae5', '#393b79', '#637939', '#8c6d31', '#843c39']
    
    for i, (model_name, model_data) in enumerate(data.items()):
        color = colors[i % len(colors)]
        ax.plot(model_data['fp_rates'], model_data['sensitivities'], 
               marker='o', label=model_name, linewidth=2, color=color, markersize=6)
    
    ax.set_xlabel("FP / 24h", fontsize=20)
    ax.set_ylabel("Sensitivity", fontsize=20)
    ax.set_title("Sensitivity vs FP Rate Comparison", fontsize=20)
    ax.grid(True)
    ax.tick_params(labelsize=20)
    ax.set_ylim(0.15, 1.0)  # 提高主图内容整体高度
    
    # 添加图例 - 分成两排显示
    ax.legend(loc='upper right', fontsize=14, frameon=True, fancybox=True, shadow=True, 
              ncol=2, columnspacing=1.0, handletextpad=0.5, handlelength=1.5)
    
    plt.tight_layout()
    plt.savefig('sensitivity_vs_fprate_comparison.png', dpi=300, bbox_inches='tight')
    # plt.show()  # 注释掉避免GUI显示
    print("ROC风格图表已保存为 sensitivity_vs_fprate_comparison.png")

if __name__ == '__main__':
    # 加载AVGS数据
    base_path = r'd:\python\dino_eval\avgs\avgs'
    
    # 使用现有的load_all_evaluation_data函数，但需要修改为适配新的数据结构
    all_data = load_all_evaluation_data()
    
    if not all_data:
        print("没有找到有效的数据")
    else:
        # 转换数据格式以适配新的绘图函数
        data = {}
        for model_name, threshold_list in all_data.items():
            if threshold_list:
                data[model_name] = {
                    'thresholds': [d['threshold'] for d in threshold_list],
                    'f1_scores': [d['f1'] for d in threshold_list],
                    'sensitivities': [d['sensitivity'] for d in threshold_list],
                    'fp_rates': [d['fpRate'] for d in threshold_list],
                    'best_threshold': max(threshold_list, key=lambda x: x['f1'])['threshold'],
                    'best_f1_score': max(threshold_list, key=lambda x: x['f1'])['f1']
                }
        
        print(f"\n=== 开始绘制图表 ===")
        
        # 绘制阈值比较图
        plot_threshold_comparison(data)
        
        # 绘制ROC风格比较图
        plot_roc_style_comparison(data)
        
        # 显示统计信息
        print(f"\n=== 模型比较统计 ===")
        for model_name, model_data in data.items():
            if model_data['thresholds']:
                print(f"\n{model_name}:")
                print(f"  最佳阈值: {model_data['best_threshold']:.1f}")
                print(f"  最佳F1: {model_data['best_f1_score']:.4f}")
                
                # 找到最佳F1对应的其他指标
                best_idx = model_data['f1_scores'].index(max(model_data['f1_scores']))
                print(f"  对应Sensitivity: {model_data['sensitivities'][best_idx]:.4f}")
                print(f"  对应FP Rate: {model_data['fp_rates'][best_idx]:.2f}")
        
        print(f"\n所有图表已生成完成！")