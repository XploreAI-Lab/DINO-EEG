import matplotlib.pyplot as plt
import pandas as pd
import json
import os
import glob
from matplotlib import rcParams
import numpy as np

# 设置字体更大
rcParams.update({'font.size': 14})

def load_tcp_evaluation_data():
    """
    从tcp文件夹中加载所有子文件夹的evaluation_summary.json数据
    """
    all_data = {}
    
    # 获取tcp文件夹路径
    base_dir = os.getcwd()
    tcp_dir = os.path.join(base_dir, 'tcp', 'tcp')
    
    if not os.path.exists(tcp_dir):
        print(f"错误: 找不到文件夹 {tcp_dir}")
        return {}
    
    # 获取所有子文件夹
    subdirs = [d for d in os.listdir(tcp_dir) 
               if os.path.isdir(os.path.join(tcp_dir, d)) and not d.startswith('.') and d != 'Untitled Folder']
    
    print(f"找到 {len(subdirs)} 个子文件夹: {subdirs}")
    
    for subdir in subdirs:
        eval_file = os.path.join(tcp_dir, subdir, 'quick_eval_results', 'evaluation_summary.json')
        
        if os.path.exists(eval_file):
            try:
                with open(eval_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                # 提取阈值数据，只记录fpRate <= 50的结果
                threshold_data = []
                if 'all_thresholds' in data:
                    for threshold_key, threshold_info in data['all_thresholds'].items():
                        threshold_val = float(threshold_key)
                        event_results = threshold_info['full_result']['event_results']
                        
                        # 只保留fpRate <= 50的数据点
                        if event_results['fpRate'] <= 50:
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
                
                print(f"成功读取 {subdir}: {len(threshold_data)} 个阈值点 (fpRate <= 50)")
                
            except Exception as e:
                print(f"读取文件 {eval_file} 时出错: {e}")
        else:
            print(f"警告: 找不到文件 {eval_file}")
    
    return all_data

def plot_tcp_threshold_comparison():
    """
    绘制TCP模型在不同阈值下的性能比较图
    """
    # 加载数据
    all_data = load_tcp_evaluation_data()
    
    if not all_data:
        print("没有找到有效的数据")
        return
    
    # 创建子图
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('TCP Model Performance Comparison Across Thresholds (fpRate ≤ 50)', fontsize=18, fontweight='bold')
    
    # 定义颜色和线型
    colors = plt.cm.Set1(np.linspace(0, 1, len(all_data)))
    line_styles = ['-', '--', '-.', ':', '-', '--', '-.', ':']
    
    metrics = ['sensitivity', 'precision', 'f1', 'fpRate']
    metric_titles = ['Sensitivity', 'Precision', 'F1 Score', 'False Positive Rate (per 24h)']
    
    for i, (metric, title) in enumerate(zip(metrics, metric_titles)):
        ax = axes[i//2, i%2]
        
        for j, (model_name, data) in enumerate(all_data.items()):
            if data:  # 确保有数据
                thresholds = [d['threshold'] for d in data]
                values = [d[metric] for d in data]
                
                ax.plot(thresholds, values, 
                       color=colors[j], 
                       linestyle=line_styles[j % len(line_styles)],
                       marker='o', markersize=6, linewidth=2.5,
                       label=model_name, alpha=0.8)
        
        ax.set_xlabel('Threshold', fontsize=12)
        ax.set_ylabel(title, fontsize=12)
        ax.set_title(f'{title} vs Threshold', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=10, loc='best')
        
        # 特殊处理FP Rate的y轴，移除对数刻度因为范围较小
        if metric == 'fpRate':
            # 不再使用对数刻度，因为fpRate <= 50
            # ax.set_yscale('log')
            ax.set_ylabel('FP Rate (per 24h, ≤50)', fontsize=12)
    
    plt.tight_layout()
    
    # 保存图片
    plt.savefig('tcp_threshold_comparison_all_models.png', dpi=300, bbox_inches='tight')
    print(f"\n图片已保存为: tcp_threshold_comparison_all_models.png")
    
    # 显示统计信息
    print(f"\n=== TCP模型比较统计 ===")
    for model_name, data in all_data.items():
        if data:
            # 找到最佳F1分数的阈值
            best_f1_data = max(data, key=lambda x: x['f1'])
            print(f"\n{model_name}:")
            print(f"  最佳阈值: {best_f1_data['threshold']:.1f}")
            print(f"  最佳F1: {best_f1_data['f1']:.4f}")
            print(f"  对应Sensitivity: {best_f1_data['sensitivity']:.4f}")
            print(f"  对应Precision: {best_f1_data['precision']:.4f}")
            print(f"  对应FP Rate: {best_f1_data['fpRate']:.2f}")
    
    plt.show()
    
    return all_data

def plot_tcp_roc_style_comparison():
    """
    绘制TCP模型的类似ROC曲线的Sensitivity vs FP Rate比较图
    """
    all_data = load_tcp_evaluation_data()
    
    if not all_data:
        print("没有找到有效的数据")
        return
    
    plt.figure(figsize=(12, 8))
    
    colors = plt.cm.Set1(np.linspace(0, 1, len(all_data)))
    
    for i, (model_name, data) in enumerate(all_data.items()):
        if data:
            fp_rates = [d['fpRate'] for d in data]
            sensitivities = [d['sensitivity'] for d in data]
            thresholds = [d['threshold'] for d in data]
            
            # 绘制曲线
            plt.plot(fp_rates, sensitivities, 
                    color=colors[i], marker='o', markersize=8, 
                    linewidth=3, label=model_name, alpha=0.8)
            
            # 为每个点添加阈值标签
            for j, (fp, sens, thresh) in enumerate(zip(fp_rates, sensitivities, thresholds)):
                if j % 2 == 0:  # 只显示部分标签避免重叠
                    plt.annotate(f'{thresh:.1f}', (fp, sens), 
                               xytext=(5, 5), textcoords='offset points',
                               fontsize=9, alpha=0.7,
                               color=colors[i])
    
    plt.xlabel('False Positive Rate (per 24h)', fontsize=14)
    plt.ylabel('Sensitivity', fontsize=14)
    plt.title('TCP Model Performance: Sensitivity vs FP Rate (fpRate ≤ 50)\n(Numbers indicate thresholds)', fontsize=16, fontweight='bold')
    plt.legend(fontsize=12, loc='best')
    plt.grid(True, alpha=0.3)
    # 移除对数刻度，因为现在fpRate范围较小 (≤50)
    # plt.xscale('log')  # FP Rate使用对数刻度
    
    plt.tight_layout()
    plt.savefig('tcp_sensitivity_vs_fprate_comparison.png', dpi=300, bbox_inches='tight')
    print(f"\nTCP ROC风格图片已保存为: tcp_sensitivity_vs_fprate_comparison.png")
    plt.show()

def compare_avgs_vs_tcp():
    """
    比较AVGS和TCP模型的最佳性能
    """
    print("\n=== AVGS vs TCP 模型性能对比 ===")
    
    # 加载AVGS数据
    base_dir = os.getcwd()
    avgs_dir = os.path.join(base_dir, 'avgs', 'avgs')
    avgs_data = {}
    
    if os.path.exists(avgs_dir):
        subdirs = [d for d in os.listdir(avgs_dir) 
                   if os.path.isdir(os.path.join(avgs_dir, d)) and not d.startswith('.')]
        
        for subdir in subdirs:
            eval_file = os.path.join(avgs_dir, subdir, 'quick_eval_results', 'evaluation_summary.json')
            if os.path.exists(eval_file):
                try:
                    with open(eval_file, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                    avgs_data[f"AVGS-{subdir}"] = data['best_f1_score']
                except:
                    pass
    
    # 加载TCP数据
    tcp_data = load_tcp_evaluation_data()
    tcp_best_f1 = {}
    
    for model_name, data in tcp_data.items():
        if data:
            best_f1 = max(d['f1'] for d in data)
            tcp_best_f1[f"TCP-{model_name}"] = best_f1
    
    # 创建对比图
    plt.figure(figsize=(14, 8))
    
    all_models = list(avgs_data.keys()) + list(tcp_best_f1.keys())
    all_f1_scores = list(avgs_data.values()) + list(tcp_best_f1.values())
    
    colors = ['skyblue'] * len(avgs_data) + ['lightcoral'] * len(tcp_best_f1)
    
    bars = plt.bar(range(len(all_models)), all_f1_scores, color=colors, alpha=0.7, edgecolor='black')
    
    plt.xlabel('Models', fontsize=14)
    plt.ylabel('Best F1 Score', fontsize=14)
    plt.title('AVGS vs TCP Models: Best F1 Score Comparison', fontsize=16, fontweight='bold')
    plt.xticks(range(len(all_models)), all_models, rotation=45, ha='right')
    plt.grid(True, alpha=0.3, axis='y')
    
    # 添加数值标签
    for i, (bar, score) in enumerate(zip(bars, all_f1_scores)):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                f'{score:.3f}', ha='center', va='bottom', fontsize=10)
    
    # 添加图例
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor='skyblue', label='AVGS Models'),
                      Patch(facecolor='lightcoral', label='TCP Models')]
    plt.legend(handles=legend_elements, loc='upper right')
    
    plt.tight_layout()
    plt.savefig('avgs_vs_tcp_comparison.png', dpi=300, bbox_inches='tight')
    print(f"\n对比图已保存为: avgs_vs_tcp_comparison.png")
    plt.show()
    
    # 打印统计信息
    print(f"\nAVGS模型最佳F1: {max(avgs_data.values()):.4f}")
    print(f"TCP模型最佳F1: {max(tcp_best_f1.values()):.4f}")
    
    best_avgs = max(avgs_data.items(), key=lambda x: x[1])
    best_tcp = max(tcp_best_f1.items(), key=lambda x: x[1])
    
    print(f"\n最佳AVGS模型: {best_avgs[0]} (F1: {best_avgs[1]:.4f})")
    print(f"最佳TCP模型: {best_tcp[0]} (F1: {best_tcp[1]:.4f})")

if __name__ == '__main__':
    print("=== TCP阈值性能比较分析 ===")
    tcp_data = plot_tcp_threshold_comparison()
    print("\n=== TCP ROC风格比较图 ===")
    plot_tcp_roc_style_comparison()
    print("\n=== AVGS vs TCP 对比分析 ===")
    compare_avgs_vs_tcp()