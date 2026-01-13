import matplotlib.pyplot as plt
import pandas as pd
import json
import os
import glob
from matplotlib import rcParams

# 设置字体更大
rcParams.update({'font.size': 16})

def load_channel_data():
    """
    从channel_summary文件夹中加载所有JSON文件的数据
    """
    channel_data = []
    
    # 获取channel_summary文件夹路径
    base_dir = os.getcwd()
    channel_dir = os.path.join(base_dir, 'drop_summary')
    
    if not os.path.exists(channel_dir):
        print(f"错误: 找不到文件夹 {channel_dir}")
        return []
    
    # 查找所有JSON文件
    json_files = glob.glob(os.path.join(channel_dir, '*.json'))
    
    print(f"找到 {len(json_files)} 个JSON文件")
    
    for json_file in json_files:
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # 提取文件名作为通道名称
            channel_name = os.path.splitext(os.path.basename(json_file))[0]
            
            # 提取best_result中的event_results数据
            if 'best_result' in data and 'event_results' in data['best_result']:
                event_results = data['best_result']['event_results']
                
                if 'sensitivity' in event_results and 'fpRate' in event_results:
                    channel_data.append({
                        'channel': channel_name,
                        'sensitivity': event_results['sensitivity'],
                        'fpRate': event_results['fpRate'],
                        'f1': event_results.get('f1', 0),
                        'precision': event_results.get('precision', 0)
                    })
                    print(f"读取: {channel_name} - Sensitivity: {event_results['sensitivity']:.4f}, FP Rate: {event_results['fpRate']:.4f}")
                else:
                    print(f"警告: {channel_name} 缺少sensitivity或fpRate数据")
            else:
                print(f"警告: {channel_name} 缺少best_result或event_results数据")
                
        except Exception as e:
            print(f"读取文件 {json_file} 时出错: {e}")
    
    return channel_data

def plot_channel_scatter():
    """
    绘制通道数据的sensitivity vs fpRate散点图
    """
    # 加载数据
    channel_data = load_channel_data()
    
    if not channel_data:
        print("没有找到有效的数据")
        return
    
    # 转换为DataFrame
    df = pd.DataFrame(channel_data)
    
    print(f"\n成功加载 {len(df)} 个通道的数据")
    print(f"Sensitivity 范围: {df['sensitivity'].min():.4f} - {df['sensitivity'].max():.4f}")
    print(f"FP Rate 范围: {df['fpRate'].min():.4f} - {df['fpRate'].max():.4f}")
    
    # 创建图形
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # 绘制散点图
    scatter = ax.scatter(df['fpRate'], df['sensitivity'], 
                        s=100, alpha=0.7, 
                        c=df['f1'], cmap='viridis',
                        edgecolors='black', linewidths=0.5)
    
    # 添加颜色条
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('F1 Score', fontsize=16)
    cbar.ax.tick_params(labelsize=14)
    
    # 为每个点添加标签
    for i, row in df.iterrows():
        ax.annotate(row['channel'], 
                   (row['fpRate'], row['sensitivity']),
                   xytext=(5, 5), textcoords='offset points',
                   fontsize=10, alpha=0.8,
                   bbox=dict(boxstyle="round,pad=0.2", fc="white", alpha=0.7))
    
    # 设置坐标轴标签和标题
    ax.set_xlabel('FP Rate (False Positives per 24h)', fontsize=16)
    ax.set_ylabel('Sensitivity', fontsize=16)
    ax.set_title('Channel Performance: Sensitivity vs FP Rate\n(Color indicates F1 Score)', fontsize=18)
    
    # 设置网格
    ax.grid(True, alpha=0.3)
    ax.tick_params(labelsize=14)
    
    # 调整布局
    plt.tight_layout()
    
    # 显示统计信息
    print(f"\n=== 统计信息 ===")
    print(f"平均 Sensitivity: {df['sensitivity'].mean():.4f} ± {df['sensitivity'].std():.4f}")
    print(f"平均 FP Rate: {df['fpRate'].mean():.4f} ± {df['fpRate'].std():.4f}")
    print(f"平均 F1 Score: {df['f1'].mean():.4f} ± {df['f1'].std():.4f}")
    
    # 找出最佳和最差的通道
    best_f1_idx = df['f1'].idxmax()
    worst_f1_idx = df['f1'].idxmin()
    
    print(f"\n=== 最佳通道 (F1 Score) ===")
    print(f"{df.loc[best_f1_idx, 'channel']}: F1={df.loc[best_f1_idx, 'f1']:.4f}, "
          f"Sensitivity={df.loc[best_f1_idx, 'sensitivity']:.4f}, "
          f"FP Rate={df.loc[best_f1_idx, 'fpRate']:.4f}")
    
    print(f"\n=== 最差通道 (F1 Score) ===")
    print(f"{df.loc[worst_f1_idx, 'channel']}: F1={df.loc[worst_f1_idx, 'f1']:.4f}, "
          f"Sensitivity={df.loc[worst_f1_idx, 'sensitivity']:.4f}, "
          f"FP Rate={df.loc[worst_f1_idx, 'fpRate']:.4f}")
    
    # 保存图片
    plt.savefig('channel_sensitivity_fprate_scatter.png', dpi=300, bbox_inches='tight')
    print(f"\n图片已保存为: channel_sensitivity_fprate_scatter.png")
    
    # 显示图形
    plt.show()
    
    return df

if __name__ == '__main__':
    df = plot_channel_scatter()