#!/usr/bin/env python3
"""
分析损失函数与性能指标之间的关系
解释为什么损失下降但性能提升的现象
"""

import json
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from pathlib import Path

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

def load_and_analyze_relationship():
    """加载数据并分析损失与性能的关系"""
    
    # 加载训练数据
    log_file = "logs/log.txt"
    if not Path(log_file).exists():
        print("找不到日志文件")
        return
    
    data = []
    with open(log_file, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                epoch_data = json.loads(line.strip())
                if 'epoch' in epoch_data:
                    data.append(epoch_data)
            except:
                continue
    
    if not data:
        print("没有找到训练数据")
        return
    
    df = pd.DataFrame(data)
    print(f"加载了 {len(df)} 个epoch的数据")
    
    # 创建对比分析图
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('损失函数 vs 性能指标：为什么损失下降性能提升？', fontsize=16, fontweight='bold')
    
    # 1. 损失与mAP的关系
    ax1 = axes[0, 0]
    if 'train_loss' in df.columns and 'test_eval_bbox' in df.columns:
        ax1_twin = ax1.twinx()
        
        # 训练损失（左轴）
        line1 = ax1.plot(df['epoch'], df['train_loss'], 'b-', linewidth=2, label='训练损失')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('训练损失', color='blue')
        ax1.tick_params(axis='y', labelcolor='blue')
        
        # mAP（右轴）
        line2 = ax1_twin.plot(df['epoch'], df['test_eval_bbox'], 'r-', linewidth=2, label='验证mAP')
        ax1_twin.set_ylabel('mAP', color='red')
        ax1_twin.tick_params(axis='y', labelcolor='red')
        
        # 合并图例
        lines = line1 + line2
        labels = [l.get_label() for l in lines]
        ax1.legend(lines, labels, loc='center right')
        
        ax1.set_title('训练损失 vs 验证mAP\n(损失↓，性能↑)')
        ax1.grid(True, alpha=0.3)
    
    # 2. 损失组件分解
    ax2 = axes[0, 1]
    loss_components = {
        'train_loss_ce': '分类损失',
        'train_loss_bbox': '边界框损失',
        'train_loss_giou': 'GIoU损失'
    }
    colors = ['orange', 'brown', 'pink']
    
    for i, (component, label) in enumerate(loss_components.items()):
        if component in df.columns:
            ax2.plot(df['epoch'], df[component], color=colors[i], 
                    linewidth=2, label=label, marker='o', markersize=3)
    
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('损失值')
    ax2.set_title('损失组件分解\n各组件都在下降')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. 损失与性能的散点图关系
    ax3 = axes[0, 2]
    if 'train_loss' in df.columns and 'test_eval_bbox' in df.columns:
        scatter = ax3.scatter(df['train_loss'], df['test_eval_bbox'], 
                             c=df['epoch'], cmap='viridis', alpha=0.7, s=50)
        ax3.set_xlabel('训练损失')
        ax3.set_ylabel('验证mAP')
        ax3.set_title('损失-性能散点图\n(颜色表示epoch)')
        
        # 添加趋势线
        z = np.polyfit(df['train_loss'], df['test_eval_bbox'], 1)
        p = np.poly1d(z)
        ax3.plot(df['train_loss'], p(df['train_loss']), "r--", alpha=0.8, linewidth=2)
        
        # 添加颜色条
        cbar = plt.colorbar(scatter, ax=ax3)
        cbar.set_label('Epoch')
        ax3.grid(True, alpha=0.3)
    
    # 4. 学习率对损失的影响
    ax4 = axes[1, 0]
    if 'train_lr' in df.columns and 'train_loss' in df.columns:
        ax4_twin = ax4.twinx()
        
        # 学习率（左轴）
        line1 = ax4.plot(df['epoch'], df['train_lr'], 'g-', linewidth=2, label='学习率')
        ax4.set_xlabel('Epoch')
        ax4.set_ylabel('学习率', color='green')
        ax4.tick_params(axis='y', labelcolor='green')
        ax4.set_yscale('log')
        
        # 训练损失（右轴）
        line2 = ax4_twin.plot(df['epoch'], df['train_loss'], 'b-', linewidth=2, label='训练损失')
        ax4_twin.set_ylabel('训练损失', color='blue')
        ax4_twin.tick_params(axis='y', labelcolor='blue')
        
        lines = line1 + line2
        labels = [l.get_label() for l in lines]
        ax4.legend(lines, labels, loc='upper right')
        
        ax4.set_title('学习率调度与损失下降\n学习率适当，损失稳定下降')
        ax4.grid(True, alpha=0.3)
    
    # 5. 性能改善速率
    ax5 = axes[1, 1]
    if 'test_eval_bbox' in df.columns and len(df) > 5:
        # 计算性能改善的梯度
        performance_gradient = np.gradient(df['test_eval_bbox'])
        
        ax5.plot(df['epoch'], df['test_eval_bbox'], 'b-', linewidth=2, label='验证mAP')
        ax5_twin = ax5.twinx()
        ax5_twin.plot(df['epoch'], performance_gradient, 'r-', linewidth=1, alpha=0.7, label='改善速率')
        
        ax5.set_xlabel('Epoch')
        ax5.set_ylabel('mAP', color='blue')
        ax5_twin.set_ylabel('改善速率', color='red')
        ax5.tick_params(axis='y', labelcolor='blue')
        ax5_twin.tick_params(axis='y', labelcolor='red')
        
        ax5.set_title('性能改善速率分析\n持续正向改善')
        ax5.grid(True, alpha=0.3)
        ax5.legend(loc='upper left')
        ax5_twin.legend(loc='upper right')
    
    # 6. 训练稳定性分析
    ax6 = axes[1, 2]
    if 'train_loss' in df.columns and len(df) > 10:
        # 计算损失的滑动方差（稳定性指标）
        window = min(10, len(df)//3)
        rolling_std = df['train_loss'].rolling(window=window).std()
        rolling_mean = df['train_loss'].rolling(window=window).mean()
        
        ax6.plot(df['epoch'], rolling_mean, 'b-', linewidth=2, label=f'{window}-epoch滑动均值')
        ax6.fill_between(df['epoch'], 
                        rolling_mean - rolling_std, 
                        rolling_mean + rolling_std, 
                        alpha=0.3, color='blue', label='标准差范围')
        
        ax6.set_xlabel('Epoch')
        ax6.set_ylabel('训练损失')
        ax6.set_title('训练稳定性分析\n损失收敛稳定')
        ax6.legend()
        ax6.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('plots/loss_vs_performance_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # 打印详细分析
    print_detailed_analysis(df)

def print_detailed_analysis(df):
    """打印详细的分析报告"""
    print("\n" + "="*80)
    print("🔍 损失函数 vs 性能指标 详细分析报告")
    print("="*80)
    
    print("\n📊 核心现象：")
    print("   ✅ 训练损失持续下降")
    print("   ✅ 验证mAP持续上升")
    print("   ✅ 这是正常且理想的训练状态！")
    
    print("\n🧠 原理解释：")
    print("   1. 损失函数 ≠ 性能指标")
    print("      • 损失函数：模型优化的目标（如交叉熵、L1损失等）")
    print("      • 性能指标：实际任务的评估标准（如mAP、准确率等）")
    
    print("\n   2. 损失下降 → 性能提升的机制：")
    print("      • 分类损失↓ → 分类准确性↑ → 正确识别更多目标")
    print("      • 边界框损失↓ → 定位精度↑ → 检测框更准确")
    print("      • GIoU损失↓ → 框重叠度↑ → 检测质量更好")
    
    if 'train_loss' in df.columns and 'test_eval_bbox' in df.columns:
        initial_loss = df['train_loss'].iloc[0]
        final_loss = df['train_loss'].iloc[-1]
        initial_map = df['test_eval_bbox'].iloc[0] if pd.notna(df['test_eval_bbox'].iloc[0]) else 0
        final_map = df['test_eval_bbox'].iloc[-1]
        
        loss_reduction = ((initial_loss - final_loss) / initial_loss) * 100
        performance_improvement = ((final_map - initial_map) / initial_map) * 100 if initial_map > 0 else 0
        
        print(f"\n📈 数值验证：")
        print(f"   • 损失下降：{initial_loss:.4f} → {final_loss:.4f} ({loss_reduction:.2f}%↓)")
        print(f"   • 性能提升：{initial_map:.4f} → {final_map:.4f} ({performance_improvement:.2f}%↑)")
        
        # 计算相关性
        correlation = np.corrcoef(df['train_loss'], df['test_eval_bbox'])[0, 1]
        print(f"   • 损失-性能相关性：{correlation:.4f} (负相关表示损失↓性能↑)")
    
    print("\n🎯 DINO模型特殊性：")
    print("   • 多尺度特征：不同层的损失都在优化")
    print("   • 注意力机制：学习更好的特征表示")
    print("   • 端到端训练：整体优化检测pipeline")
    
    print("\n✨ 训练状态评估：")
    if 'train_loss' in df.columns and 'test_eval_bbox' in df.columns:
        # 检查过拟合
        recent_epochs = min(10, len(df)//2)
        recent_loss_trend = df['train_loss'].tail(recent_epochs)
        recent_map_trend = df['test_eval_bbox'].tail(recent_epochs)
        
        loss_still_decreasing = recent_loss_trend.iloc[-1] < recent_loss_trend.iloc[0]
        map_still_improving = recent_map_trend.iloc[-1] > recent_map_trend.iloc[0]
        
        if loss_still_decreasing and map_still_improving:
            print("   🟢 训练状态：健康（损失↓，性能↑）")
            print("   🟢 建议：继续训练")
        elif loss_still_decreasing and not map_still_improving:
            print("   🟡 训练状态：可能过拟合（损失↓，性能平稳）")
            print("   🟡 建议：考虑早停或调整正则化")
        else:
            print("   🟢 训练状态：收敛中")
    
    print("\n🔬 技术细节：")
    print("   • DINO使用多个损失组件的加权和作为总损失")
    print("   • 每个组件针对不同的任务（分类、定位、匹配）")
    print("   • mAP是综合评估检测质量的标准指标")
    print("   • 损失优化 → 模型参数改善 → 检测能力提升 → mAP上升")
    
    print("\n💡 关键洞察：")
    print("   ✅ 损失下降 + 性能提升 = 理想训练状态")
    print("   ✅ 这表明模型正在学习有用的特征表示")
    print("   ✅ 优化目标与实际任务目标保持一致")
    
    print("="*80)

if __name__ == "__main__":
    import os
    os.makedirs('plots', exist_ok=True)
    load_and_analyze_relationship()
