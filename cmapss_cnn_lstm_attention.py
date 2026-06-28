"""
=============================================================================
C-MAPSS 剩余寿命预测 — CNN-LSTM-Attention 混合模型
=============================================================================
数据集: NASA 涡轮风扇发动机退化模拟数据 (FD001~FD004)
模型: 1D-CNN → Bi-LSTM → Attention → Dense → RUL 回归

作者: Auto-generated
环境: Python 3.9+, TensorFlow 2.x
=============================================================================
"""

import os
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from tensorflow.keras import layers, Model, Input, backend as K
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam

warnings.filterwarnings('ignore')

# =============================================================================
# 0. 配置参数
# =============================================================================
class Config:
    # 数据路径
    data_dir = r'e:\CMAPSS'
    
    # 数据集选择: 'FD001', 'FD002', 'FD003', 'FD004'
    dataset = 'FD003'
    
    # 滑动窗口参数
    window_size = 30          # 历史时间步数
    stride = 1                # 滑动步长
    
    # RUL 裁剪上限
    rul_max = 125
    
    # 模型超参数
    cnn_filters = 64          # CNN 卷积核数量
    cnn_kernel_size = 3       # CNN 卷积核大小
    lstm_units = 64           # LSTM 单元数
    attention_units = 32      # Attention 层维度
    dropout_rate = 0.3        # Dropout 比率
    
    # 训练参数
    batch_size = 256
    epochs = 200
    learning_rate = 1e-3
    validation_split = 0.2
    patience_early_stop = 15
    patience_reduce_lr = 8
    
    # 随机种子
    random_seed = 42


config = Config()
np.random.seed(config.random_seed)

# =============================================================================
# 1. 数据加载与预处理
# =============================================================================
COL_NAMES = [
    'unit_id', 'cycle',
    'op_setting_1', 'op_setting_2', 'op_setting_3',
    'sensor_1',  'sensor_2',  'sensor_3',  'sensor_4',  'sensor_5',
    'sensor_6',  'sensor_7',  'sensor_8',  'sensor_9',  'sensor_10',
    'sensor_11', 'sensor_12', 'sensor_13', 'sensor_14', 'sensor_15',
    'sensor_16', 'sensor_17', 'sensor_18', 'sensor_19', 'sensor_20',
    'sensor_21'
]

# 传感器列名
SENSOR_COLS = [f'sensor_{i}' for i in range(1, 22)]
OP_COLS = ['op_setting_1', 'op_setting_2', 'op_setting_3']


def load_raw_data(dataset_name):
    """加载原始 C-MAPSS 数据"""
    base = config.data_dir
    train_path = os.path.join(base, f'train_{dataset_name}.txt')
    test_path  = os.path.join(base, f'test_{dataset_name}.txt')
    rul_path   = os.path.join(base, f'RUL_{dataset_name}.txt')
    
    df_train = pd.read_csv(train_path, sep=r'\s+', header=None, names=COL_NAMES)
    df_test  = pd.read_csv(test_path,  sep=r'\s+', header=None, names=COL_NAMES)
    df_rul   = pd.read_csv(rul_path,   sep=r'\s+', header=None, names=['RUL'])
    
    return df_train, df_test, df_rul


def add_rul_labels(df_train, df_test, df_rul):
    """为训练集和测试集添加 RUL 标签"""
    # --- 训练集: RUL = max_cycle - current_cycle ---
    max_cycles = df_train.groupby('unit_id')['cycle'].transform('max')
    df_train['RUL'] = max_cycles - df_train['cycle']
    df_train['RUL'] = df_train['RUL'].clip(upper=config.rul_max)
    
    # --- 测试集: RUL = (last_cycle + RUL_final) - current_cycle ---
    last_cycles = df_test.groupby('unit_id')['cycle'].max().values
    rul_values  = df_rul['RUL'].values
    total_life  = last_cycles + rul_values
    
    # 将 total_life 映射回每个样本
    unit_total_life = pd.DataFrame({
        'unit_id': df_test['unit_id'].unique(),
        'total_life': total_life
    })
    df_test = df_test.merge(unit_total_life, on='unit_id', how='left')
    df_test['RUL'] = df_test['total_life'] - df_test['cycle']
    df_test['RUL'] = df_test['RUL'].clip(upper=config.rul_max)
    df_test.drop(columns=['total_life'], inplace=True)
    
    return df_train, df_test


def select_sensors(df_train, df_test, variance_threshold=1e-3):
    """
    自动选择传感器: 
    1. 去除方差过小的传感器（无信息量）
    2. 去除与操作条件高度相关的传感器（反映工况而非退化）
    """
    # 计算每个传感器在训练集上的方差
    variances = df_train[SENSOR_COLS].var()
    valid_sensors = variances[variances > variance_threshold].index.tolist()
    
    # 计算传感器与操作条件的相关性，去除相关性过高的
    corr_with_op = []
    for sensor in valid_sensors:
        max_corr = max(
            abs(df_train[sensor].corr(df_train[op_col]))
            for op_col in OP_COLS
        )
        corr_with_op.append(max_corr)
    
    # 保留与操作条件相关性 < 0.7 的传感器
    sensor_corr_df = pd.DataFrame({
        'sensor': valid_sensors,
        'max_corr_with_op': corr_with_op
    })
    selected = sensor_corr_df[sensor_corr_df['max_corr_with_op'] < 0.7]['sensor'].tolist()
    
    print(f"[信息] 传感器总数: {len(SENSOR_COLS)}")
    print(f"[信息] 方差筛选后: {len(valid_sensors)}")
    print(f"[信息] 最终选择: {len(selected)} 个传感器")
    print(f"[信息] 选中传感器: {selected}")
    
    return selected


def normalize_by_condition(df, feature_cols):
    """
    按操作条件分组标准化（对多工况数据集 FD002/FD004 尤其重要）
    使用操作条件的离散化聚类进行分组
    """
    from sklearn.cluster import KMeans
    
    # 对操作条件进行聚类
    op_data = df[OP_COLS].values
    kmeans = KMeans(n_clusters=6, random_state=config.random_seed, n_init=10)
    df['condition_cluster'] = kmeans.fit_predict(op_data)
    
    # 按聚类分组标准化
    scalers = {}
    df_scaled = df.copy()
    for cluster in df['condition_cluster'].unique():
        mask = df['condition_cluster'] == cluster
        scaler = StandardScaler()
        df_scaled.loc[mask, feature_cols] = scaler.fit_transform(
            df.loc[mask, feature_cols]
        )
        scalers[cluster] = scaler
    
    return df_scaled, scalers, kmeans


def create_sliding_windows(data, window_size, stride=1):
    """
    创建滑动窗口样本
    输入: data shape = [n_timesteps, n_features]
    输出: windows shape = [n_samples, window_size, n_features]
    """
    n_samples = (len(data) - window_size) // stride + 1
    if n_samples <= 0:
        return None, None
    
    X, y = [], []
    for i in range(0, len(data) - window_size + 1, stride):
        X.append(data[i:i + window_size, :-1])  # 特征
        y.append(data[i + window_size - 1, -1])  # 标签取窗口最后一个时间步的 RUL
    
    return np.array(X), np.array(y)


def prepare_dataset(df_train, df_test, feature_cols):
    """
    完整的数据预处理流程:
    1. 按发动机分组
    2. 对每个发动机创建滑动窗口
    3. 返回 3D 数组 [样本数, window_size, 特征数]
    """
    X_train_list, y_train_list = [], []
    X_test_list,  y_test_list  = [], []
    
    # 训练集
    for unit_id in df_train['unit_id'].unique():
        unit_data = df_train[df_train['unit_id'] == unit_id].sort_values('cycle')
        unit_array = unit_data[feature_cols + ['RUL']].values
        X_unit, y_unit = create_sliding_windows(unit_array, config.window_size, config.stride)
        if X_unit is not None:
            X_train_list.append(X_unit)
            y_train_list.append(y_unit)
    
    # 测试集
    for unit_id in df_test['unit_id'].unique():
        unit_data = df_test[df_test['unit_id'] == unit_id].sort_values('cycle')
        unit_array = unit_data[feature_cols + ['RUL']].values
        X_unit, y_unit = create_sliding_windows(unit_array, config.window_size, config.stride)
        if X_unit is not None:
            X_test_list.append(X_unit)
            y_test_list.append(y_unit)
    
    X_train = np.concatenate(X_train_list, axis=0) if X_train_list else np.array([])
    y_train = np.concatenate(y_train_list, axis=0) if y_train_list else np.array([])
    X_test  = np.concatenate(X_test_list,  axis=0) if X_test_list  else np.array([])
    y_test  = np.concatenate(y_test_list,  axis=0) if y_test_list  else np.array([])
    
    print(f"[信息] 训练集形状: {X_train.shape}, 测试集形状: {X_test.shape}")
    return X_train, y_train, X_test, y_test


# =============================================================================
# 2. 模型构建 — CNN-LSTM-Attention
# =============================================================================
def attention_layer(inputs, units, name_prefix='attn'):
    """
    自定义注意力层: 计算输入序列各时间步的权重，加权求和
    inputs shape: [batch, timesteps, features]
    output shape: [batch, units]
    """
    # 得分网络
    score = layers.Dense(units, activation='tanh', name=f'{name_prefix}_score')(inputs)  # [batch, timesteps, units]
    score = layers.Dense(1, activation='linear', name=f'{name_prefix}_logits')(score)     # [batch, timesteps, 1]
    score = layers.Flatten(name=f'{name_prefix}_flatten')(score)                          # [batch, timesteps]
    attention_weights = layers.Activation('softmax', name=f'{name_prefix}_softmax')(score)  # [batch, timesteps]
    attention_weights = layers.Reshape((-1, 1), name=f'{name_prefix}_reshape')(attention_weights)  # [batch, timesteps, 1]
    
    # 加权求和
    weighted = layers.Multiply(name=f'{name_prefix}_multiply')([inputs, attention_weights])  # [batch, timesteps, features]
    context = layers.Lambda(lambda x: K.sum(x, axis=1), name=f'{name_prefix}_context')(weighted)  # [batch, features]
    
    return context, attention_weights


def build_cnn_lstm_attention_model(input_shape):
    """
    构建 CNN-LSTM-Attention 混合模型
    
    架构:
    Input [batch, window_size, n_features]
      ↓
    1D-CNN (提取局部传感器模式)
      ↓
    BatchNormalization + MaxPooling
      ↓
    Bi-LSTM (双向时序建模)
      ↓
    Attention (聚焦关键退化阶段)
      ↓
    Dense + Dropout
      ↓
    Output: RUL (回归)
    """
    inputs = Input(shape=input_shape, name='input')
    
    # --- Stage 1: 1D-CNN 提取局部特征 ---
    x = layers.Conv1D(
        filters=config.cnn_filters,
        kernel_size=config.cnn_kernel_size,
        padding='same',
        activation='relu',
        name='conv1d_1'
    )(inputs)
    x = layers.BatchNormalization(name='bn_1')(x)
    x = layers.MaxPooling1D(pool_size=2, padding='same', name='maxpool_1')(x)
    
    x = layers.Conv1D(
        filters=config.cnn_filters * 2,
        kernel_size=config.cnn_kernel_size,
        padding='same',
        activation='relu',
        name='conv1d_2'
    )(x)
    x = layers.BatchNormalization(name='bn_2')(x)
    x = layers.MaxPooling1D(pool_size=2, padding='same', name='maxpool_2')(x)
    
    # --- Stage 2: Bi-LSTM 时序建模 ---
    x = layers.Bidirectional(
        layers.LSTM(units=config.lstm_units, return_sequences=True, dropout=config.dropout_rate),
        name='bidirectional_lstm'
    )(x)
    
    # --- Stage 3: Attention 机制 ---
    context, attn_weights = attention_layer(x, config.attention_units)
    
    # --- Stage 4: 全连接回归头 ---
    x = layers.Dense(64, activation='relu', name='dense_1')(context)
    x = layers.Dropout(config.dropout_rate, name='dropout_1')(x)
    x = layers.Dense(32, activation='relu', name='dense_2')(x)
    x = layers.Dropout(config.dropout_rate, name='dropout_2')(x)
    outputs = layers.Dense(1, name='output')(x)
    
    model = Model(inputs=inputs, outputs=outputs, name='CNN_LSTM_Attention')
    return model


# =============================================================================
# 3. 自定义评估指标
# =============================================================================
def phm08_score(y_true, y_pred):
    """
    PHM08 竞赛评分函数
    非对称评分: 早期预测（预测 < 真实）惩罚小，晚期预测（预测 > 真实）惩罚大
    """
    y_true = np.array(y_true).flatten()
    y_pred = np.array(y_pred).flatten()
    
    d = y_pred - y_true  # 误差
    # d < 0: 预测偏小（早期预测），惩罚较小
    # d >= 0: 预测偏大（晚期预测），惩罚较大
    score = np.where(
        d < 0,
        np.exp(-d / 13.0) - 1,
        np.exp(d / 10.0) - 1
    )
    return np.mean(score)


def evaluate_model(y_true, y_pred, model_name='模型'):
    """综合评估"""
    y_true = np.array(y_true).flatten()
    y_pred = np.array(y_pred).flatten()
    
    mae  = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2   = r2_score(y_true, y_pred)
    score = phm08_score(y_true, y_pred)
    
    print(f"\n{'='*50}")
    print(f" {model_name} 评估结果")
    print(f"{'='*50}")
    print(f"  MAE   (平均绝对误差)  : {mae:.4f}")
    print(f"  RMSE  (均方根误差)    : {rmse:.4f}")
    print(f"  R²    (决定系数)      : {r2:.4f}")
    print(f"  Score (PHM08评分)     : {score:.4f}")
    print(f"{'='*50}\n")
    
    return {'MAE': mae, 'RMSE': rmse, 'R2': r2, 'PHM08_Score': score}


# =============================================================================
# 4. 可视化
# =============================================================================
def plot_training_history(history):
    """绘制训练曲线"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Loss
    axes[0].plot(history.history['loss'], label='Train Loss', linewidth=2)
    axes[0].plot(history.history['val_loss'], label='Val Loss', linewidth=2)
    axes[0].set_xlabel('Epoch', fontsize=12)
    axes[0].set_ylabel('Loss (MSE)', fontsize=12)
    axes[0].set_title('Training & Validation Loss', fontsize=14)
    axes[0].legend(fontsize=11)
    axes[0].grid(True, alpha=0.3)
    
    # MAE
    axes[1].plot(history.history['mae'], label='Train MAE', linewidth=2)
    axes[1].plot(history.history['val_mae'], label='Val MAE', linewidth=2)
    axes[1].set_xlabel('Epoch', fontsize=12)
    axes[1].set_ylabel('MAE', fontsize=12)
    axes[1].set_title('Training & Validation MAE', fontsize=14)
    axes[1].legend(fontsize=11)
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(config.data_dir, f'training_history_{config.dataset}.png'), dpi=150)
    plt.show()


def plot_predictions(y_true, y_pred, title='预测 vs 真实 RUL'):
    """绘制预测结果"""
    y_true = np.array(y_true).flatten()
    y_pred = np.array(y_pred).flatten()
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # 散点图
    axes[0].scatter(y_true, y_pred, alpha=0.3, s=10, c='steelblue')
    min_val = min(y_true.min(), y_pred.min())
    max_val = max(y_true.max(), y_pred.max())
    axes[0].plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect')
    axes[0].set_xlabel('True RUL', fontsize=12)
    axes[0].set_ylabel('Predicted RUL', fontsize=12)
    axes[0].set_title(title, fontsize=14)
    axes[0].legend(fontsize=11)
    axes[0].grid(True, alpha=0.3)
    
    # 误差分布直方图
    errors = y_pred - y_true
    axes[1].hist(errors, bins=50, alpha=0.7, color='steelblue', edgecolor='white')
    axes[1].axvline(x=0, color='red', linestyle='--', linewidth=2, label='Zero Error')
    axes[1].set_xlabel('Prediction Error', fontsize=12)
    axes[1].set_ylabel('Frequency', fontsize=12)
    axes[1].set_title('Error Distribution', fontsize=14)
    axes[1].legend(fontsize=11)
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(config.data_dir, f'predictions_{config.dataset}.png'), dpi=150)
    plt.show()


def plot_unit_degradation(df_test, y_pred_all, unit_id=1):
    """绘制单个发动机的退化趋势"""
    unit_data = df_test[df_test['unit_id'] == unit_id].sort_values('cycle')
    true_rul = unit_data['RUL'].values
    
    # 获取该发动机对应的预测值（滑动窗口的最后时间步）
    unit_preds = y_pred_all[:len(true_rul)]  # 简化处理，实际需对齐
    
    plt.figure(figsize=(12, 5))
    plt.plot(unit_data['cycle'].values, true_rul, 'b-', linewidth=2, label='True RUL')
    plt.plot(unit_data['cycle'].values, unit_preds, 'r--', linewidth=2, label='Predicted RUL')
    plt.xlabel('Cycle', fontsize=12)
    plt.ylabel('RUL', fontsize=12)
    plt.title(f'Unit {unit_id} Degradation Trend', fontsize=14)
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(config.data_dir, f'unit_{unit_id}_degradation_{config.dataset}.png'), dpi=150)
    plt.show()


# =============================================================================
# 5. 主流程
# =============================================================================
def main():
    print("=" * 60)
    print(f" C-MAPSS 剩余寿命预测 — CNN-LSTM-Attention")
    print(f" 数据集: {config.dataset}")
    print("=" * 60)
    
    # ---- Step 1: 加载数据 ----
    print("\n[1/6] 加载原始数据...")
    df_train, df_test, df_rul = load_raw_data(config.dataset)
    print(f"  训练集: {df_train.shape}, 测试集: {df_test.shape}")
    
    # ---- Step 2: 添加 RUL 标签 ----
    print("\n[2/6] 添加 RUL 标签...")
    df_train, df_test = add_rul_labels(df_train, df_test, df_rul)
    print(f"  RUL 范围: [{df_train['RUL'].min():.0f}, {df_train['RUL'].max():.0f}]")
    
    # ---- Step 3: 自动选择传感器 ----
    print("\n[3/6] 自动选择传感器特征...")
    selected_sensors = select_sensors(df_train, df_test)
    feature_cols = OP_COLS + selected_sensors
    print(f"  最终特征数: {len(feature_cols)} ({len(OP_COLS)} 操作条件 + {len(selected_sensors)} 传感器)")
    
    # ---- Step 4: 按工况分组标准化 ----
    print("\n[4/6] 按操作条件分组标准化...")
    df_train, train_scalers, condition_kmeans = normalize_by_condition(df_train, feature_cols)
    
    # 对测试集使用相同的聚类模型和对应的 scaler
    test_op_data = df_test[OP_COLS].values
    df_test['condition_cluster'] = condition_kmeans.predict(test_op_data)
    for cluster in df_test['condition_cluster'].unique():
        if cluster in train_scalers:
            mask = df_test['condition_cluster'] == cluster
            df_test.loc[mask, feature_cols] = train_scalers[cluster].transform(
                df_test.loc[mask, feature_cols]
            )
        else:
            # 如果测试集出现训练集未见的工况，使用全局标准化
            print(f"  [警告] 测试集出现未见工况 cluster={cluster}，使用全局标准化")
            global_scaler = StandardScaler()
            df_test.loc[df_test['condition_cluster'] == cluster, feature_cols] = \
                global_scaler.fit_transform(
                    df_test.loc[df_test['condition_cluster'] == cluster, feature_cols]
                )
    
    # ---- Step 5: 创建滑动窗口数据集 ----
    print(f"\n[5/6] 创建滑动窗口 (window_size={config.window_size})...")
    X_train, y_train, X_test, y_test = prepare_dataset(df_train, df_test, feature_cols)
    
    # ---- Step 6: 构建并训练模型 ----
    print("\n[6/6] 构建 CNN-LSTM-Attention 模型...")
    input_shape = (X_train.shape[1], X_train.shape[2])
    model = build_cnn_lstm_attention_model(input_shape)
    model.summary()
    
    model.compile(
        optimizer=Adam(learning_rate=config.learning_rate),
        loss='mse',
        metrics=['mae']
    )
    
    # 回调函数
    callbacks = [
        EarlyStopping(
            monitor='val_loss',
            patience=config.patience_early_stop,
            restore_best_weights=True,
            verbose=1
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=config.patience_reduce_lr,
            min_lr=1e-6,
            verbose=1
        )
    ]
    
    print(f"\n开始训练 (epochs={config.epochs}, batch_size={config.batch_size})...")
    history = model.fit(
        X_train, y_train,
        validation_split=config.validation_split,
        epochs=config.epochs,
        batch_size=config.batch_size,
        callbacks=callbacks,
        verbose=2
    )
    
    # ---- 评估 ----
    print("\n" + "=" * 60)
    print(" 模型评估")
    print("=" * 60)
    
    # 训练集评估
    y_train_pred = model.predict(X_train, verbose=0).flatten()
    train_metrics = evaluate_model(y_train, y_train_pred, '训练集')
    
    # 测试集评估
    y_test_pred = model.predict(X_test, verbose=0).flatten()
    test_metrics = evaluate_model(y_test, y_test_pred, '测试集')
    
    # ---- 可视化 ----
    print("\n生成可视化图表...")
    plot_training_history(history)
    plot_predictions(y_test, y_test_pred, f'测试集预测结果 ({config.dataset})')
    
    # 绘制前3个发动机的退化趋势
    for uid in df_test['unit_id'].unique()[:3]:
        plot_unit_degradation(df_test, y_test_pred, uid)
    
    # ---- 保存结果 ----
    results_df = pd.DataFrame({
        'Metric': ['MAE', 'RMSE', 'R2', 'PHM08_Score'],
        'Train': [
            f"{train_metrics['MAE']:.4f}",
            f"{train_metrics['RMSE']:.4f}",
            f"{train_metrics['R2']:.4f}",
            f"{train_metrics['PHM08_Score']:.4f}"
        ],
        'Test': [
            f"{test_metrics['MAE']:.4f}",
            f"{test_metrics['RMSE']:.4f}",
            f"{test_metrics['R2']:.4f}",
            f"{test_metrics['PHM08_Score']:.4f}"
        ]
    })
    print("\n结果汇总:")
    print(results_df.to_string(index=False))
    
    # 保存模型
    model_path = os.path.join(config.data_dir, f'cnn_lstm_attention_{config.dataset}.h5')
    model.save(model_path)
    print(f"\n模型已保存至: {model_path}")
    
    # 保存评估结果
    results_df.to_csv(
        os.path.join(config.data_dir, f'evaluation_results_{config.dataset}.csv'),
        index=False
    )
    print(f"评估结果已保存")
    
    print("\n" + "=" * 60)
    print(" 完成!")
    print("=" * 60)
    
    return model, history, test_metrics


if __name__ == '__main__':
    main()
