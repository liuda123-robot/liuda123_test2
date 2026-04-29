import os
import json
import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report, f1_score, accuracy_score, balanced_accuracy_score, confusion_matrix
from ecg_model import build_ecg_model

# 路径设置
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
DATA_PATH = os.path.join(PROJECT_ROOT, "data", "processed")
SAVE_DIR = os.path.join(CURRENT_DIR, "saved_model")
os.makedirs(SAVE_DIR, exist_ok=True)

#设置种子，保证每次运行结果相同
SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)

VALID_LABELS = ['N', 'V', 'S']
VAL_RECORDS = ['102', '114']#验证集(参与多轮训练评估并调整模型)
TEST_RECORDS = ['104', '201']#测试集(最终测试模型)
# 数据加载
def load_data():
    X = np.load(os.path.join(DATA_PATH, "X_all.npy")).astype(np.float32)#获取心拍片段
    y = np.load(os.path.join(DATA_PATH, "y_all.npy")).astype(np.int32)#获取标签
    record_ids = np.load(os.path.join(DATA_PATH, "record_ids_all.npy"), allow_pickle=True)#获取对应记录名+
    X = X[..., np.newaxis]#修改维度，保证适配Conv1D输入(单导联channels=1)
    return X, y, record_ids

def build_record_split(record_ids):
    unique_records = sorted(np.unique(record_ids).tolist())
    reserved = set(VAL_RECORDS) | set(TEST_RECORDS)#并集
    train_records = sorted(list(set(unique_records) - reserved))#从待训练集中去除验证集和测试集
    return {"train_records": train_records, "val_records": VAL_RECORDS, "test_records": TEST_RECORDS}

#生成掩码
def mask_by_records(record_ids, selected_records):
    return np.isin(record_ids, np.array(selected_records))

# 弱增强函数
def augment_class(X, y, target_class, copies=1, noise_std=0.004, scale_jitter=0.01):
    idx = np.where(y == target_class)[0]
    if len(idx) == 0 or copies <= 0:
        return X, y
    base = X[idx]
    xs, ys = [X], [y]
    for _ in range(copies):
        # 添加高斯噪声，让模型对噪声鲁棒
        noise = np.random.normal(0, noise_std, size=base.shape).astype(np.float32)
        scale = np.random.uniform(1.0 - scale_jitter, 1.0 + scale_jitter, size=(len(base),1,1)).astype(np.float32)
        xs.append(base*scale + noise)
        ys.append(np.full((len(base),), target_class, dtype=y.dtype))
    X_out = np.concatenate(xs, axis=0)# 拼接原始和增强数据
    y_out = np.concatenate(ys, axis=0)
    perm = np.random.permutation(len(X_out))
    return X_out[perm], y_out[perm]

# 保存切分后的数据
def save_split_data(X_train, X_val, X_test, y_train, y_val, y_test, split_info):
    np.save(os.path.join(DATA_PATH, "X_train.npy"), X_train)
    np.save(os.path.join(DATA_PATH, "X_val.npy"), X_val)
    np.save(os.path.join(DATA_PATH, "X_test.npy"), X_test)
    np.save(os.path.join(DATA_PATH, "y_train.npy"), y_train)
    np.save(os.path.join(DATA_PATH, "y_val.npy"), y_val)
    np.save(os.path.join(DATA_PATH, "y_test.npy"), y_test)
    with open(os.path.join(DATA_PATH, "split_records.json"), "w", encoding="utf-8") as f:
        json.dump(split_info, f, ensure_ascii=False, indent=2)

#评估函数
def evaluate_and_print(model, X, y, title="Evaluation"):
    y_prob = model.predict(X, verbose=0)
    y_pred = np.argmax(y_prob, axis=1)
    print(f"\n{title}")
    print(f"Accuracy: {accuracy_score(y, y_pred):.4f}, Balanced Accuracy: {balanced_accuracy_score(y, y_pred):.4f}, F1 Macro: {f1_score(y, y_pred, average='macro'):.4f}")
    print(classification_report(y, y_pred, target_names=VALID_LABELS, zero_division=0))

# 主函数
def main():#整体获取心拍数据
    X, y, record_ids = load_data()
    split_info = build_record_split(record_ids)
    train_mask = mask_by_records(record_ids, split_info["train_records"])
    val_mask = mask_by_records(record_ids, split_info["val_records"])
    test_mask = mask_by_records(record_ids, split_info["test_records"])
    X_train, y_train = X[train_mask], y[train_mask]
    X_val, y_val = X[val_mask], y[val_mask]
    X_test, y_test = X[test_mask], y[test_mask]

    # 弱增强：N/V/S，N，V增强一倍，S增强2倍
    X_train, y_train = augment_class(X_train, y_train, 0, copies=1)
    X_train, y_train = augment_class(X_train, y_train, 1, copies=1)
    X_train, y_train = augment_class(X_train, y_train, 2, copies=2)

    save_split_data(X_train, X_val, X_test, y_train, y_val, y_test, split_info)

    model = build_ecg_model(input_shape=(360,1), num_classes=3, transformer_blocks=2, embed_dim=128, num_heads=4, ff_dim=256, dropout=0.2)
    model.compile(optimizer=tf.keras.optimizers.AdamW(3e-4, weight_decay=1e-4),#学习率 3e-4，权重衰减 1e-4，有助于正则化。
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(),
                  metrics=[tf.keras.metrics.SparseCategoricalAccuracy(name='accuracy')])

    best_model_path = os.path.join(SAVE_DIR, "best_model.keras")
    final_model_path = os.path.join(SAVE_DIR, "ecg_1dcnn_se_transformer.keras")

    class_weight = {0:1.05, 1:1.3, 2:1.3}#给V类和S类更高的权重，提高对其的观众

    #配置回调函数
    callbacks = [
        tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=12, restore_best_weights=True, verbose=1),#验证损失连续 12 轮不下降则停止训练
        tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=4, min_lr=1e-6, verbose=1),#验证损失 4 轮不下降则将学习率减半，帮助收敛
        tf.keras.callbacks.ModelCheckpoint(filepath=best_model_path, monitor='val_loss', save_best_only=True, verbose=1)#保存验证损失最低的模型（best_model.keras）
    ]

    model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=60, batch_size=64, class_weight=class_weight, callbacks=callbacks, verbose=1)#训练 60 个 epoch，批次大小 64
    model.save(final_model_path)

    evaluate_and_print(model, X_val, y_val, "Validation Results")
    evaluate_and_print(model, X_test, y_test, "Independent Test Results")

if __name__ == "__main__":
    os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
    main()
    
    
'''
在每次训练轮次中：
loss 和 accuracy：来自训练集。
它们是在当前 epoch 内，模型对训练集的所有批次（batch）进行前向传播和反向传播后，计算得到的平均损失和平均准确率。这些值反映模型对已见数据的拟合程度。

val_loss 和 val_accuracy：来自验证集。
它们是在当前 epoch 结束后，模型在验证集上（不进行反向传播，仅前向传播）计算得到的损失和准确率。这些值用于监控模型是否过拟合、是否还在泛化，并作为早停、学习率衰减、模型保存的依据。
'''