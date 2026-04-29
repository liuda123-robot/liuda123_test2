import os
import json
import numpy as np
import wfdb

from scipy.signal import butter, filtfilt, medfilt, resample_poly
from collections import defaultdict
from math import gcd

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, 'data', 'raw')#输入路径
OUTPUT_PATH = os.path.join(BASE_DIR, 'data', 'processed')#输出路径

SEGMENT_LENGTH = 360
TARGET_FS = 360
NOISE_THRESHOLD = 0.1

# 全局上限：最后兜底
MAX_SAMPLES_PER_CLASS = {
    'N': 8000,#限制N类参与训练的数量，对多数类进行欠采样
    'V': 3000,
    'S': 3000
}

# 每条记录每类上限：真正改善多记录覆盖,改变了之前代码依次获取数据的缺陷，尽量涉及所有参与训练的训练集
MAX_SAMPLES_PER_RECORD = {
    'N': 300,
    'V': 200,
    'S': 200
}

VALID_LABELS = ['N', 'V', 'S']

LABEL_MAPPING = {
    'N': 'N', 'L': 'N', 'R': 'N', 'e': 'N', 'j': 'N',
    'V': 'V', 'E': 'V',
    'A': 'S', 'a': 'S', 'J': 'S', 'S': 'S'
}

os.makedirs(OUTPUT_PATH, exist_ok=True)

def resample_to_target_fs(signal_data, orig_fs, target_fs=360):#采用MIT数据集进行训练，固定采样率为360Hz,进行重采样
    signal_data = np.asarray(signal_data).astype(np.float32)
    if int(orig_fs) == int(target_fs):
        return signal_data

    g = gcd(int(orig_fs), int(target_fs))
    up = int(target_fs) // g
    down = int(orig_fs) // g
    return resample_poly(signal_data, up, down).astype(np.float32)


def remap_annotation_samples(annotation_samples, orig_fs, target_fs=360):#R峰重映射
    if int(orig_fs) == int(target_fs):
        return np.asarray(annotation_samples, dtype=np.int32)
    ratio = float(target_fs) / float(orig_fs)
    return np.round(np.asarray(annotation_samples) * ratio).astype(np.int32)


def denoise_ecg(signal_data, fs=360):#去噪
    b, a = butter(4, [0.5, 40], btype='bandpass', fs=fs)#只保留0.5-40Hz的波段，滤除基线漂移（<0.5 Hz）和工频/肌电噪声（>40 Hz）
    filtered = filtfilt(b, a, signal_data)#零相位滤波
    baseline = medfilt(filtered, kernel_size=201)
    denoised = filtered - baseline
    denoised[np.abs(denoised) < NOISE_THRESHOLD] = 0#小于0.1的样点置0
    return denoised


def normalize_signal(signal):#归一化
    mean = np.mean(signal)
    std = np.std(signal)
    return (signal - mean) / (std + 1e-8)


def segment_beat(signal, r_peak, half_len=180):#心拍切分，以 R 峰为中心，左右各取 180 点（共 360 点，约 1 秒）。此长度能覆盖一个完整心拍（包括前后部分 P、T 波），又避免引入相邻心拍。
    start = int(r_peak) - half_len
    end = int(r_peak) + half_len
    if start < 0 or end > len(signal):
        return None
    seg = signal[start:end]
    if len(seg) != 2 * half_len:
        return None
    return seg


def discover_record_ids(data_dir):
    record_ids = sorted(list(set(
        f.split('.')[0] for f in os.listdir(data_dir) if f.endswith('.dat')
    )))
    return record_ids


def preprocess_data():#主预处理函数
    record_ids = discover_record_ids(DATA_PATH)

    if len(record_ids) == 0:
        raise FileNotFoundError(f'未在 {DATA_PATH} 找到 .dat 数据文件')

    class_counter = defaultdict(int)
    skipped_counter = defaultdict(int)

    all_segments = []
    all_labels = []
    all_record_ids = []
    all_raw_symbols = []
    all_source_fs = []

    for rec in record_ids:
        try:
            record = wfdb.rdrecord(os.path.join(DATA_PATH, rec))#读取 .dat 和 .hea 文件
            annotation = wfdb.rdann(os.path.join(DATA_PATH, rec), 'atr')#读取 .atr 注释文件，返回 R 峰位置 (annotation.sample) 和对应的标注符号

            if getattr(record, 'p_signal', None) is None:
                print(f'跳过 {rec}: 无 p_signal')
                continue

            raw_signal = record.p_signal[:, 0].astype(np.float32)#只取第一导联
            orig_fs = float(record.fs) if getattr(record, 'fs', None) is not None else 360.0#获取原始采样率

            signal = resample_to_target_fs(raw_signal, orig_fs=orig_fs, target_fs=TARGET_FS)#重采样
            ann_samples = remap_annotation_samples(annotation.sample, orig_fs=orig_fs, target_fs=TARGET_FS)#R峰重定位

            signal = denoise_ecg(signal, fs=TARGET_FS)#去噪+归一化
            signal = normalize_signal(signal)

            # 每条记录内部单独计数
            record_class_counter = defaultdict(int)

            # 先收集当前记录全部候选样本，再按顺序筛
            candidate_items = []
            for i, r_peak in enumerate(ann_samples):
                raw_label = annotation.symbol[i]
                label = LABEL_MAPPING.get(raw_label, None)
                if label not in VALID_LABELS:
                    skipped_counter['invalid_label'] += 1
                    continue

                seg = segment_beat(signal, r_peak, half_len=SEGMENT_LENGTH // 2)
                if seg is None or len(seg) != SEGMENT_LENGTH:
                    skipped_counter['boundary_or_length'] += 1
                    continue

                candidate_items.append((seg.astype(np.float32), label, raw_label))

            # 打乱当前记录候选样本，减少同一时段连续片段偏置，降低过拟合风险
            rng = np.random.default_rng(seed=42 + int(rec))
            rng.shuffle(candidate_items)

            #双层上限，全局上限和每条记录的上限
            for seg, label, raw_label in candidate_items:
                if class_counter[label] >= MAX_SAMPLES_PER_CLASS[label]:
                    skipped_counter[f'class_cap_{label}'] += 1
                    continue

                if record_class_counter[label] >= MAX_SAMPLES_PER_RECORD[label]:
                    skipped_counter[f'record_cap_{label}'] += 1
                    continue

                all_segments.append(seg)
                all_labels.append(VALID_LABELS.index(label))
                all_record_ids.append(rec)
                all_raw_symbols.append(raw_label)
                all_source_fs.append(orig_fs)

                class_counter[label] += 1
                record_class_counter[label] += 1

            print(f'处理完成 {rec} | 当前统计: {dict(class_counter)} | 当前记录统计: {dict(record_class_counter)}')

        except Exception as e:
            print(f'跳过 {rec}: {str(e)}')

    if len(all_segments) == 0:
        raise RuntimeError('未提取到任何有效心拍片段，请检查数据路径、标注或标签映射')

    X_all = np.array(all_segments, dtype=np.float32)
    y_all = np.array(all_labels, dtype=np.int32)
    record_ids_all = np.array(all_record_ids)
    raw_symbols_all = np.array(all_raw_symbols, dtype=object)
    source_fs_all = np.array(all_source_fs, dtype=np.float32)

    np.save(os.path.join(OUTPUT_PATH, 'X_all.npy'), X_all)
    np.save(os.path.join(OUTPUT_PATH, 'y_all.npy'), y_all)
    np.save(os.path.join(OUTPUT_PATH, 'record_ids_all.npy'), record_ids_all)
    np.save(os.path.join(OUTPUT_PATH, 'raw_symbols_all.npy'), raw_symbols_all, allow_pickle=True)
    np.save(os.path.join(OUTPUT_PATH, 'source_fs_all.npy'), source_fs_all)

    class_distribution = {
        VALID_LABELS[i]: int(np.sum(y_all == i))
        for i in range(len(VALID_LABELS))
    }

    record_distribution = {
        rec: int(np.sum(record_ids_all == rec))
        for rec in sorted(np.unique(record_ids_all))
    }

    meta = {
        'total_samples': int(len(X_all)),
        'segment_length': SEGMENT_LENGTH,
        'target_fs': TARGET_FS,
        'label_order': VALID_LABELS,
        'class_distribution': class_distribution,
        'record_count': int(len(np.unique(record_ids_all))),
        'record_distribution': record_distribution,
        'skipped_counter': dict(skipped_counter),
        'max_samples_per_class': MAX_SAMPLES_PER_CLASS,
        'max_samples_per_record': MAX_SAMPLES_PER_RECORD
    }

    np.save(os.path.join(OUTPUT_PATH, 'dataset_meta.npy'), meta, allow_pickle=True)

    with open(os.path.join(OUTPUT_PATH, 'dataset_meta.json'), 'w', encoding='utf-8') as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    print('\n预处理全部完成')
    print(f'总样本数: {len(X_all)}')
    print(f'类别分布: {class_distribution}')
    print(f'记录数: {len(np.unique(record_ids_all))}')
    print(f'跳过统计: {dict(skipped_counter)}')


if __name__ == '__main__':
    preprocess_data()