import tensorflow as tf
from tensorflow.keras import layers, models

#SE注意力，用于通过学习通道重要性，分配给不同通道注意力权重
class SEBlock1D(layers.Layer):
    def __init__(self, channels, ratio=8, **kwargs):
        super().__init__(**kwargs)
        self.channels = channels#通道数
        self.ratio = ratio#压缩的比例
        hidden_units = max(channels // ratio, 4)#中间瓶颈层维度被压缩成原通道数的 1/8，但至少保留 4 个神经元，减少参数量，避免 SE 本身太重。

        self.global_avg = layers.GlobalAveragePooling1D()#实现 Squeeze 操作，每个通道用一个标量描述全局特征。
        self.fc1 = layers.Dense(hidden_units, activation='relu')#降维
        self.fc2 = layers.Dense(channels, activation='sigmoid')#升维
        self.reshape = layers.Reshape((1, channels))

    def call(self, inputs):
        x = self.global_avg(inputs)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.reshape(x)
        return inputs * x

    def get_config(self):
        config = super().get_config()
        config.update({
            'channels': self.channels,
            'ratio': self.ratio
        })
        return config
    
#先让 CNN 做局部特征抽取，再让 Transformer 在高层特征上建模全局关系。
#Transformer经典架构
class TransformerEncoder1D(layers.Layer):
    def __init__(self, embed_dim, num_heads, ff_dim, dropout=0.1, **kwargs):
        super().__init__(**kwargs)
        self.embed_dim = embed_dim#嵌入模型维度/宽度
        self.num_heads = num_heads#注意力头数
        self.ff_dim = ff_dim#前馈网络隐藏层维度
        self.dropout_rate = dropout

        self.att = layers.MultiHeadAttention(
            num_heads=num_heads,
            key_dim=embed_dim // num_heads,# 每个头的键维度
            dropout=dropout#防止过拟合
        )
        self.ffn = models.Sequential([
            layers.Dense(ff_dim, activation='gelu'),# 升维
            layers.Dropout(dropout),# 正则化
            layers.Dense(embed_dim)# 降维回原始维度
        ])

        self.norm1 = layers.LayerNormalization(epsilon=1e-6)
        self.norm2 = layers.LayerNormalization(epsilon=1e-6)
        self.drop1 = layers.Dropout(dropout)
        self.drop2 = layers.Dropout(dropout)

    def call(self, inputs, training=None):
        # 1. 多头自注意力 + 残差 + 层归一化（Pre-LN）
        attn_output = self.att(inputs, inputs, training=training)
        attn_output = self.drop1(attn_output, training=training)
        out1 = self.norm1(inputs + attn_output)

        # 2. 前馈网络 + 残差 + 层归一化
        ffn_output = self.ffn(out1, training=training)
        ffn_output = self.drop2(ffn_output, training=training)
        return self.norm2(out1 + ffn_output)

    def get_config(self):
        config = super().get_config()
        config.update({
            'embed_dim': self.embed_dim,
            'num_heads': self.num_heads,
            'ff_dim': self.ff_dim,
            'dropout': self.dropout_rate
        })
        return config

#位置编码
class PositionalEncoding1D(layers.Layer):
    def __init__(self, max_len, embed_dim, **kwargs):
        super().__init__(**kwargs)
        self.max_len = max_len
        self.embed_dim = embed_dim

    def build(self, input_shape):
        position = tf.range(self.max_len, dtype=tf.float32)[:, tf.newaxis]
        div_term = tf.exp(
            tf.range(0, self.embed_dim, 2, dtype=tf.float32)
            * -(tf.math.log(10000.0) / self.embed_dim)
        )

        pe_sin = tf.sin(position * div_term)
        pe_cos = tf.cos(position * div_term)

        if self.embed_dim % 2 == 0:
            pe = tf.reshape(
                tf.stack([pe_sin, pe_cos], axis=-1),
                (self.max_len, self.embed_dim)
            )
        else:
            pe_cos = pe_cos[:, :tf.shape(pe_sin)[1] - 1]
            pe = tf.reshape(
                tf.stack([pe_sin[:, :tf.shape(pe_cos)[1]], pe_cos], axis=-1),
                (self.max_len, self.embed_dim - 1)
            )
            pe = tf.concat([pe, tf.zeros((self.max_len, 1))], axis=-1)

        self.pe = pe[tf.newaxis, ...]
        super().build(input_shape)

    def call(self, inputs):
        seq_len = tf.shape(inputs)[1]
        return inputs + self.pe[:, :seq_len, :]

    def get_config(self):
        config = super().get_config()
        config.update({
            'max_len': self.max_len,
            'embed_dim': self.embed_dim
        })
        return config

#损失函数，Focal loss 会降低“已经很好分”的样本的影响，把更多训练注意力放到难样本和少数类上。
class FocalLoss(tf.keras.losses.Loss):
    def __init__(self, alpha=None, gamma=2.0, num_classes=3, label_smoothing=0.0,
                 reduction=tf.keras.losses.Reduction.SUM_OVER_BATCH_SIZE,
                 name='focal_loss'):
        super().__init__(reduction=reduction, name=name)
        self.alpha = alpha
        self.gamma = gamma
        self.num_classes = num_classes
        self.label_smoothing = label_smoothing

    def call(self, y_true, y_pred):
        y_true = tf.cast(tf.reshape(y_true, [-1]), tf.int32)
        y_true = tf.one_hot(y_true, depth=self.num_classes, dtype=tf.float32)

        if self.label_smoothing > 0:
            smooth = self.label_smoothing
            y_true = y_true * (1.0 - smooth) + smooth / self.num_classes

        y_pred = tf.clip_by_value(y_pred, 1e-7, 1.0 - 1e-7)

        cross_entropy = -y_true * tf.math.log(y_pred)
        focal_weight = tf.pow(1.0 - y_pred, self.gamma)

        if self.alpha is None:
            alpha = tf.ones((self.num_classes,), dtype=tf.float32)
        else:
            alpha = tf.constant(self.alpha, dtype=tf.float32)

        alpha_factor = y_true * alpha
        loss = alpha_factor * focal_weight * cross_entropy
        loss = tf.reduce_sum(loss, axis=-1)
        return tf.reduce_mean(loss)

    def get_config(self):
        config = super().get_config()
        config.update({
            'alpha': self.alpha,
            'gamma': self.gamma,
            'num_classes': self.num_classes,
            'label_smoothing': self.label_smoothing
        })
        return config

#卷积+SE+池化块
def conv_se_block(x, filters, kernel_size=7, pool_size=2, dropout=0.15):
    #一层卷积
    x = layers.Conv1D(filters, kernel_size, padding='same', use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)

    #二层卷积
    x = layers.Conv1D(filters, kernel_size, padding='same', use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)

    x = SEBlock1D(filters)(x)
    x = layers.MaxPooling1D(pool_size=pool_size)(x)#池化
    x = layers.Dropout(dropout)(x)
    return x


def build_ecg_model(
    input_shape=(360, 1),
    num_classes=3,
    transformer_blocks=2,
    embed_dim=128,
    num_heads=4,
    ff_dim=256,
    dropout=0.2
):
    inputs = layers.Input(shape=input_shape, name='ecg_input')

    #通道数逐层增加，卷积核大小逐渐减小，同时池化降维
    x = conv_se_block(inputs, 32, kernel_size=9, pool_size=2, dropout=0.10)#一层卷积块
    x = conv_se_block(x, 64, kernel_size=7, pool_size=2, dropout=0.15)#二层卷积块
    x = conv_se_block(x, embed_dim, kernel_size=5, pool_size=2, dropout=0.20)#三层卷积块

    x = PositionalEncoding1D(max_len=45, embed_dim=embed_dim)(x)#三次pool_size=2卷积池化后由360->180->90->45

    for _ in range(transformer_blocks):
        x = TransformerEncoder1D(
            embed_dim=embed_dim,
            num_heads=num_heads,
            ff_dim=ff_dim,
            dropout=dropout
        )(x)

    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dense(128, activation='relu')(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(num_classes, activation='softmax', name='classifier')(x)

    model = tf.keras.Model(inputs, outputs, name='ECG_1DCNN_SE_Transformer')
    return model


def calibrate_probs(probs, temperature=1.2):#温度校准，防止模型过度自信
    import numpy as np

    probs = np.asarray(probs, dtype=np.float64)
    probs = np.clip(probs, 1e-8, 1.0)
    logits = np.log(probs)
    logits = logits / temperature
    logits = logits - np.max(logits, axis=-1, keepdims=True)
    exp_logits = np.exp(logits)
    calibrated = exp_logits / np.sum(exp_logits, axis=-1, keepdims=True)
    return calibrated
