# -------------------------
# Transformer-based model
# -------------------------
def get_sinusoidal_positional_encoding(seq_len, d_model):
    """Return (1, seq_len, d_model) positional encoding"""
    position = np.arange(seq_len)[:, np.newaxis]  # (seq_len, 1)
    div_term = np.exp(np.arange(0, d_model, 2) * -(np.log(10000.0) / d_model))
    pe = np.zeros((seq_len, d_model))
    pe[:, 0::2] = np.sin(position * div_term)
    pe[:, 1::2] = np.cos(position * div_term)
    pe = pe[np.newaxis, ...].astype(np.float32)  # (1, seq_len, d_model)
    return tf.constant(pe)

class TransformerEncoderLayer(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads, ff_dim, dropout=0.1, **kwargs):
        super().__init__(**kwargs)
        self.mha = tf.keras.layers.MultiHeadAttention(num_heads=num_heads, key_dim=d_model//num_heads, dropout=dropout)
        self.norm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = tf.keras.layers.Dropout(dropout)

        self.ffn = tf.keras.Sequential([
            tf.keras.layers.Dense(ff_dim, activation='relu'),
            tf.keras.layers.Dense(d_model),
        ])
        self.norm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.dropout2 = tf.keras.layers.Dropout(dropout)

    def call(self, x, training=False, mask=None):
        attn_out = self.mha(x, x, attention_mask=mask, training=training)
        attn_out = self.dropout1(attn_out, training=training)
        out1 = self.norm1(x + attn_out)

        ffn_out = self.ffn(out1)
        ffn_out = self.dropout2(ffn_out, training=training)
        return self.norm2(out1 + ffn_out)

# Hyperparameters you can tune
d_model = 128        # embedding dim
num_heads = 8
ff_dim = 256
num_layers = 3
dropout = 0.1

num_lidar_range_values = lidar.shape[1]  # existing variable in script
# Input: (batch, seq_len, 1)
inp = tf.keras.Input(shape=(num_lidar_range_values, 1), name='lidar_input')

# Project scalar range -> d_model embedding per token
x = tf.keras.layers.Dense(d_model)(inp)  # now (batch, seq_len, d_model)

# add sinusoidal positional encodings (non-trainable)
pos_enc = get_sinusoidal_positional_encoding(num_lidar_range_values, d_model)
x = x + pos_enc  # broadcasting over batch

# optional small dropout
x = tf.keras.layers.Dropout(dropout)(x)

# stack Transformer encoder layers
for i in range(num_layers):
    x = TransformerEncoderLayer(d_model=d_model, num_heads=num_heads, ff_dim=ff_dim, dropout=dropout, name=f'trans_encoder_{i}')(x)

# Pooling: global avg over sequence (you can also use attention pooling or CLS token)
x = tf.keras.layers.GlobalAveragePooling1D()(x)  # (batch, d_model)

# small dense head
x = tf.keras.layers.Dense(128, activation='relu')(x)
x = tf.keras.layers.Dropout(0.2)(x)
x = tf.keras.layers.Dense(64, activation='relu')(x)
x = tf.keras.layers.Dense(16, activation='relu')(x)

# Output: (steering, speed)
# Note: your script used activation='tanh' previously. Keep tanh for steering; consider sigmoid for speed.
# To preserve prior behavior we use tanh for both, but I'll comment an alternative below.
out = tf.keras.layers.Dense(2, activation='tanh', name='control_output')(x)

# If you prefer steering in [-1,1] and speed in [0,1], you can instead:
# steering = tf.keras.layers.Dense(1, activation='tanh', name='steer')(x)
# speed_out = tf.keras.layers.Dense(1, activation='sigmoid', name='speed')(x)
# out = tf.keras.layers.Concatenate(name='control_output')([steering, speed_out])

model = tf.keras.Model(inputs=inp, outputs=out, name='lidar_transformer_model')

# Compile (keeps your optimizer & custom_loss)
optimizer = tf.keras.optimizers.Adam(lr)
model.compile(optimizer=optimizer, loss=custom_loss)
model.summary()
