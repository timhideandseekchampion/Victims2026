"""FULL deep-learning battery (PyTorch + TensorFlow/Keras).

All strictly walk-forward (train on first 80%, test on last 20%). We report
OOS R^2 and directional accuracy for each architecture. Autoencoders report
reconstruction R^2 (can the return cross-section be compressed = structure?).

PyTorch:   MLP, GRU, 1D-CNN, Autoencoder
TensorFlow: LSTM, GRU, Conv1D, Autoencoder, tiny Transformer
"""
import warnings, os
warnings.filterwarnings("ignore")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"; os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import numpy as np
from sklearn.metrics import r2_score, accuracy_score
from common import load, log_returns, section, Recorder

rec = Recorder("deep_learning_full")
df, tickers = load()
rets = log_returns(df)
R = rets.values
T, N = R.shape
mkt = R.mean(axis=1)

# pooled tabular panel (5 own lags + mkt + vol) -> next return
Xp, yp = [], []
for i in range(N):
    r = R[:, i]
    for t in range(5, T-1):
        Xp.append(list(r[t-5:t][::-1])+[mkt[t-1], r[t-5:t].std()]); yp.append(r[t])
Xp = np.array(Xp, np.float32); yp = np.array(yp, np.float32)
sp = int(len(Xp)*0.8)
mu, sd = Xp[:sp].mean(0), Xp[:sp].std(0)+1e-8
Xn = (Xp-mu)/sd

# pooled sequence panel (last 10 days own+mkt) -> next return
seqlen = 10
Xs, ys = [], []
for i in range(N):
    r = R[:, i]
    for t in range(seqlen, T-1):
        Xs.append(np.column_stack([r[t-seqlen:t], mkt[t-seqlen:t]])); ys.append(r[t])
Xs = np.array(Xs, np.float32); ys = np.array(ys, np.float32)
sps = int(len(Xs)*0.8)

def report(lib, arch, ytrue, ypred):
    r2 = r2_score(ytrue, ypred); acc = accuracy_score(ytrue>0, ypred.ravel()>0)
    rec.add(lib, "dl_predict", f"{arch}_oos_r2", "pooled", r2, np.nan, note=f"dir_acc={acc:.4f}")
    print(f"  {lib}/{arch}: OOS R2={r2:.5f}  dir_acc={acc:.4f}")
    return r2, acc

section("14A. PyTorch models")
import torch, torch.nn as nn
torch.manual_seed(0)
Xt = torch.tensor(Xn); yt = torch.tensor(yp).view(-1,1)
Xseq = torch.tensor(Xs); yseq = torch.tensor(ys).view(-1,1)

def train_torch(net, X, y, sp, epochs=60, lr=1e-3, flatten=True):
    opt = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=1e-4); lf = nn.MSELoss()
    for _ in range(epochs):
        net.train(); opt.zero_grad(); loss = lf(net(X[:sp]), y[:sp]); loss.backward(); opt.step()
    net.eval()
    with torch.no_grad():
        out = net(X[sp:]).numpy()
    return out.ravel() if flatten else out

# MLP
mlp = nn.Sequential(nn.Linear(7,32),nn.ReLU(),nn.Linear(32,16),nn.ReLU(),nn.Linear(16,1))
report("pytorch","MLP", yp[sp:], train_torch(mlp, Xt, yt, sp))

# GRU
class GRUNet(nn.Module):
    def __init__(s): super().__init__(); s.g=nn.GRU(2,16,batch_first=True); s.f=nn.Linear(16,1)
    def forward(s,x): o,_=s.g(x); return s.f(o[:,-1])
report("pytorch","GRU", ys[sps:], train_torch(GRUNet(), Xseq, yseq, sps, epochs=40))

# 1D-CNN
class CNN(nn.Module):
    def __init__(s):
        super().__init__(); s.c=nn.Conv1d(2,8,3,padding=1); s.p=nn.AdaptiveAvgPool1d(1); s.f=nn.Linear(8,1)
    def forward(s,x): x=x.transpose(1,2); x=torch.relu(s.c(x)); return s.f(s.p(x).squeeze(-1))
report("pytorch","Conv1D", ys[sps:], train_torch(CNN(), Xseq, yseq, sps, epochs=40))

# Autoencoder on the 51-dim daily return cross-section (reconstruction R^2)
Rt = torch.tensor((R - R.mean(0))/(R.std(0)+1e-8), dtype=torch.float32)
ae = nn.Sequential(nn.Linear(N,8),nn.ReLU(),nn.Linear(8,N))
spa = int(T*0.8)
recon = train_torch(ae, Rt, Rt, spa, epochs=200, lr=5e-3, flatten=False)
r2ae = r2_score(Rt[spa:].numpy(), recon)
rec.add("pytorch","dl_autoencoder","AE_recon_r2_8dim","cross-section", r2ae, np.nan)
print(f"  pytorch/Autoencoder(8-dim): OOS reconstruction R2={r2ae:.4f} "
      f"(high => cross-section compresses to few factors)")

section("14B. TensorFlow / Keras models")
import tensorflow as tf
tf.random.set_seed(0)
def keras_seq(build, X, y, sp, epochs=40):
    m = build(); m.compile(optimizer=tf.keras.optimizers.Adam(1e-3), loss="mse")
    m.fit(X[:sp], y[:sp], epochs=epochs, batch_size=64, verbose=0)
    return m.predict(X[sp:], verbose=0).ravel()

# LSTM
report("tensorflow","LSTM", ys[sps:], keras_seq(
    lambda: tf.keras.Sequential([tf.keras.layers.Input((seqlen,2)),
             tf.keras.layers.LSTM(16), tf.keras.layers.Dense(1)]), Xs, ys, sps))
# GRU
report("tensorflow","GRU", ys[sps:], keras_seq(
    lambda: tf.keras.Sequential([tf.keras.layers.Input((seqlen,2)),
             tf.keras.layers.GRU(16), tf.keras.layers.Dense(1)]), Xs, ys, sps))
# Conv1D
report("tensorflow","Conv1D", ys[sps:], keras_seq(
    lambda: tf.keras.Sequential([tf.keras.layers.Input((seqlen,2)),
             tf.keras.layers.Conv1D(8,3,activation="relu",padding="same"),
             tf.keras.layers.GlobalAveragePooling1D(), tf.keras.layers.Dense(1)]), Xs, ys, sps))
# tiny Transformer
def build_tx():
    inp = tf.keras.layers.Input((seqlen,2))
    x = tf.keras.layers.Dense(16)(inp)
    a = tf.keras.layers.MultiHeadAttention(num_heads=2, key_dim=8)(x, x)
    x = tf.keras.layers.LayerNormalization()(x + a)
    x = tf.keras.layers.GlobalAveragePooling1D()(x)
    return tf.keras.Model(inp, tf.keras.layers.Dense(1)(x))
report("tensorflow","Transformer", ys[sps:], keras_seq(build_tx, Xs, ys, sps))
# MLP on tabular
report("tensorflow","MLP", yp[sp:], keras_seq(
    lambda: tf.keras.Sequential([tf.keras.layers.Input((7,)),
             tf.keras.layers.Dense(32,activation="relu"),
             tf.keras.layers.Dense(16,activation="relu"), tf.keras.layers.Dense(1)]),
    Xn, yp, sp))
# Autoencoder (cross-section)
Rzt = (R - R.mean(0))/(R.std(0)+1e-8)
aem = tf.keras.Sequential([tf.keras.layers.Input((N,)),
        tf.keras.layers.Dense(8,activation="relu"), tf.keras.layers.Dense(N)])
aem.compile(optimizer=tf.keras.optimizers.Adam(5e-3), loss="mse")
aem.fit(Rzt[:spa], Rzt[:spa], epochs=200, batch_size=32, verbose=0)
r2ae2 = r2_score(Rzt[spa:], aem.predict(Rzt[spa:], verbose=0))
rec.add("tensorflow","dl_autoencoder","AE_recon_r2_8dim","cross-section", r2ae2, np.nan)
print(f"  tensorflow/Autoencoder(8-dim): OOS reconstruction R2={r2ae2:.4f}")

rec.save()
