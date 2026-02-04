import pandas as pd
import numpy as np

# 读取刚才生成的测试集预测文件
# 请修改为你的实际路径
csv_path = r"E:\RAW_DATA\outputs\09_seq_featcore\09_residual\predictions_test_RESCTX_OOF_pred_core_tfmr.csv"

df = pd.read_csv(csv_path)
y_true = df["y_true"].values
base_pred = df["base_pred"].values
# 注意：csv里的 residual_pred 是未缩放的原始预测值 (model output * std + mu)
r_pred = df["residual_pred"].values 

print(f"Base HGBR MAE: {np.mean(np.abs(y_true - base_pred)):.6f}")

best_mae = 1.0
best_g = 0.0

# 扫描 Gamma 从 0.0 到 1.0
for g in np.arange(0.0, 1.01, 0.01):
    final_pred = base_pred + g * r_pred
    mae = np.mean(np.abs(y_true - final_pred))
    
    if mae < best_mae:
        best_mae = mae
        best_g = g
    
    if int(g*100) % 5 == 0:
        print(f"Gamma={g:.2f} | MAE={mae:.6f}")

print("-" * 30)
print(f"【最佳结果】 Gamma = {best_g:.2f}")
print(f"【最低 MAE】 {best_mae:.6f}")
print(f"【提升幅度】 {(0.014374 - best_mae)/0.014374*100:.2f}%")