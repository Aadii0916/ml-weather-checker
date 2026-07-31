import os
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier, GradientBoostingRegressor
from sklearn.metrics import accuracy_score, classification_report, mean_squared_error, r2_score

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

from dataset_generator import generate_synthetic_weather_data

# Define PyTorch Multi-Task Weather Neural Network
class PyTorchWeatherNN(nn.Module):
    def __init__(self, input_dim, num_classes):
        super(PyTorchWeatherNN, self).__init__()
        
        # Shared feature extraction backbone
        self.shared = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        
        # Classification Head (Weather Condition)
        self.clf_head = nn.Sequential(
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, num_classes)
        )
        
        # Regression Head (Next Temp prediction)
        self.reg_head = nn.Sequential(
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1)
        )

    def forward(self, x):
        features = self.shared(x)
        logits = self.clf_head(features)
        temp_pred = self.reg_head(features)
        return logits, temp_pred

def train_and_save_models():
    base_dir = os.path.dirname(__file__)
    data_dir = os.path.join(base_dir, 'data')
    models_dir = os.path.join(base_dir, 'models')
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(data_dir, exist_ok=True)
    
    csv_path = os.path.join(data_dir, 'weather_dataset.csv')
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
    else:
        df = generate_synthetic_weather_data()
        df.to_csv(csv_path, index=False)
        
    feature_cols = ['month', 'hour', 'temperature_c', 'humidity_pct', 'pressure_hpa', 
                    'wind_speed_kmh', 'cloud_cover_pct', 'uv_index']
    
    X = df[feature_cols].values
    y_cls_raw = df['weather_condition'].values
    y_reg = df['next_temp_c'].values
    
    # Encoders & Scalers
    label_encoder = LabelEncoder()
    y_cls = label_encoder.fit_transform(y_cls_raw)
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Save preprocessing objects
    joblib.dump(scaler, os.path.join(models_dir, 'scaler.joblib'))
    joblib.dump(label_encoder, os.path.join(models_dir, 'label_encoder.joblib'))
    
    # Train-test split
    X_train, X_test, y_cls_train, y_cls_test, y_reg_train, y_reg_test = train_test_split(
        X_scaled, y_cls, y_reg, test_size=0.2, random_state=42, stratify=y_cls
    )
    
    # -------------------------------------------------------------
    # 1. Scikit-Learn Machine Learning Models
    # -------------------------------------------------------------
    print("--- Training Scikit-Learn Models ---")
    rf_clf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_clf.fit(X_train, y_cls_train)
    
    gb_reg = GradientBoostingRegressor(n_estimators=100, random_state=42)
    gb_reg.fit(X_train, y_reg_train)
    
    rf_acc = accuracy_score(y_cls_test, rf_clf.predict(X_test))
    gb_r2 = r2_score(y_reg_test, gb_reg.predict(X_test))
    
    joblib.dump(rf_clf, os.path.join(models_dir, 'rf_classifier.joblib'))
    joblib.dump(gb_reg, os.path.join(models_dir, 'gb_regressor.joblib'))
    
    print(f"Random Forest Accuracy: {rf_acc * 100:.2f}%")
    print(f"Gradient Boosting R2 Score: {gb_r2:.4f}")
    
    # -------------------------------------------------------------
    # 2. PyTorch Deep Learning Model
    # -------------------------------------------------------------
    print("--- Training PyTorch Deep Learning Model ---")
    train_dataset = TensorDataset(
        torch.tensor(X_train, dtype=torch.float32),
        torch.tensor(y_cls_train, dtype=torch.long),
        torch.tensor(y_reg_train, dtype=torch.float32).unsqueeze(1)
    )
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    
    num_classes = len(label_encoder.classes_)
    nn_model = PyTorchWeatherNN(input_dim=len(feature_cols), num_classes=num_classes)
    
    criterion_clf = nn.CrossEntropyLoss()
    criterion_reg = nn.MSELoss()
    optimizer = optim.Adam(nn_model.parameters(), lr=0.005, weight_decay=1e-4)
    
    epochs = 40
    history = {'epoch': [], 'loss': [], 'clf_loss': [], 'reg_loss': [], 'accuracy': []}
    
    for epoch in range(1, epochs + 1):
        nn_model.train()
        total_loss = 0.0
        total_clf_loss = 0.0
        total_reg_loss = 0.0
        correct = 0
        total_samples = 0
        
        for bx, by_cls, by_reg in train_loader:
            optimizer.zero_grad()
            logits, temp_pred = nn_model(bx)
            
            loss_c = criterion_clf(logits, by_cls)
            loss_r = criterion_reg(temp_pred, by_reg)
            loss = loss_c + 0.5 * loss_r
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item() * bx.size(0)
            total_clf_loss += loss_c.item() * bx.size(0)
            total_reg_loss += loss_r.item() * bx.size(0)
            
            _, preds = torch.max(logits, 1)
            correct += (preds == by_cls).sum().item()
            total_samples += bx.size(0)
            
        epoch_loss = total_loss / total_samples
        epoch_acc = correct / total_samples
        
        history['epoch'].append(epoch)
        history['loss'].append(epoch_loss)
        history['clf_loss'].append(total_clf_loss / total_samples)
        history['reg_loss'].append(total_reg_loss / total_samples)
        history['accuracy'].append(epoch_acc)
        
        if epoch % 10 == 0 or epoch == 1:
            print(f"Epoch {epoch}/{epochs} | Loss: {epoch_loss:.4f} | Accuracy: {epoch_acc*100:.2f}%")
            
    # Evaluation on PyTorch Test set
    nn_model.eval()
    with torch.no_grad():
        test_x_t = torch.tensor(X_test, dtype=torch.float32)
        test_logits, test_reg = nn_model(test_x_t)
        _, test_preds = torch.max(test_logits, 1)
        nn_acc = accuracy_score(y_cls_test, test_preds.numpy())
        
    print(f"PyTorch Neural Network Final Test Accuracy: {nn_acc * 100:.2f}%")
    
    # Save PyTorch Model weights & history
    torch.save(nn_model.state_dict(), os.path.join(models_dir, 'pytorch_weather_net.pth'))
    joblib.dump(history, os.path.join(models_dir, 'dl_history.joblib'))
    
    print("All ML & Deep Learning models trained and saved successfully!")
    return {
        'rf_acc': rf_acc,
        'gb_r2': gb_r2,
        'nn_acc': nn_acc,
        'history': history
    }

if __name__ == '__main__':
    train_and_save_models()
