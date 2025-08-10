import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

class TimeSeriesLiquidCell(nn.Module):
    """
    Liquid Neural Network cell optimized for time series
    """
    def __init__(self, input_size, hidden_size, num_steps=5):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_steps = num_steps
        
        # Core liquid dynamics
        self.W_rec = nn.Linear(hidden_size, hidden_size, bias=False)
        self.W_in = nn.Linear(input_size, hidden_size)
        self.bias = nn.Parameter(torch.zeros(hidden_size))
        
        # Adaptive time constants based on input volatility
        self.tau_base = nn.Parameter(torch.ones(hidden_size) * 2.0)
        self.volatility_gate = nn.Sequential(
            nn.Linear(input_size, hidden_size // 4),
            nn.ReLU(),
            nn.Linear(hidden_size // 4, hidden_size),
            nn.Sigmoid()
        )
        
        # Better initialization for time series
        nn.init.orthogonal_(self.W_rec.weight, gain=0.9)
        nn.init.xavier_normal_(self.W_in.weight, gain=1.2)
        
    def forward(self, x, h_prev=None, dt=0.1):
        batch_size, seq_len, _ = x.shape
        device = x.device
        
        if h_prev is None:
            h = torch.zeros(batch_size, self.hidden_size, device=device)
        else:
            h = h_prev
            
        outputs = []
        
        for t in range(seq_len):
            x_t = x[:, t, :]  # Current timestep input
            
            # Compute volatility-based time constants
            volatility = self.volatility_gate(x_t)
            # High volatility -> fast adaptation (small tau)
            # Low volatility -> slow adaptation (large tau)
            tau = self.tau_base * (0.2 + 1.8 * (1 - volatility))
            tau = torch.clamp(tau, 0.1, 10.0)
            
            # Liquid ODE integration
            for step in range(self.num_steps):
                recurrent = self.W_rec(h)
                input_contrib = self.W_in(x_t)
                
                activation_input = recurrent + input_contrib + self.bias
                target_state = torch.tanh(activation_input)
                
                # ODE: τ * dh/dt = -h + target_state
                dhdt = (-h + target_state) / tau
                
                # Adaptive step size based on state change magnitude
                step_magnitude = torch.norm(dhdt, dim=1, keepdim=True)
                adaptive_dt = dt / (1.0 + 0.2 * step_magnitude)
                
                h = h + adaptive_dt * dhdt
            
            outputs.append(h.unsqueeze(1))
        
        return torch.cat(outputs, dim=1), h

class MultiScaleLiquidForecaster(nn.Module):
    """
    Multi-scale liquid neural network for time series forecasting
    """
    def __init__(self, input_size, hidden_size=64, output_size=1, num_scales=3):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.num_scales = num_scales
        
        # Input projection
        self.input_proj = nn.Linear(input_size, hidden_size)
        
        # Multi-scale liquid cells with different time horizons
        self.liquid_cells = nn.ModuleList([
            TimeSeriesLiquidCell(
                hidden_size, 
                hidden_size, 
                num_steps=3 + 2*i  # 3, 5, 7 steps for different scales
            ) for i in range(num_scales)
        ])
        
        # Scale-specific feature extraction
        self.scale_projectors = nn.ModuleList([
            nn.Linear(hidden_size, hidden_size // num_scales)
            for _ in range(num_scales)
        ])
        
        # Fusion and prediction layers
        self.fusion = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, output_size)
        )
        
        # Uncertainty estimation (optional)
        self.uncertainty = nn.Linear(hidden_size, output_size)
        
    def forward(self, x, return_uncertainty=False):
        # x shape: (batch, seq_len, features)
        batch_size, seq_len, _ = x.shape
        
        # Project input
        x_proj = self.input_proj(x)
        
        # Multi-scale processing
        scale_outputs = []
        for i, (liquid_cell, projector) in enumerate(zip(self.liquid_cells, self.scale_projectors)):
            # Each scale processes the same input but with different dynamics
            scale_hidden, _ = liquid_cell(x_proj)
            
            # Take the last hidden state for this scale
            scale_final = scale_hidden[:, -1, :]  # (batch, hidden_size)
            scale_feature = projector(scale_final)  # (batch, hidden_size//num_scales)
            scale_outputs.append(scale_feature)
        
        # Fuse multi-scale features
        fused = torch.cat(scale_outputs, dim=-1)  # (batch, hidden_size)
        
        # Final prediction
        prediction = self.fusion(fused)
        
        if return_uncertainty:
            uncertainty = F.softplus(self.uncertainty(fused))
            return prediction, uncertainty
        
        return prediction

class TimeSeriesDataset:
    """
    Generate synthetic time series data for testing
    """
    @staticmethod
    def generate_complex_series(n_samples=1000, seq_length=50):
        """Generate complex synthetic time series with multiple patterns"""
        t = np.linspace(0, 4*np.pi, n_samples + seq_length)
        
        # Multiple components
        trend = 0.1 * t
        seasonal = 2 * np.sin(0.5 * t) + 0.5 * np.sin(2 * t)
        volatility_regime = 1 + 0.5 * np.sin(0.1 * t)  # Changing volatility
        noise = np.random.normal(0, 0.2 * volatility_regime, len(t))
        
        # Occasional spikes (like news events)
        spikes = np.random.exponential(0.1, len(t)) * np.random.choice([-1, 1], len(t))
        spike_mask = np.random.random(len(t)) < 0.05  # 5% chance of spike
        spikes = spikes * spike_mask
        
        series = trend + seasonal + noise + spikes
        
        # Create sequences
        X, y = [], []
        for i in range(n_samples):
            X.append(series[i:i+seq_length])
            y.append(series[i+seq_length])
        
        return np.array(X), np.array(y)
    
    @staticmethod
    def add_features(X):
        """Add technical indicators as features"""
        # X shape: (n_samples, seq_length)
        features = []
        
        for i in range(X.shape[0]):
            seq = X[i]
            
            # Original value
            values = seq.reshape(-1, 1)
            
            # Moving averages
            ma_5 = np.convolve(seq, np.ones(5)/5, mode='same').reshape(-1, 1)
            ma_10 = np.convolve(seq, np.ones(10)/10, mode='same').reshape(-1, 1)
            
            # Returns
            returns = np.diff(seq, prepend=seq[0]).reshape(-1, 1)
            
            # Volatility (rolling std)
            volatility = np.array([np.std(seq[max(0, j-5):j+1]) for j in range(len(seq))]).reshape(-1, 1)
            
            # Combine all features
            seq_features = np.concatenate([values, ma_5, ma_10, returns, volatility], axis=1)
            features.append(seq_features)
        
        return np.array(features)

def train_liquid_forecaster():
    """
    Train liquid neural network for time series forecasting
    """
    print("📈 LIQUID NEURAL NETWORK TIME SERIES FORECASTING")
    print("=" * 50)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Generate synthetic time series data
    print("Generating synthetic time series data...")
    X_raw, y_raw = TimeSeriesDataset.generate_complex_series(n_samples=2000, seq_length=30)
    
    # Add technical features
    X_features = TimeSeriesDataset.add_features(X_raw)
    
    # Convert to tensors
    X = torch.FloatTensor(X_features)
    y = torch.FloatTensor(y_raw).unsqueeze(-1)
    
    # Train/test split
    train_size = int(0.8 * len(X))
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]
    
    # Data loaders
    train_dataset = TensorDataset(X_train, y_train)
    test_dataset = TensorDataset(X_test, y_test)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    print(f"Data shape: {X.shape}")
    print(f"Features: {X.shape[2]} (value, MA5, MA10, returns, volatility)")
    print(f"Training samples: {len(X_train)}")
    print(f"Test samples: {len(X_test)}")
    
    # Model
    model = MultiScaleLiquidForecaster(
        input_size=X.shape[2],  # Number of features
        hidden_size=64,
        output_size=1,
        num_scales=3
    ).to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {total_params:,}")
    
    # Training setup
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, factor=0.5)
    criterion = nn.MSELoss()
    
    # Training loop
    print("\nTraining liquid forecaster...")
    train_losses = []
    val_losses = []
    
    for epoch in range(50):
        # Training
        model.train()
        epoch_train_loss = 0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            
            optimizer.zero_grad()
            predictions = model(X_batch)
            loss = criterion(predictions, y_batch)
            loss.backward()
            
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            epoch_train_loss += loss.item()
        
        # Validation
        model.eval()
        epoch_val_loss = 0
        with torch.no_grad():
            for X_batch, y_batch in test_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                predictions = model(X_batch)
                loss = criterion(predictions, y_batch)
                epoch_val_loss += loss.item()
        
        avg_train_loss = epoch_train_loss / len(train_loader)
        avg_val_loss = epoch_val_loss / len(test_loader)
        
        train_losses.append(avg_train_loss)
        val_losses.append(avg_val_loss)
        
        scheduler.step(avg_val_loss)
        
        if epoch % 10 == 0:
            print(f"Epoch {epoch:2d} - Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")
    
    # Generate predictions for visualization
    model.eval()
    with torch.no_grad():
        # Take a test sequence
        test_idx = 50
        test_input = X_test[test_idx:test_idx+1].to(device)
        test_true = y_test[test_idx:test_idx+1].to(device)
        
        # Multi-step prediction
        predictions = []
        current_input = test_input.clone()
        
        for step in range(20):  # Predict 20 steps ahead
            pred = model(current_input)
            predictions.append(pred.cpu().numpy()[0, 0])
            
            # Update input for next prediction (sliding window)
            new_features = np.array([[pred.cpu().numpy()[0, 0], 0, 0, 0, 0]])  # Simplified
            new_input = torch.FloatTensor(new_features).unsqueeze(0).to(device)
            
            # Slide window
            current_input = torch.cat([
                current_input[:, 1:, :],  # Remove first timestep
                new_input.unsqueeze(1)    # Add prediction as last timestep
            ], dim=1)
    
    # Visualization
    plt.figure(figsize=(15, 10))
    
    # Training curves
    plt.subplot(2, 2, 1)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.title('Training Curves')
    plt.xlabel('Epoch')
    plt.ylabel('MSE Loss')
    plt.legend()
    plt.yscale('log')
    
    # Original time series
    plt.subplot(2, 2, 2)
    plt.plot(X_raw[test_idx], label='Historical')
    plt.plot(range(30, 30 + len(predictions)), predictions, 'r--', label='Predictions')
    plt.axvline(x=30, color='k', linestyle=':', alpha=0.7, label='Prediction Start')
    plt.title('Multi-Step Forecasting')
    plt.xlabel('Time Steps')
    plt.ylabel('Value')
    plt.legend()
    
    # Feature importance (volatility adaptation)
    plt.subplot(2, 2, 3)
    sample_volatility = X_features[test_idx, :, -1]  # Last feature is volatility
    plt.plot(sample_volatility, label='Volatility')
    plt.title('Input Volatility (Drives Adaptive Time Constants)')
    plt.xlabel('Time Steps')
    plt.ylabel('Volatility')
    plt.legend()
    
    # Prediction vs actual
    plt.subplot(2, 2, 4)
    all_predictions = []
    all_targets = []
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch = X_batch.to(device)
            pred = model(X_batch).cpu().numpy()
            all_predictions.extend(pred.flatten())
            all_targets.extend(y_batch.numpy().flatten())
    
    plt.scatter(all_targets[:200], all_predictions[:200], alpha=0.6)
    plt.plot([min(all_targets), max(all_targets)], [min(all_targets), max(all_targets)], 'r--')
    plt.title('Predictions vs Actual')
    plt.xlabel('Actual Values')
    plt.ylabel('Predicted Values')
    
    plt.tight_layout()
    plt.savefig('liquid_time_series_results.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    # Calculate metrics
    mse = np.mean((np.array(all_predictions) - np.array(all_targets))**2)
    mae = np.mean(np.abs(np.array(all_predictions) - np.array(all_targets)))
    
    print(f"\n✅ Liquid Forecasting Complete!")
    print(f"Final MSE: {mse:.4f}")
    print(f"Final MAE: {mae:.4f}")
    print("\nKey Features:")
    print("🌊 Adaptive time constants based on volatility")
    print("📊 Multi-scale temporal processing")
    print("🎯 Continuous-time ODE dynamics")
    print("📈 Multi-step ahead forecasting")
    
    return model, (train_losses, val_losses)

if __name__ == "__main__":
    train_liquid_forecaster()
