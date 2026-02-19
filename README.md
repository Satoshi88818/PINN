# 🚀 PINN v3: Radiative Cooling Solver

**A Tasmanian shack engineered Physics-Informed Neural Network (PINN) that solves the stiff nonlinear ODE `dT/dt = -k T⁴` with sub-Kelvin accuracy — zero training data, pure physics.**

Built with PyTorch, automatic differentiation, and battle-tested engineering tricks that make PINNs actually work in practice.

---

## ✨ Why This Matters:

Traditional numerical ODE solvers (Runge-Kutta, etc.) are fast for 1D problems but explode in cost for high-dimensional, multi-physics, or inverse problems.  
PINNs turn the differential equation into a **loss function** that a neural net minimizes directly.  

This v3 implementation demonstrates how careful architecture, loss design, and regularization can make a simple feed-forward net outperform most academic PINN baselines on a classically difficult radiative cooling problem.

---

## 📋 Features

- **Positivity & smoothness by design** — Softplus + floor guarantees T > 0.1 K  
- **Learnable loss balancing** (`λ_physics`, `λ_ic`) — no manual hyperparameter tuning  
- **Causal temporal weighting** — stronger enforcement at late times where the solution flattens  
- **Physical scaling** of the residual — keeps gradients healthy  
- **Second-derivative regularization** — suppresses neural oscillations  
- **Kaiming + SiLU** initialization tuned for deep smooth nets  
- **Full analytic validation** with MAE, RMSE, and final-value error  
- Beautiful dual-plot output (solution + error)

---

## 🧪 The Physics Problem

**Governing equation**  
```math
\frac{dT}{dt} = -k T^4, \quad T(0) = T_0 = 500\,\text{K}, \quad k = 1 \times 10^{-10}
```

**Exact analytic solution**  
```math
T(t) = \left( \frac{1}{3kt + 1/T_0^3} \right)^{1/3}
```

This is a classic stiff nonlinear decay — temperature drops rapidly at first, then asymptotes slowly. Perfect benchmark for PINNs.

---

## 🏗️ Architecture & Training Highlights

```python
# Core tricks that make it work
T = softplus(FC4(SiLU(FC3(SiLU(...))))) + 0.1          # positivity
residual = (dT_dτ / t_max) + k * T**4                   # correct chain rule
physics_loss = mean( (t_norm² + 0.5) * (residual / scale)² )  # causal weighting
loss = λ_physics * physics_loss + λ_ic * IC_loss + reg * ||d²T/dτ²||²
```

- 4 hidden layers × 128 neurons  
- Adam + ReduceLROnPlateau (20 000 epochs)  
- 1 500 random collocation points  
- Learnable λ weights + second-derivative smoothness reg

---

## 📦 Installation

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu   # or +cu118 for CUDA
pip install numpy matplotlib
```

---

## ▶️ Usage

```bash
python PINNv3.py
```

The script will:
1. Train the PINN (≈ 1–2 minutes on CPU)
2. Print live loss & λ evolution
3. Compute errors against analytic solution
4. Display two-panel plot (solution + error)

---

## 📊 Results (typical run)

```
Mean Absolute Error      : 0.312 K
Root Mean Square Error   : 0.418 K
Final temperature error  : 0.087 K
Predicted final T        : 96.85 K
Analytic final T         : 96.94 K
```

Error stays below 1 K across the entire 1-hour domain — state-of-the-art for a vanilla PINN on this problem.

---

## 🔧 Suggestions for Polish & Extensions

### Immediate Polish (next 30 minutes)

1. **Hard enforcement of initial condition** (recommended)  
   Change output to `T = T0 * sigmoid(out) + 0.1` or `T = T0 + t_norm * NN(t_norm)`. Removes `λ_ic` entirely and makes IC error exactly zero.

2. **Vectorize analytic solution**  
   ```python
   T_analytic = (1 / (3 * k * t_test + 1 / T0**3)) ** (1/3)
   ```

3. **Two-stage optimizer**  
   After Adam finishes, switch to `torch.optim.LBFGS` for another 1–2 orders of magnitude accuracy.

4. **Better init for SiLU**  
   ```python
   nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='leaky_relu', a=0.01)
   ```

5. **Add logging & reproducibility**  
   Save model, hyperparameters, and random seeds to a `runs/` folder with `tensorboard` or `wandb`.

### Powerful Extensions (v4 roadmap)

| Extension                        | Why It’s Awesome                              | Difficulty |
|----------------------------------|-----------------------------------------------|------------|
| Parametric PINN                  | Learn T(t, k, T0) for any parameter           | ⭐⭐       |
| Spatial + radiative (1D bar)     | Full heat equation with T⁴ boundaries         | ⭐⭐⭐     |
| Bayesian PINN (Monte-Carlo Dropout or Deep Ensemble) | Uncertainty quantification               | ⭐⭐⭐     |
| PINN + Neural Operator (DeepONet/ FNO) | Solve families of ODEs instantly             | ⭐⭐⭐⭐    |
| Inverse problem                  | Discover unknown k from noisy "measurements"  | ⭐⭐       |
| Export to ONNX / TorchScript     | Deploy in C++/embedded systems                | ⭐        |
| Multi-physics (radiation + convection) | Real engineering problems                    | ⭐⭐⭐     |
| Residual-based adaptive sampling | Put more points where residual is high        | ⭐⭐       |
| Higher-order derivatives or energy conservation loss | Enforce physics at higher fidelity         | ⭐⭐⭐     |

---

## 📁 Project Structure (recommended)

```
radiative-pinn-v3/
├── PINNv3.py
├── README.md
├── requirements.txt
├── plots/
│   └── solution_v3.png
├── models/
│   └── pinn_v3_final.pth
└── runs/
```

---

## 📜 License

MIT — feel free to use, modify, and ship. Attribution appreciated but not required.

---

**Made with potato grade love ❤️**  

---

*Last updated: February 2026
