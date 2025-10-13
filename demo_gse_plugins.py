#!/usr/bin/env python
"""
Demo script để minh họa hoạt động của GSE Balance & Constrained Plugin algorithms
Tạo synthetic data để hiểu rõ flow và input/output variables
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Tạo synthetic data để demo
np.random.seed(42)

class GSEPluginDemo:
    def __init__(self):
        self.num_classes = 100
        self.num_groups = 2  # head vs tail
        self.num_experts = 3
        self.n_samples_s1 = 2500  # tuneV split
        self.n_samples_s2 = 2500  # val_lt split
        
        # Grouping: classes 0-33 = head, classes 34-99 = tail
        self.class_to_group = np.array([0 if i < 34 else 1 for i in range(self.num_classes)])
        
        # Tạo synthetic mixture posteriors và labels
        self.eta_s1, self.y_s1 = self._generate_mixture_posteriors(self.n_samples_s1, "S1")
        self.eta_s2, self.y_s2 = self._generate_mixture_posteriors(self.n_samples_s2, "S2")
        
        print("🎯 GSE Plugin Demo Initialized")
        print(f"   - Classes: {self.num_classes} (34 head + 66 tail)")
        print(f"   - Experts: {self.num_experts}")
        print(f"   - S1 (tuneV): {self.n_samples_s1} samples")
        print(f"   - S2 (val_lt): {self.n_samples_s2} samples")
        print()

    def _generate_mixture_posteriors(self, n_samples, split_name):
        """Tạo synthetic mixture posteriors η̃(x) và labels y"""
        
        # Long-tail distribution cho labels
        probs = np.array([max(0.1, 1.0 - i*0.01) for i in range(self.num_classes)])
        probs = probs / probs.sum()
        
        y = np.random.choice(self.num_classes, size=n_samples, p=probs)
        
        # Mixture posteriors: higher confidence for head classes
        eta = np.random.dirichlet(np.ones(self.num_classes) * 0.5, size=n_samples)
        
        # Boost confidence for true labels
        for i in range(n_samples):
            true_class = y[i]
            group = self.class_to_group[true_class]
            
            # Head classes có confidence cao hơn
            boost = 3.0 if group == 0 else 1.5
            eta[i, true_class] += boost
            
        # Normalize lại
        eta = eta / eta.sum(axis=1, keepdims=True)
        
        return eta, y

    def compute_raw_margin(self, eta, alpha, mu):
        """Tính raw margin scores"""
        n_samples = eta.shape[0]
        
        # score = max_y α_{g(y)} * η̃_y
        alpha_per_class = alpha[self.class_to_group]
        scores = (alpha_per_class * eta).max(axis=1)
        
        # threshold_term = Σ_y (1/α_{g(y)} - μ_{g(y)}) * η̃_y  
        coeff = 1.0 / alpha_per_class - mu[self.class_to_group]
        threshold_term = (coeff * eta).sum(axis=1)
        
        return scores - threshold_term

    def compute_group_metrics(self, eta, y, alpha, mu, t):
        """Tính per-group error và coverage"""
        raw_margins = self.compute_raw_margin(eta, alpha, mu)
        accepted = raw_margins >= t
        
        # Predictions
        alpha_per_class = alpha[self.class_to_group]
        preds = (alpha_per_class * eta).argmax(axis=1)
        
        # Per-group metrics
        e_k = np.zeros(self.num_groups)
        cov_k = np.zeros(self.num_groups)
        
        y_groups = self.class_to_group[y]
        
        for k in range(self.num_groups):
            group_mask = (y_groups == k)
            n_k = group_mask.sum()
            
            if n_k > 0:
                # Coverage
                cov_k[k] = (accepted & group_mask).sum() / n_k
                
                # Error (trên accepted samples)
                accepted_group = accepted & group_mask
                if accepted_group.sum() > 0:
                    correct_group = (preds == y) & accepted_group
                    e_k[k] = 1.0 - correct_group.sum() / accepted_group.sum()
                else:
                    e_k[k] = 1.0
            else:
                cov_k[k] = 0.0
                e_k[k] = 1.0
                
        return e_k, cov_k, accepted, preds

    def gse_balanced_demo(self):
        """Demo GSE-Balanced Plugin algorithm"""
        print("=" * 50)
        print("🔵 GSE-BALANCED PLUGIN DEMO")
        print("=" * 50)
        
        # Initialize parameters
        alpha = np.ones(self.num_groups)
        mu = np.zeros(self.num_groups) 
        t = 0.0
        
        # Configuration
        T = 20  # Reduced for demo
        lambda_grid = np.linspace(-1.0, 1.0, 11)
        ema_alpha = 0.7
        
        print(f"📊 Configuration:")
        print(f"   - Iterations: {T}")
        print(f"   - Lambda grid: {lambda_grid}")
        print(f"   - EMA alpha: {ema_alpha}")
        print()
        
        history = {'balanced_error': [], 'worst_error': [], 'objective': []}
        
        for iter in range(T):
            # 1. Fit threshold on S1
            raw_s1 = self.compute_raw_margin(self.eta_s1, alpha, mu)
            target_coverage = 0.7
            t = np.quantile(raw_s1, 1.0 - target_coverage)
            
            # 2. Compute metrics on S2
            e_k, cov_k, accepted, preds = self.compute_group_metrics(
                self.eta_s2, self.y_s2, alpha, mu, t
            )
            
            balanced_error = e_k.mean()
            worst_error = e_k.max()
            objective = balanced_error  # balanced objective
            
            history['balanced_error'].append(balanced_error)
            history['worst_error'].append(worst_error)
            history['objective'].append(objective)
            
            print(f"[{iter+1:2d}] bal={balanced_error:.4f}, worst={worst_error:.4f}, "
                  f"cov={cov_k.mean():.3f}, obj={objective:.4f}")
            
            # 3. Update alpha (fixed-point)
            if iter < T - 1:
                y_groups_s1 = self.class_to_group[self.y_s1]
                alpha_new = np.zeros(self.num_groups)
                
                for k in range(self.num_groups):
                    group_mask = (y_groups_s1 == k)
                    if group_mask.sum() > 0:
                        raw_s1 = self.compute_raw_margin(self.eta_s1, alpha, mu)
                        accepted_s1 = (raw_s1 >= t)
                        acceptance_rate = (accepted_s1 & group_mask).sum() / group_mask.sum()
                        alpha_new[k] = acceptance_rate + 0.001  # Small regularization
                    else:
                        alpha_new[k] = 1.0
                
                # EMA smoothing
                alpha = ema_alpha * alpha + (1 - ema_alpha) * alpha_new
                alpha = np.clip(alpha, 0.75, 1.35)  # Project to valid range
                
                # 4. Update mu (grid search)
                best_mu = mu.copy()
                best_obj = objective
                
                for lam in lambda_grid:
                    mu_candidate = np.array([+lam/2, -lam/2])  # For K=2
                    
                    e_k_cand, _, _, _ = self.compute_group_metrics(
                        self.eta_s2, self.y_s2, alpha, mu_candidate, t
                    )
                    obj_cand = e_k_cand.mean()  # balanced objective
                    
                    if obj_cand < best_obj:
                        best_obj = obj_cand
                        best_mu = mu_candidate
                
                mu = 0.8 * mu + 0.2 * best_mu  # EMA update
        
        print()
        print(f"✅ Final Results (GSE-Balanced):")
        print(f"   α* = [{alpha[0]:.4f}, {alpha[1]:.4f}]")
        print(f"   μ* = [{mu[0]:.4f}, {mu[1]:.4f}]")
        print(f"   t* = {t:.4f}")
        print(f"   Balanced Error = {balanced_error:.4f}")
        print(f"   Worst Error = {worst_error:.4f}")
        print(f"   Coverage = {cov_k.mean():.3f}")
        print(f"   Per-group errors: [{e_k[0]:.4f}, {e_k[1]:.4f}]")
        print()
        
        return alpha, mu, t, history

    def gse_constrained_demo(self):
        """Demo GSE-Constrained Plugin algorithm"""
        print("=" * 50)
        print("🔒 GSE-CONSTRAINED PLUGIN DEMO")
        print("=" * 50)
        
        # Initialize parameters
        alpha = np.ones(self.num_groups)
        mu = np.zeros(self.num_groups)
        t = 0.0
        
        # Initialize dual variables
        lambda_cov = 0.0
        nu = np.zeros(self.num_groups)
        
        # Configuration  
        T = 20
        tau = 0.65  # Coverage constraint
        delta_multiplier = 1.3
        eta_dual = 0.05
        lambda_grid = np.linspace(-1.0, 1.0, 11)
        warmup_iters = 5
        
        print(f"📊 Configuration:")
        print(f"   - Coverage constraint: τ ≥ {tau}")
        print(f"   - Delta multiplier: {delta_multiplier}")
        print(f"   - Dual step size: {eta_dual}")
        print(f"   - Warmup iterations: {warmup_iters}")
        print()
        
        # Initial delta
        e_k_init, _, _, _ = self.compute_group_metrics(
            self.eta_s2, self.y_s2, alpha, mu, t
        )
        delta = delta_multiplier * e_k_init.mean()
        print(f"Initial δ = {delta:.3f}")
        print()
        
        history = {
            'balanced_error': [], 'worst_error': [], 'coverage': [],
            'lagrangian': [], 'lambda_cov': [], 'nu_max': []
        }
        
        for iter in range(T):
            # 1. Fit threshold
            raw_s1 = self.compute_raw_margin(self.eta_s1, alpha, mu)
            t_new = np.quantile(raw_s1, 1.0 - tau)
            t = 0.7 * t + 0.3 * t_new  # Smooth update
            
            # 2. Compute metrics
            e_k, cov_k, _, _ = self.compute_group_metrics(
                self.eta_s2, self.y_s2, alpha, mu, t
            )
            
            balanced_error = e_k.mean()
            worst_error = e_k.max()
            avg_coverage = cov_k.mean()
            
            # 3. Lagrangian
            coverage_violation = tau - avg_coverage
            fairness_violations = e_k - delta
            
            lagrangian = balanced_error
            if iter >= warmup_iters:
                lagrangian += lambda_cov * coverage_violation
                lagrangian += np.sum(nu * np.maximum(fairness_violations, 0))
            
            history['balanced_error'].append(balanced_error)
            history['worst_error'].append(worst_error)
            history['coverage'].append(avg_coverage)
            history['lagrangian'].append(lagrangian)
            history['lambda_cov'].append(lambda_cov)
            history['nu_max'].append(nu.max())
            
            constraint_info = ""
            if iter >= warmup_iters:
                constraint_info = f", λ={lambda_cov:.3f}, ν_max={nu.max():.3f}"
            
            print(f"[{iter+1:2d}] bal={balanced_error:.4f}, worst={worst_error:.4f}, "
                  f"cov={avg_coverage:.3f}, L={lagrangian:.4f}{constraint_info}")
            
            # 4. Update primal (simplified grid search)
            if iter < T - 1:
                best_alpha_new = alpha.copy()
                best_mu_new = mu.copy()
                best_loss = lagrangian
                
                for lam in lambda_grid:
                    mu_candidate = np.array([+lam/2, -lam/2])
                    
                    # Simple alpha update
                    alpha_candidate = np.clip(alpha * (1 + 0.01 * np.random.randn(2)), 0.75, 1.35)
                    
                    e_k_cand, cov_k_cand, _, _ = self.compute_group_metrics(
                        self.eta_s2, self.y_s2, alpha_candidate, mu_candidate, t
                    )
                    
                    loss_cand = e_k_cand.mean()
                    if iter >= warmup_iters:
                        cov_violation = tau - cov_k_cand.mean()
                        fair_violations = e_k_cand - delta
                        loss_cand += lambda_cov * cov_violation
                        loss_cand += np.sum(nu * np.maximum(fair_violations, 0))
                    
                    if loss_cand < best_loss:
                        best_loss = loss_cand
                        best_alpha_new = alpha_candidate
                        best_mu_new = mu_candidate
                
                alpha = 0.8 * alpha + 0.2 * best_alpha_new
                mu = 0.8 * mu + 0.2 * best_mu_new
                alpha = np.clip(alpha, 0.75, 1.35)
            
            # 5. Update dual variables
            if iter >= warmup_iters:
                lambda_cov = max(0.0, lambda_cov + eta_dual * coverage_violation)
                nu = np.maximum(0.0, nu + eta_dual * fairness_violations)
        
        print()
        print(f"✅ Final Results (GSE-Constrained):")
        print(f"   α* = [{alpha[0]:.4f}, {alpha[1]:.4f}]")
        print(f"   μ* = [{mu[0]:.4f}, {mu[1]:.4f}]")
        print(f"   t* = {t:.4f}")
        print(f"   Balanced Error = {balanced_error:.4f}")
        print(f"   Worst Error = {worst_error:.4f}")
        print(f"   Coverage = {avg_coverage:.3f} (≥ {tau} {'✓' if avg_coverage >= tau else '✗'})")
        print(f"   Per-group errors: [{e_k[0]:.4f}, {e_k[1]:.4f}]")
        print(f"   Fairness satisfied: {all(e_k <= delta + 0.01)} (δ={delta:.3f})")
        print()
        
        return alpha, mu, t, history

    def plot_comparison(self, hist_balanced, hist_constrained):
        """Vẽ biểu đồ so sánh hai thuật toán"""
        try:
            fig, axes = plt.subplots(2, 2, figsize=(12, 8))
            
            # Plot 1: Balanced Error
            axes[0,0].plot(hist_balanced['balanced_error'], label='Balanced Plugin', color='blue')
            axes[0,0].plot(hist_constrained['balanced_error'], label='Constrained Plugin', color='red')
            axes[0,0].set_xlabel('Iteration')
            axes[0,0].set_ylabel('Balanced Error')
            axes[0,0].set_title('Balanced Error Evolution')
            axes[0,0].legend()
            axes[0,0].grid(True)
            
            # Plot 2: Worst Error
            axes[0,1].plot(hist_balanced['worst_error'], label='Balanced Plugin', color='blue')
            axes[0,1].plot(hist_constrained['worst_error'], label='Constrained Plugin', color='red')
            axes[0,1].set_xlabel('Iteration')
            axes[0,1].set_ylabel('Worst Error')
            axes[0,1].set_title('Worst Error Evolution')
            axes[0,1].legend()
            axes[0,1].grid(True)
            
            # Plot 3: Coverage
            axes[1,0].plot(hist_balanced.get('coverage', [0]*20), label='Balanced Plugin', color='blue')
            axes[1,0].plot(hist_constrained['coverage'], label='Constrained Plugin', color='red')
            axes[1,0].axhline(y=0.65, color='green', linestyle='--', label='Target τ=0.65')
            axes[1,0].set_xlabel('Iteration')
            axes[1,0].set_ylabel('Coverage')
            axes[1,0].set_title('Coverage Evolution')
            axes[1,0].legend()
            axes[1,0].grid(True)
            
            # Plot 4: Objective/Lagrangian
            axes[1,1].plot(hist_balanced['objective'], label='Balanced Objective', color='blue')
            axes[1,1].plot(hist_constrained['lagrangian'], label='Constrained Lagrangian', color='red')
            axes[1,1].set_xlabel('Iteration')
            axes[1,1].set_ylabel('Objective')
            axes[1,1].set_title('Objective Evolution')
            axes[1,1].legend()
            axes[1,1].grid(True)
            
            plt.tight_layout()
            
            # Tạo thư mục nếu chưa có
            Path("./demo_outputs").mkdir(exist_ok=True)
            plt.savefig("./demo_outputs/gse_plugins_comparison.png", dpi=150, bbox_inches='tight')
            plt.close()
            
            print("📊 Saved comparison plot to: ./demo_outputs/gse_plugins_comparison.png")
            
        except Exception as e:
            print(f"⚠️ Could not generate plot: {e}")

def main():
    """Main demo function"""
    print("🚀 AR-GSE Plugin Algorithms Demo")
    print("=" * 60)
    print("This demo illustrates the GSE-Balanced and GSE-Constrained")
    print("plugin algorithms with synthetic data to show input/output")
    print("variables and optimization flow.")
    print("=" * 60)
    print()
    
    # Initialize demo
    demo = GSEPluginDemo()
    
    # Run GSE-Balanced Plugin
    alpha_bal, mu_bal, t_bal, hist_bal = demo.gse_balanced_demo()
    
    # Run GSE-Constrained Plugin
    alpha_con, mu_con, t_con, hist_con = demo.gse_constrained_demo()
    
    # Generate comparison plot
    demo.plot_comparison(hist_bal, hist_con)
    
    print("🎯 SUMMARY COMPARISON")
    print("=" * 50)
    print(f"{'Metric':<20} {'Balanced':<15} {'Constrained':<15}")
    print("-" * 50)
    print(f"{'Final α[0]':<20} {alpha_bal[0]:<15.4f} {alpha_con[0]:<15.4f}")
    print(f"{'Final α[1]':<20} {alpha_bal[1]:<15.4f} {alpha_con[1]:<15.4f}")
    print(f"{'Final μ[0]':<20} {mu_bal[0]:<15.4f} {mu_con[0]:<15.4f}")
    print(f"{'Final μ[1]':<20} {mu_bal[1]:<15.4f} {mu_con[1]:<15.4f}")
    print(f"{'Final threshold':<20} {t_bal:<15.4f} {t_con:<15.4f}")
    print()
    print("✅ Demo completed successfully!")
    print("📖 See docs/ensemble_training.md for detailed documentation")

if __name__ == "__main__":
    main()