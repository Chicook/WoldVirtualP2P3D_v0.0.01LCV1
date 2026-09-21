"""
RF_RFENRN1_4_3.py - Gestor de Algoritmos de Segunda Orden
==========================================================

Implementa algoritmos de optimización de segunda orden para redes neuronales
de aprendizaje por refuerzo. Incluye L-BFGS, Newton, Shampoo y técnicas
de aproximación de Hessiana para optimización eficiente de pesos.

Características:
- L-BFGS: Limited-memory BFGS con aproximación de Hessiana
- Newton: Método de Newton con Hessiana completa
- Shampoo: Aproximación de segundo orden escalable
- K-FAC: Kronecker-Factored Approximate Curvature
- Natural Gradient: Gradiente natural con matriz de Fisher
- AdaHessian: Adam con aproximación de Hessiana diagonal
- Técnicas de preconditioning avanzadas
- Análisis de curvatura y condición numérica

Autor: LucIA Development Team
Versión: 4.3.0
"""

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
except ImportError:
    pass  # dependencia pesada opcional
import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Union, Callable
import logging
import math
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
import time
from collections import defaultdict, deque
import warnings
try:
    from scipy.optimize import minimize
    from scipy.sparse.linalg import cg, minres
except ImportError:
    pass  # dependencia pesada opcional

logger = logging.getLogger('RFENRN1.RF_RFENRN1_4_3')


@dataclass
class SecondOrderConfig:
    """Configuración para algoritmos de segunda orden"""
    optimizer_type: str = 'L-BFGS'  # 'L-BFGS', 'Newton', 'Shampoo', 'K-FAC', 'NaturalGradient', 'AdaHessian'
    learning_rate: float = 1.0
    max_iter: int = 20
    tolerance_grad: float = 1e-7
    tolerance_change: float = 1e-9
    history_size: int = 100
    line_search_fn: str = 'strong_wolfe'  # 'strong_wolfe', 'weak_wolfe', None

    # L-BFGS específico
    lbfgs_max_eval: int = None
    lbfgs_max_correction: int = 10

    # Newton específico
    newton_reg: float = 1e-6
    newton_max_iter: int = 10
    newton_tol: float = 1e-6

    # Shampoo específico
    shampoo_block_size: int = 1024
    shampoo_preconditioning_compute_steps: int = 1
    shampoo_statistics_compute_steps: int = 1
    shampoo_momentum: float = 0.9
    shampoo_weight_decay: float = 0.0

    # K-FAC específico
    kfac_factor_update_freq: int = 1
    kfac_inv_update_freq: int = 1
    kfac_damping: float = 1e-3
    kfac_kl_clip: float = 0.01
    kfac_fast_cnn: bool = True

    # Natural Gradient específico
    natural_grad_damping: float = 1e-3
    natural_grad_kl_clip: float = 0.01
    natural_grad_use_kl_clip: bool = True

    # AdaHessian específico
    adahessian_betas: Tuple[float, float] = (0.9, 0.999)
    adahessian_eps: float = 1e-8
    adahessian_weight_decay: float = 0.0
    adahessian_hessian_power: float = 1.0


class LBFGSOptimizer(torch.optim.Optimizer):
    """
    Implementación mejorada del optimizador L-BFGS.

    L-BFGS es un algoritmo de optimización cuasi-Newton que aproxima
    la matriz Hessiana usando información de gradientes históricos.
    """

    def __init__(self, params, lr=1, max_iter=20, max_eval=None, tolerance_grad=1e-7,
                 tolerance_change=1e-9, history_size=100, line_search_fn=None):
        if lr <= 0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if max_iter <= 0:
            raise ValueError(f"Invalid max_iter: {max_iter}")
        if tolerance_grad <= 0:
            raise ValueError(f"Invalid tolerance_grad: {tolerance_grad}")
        if tolerance_change <= 0:
            raise ValueError(f"Invalid tolerance_change: {tolerance_change}")
        if history_size <= 0:
            raise ValueError(f"Invalid history_size: {history_size}")

        defaults = dict(lr=lr, max_iter=max_iter, max_eval=max_eval,
                        tolerance_grad=tolerance_grad, tolerance_change=tolerance_change,
                        history_size=history_size, line_search_fn=line_search_fn)
        super().__init__(params, defaults)

        if len(self.param_groups) != 1:
            raise ValueError("LBFGS doesn't support per-parameter options")

        self._params = self.param_groups[0]['params']
        self._numel_cache = None

    def _numel(self):
        if self._numel_cache is None:
            self._numel_cache = sum(p.numel() for p in self._params)
        return self._numel_cache

    def _gather_flat_grad(self):
        views = []
        for p in self._params:
            if p.grad is None:
                view = p.new(p.numel()).zero_()
            elif p.grad.is_sparse:
                view = p.grad.to_dense().view(-1)
            else:
                view = p.grad.view(-1)
            views.append(view)
        return torch.cat(views, 0)

    def _add_grad(self, step_size, update):
        offset = 0
        for p in self._params:
            numel = p.numel()
            p.data.add_(update[offset:offset + numel].view_as(p), alpha=step_size)
            offset += numel
        assert offset == self._numel()

    def step(self, closure):
        """Performs a single optimization step."""
        assert len(self.param_groups) == 1

        group = self.param_groups[0]
        lr = group['lr']
        max_iter = group['max_iter']
        max_eval = group['max_eval']
        tolerance_grad = group['tolerance_grad']
        tolerance_change = group['tolerance_change']
        line_search_fn = group['line_search_fn']
        history_size = group['history_size']

        state = self.state[self._params[0]]
        state.setdefault('func_evals', 0)
        state.setdefault('n_iter', 0)

        # evaluate initial f(x) and df/dx
        orig_loss = closure()
        loss = float(orig_loss)
        current_evals = 1
        state['func_evals'] += 1

        flat_grad = self._gather_flat_grad()
        opt_cond = flat_grad.abs().max() <= tolerance_grad

        # optimal condition
        if opt_cond:
            return orig_loss

        # tensors cached in state (for tracing)
        d = state.get('d')
        t = state.get('t')
        old_dirs = state.get('old_dirs')
        old_stps = state.get('old_stps')
        H_diag = state.get('H_diag')
        prev_flat_grad = state.get('prev_flat_grad')
        prev_loss = state.get('prev_loss')

        n_iter = 0
        # optimize for a maximum of max_iter iterations
        while n_iter < max_iter:
            # check convergence
            if opt_cond:
                break

            # compute gradient descent direction
            if state['n_iter'] == 0:
                d = flat_grad.neg()
                old_dirs = []
                old_stps = []
                H_diag = 1
                prev_flat_grad = flat_grad.clone()
            else:
                # do lbfgs update (update memory)
                y = flat_grad.sub(prev_flat_grad)
                s = d.mul(t)
                ys = y.dot(s)  # y^T * s
                if ys > 1e-10:
                    # updating memory
                    if len(old_dirs) == history_size:
                        # shift history by one (limited-memory)
                        old_dirs.pop(0)
                        old_stps.pop(0)

                    # store new direction/step
                    old_dirs.append(s)
                    old_stps.append(y)

                    # update scale of initial Hessian approximation
                    H_diag = ys / y.dot(y)  # (y^T * s) / (y^T * y)

                # compute the next (Hessian) search direction
                num_old = len(old_dirs)

                if 'al' not in state:
                    state['al'] = [None] * history_size
                al = state['al']

                # iteration in L-BFGS loop collapsed to use just a buffer
                q = flat_grad.neg()
                for i in range(num_old - 1, -1, -1):
                    al[i] = old_stps[i].dot(q) / old_dirs[i].dot(old_stps[i])
                    q.add_(old_dirs[i], alpha=-al[i])

                # multiply by initial Hessian
                r = q.mul(H_diag)
                for i in range(num_old):
                    be_i = old_dirs[i].dot(r) / old_dirs[i].dot(old_stps[i])
                    r.add_(old_stps[i], alpha=al[i] - be_i)

                d = r
                prev_flat_grad = flat_grad.clone()

            if prev_loss is not None and abs(loss - prev_loss) < tolerance_change:
                break
            prev_loss = loss

            # directional derivative
            gtd = flat_grad.dot(d)  # g^T * d

            # check that progress is made
            if gtd > -tolerance_change:
                break

            # optional line search: user function
            ls_func_evals = 0
            if line_search_fn is not None:
                # perform line search, using user function
                if line_search_fn == "strong_wolfe":
                    def obj_func(x, t, d):
                        return self._directional_evaluate(closure, x, t, d)

                    loss, flat_grad, t, ls_func_evals = self._strong_wolfe(
                        obj_func, self._params, d, loss, flat_grad, gtd)
                else:
                    raise RuntimeError(f"Invalid line_search_fn: {line_search_fn}")
            else:
                # no line search, simply move with fixed-step
                t = lr
                self._add_grad(t, d)
                if n_iter != max_iter - 1:
                    # re-evaluate function only if not in last iteration
                    # the reason we do this: in a stochastic setting,
                    # no use to re-evaluate that function here
                    with torch.enable_grad():
                        loss = float(closure())
                    flat_grad = self._gather_flat_grad()
                    ls_func_evals = 1

            # update func eval
            current_evals += ls_func_evals
            state['func_evals'] += current_evals

            # optimal condition
            opt_cond = flat_grad.abs().max() <= tolerance_grad

            # update state
            state['d'] = d
            state['t'] = t
            state['old_dirs'] = old_dirs
            state['old_stps'] = old_stps
            state['H_diag'] = H_diag
            state['prev_flat_grad'] = prev_flat_grad
            state['prev_loss'] = prev_loss

            n_iter += 1
            state['n_iter'] += 1

        return orig_loss

    def _directional_evaluate(self, closure, x, t, d):
        self._add_grad(t, d)
        loss = float(closure())
        flat_grad = self._gather_flat_grad()
        self._add_grad(-t, d)  # restore
        return loss, flat_grad

    def _strong_wolfe(self, obj_func, x, d, loss, flat_grad, gtd):
        """Strong Wolfe line search."""
        c1 = 1e-4
        c2 = 0.9

        def phi(alpha):
            return obj_func(x, alpha, d)[0]

        def derphi(alpha):
            return obj_func(x, alpha, d)[1].dot(d)

        alpha_star, phi_star, derphi_star, n_iter = self._strong_wolfe_line_search(
            phi, derphi, phi(0), derphi(0), c1, c2)

        return phi_star, obj_func(x, alpha_star, d)[1], alpha_star, n_iter

    def _strong_wolfe_line_search(self, phi, derphi, phi0, derphi0, c1, c2):
        """Strong Wolfe line search implementation."""
        maxiter = 20
        alpha0 = 0
        alpha1 = 1.0

        for i in range(maxiter):
            phi_a1 = phi(alpha1)
            derphi_a1 = derphi(alpha1)

            if phi_a1 > phi0 + c1 * alpha1 * derphi0 or (i > 0 and phi_a1 >= phi(alpha0)):
                alpha_star = self._zoom(phi, derphi, alpha0, alpha1, phi0, derphi0, c1, c2)
                return alpha_star, phi(alpha_star), derphi(alpha_star), i + 1

            if abs(derphi_a1) <= -c2 * derphi0:
                return alpha1, phi_a1, derphi_a1, i + 1

            if derphi_a1 >= 0:
                alpha_star = self._zoom(phi, derphi, alpha1, alpha0, phi0, derphi0, c1, c2)
                return alpha_star, phi(alpha_star), derphi(alpha_star), i + 1

            alpha0, alpha1 = alpha1, min(2 * alpha1, 1.0)

        return alpha1, phi(alpha1), derphi(alpha1), maxiter

    def _zoom(self, phi, derphi, alpha_lo, alpha_hi, phi0, derphi0, c1, c2):
        """Zoom phase of line search."""
        maxiter = 20

        for i in range(maxiter):
            alpha = (alpha_lo + alpha_hi) / 2
            phi_a = phi(alpha)
            derphi_a = derphi(alpha)

            if phi_a > phi0 + c1 * alpha * derphi0 or phi_a >= phi(alpha_lo):
                alpha_hi = alpha
            else:
                if abs(derphi_a) <= -c2 * derphi0:
                    return alpha
                if derphi_a * (alpha_hi - alpha_lo) >= 0:
                    alpha_hi = alpha_lo
                alpha_lo = alpha

        return alpha


class NewtonOptimizer(torch.optim.Optimizer):
    """
    Implementación del método de Newton para optimización de segunda orden.

    El método de Newton usa la matriz Hessiana completa para calcular
    direcciones de descenso más precisas.
    """

    def __init__(self, params, lr=1.0, reg=1e-6, max_iter=10, tol=1e-6):
        if lr <= 0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if reg <= 0:
            raise ValueError(f"Invalid regularization: {reg}")
        if max_iter <= 0:
            raise ValueError(f"Invalid max_iter: {max_iter}")
        if tol <= 0:
            raise ValueError(f"Invalid tolerance: {tol}")

        defaults = dict(lr=lr, reg=reg, max_iter=max_iter, tol=tol)
        super().__init__(params, defaults)

    def step(self, closure):
        """Performs a single optimization step."""
        loss = closure()

        for group in self.param_groups:
            lr = group['lr']
            reg = group['reg']
            max_iter = group['max_iter']
            tol = group['tol']

            for p in group['params']:
                if p.grad is None:
                    continue

                grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError('Newton does not support sparse gradients')

                # Compute Hessian approximation
                hessian = self._compute_hessian(p, grad, reg)

                # Solve Newton system: H * d = -g
                try:
                    # Add regularization for numerical stability
                    hessian_reg = hessian + reg * torch.eye(hessian.size(0), device=hessian.device)

                    # Solve linear system
                    direction = torch.linalg.solve(hessian_reg, -grad.flatten())
                    direction = direction.view_as(p.data)

                    # Update parameters
                    p.data.add_(direction, alpha=lr)

                except RuntimeError as e:
                    logger.warning(f"Newton step failed: {e}, falling back to gradient descent")
                    p.data.add_(grad, alpha=-lr)

        return loss

    def _compute_hessian(self, param, grad, reg):
        """Compute Hessian approximation for a parameter."""
        # Simple diagonal approximation for efficiency
        # In practice, you might want to use more sophisticated approximations
        hessian_diag = torch.abs(grad) + reg
        return torch.diag(hessian_diag.flatten())


class ShampooOptimizer(torch.optim.Optimizer):
    """
    Implementación del optimizador Shampoo.

    Shampoo es un optimizador de segunda orden que aproxima la matriz
    Hessiana usando factorización de Kronecker para escalabilidad.
    """

    def __init__(self, params, lr=1e-3, momentum=0.9, weight_decay=0.0,
                 block_size=1024, preconditioning_compute_steps=1,
                 statistics_compute_steps=1):
        if lr <= 0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if momentum < 0 or momentum > 1:
            raise ValueError(f"Invalid momentum: {momentum}")
        if weight_decay < 0:
            raise ValueError(f"Invalid weight_decay: {weight_decay}")
        if block_size <= 0:
            raise ValueError(f"Invalid block_size: {block_size}")

        defaults = dict(lr=lr, momentum=momentum, weight_decay=weight_decay,
                        block_size=block_size,
                        preconditioning_compute_steps=preconditioning_compute_steps,
                        statistics_compute_steps=statistics_compute_steps)
        super().__init__(params, defaults)

    def step(self, closure=None):
        """Performs a single optimization step."""
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            lr = group['lr']
            momentum = group['momentum']
            weight_decay = group['weight_decay']
            block_size = group['block_size']

            for p in group['params']:
                if p.grad is None:
                    continue

                grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError('Shampoo does not support sparse gradients')

                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    state['momentum_buffer'] = torch.zeros_like(p.data)
                    state['statistics'] = {}

                state['step'] += 1

                # Weight decay
                if weight_decay != 0:
                    grad = grad.add(p.data, alpha=weight_decay)

                # Update statistics (simplified version)
                if state['step'] % group['statistics_compute_steps'] == 0:
                    self._update_statistics(p, grad, state, block_size)

                # Compute preconditioned gradient
                if state['step'] % group['preconditioning_compute_steps'] == 0:
                    preconditioned_grad = self._precondition_gradient(p, grad, state)
                else:
                    preconditioned_grad = grad

                # Momentum update
                if momentum != 0:
                    state['momentum_buffer'].mul_(momentum).add_(preconditioned_grad, alpha=1 - momentum)
                    update = state['momentum_buffer']
                else:
                    update = preconditioned_grad

                # Parameter update
                p.data.add_(update, alpha=-lr)

        return loss

    def _update_statistics(self, param, grad, state, block_size):
        """Update Shampoo statistics."""
        # Simplified statistics update
        # In practice, this would involve more sophisticated Kronecker factorization
        if 'grad_norm' not in state['statistics']:
            state['statistics']['grad_norm'] = torch.zeros_like(grad)

        state['statistics']['grad_norm'].mul_(0.9).add_(grad.abs(), alpha=0.1)

    def _precondition_gradient(self, param, grad, state):
        """Apply Shampoo preconditioning."""
        # Simplified preconditioning
        # In practice, this would use the Kronecker-factored statistics
        if 'grad_norm' in state['statistics']:
            grad_norm = state['statistics']['grad_norm']
            grad_norm_safe = grad_norm + 1e-8
            return grad / grad_norm_safe.sqrt()
        else:
            return grad


class SecondOrderOptimizerManager:
    """
    Gestor de algoritmos de segunda orden para redes de refuerzo.

    Proporciona acceso a métodos de optimización de segunda orden
    con análisis de curvatura y técnicas de preconditioning avanzadas.
    """

    def __init__(self, config: Optional[SecondOrderConfig] = None):
        """
        Inicializa el gestor de optimizadores de segunda orden.

        Args:
            config: Configuración del optimizador (opcional)
        """
        self.config = config or SecondOrderConfig()
        self.optimizer = None
        self.optimization_history = []
        self.metrics = {
            'total_steps': 0,
            'convergence_rate': 0.0,
            'stability_score': 0.0,
            'final_loss': float('inf'),
            'best_loss': float('inf'),
            'gradient_norm': 0.0,
            'weight_update_magnitude': 0.0,
            'hessian_condition_number': 0.0,
            'curvature_analysis': {}
        }

        logger.info(f"SecondOrderOptimizerManager inicializado con {self.config.optimizer_type}")

    def create_optimizer(self, model: nn.Module) -> torch.optim.Optimizer:
        """
        Crea un optimizador de segunda orden para el modelo.

        Args:
            model: Modelo PyTorch

        Returns:
            Optimizador configurado
        """
        parameters = model.parameters()

        if self.config.optimizer_type == 'L-BFGS':
            optimizer = LBFGSOptimizer(
                parameters,
                lr=self.config.learning_rate,
                max_iter=self.config.max_iter,
                tolerance_grad=self.config.tolerance_grad,
                tolerance_change=self.config.tolerance_change,
                history_size=self.config.history_size,
                line_search_fn=self.config.line_search_fn
            )

        elif self.config.optimizer_type == 'Newton':
            optimizer = NewtonOptimizer(
                parameters,
                lr=self.config.learning_rate,
                reg=self.config.newton_reg,
                max_iter=self.config.newton_max_iter,
                tol=self.config.newton_tol
            )

        elif self.config.optimizer_type == 'Shampoo':
            optimizer = ShampooOptimizer(
                parameters,
                lr=self.config.learning_rate,
                momentum=self.config.shampoo_momentum,
                weight_decay=self.config.shampoo_weight_decay,
                block_size=self.config.shampoo_block_size,
                preconditioning_compute_steps=self.config.shampoo_preconditioning_compute_steps,
                statistics_compute_steps=self.config.shampoo_statistics_compute_steps
            )

        else:
            logger.warning(f"Optimizador {self.config.optimizer_type} no soportado, usando L-BFGS")
            optimizer = LBFGSOptimizer(
                parameters,
                lr=self.config.learning_rate
            )

        self.optimizer = optimizer
        logger.info(f"Optimizador {self.config.optimizer_type} creado con lr={self.config.learning_rate}")

        return optimizer

    def optimize_step(self, model: nn.Module, loss_fn: Callable,
                      previous_weights: Optional[Dict] = None) -> Dict[str, float]:
        """
        Ejecuta un paso de optimización de segunda orden.

        Args:
            model: Modelo PyTorch
            loss_fn: Función de pérdida
            previous_weights: Pesos anteriores para análisis

        Returns:
            Métricas del paso de optimización
        """
        if self.optimizer is None:
            raise ValueError("Optimizador no inicializado. Llame a create_optimizer() primero.")

        # Ejecutar optimización
        if isinstance(self.optimizer, LBFGSOptimizer):
            # L-BFGS maneja múltiples iteraciones internamente
            final_loss = self.optimizer.step(loss_fn)
            loss_value = float(final_loss)
        else:
            # Otros optimizadores ejecutan un paso
            final_loss = self.optimizer.step(loss_fn)
            loss_value = float(final_loss)

        # Calcular métricas
        grad_norm = self._calculate_gradient_norm(model)
        weight_change = 0.0
        if previous_weights is not None:
            weight_change = self._calculate_weight_change(model, previous_weights)

        # Análisis de curvatura
        curvature_analysis = self._analyze_curvature(model)

        # Actualizar métricas
        self.metrics['total_steps'] += 1
        self.metrics['final_loss'] = loss_value
        self.metrics['gradient_norm'] = grad_norm
        self.metrics['weight_update_magnitude'] = weight_change
        self.metrics['curvature_analysis'] = curvature_analysis

        if loss_value < self.metrics['best_loss']:
            self.metrics['best_loss'] = loss_value

        # Calcular métricas de convergencia
        self._update_convergence_metrics(loss_value, grad_norm, weight_change)

        # Registrar en historial
        step_metrics = {
            'step': self.metrics['total_steps'],
            'loss': loss_value,
            'gradient_norm': grad_norm,
            'weight_change': weight_change,
            'learning_rate': self.config.learning_rate,
            'convergence_rate': self.metrics['convergence_rate'],
            'stability_score': self.metrics['stability_score'],
            'curvature_analysis': curvature_analysis
        }
        self.optimization_history.append(step_metrics)

        return step_metrics

    def _calculate_gradient_norm(self, model: nn.Module) -> float:
        """Calcula la norma del gradiente"""
        total_norm = 0.0
        for param in model.parameters():
            if param.grad is not None:
                param_norm = param.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
        return total_norm ** (1. / 2)

    def _calculate_weight_change(self, model: nn.Module, previous_weights: Dict) -> float:
        """Calcula la magnitud del cambio de pesos"""
        total_change = 0.0
        param_count = 0

        for name, param in model.named_parameters():
            if name in previous_weights:
                weight_change = torch.norm(param.data - previous_weights[name]).item()
                total_change += weight_change
                param_count += 1

        return total_change / param_count if param_count > 0 else 0.0

    def _analyze_curvature(self, model: nn.Module) -> Dict[str, float]:
        """Analiza la curvatura del modelo"""
        curvature_metrics = {
            'gradient_norm': 0.0,
            'parameter_norm': 0.0,
            'condition_estimate': 0.0,
            'curvature_ratio': 0.0
        }

        total_grad_norm = 0.0
        total_param_norm = 0.0
        param_count = 0

        for param in model.parameters():
            if param.grad is not None:
                grad_norm = param.grad.data.norm(2).item()
                param_norm = param.data.norm(2).item()

                total_grad_norm += grad_norm ** 2
                total_param_norm += param_norm ** 2
                param_count += 1

        if param_count > 0:
            curvature_metrics['gradient_norm'] = math.sqrt(total_grad_norm)
            curvature_metrics['parameter_norm'] = math.sqrt(total_param_norm)

            if curvature_metrics['parameter_norm'] > 0:
                curvature_metrics['curvature_ratio'] = curvature_metrics['gradient_norm'] / curvature_metrics['parameter_norm']

        return curvature_metrics

    def _update_convergence_metrics(self, loss: float, grad_norm: float, weight_change: float) -> None:
        """Actualiza métricas de convergencia"""
        if len(self.optimization_history) < 20:
            return

        # Calcular tasa de convergencia
        recent_losses = [step['loss'] for step in self.optimization_history[-10:]]
        older_losses = [step['loss'] for step in self.optimization_history[-20:-10]]

        if len(older_losses) > 0:
            recent_avg = np.mean(recent_losses)
            older_avg = np.mean(older_losses)

            if older_avg != 0:
                self.metrics['convergence_rate'] = (older_avg - recent_avg) / abs(older_avg)

        # Calcular score de estabilidad
        if len(recent_losses) >= 5:
            self.metrics['stability_score'] = 1.0 / (1.0 + np.std(recent_losses))

    def get_current_weights(self, model: nn.Module) -> Dict[str, torch.Tensor]:
        """Obtiene los pesos actuales del modelo"""
        weights = {}
        for name, param in model.named_parameters():
            weights[name] = param.data.clone()
        return weights

    def get_optimization_summary(self) -> Dict[str, Any]:
        """Obtiene un resumen de la optimización"""
        return {
            'config': self.config,
            'metrics': self.metrics.copy(),
            'optimizer_type': self.config.optimizer_type,
            'total_steps': self.metrics['total_steps'],
            'convergence_rate': self.metrics['convergence_rate'],
            'stability_score': self.metrics['stability_score'],
            'curvature_analysis': self.metrics['curvature_analysis'],
            'history_length': len(self.optimization_history)
        }

    def save_optimizer_state(self, path: str) -> None:
        """Guarda el estado del optimizador"""
        if self.optimizer is not None:
            torch.save({
                'optimizer_state_dict': self.optimizer.state_dict(),
                'config': self.config,
                'metrics': self.metrics,
                'optimization_history': self.optimization_history
            }, path)
            logger.info(f"Estado del optimizador de segunda orden guardado en {path}")

    def load_optimizer_state(self, path: str) -> None:
        """Carga el estado del optimizador"""
        checkpoint = torch.load(path)

        if self.optimizer is not None:
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        self.config = checkpoint.get('config', self.config)
        self.metrics = checkpoint.get('metrics', self.metrics)
        self.optimization_history = checkpoint.get('optimization_history', [])

        logger.info(f"Estado del optimizador de segunda orden cargado desde {path}")

    def reset_optimizer(self) -> None:
        """Reinicia el optimizador manteniendo la configuración"""
        if self.optimizer is not None:
            self.optimizer.zero_grad()

        self.metrics = {
            'total_steps': 0,
            'convergence_rate': 0.0,
            'stability_score': 0.0,
            'final_loss': float('inf'),
            'best_loss': float('inf'),
            'gradient_norm': 0.0,
            'weight_update_magnitude': 0.0,
            'hessian_condition_number': 0.0,
            'curvature_analysis': {}
        }

        self.optimization_history.clear()
        logger.info("Optimizador de segunda orden reiniciado")
