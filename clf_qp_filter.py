"""
Control Lyapunov Function - Quadratic Programming (CLF-QP) Filter
for UAV Target Tracking with Stability Guarantees
"""

import numpy as np
import cvxpy as cp
from typing import Tuple, Optional


class CLFQPFilter:
    """
    CLF-QP filter that ensures stability while respecting safety constraints.
    
    The filter solves a QP to find the control input that:
    1. Stays close to the RL policy output (u_rl)
    2. Satisfies the CLF decrease condition for stability
    3. Respects control input bounds
    """
    
    def __init__(
        self,
        u_min: np.ndarray,
        u_max: np.ndarray,
        lambda_clf: float = 1.0,
        gamma_clf: float = 0.5,
        epsilon: float = 1e-6
    ):
        """
        Initialize CLF-QP filter.
        
        Args:
            u_min: Minimum control input [vx_min, vy_min]
            u_max: Maximum control input [vx_max, vy_max]
            lambda_clf: Weight for tracking RL policy
            gamma_clf: CLF decrease rate (0 < gamma_clf < 1)
            epsilon: Small constant for numerical stability
        """
        self.u_min = u_min
        self.u_max = u_max
        self.lambda_clf = lambda_clf
        self.gamma_clf = gamma_clf
        self.epsilon = epsilon
        
        # Control input dimension
        self.u_dim = 2
        
    def compute_clf(self, state: dict) -> float:
        """
        Compute Control Lyapunov Function value.
        
        V(x) = 0.5 * ||p_A - p_T||^2
        
        where p_A is agent position and p_T is target position.
        
        Args:
            state: Dictionary containing agent and target states
            
        Returns:
            CLF value V(x)
        """
        p_diff = state['agent_pos'][:2] - state['target_pos'][:2]
        V = 0.5 * np.dot(p_diff, p_diff)
        return V
    
    def compute_clf_derivative(
        self,
        state: dict,
        u: np.ndarray
    ) -> float:
        """
        Compute time derivative of CLF.
        
        dV/dt = ∇V^T * f(x) + ∇V^T * g(x) * u
        
        For our system:
        - f(x) = 0 (no drift dynamics in velocity control)
        - g(x) = I (direct velocity control)
        - ∇V = [x_A - x_T, y_A - y_T]
        
        So: dV/dt = (p_A - p_T)^T * u_A
        
        Args:
            state: Dictionary containing agent and target states
            u: Control input [u_vx, u_vy]
            
        Returns:
            Time derivative of CLF
        """
        p_diff = state['agent_pos'][:2] - state['target_pos'][:2]
        dV_dt = np.dot(p_diff, u)
        return dV_dt
    
    def filter_action(
        self,
        u_rl: np.ndarray,
        state: dict,
        verbose: bool = False
    ) -> Tuple[np.ndarray, dict]:
        """
        Filter the RL policy action through CLF-QP.
        
        Solves:
            min  0.5 * ||u - u_rl||^2
            s.t. dV/dt + γ * V(x) ≤ 0  (CLF condition)
                 u_min ≤ u ≤ u_max
        
        Args:
            u_rl: Desired control from RL policy [vx, vy]
            state: Current state information
            verbose: Print debug information
            
        Returns:
            u_safe: Filtered safe control input
            info: Dictionary with solver information
        """
        # Define optimization variable
        u = cp.Variable(self.u_dim)
        
        # Compute current CLF value
        V_current = self.compute_clf(state)
        
        # Objective: minimize deviation from RL policy
        objective = cp.Minimize(
            0.5 * self.lambda_clf * cp.sum_squares(u - u_rl)
        )
        
        # Constraint 1: CLF decrease condition
        # dV/dt + γ * V(x) ≤ 0
        # For our system: (p_A - p_T)^T * u + γ * V(x) ≤ 0
        p_diff = state['agent_pos'][:2] - state['target_pos'][:2]
        clf_constraint = p_diff @ u + self.gamma_clf * V_current <= 0
        
        # Constraint 2: Control bounds
        control_constraints = [
            u >= self.u_min,
            u <= self.u_max
        ]
        
        # Combine all constraints
        constraints = [clf_constraint] + control_constraints
        
        # Solve QP
        problem = cp.Problem(objective, constraints)
        
        try:
            problem.solve(solver=cp.OSQP, verbose=verbose)
            
            if problem.status in [cp.OPTIMAL, cp.OPTIMAL_INACCURATE]:
                u_safe = u.value
                success = True
            else:
                # Fallback: use RL action clipped to bounds
                u_safe = np.clip(u_rl, self.u_min, self.u_max)
                success = False
                if verbose:
                    print(f"QP solver failed with status: {problem.status}")
        except Exception as e:
            # Fallback: use RL action clipped to bounds
            u_safe = np.clip(u_rl, self.u_min, self.u_max)
            success = False
            if verbose:
                print(f"QP solver exception: {e}")
        
        # Compute CLF derivative with filtered action
        dV_dt = self.compute_clf_derivative(state, u_safe)
        
        # Prepare info dictionary
        info = {
            'qp_success': success,
            'V_current': V_current,
            'dV_dt': dV_dt,
            'clf_satisfied': dV_dt + self.gamma_clf * V_current <= self.epsilon,
            'u_rl': u_rl.copy(),
            'u_safe': u_safe.copy(),
            'action_deviation': np.linalg.norm(u_safe - u_rl)
        }
        
        return u_safe, info


class AdaptiveCLFQPFilter(CLFQPFilter):
    """
    Adaptive CLF-QP filter that adjusts gamma based on distance to target.
    
    Uses higher gamma (faster convergence) when far from target,
    and lower gamma (smoother control) when close to target.
    """
    
    def __init__(
        self,
        u_min: np.ndarray,
        u_max: np.ndarray,
        lambda_clf: float = 1.0,
        gamma_min: float = 0.1,
        gamma_max: float = 0.8,
        distance_threshold: float = 10.0,
        epsilon: float = 1e-6
    ):
        """
        Initialize adaptive CLF-QP filter.
        
        Args:
            u_min: Minimum control input
            u_max: Maximum control input
            lambda_clf: Weight for tracking RL policy
            gamma_min: Minimum CLF decrease rate (used when close)
            gamma_max: Maximum CLF decrease rate (used when far)
            distance_threshold: Distance for gamma adaptation
            epsilon: Small constant for numerical stability
        """
        super().__init__(u_min, u_max, lambda_clf, gamma_max, epsilon)
        self.gamma_min = gamma_min
        self.gamma_max = gamma_max
        self.distance_threshold = distance_threshold
        
    def adapt_gamma(self, state: dict) -> float:
        """
        Adapt gamma based on distance to target.
        
        γ = γ_min + (γ_max - γ_min) * min(d / d_thr, 1.0)
        
        Args:
            state: Current state information
            
        Returns:
            Adapted gamma value
        """
        distance = np.linalg.norm(
            state['agent_pos'][:2] - state['target_pos'][:2]
        )
        
        ratio = min(distance / self.distance_threshold, 1.0)
        gamma_adapted = self.gamma_min + (self.gamma_max - self.gamma_min) * ratio
        
        return gamma_adapted
    
    def filter_action(
        self,
        u_rl: np.ndarray,
        state: dict,
        verbose: bool = False
    ) -> Tuple[np.ndarray, dict]:
        """
        Filter action with adaptive gamma.
        
        Args:
            u_rl: Desired control from RL policy
            state: Current state information
            verbose: Print debug information
            
        Returns:
            u_safe: Filtered safe control input
            info: Dictionary with solver information
        """
        # Adapt gamma based on distance
        self.gamma_clf = self.adapt_gamma(state)
        
        # Call parent filter method
        u_safe, info = super().filter_action(u_rl, state, verbose)
        
        # Add gamma to info
        info['gamma_clf'] = self.gamma_clf
        
        return u_safe, info


def create_clf_filter(
    agent_vel_max: float = 2.0,
    filter_type: str = "adaptive",
    lambda_clf: float = 1.0,
    gamma_clf: float = 0.5,
    gamma_min: float = 0.1,
    gamma_max: float = 0.8,
    distance_threshold: float = 10.0
) -> CLFQPFilter:
    """
    Factory function to create CLF-QP filter.
    
    Args:
        agent_vel_max: Maximum agent velocity
        filter_type: "standard" or "adaptive"
        lambda_clf: Weight for tracking RL policy
        gamma_clf: CLF decrease rate (for standard filter)
        gamma_min: Minimum gamma (for adaptive filter)
        gamma_max: Maximum gamma (for adaptive filter)
        distance_threshold: Distance threshold (for adaptive filter)
        
    Returns:
        CLF-QP filter instance
    """
    u_min = np.array([-agent_vel_max, -agent_vel_max])
    u_max = np.array([agent_vel_max, agent_vel_max])
    
    if filter_type == "adaptive":
        return AdaptiveCLFQPFilter(
            u_min=u_min,
            u_max=u_max,
            lambda_clf=lambda_clf,
            gamma_min=gamma_min,
            gamma_max=gamma_max,
            distance_threshold=distance_threshold
        )
    else:
        return CLFQPFilter(
            u_min=u_min,
            u_max=u_max,
            lambda_clf=lambda_clf,
            gamma_clf=gamma_clf
        )
