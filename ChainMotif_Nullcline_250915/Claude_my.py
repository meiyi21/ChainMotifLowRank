import numpy as np
from scipy import integrate
import matplotlib.pyplot as plt

class KappaSolver:
    def __init__(self, N_E, g, N_T, J_0, sigma, N_Tc):
        """
        Initialize the solver with system parameters
        
        Parameters:
        N_E, N_T: Population sizes
        g: Coupling strength
        J_0: Connection strength
        sigma: Noise strength
        N_Tc: Another population size parameter
        """
        self.N_E = N_E
        self.g = g
        self.N_T = N_T
        self.J_0 = J_0
        self.sigma = sigma
        self.N_Tc = N_Tc
    
    def phi(self, x):
        """Activation function φ(x) = tanh(x)"""
        return np.tanh(x)
    
    def phi_prime(self, x):
        """Derivative of activation function φ'(x) = 1 - tanh²(x)"""
        return 1 - np.tanh(x)**2
    
    def gaussian_average_phi(self, kappa1, kappa2_squared):
        """
        Compute <φ(κ₁^rec, (κ₂^rec)²)> where the average is over Gaussian random variables
        
        For tanh activation, this integral can be approximated using Gauss-Hermite quadrature
        or Monte Carlo sampling. Here we use numerical integration.
        """
        def integrand(z1, z2):
            # z1, z2 are standard Gaussian variables
            arg = kappa1 + np.sqrt(kappa2_squared) * z1  # Simplified example
            return self.phi(arg) * np.exp(-(z1**2 + z2**2)/2) / (2*np.pi)
        
        # Use numerical integration over a reasonable range
        result, _ = integrate.dblquad(integrand, -4, 4, -4, 4)
        return result
    
    def gaussian_average_phi_prime(self, kappa1, kappa2_squared):
        """
        Compute <φ'(κ₁^rec, (κ₂^rec)²)> where the average is over Gaussian random variables
        """
        def integrand(z1, z2):
            arg = kappa1 + np.sqrt(max(kappa2_squared, 0)) * z1
            return self.phi_prime(arg) * np.exp(-(z1**2 + z2**2)/2) / (2*np.pi)
        
        result, _ = integrate.dblquad(integrand, -4, 4, -4, 4)
        return result
    
    def system_equations(self, kappa1, kappa2):
        """
        Compute the right-hand side of the dynamical system
        Returns (dκ₁/dt, dκ₂/dt)
        """
        kappa2_squared = kappa2**2
        
        # Compute Gaussian averages
        avg_phi = self.gaussian_average_phi(kappa1, kappa2_squared)
        avg_phi_prime = self.gaussian_average_phi_prime(kappa1, kappa2_squared)
        
        # First equation: dκ₁/dt = -κ₁ + (N_E - gN_T)J₀ · <φ(κ₁^rec, (κ₂^rec)²)> + σ√N_Tc · <φ'(κ₁^rec, (κ₂^rec)²)>κ₂
        dkappa1_dt = (-kappa1 + 
                     (self.N_E - self.g * self.N_T) * self.J_0 * avg_phi + 
                     self.sigma * np.sqrt(self.N_Tc) * avg_phi_prime * kappa2)
        
        # Second equation: dκ₂/dt = -κ₂ + σ√N_Tc · <φ(κ₁^rec, (κ₂^rec)²)>
        dkappa2_dt = -kappa2 + self.sigma * np.sqrt(self.N_Tc) * avg_phi
        
        return dkappa1_dt, dkappa2_dt
    
    def solve_iterative(self, initial_guess=(0.1, 0.1), max_iter=1000, tol=1e-6, step_size=0.01):
        """
        Solve for steady state using fixed-point iteration
        
        Parameters:
        initial_guess: Starting values for (κ₁, κ₂)
        max_iter: Maximum number of iterations
        tol: Convergence tolerance
        step_size: Step size for updates
        """
        kappa1, kappa2 = initial_guess
        history = [(kappa1, kappa2)]
        
        print(f"Starting iteration with initial guess: κ₁={kappa1:.4f}, κ₂={kappa2:.4f}")
        
        for i in range(max_iter):
            # Compute derivatives
            dkappa1_dt, dkappa2_dt = self.system_equations(kappa1, kappa2)
            
            # Update using small step (Euler method towards equilibrium)
            kappa1_new = kappa1 + step_size * dkappa1_dt
            kappa2_new = kappa2 + step_size * dkappa2_dt
            
            # Check convergence
            error = np.sqrt((kappa1_new - kappa1)**2 + (kappa2_new - kappa2)**2)
            
            kappa1, kappa2 = kappa1_new, kappa2_new
            history.append((kappa1, kappa2))
            
            if i % 100 == 0 or error < tol:
                print(f"Iteration {i}: κ₁={kappa1:.6f}, κ₂={kappa2:.6f}, error={error:.2e}")
            
            if error < tol:
                print(f"Converged after {i+1} iterations!")
                break
        else:
            print(f"Did not converge after {max_iter} iterations. Final error: {error:.2e}")
        
        return kappa1, kappa2, history
    
    def solve_newton(self, initial_guess=(0.1, 0.1), max_iter=50, tol=1e-8):
        """
        Solve using Newton's method for better convergence
        """
        from scipy.optimize import fsolve
        
        def equations(vars):
            k1, k2 = vars
            dk1_dt, dk2_dt = self.system_equations(k1, k2)
            return [dk1_dt, dk2_dt]  # We want these to be zero
        
        solution = fsolve(equations, initial_guess, xtol=tol)
        return solution[0], solution[1]
    
    def plot_trajectory(self, history):
        """Plot the convergence trajectory"""
        history = np.array(history)
        
        plt.figure(figsize=(12, 4))
        
        plt.subplot(1, 3, 1)
        plt.plot(history[:, 0], label='κ₁')
        plt.plot(history[:, 1], label='κ₂')
        plt.xlabel('Iteration')
        plt.ylabel('Value')
        plt.legend()
        plt.title('Convergence vs Iteration')
        plt.grid(True)
        
        plt.subplot(1, 3, 2)
        plt.plot(history[:, 0], history[:, 1])
        plt.xlabel('κ₁')
        plt.ylabel('κ₂')
        plt.title('Phase Space Trajectory')
        plt.grid(True)
        plt.plot(history[0, 0], history[0, 1], 'go', markersize=8, label='Start')
        plt.plot(history[-1, 0], history[-1, 1], 'ro', markersize=8, label='End')
        plt.legend()
        
        plt.subplot(1, 3, 3)
        errors = np.sqrt(np.diff(history[:, 0])**2 + np.diff(history[:, 1])**2)
        plt.semilogy(errors)
        plt.xlabel('Iteration')
        plt.ylabel('Step Size (log scale)')
        plt.title('Convergence Rate')
        plt.grid(True)
        
        plt.tight_layout()
        plt.show()

# Example usage
if __name__ == "__main__":
    # Set system parameters (you should adjust these to your specific problem)
    N_E = 100
    g = 0.5
    N_T = 80
    J_0 = 0.1
    sigma = 0.2
    N_Tc = 50
    
    # Create solver instance
    solver = KappaSolver(N_E, g, N_T, J_0, sigma, N_Tc)
    
    # Solve using iterative method
    print("=== Iterative Method ===")
    kappa1_iter, kappa2_iter, history = solver.solve_iterative(
        initial_guess=(0.1, 0.1), 
        max_iter=2000, 
        tol=1e-8,
        step_size=0.01
    )
    
    print(f"\nFinal solution (iterative): κ₁={kappa1_iter:.8f}, κ₂={kappa2_iter:.8f}")
    
    # Verify solution
    dk1, dk2 = solver.system_equations(kappa1_iter, kappa2_iter)
    print(f"Verification: dκ₁/dt={dk1:.2e}, dκ₂/dt={dk2:.2e}")
    
    # Try Newton's method for comparison
    print("\n=== Newton's Method ===")
    try:
        kappa1_newton, kappa2_newton = solver.solve_newton(initial_guess=(0.1, 0.1))
        print(f"Final solution (Newton): κ₁={kappa1_newton:.8f}, κ₂={kappa2_newton:.8f}")
        
        # Verify Newton solution
        dk1, dk2 = solver.system_equations(kappa1_newton, kappa2_newton)
        print(f"Verification: dκ₁/dt={dk1:.2e}, dκ₂/dt={dk2:.2e}")
    except Exception as e:
        print(f"Newton's method failed: {e}")
    
    # Plot convergence
    solver.plot_trajectory(history)