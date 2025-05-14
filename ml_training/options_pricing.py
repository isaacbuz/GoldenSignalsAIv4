import numpy as np
import scipy.stats as stats
from typing import Dict, Any, Tuple

class AdvancedOptionsPricingModel:
    """
    Comprehensive options pricing model with machine learning enhancements.
    """
    
    @staticmethod
    def black_scholes_price(
        S: float,  # Current stock price
        K: float,  # Strike price
        T: float,  # Time to expiration (in years)
        r: float,  # Risk-free interest rate
        sigma: float,  # Volatility
        option_type: str = 'call'
    ) -> float:
        """
        Calculate option price using Black-Scholes model.
        
        Args:
            S (float): Current stock price
            K (float): Strike price
            T (float): Time to expiration
            r (float): Risk-free rate
            sigma (float): Volatility
            option_type (str): 'call' or 'put'
        
        Returns:
            float: Theoretical option price
        """
        # Calculate d1 and d2 parameters
        d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)
        
        # Standard normal cumulative distribution
        N = stats.norm.cdf
        
        if option_type.lower() == 'call':
            price = S * N(d1) - K * np.exp(-r * T) * N(d2)
        else:  # put option
            price = K * np.exp(-r * T) * N(-d2) - S * N(-d1)
        
        return price
    
    @classmethod
    def monte_carlo_options_pricing(
        cls,
        S: float,
        K: float,
        T: float,
        r: float,
        sigma: float,
        option_type: str = 'call',
        num_simulations: int = 100000
    ) -> Dict[str, Any]:
        """
        Advanced Monte Carlo simulation for options pricing.
        
        Args:
            S (float): Current stock price
            K (float): Strike price
            T (float): Time to expiration
            r (float): Risk-free rate
            sigma (float): Volatility
            option_type (str): 'call' or 'put'
            num_simulations (int): Number of price path simulations
        
        Returns:
            Dict[str, Any]: Comprehensive pricing analysis
        """
        # Simulate stock price paths
        np.random.seed(42)
        
        # Geometric Brownian Motion simulation
        Z = np.random.standard_normal(num_simulations)
        stock_paths = S * np.exp(
            (r - 0.5 * sigma**2) * T + sigma * np.sqrt(T) * Z
        )
        
        # Calculate option payoffs
        if option_type.lower() == 'call':
            payoffs = np.maximum(stock_paths - K, 0)
        else:  # put option
            payoffs = np.maximum(K - stock_paths, 0)
        
        # Discounted expected payoff
        option_price = np.mean(payoffs) * np.exp(-r * T)
        
        # Advanced pricing analysis
        pricing_analysis = {
            'option_price': option_price,
            'price_std_dev': np.std(payoffs) * np.exp(-r * T),
            'confidence_interval': cls._calculate_confidence_interval(payoffs, r, T),
            'price_distribution': {
                'mean': np.mean(stock_paths),
                'median': np.median(stock_paths),
                'std_dev': np.std(stock_paths)
            }
        }
        
        return pricing_analysis
    
    @staticmethod
    def _calculate_confidence_interval(
        payoffs: np.ndarray, 
        r: float, 
        T: float, 
        confidence: float = 0.95
    ) -> Tuple[float, float]:
        """
        Calculate confidence interval for option pricing.
        
        Args:
            payoffs (np.ndarray): Option payoffs
            r (float): Risk-free rate
            T (float): Time to expiration
            confidence (float): Confidence level
        
        Returns:
            Tuple[float, float]: Lower and upper confidence bounds
        """
        discounted_payoffs = payoffs * np.exp(-r * T)
        
        # Calculate confidence interval
        mean = np.mean(discounted_payoffs)
        std_error = stats.sem(discounted_payoffs)
        
        # T-distribution confidence interval
        confidence_interval = stats.t.interval(
            confidence, 
            len(discounted_payoffs) - 1, 
            loc=mean, 
            scale=std_error
        )
        
        return confidence_interval
    
    @classmethod
    def implied_volatility_estimation(
        cls,
        market_price: float,
        S: float,
        K: float,
        T: float,
        r: float,
        option_type: str = 'call',
        max_iterations: int = 100,
        tolerance: float = 1e-5
    ) -> float:
        """
        Estimate implied volatility using Newton-Raphson method.
        
        Args:
            market_price (float): Observed market option price
            S (float): Current stock price
            K (float): Strike price
            T (float): Time to expiration
            r (float): Risk-free rate
            option_type (str): 'call' or 'put'
            max_iterations (int): Maximum iterations for convergence
            tolerance (float): Convergence tolerance
        
        Returns:
            float: Estimated implied volatility
        """
        sigma = 0.5  # Initial volatility guess
        
        for _ in range(max_iterations):
            # Calculate option price and vega
            price = cls.black_scholes_price(
                S, K, T, r, sigma, option_type
            )
            
            # Vega calculation (sensitivity to volatility)
            d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
            vega = S * stats.norm.pdf(d1) * np.sqrt(T)
            
            # Newton-Raphson update
            diff = market_price - price
            if abs(diff) < tolerance:
                return sigma
            
            sigma += diff / vega
        
        return sigma

def main():
    """
    Demonstrate advanced options pricing techniques.
    """
    # Example options pricing scenario
    current_price = 100.0
    strike_price = 105.0
    time_to_expiration = 1.0  # 1 year
    risk_free_rate = 0.02
    volatility = 0.3

    # Black-Scholes Option Pricing
    call_price = AdvancedOptionsPricingModel.black_scholes_price(
        current_price, strike_price, time_to_expiration, 
        risk_free_rate, volatility, 'call'
    )
    
    put_price = AdvancedOptionsPricingModel.black_scholes_price(
        current_price, strike_price, time_to_expiration, 
        risk_free_rate, volatility, 'put'
    )
    
    # Monte Carlo Simulation
    call_analysis = AdvancedOptionsPricingModel.monte_carlo_options_pricing(
        current_price, strike_price, time_to_expiration, 
        risk_free_rate, volatility, 'call'
    )
    
    # Implied Volatility Estimation
    implied_vol = AdvancedOptionsPricingModel.implied_volatility_estimation(
        call_price, current_price, strike_price, 
        time_to_expiration, risk_free_rate, 'call'
    )
    
    # Display results
    print("Call Option Price (Black-Scholes):", call_price)
    print("Put Option Price (Black-Scholes):", put_price)
    print("\nMonte Carlo Call Option Analysis:")
    for key, value in call_analysis.items():
        print(f"{key}: {value}")
    print("\nImplied Volatility:", implied_vol)

if __name__ == '__main__':
    main()
