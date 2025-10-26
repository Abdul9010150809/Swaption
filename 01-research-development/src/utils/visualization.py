"""
Visualization utilities for the price matrix system.

This module provides plotting and visualization functions for
financial data, model performance, risk metrics, and experiment results.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any, Union
from pathlib import Path
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# Set default plotting style
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")


class FinancialVisualizer:
    """
    Visualization class for financial data and models.
    """

    def __init__(self, output_dir: str = 'results/plots'):
        """
        Initialize visualizer.

        Args:
            output_dir: Directory to save plots
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def plot_yield_curve(self, yields: pd.DataFrame,
                        title: str = 'Yield Curve',
                        save_path: Optional[str] = None) -> plt.Figure:
        """
        Plot yield curve(s).

        Args:
            yields: DataFrame with yield data (tenors as columns)
            title: Plot title
            save_path: Path to save plot (optional)

        Returns:
            Matplotlib figure
        """
        fig, ax = plt.subplots(figsize=(12, 8))

        tenors = [col.replace('maturity_', '') for col in yields.columns if col.startswith('maturity_')]
        tenors = [float(t) for t in tenors]

        if len(yields) == 1:
            # Single yield curve
            rates = yields.iloc[0].values
            ax.plot(tenors, rates, 'b-', linewidth=2, marker='o')
        else:
            # Multiple yield curves
            for i, (_, row) in enumerate(yields.head(10).iterrows()):  # Plot first 10
                rates = row.values
                ax.plot(tenors, rates, alpha=0.7, label=f'Curve {i+1}')

            ax.legend()

        ax.set_xlabel('Tenor (Years)')
        ax.set_ylabel('Yield (%)')
        ax.set_title(title)
        ax.grid(True, alpha=0.3)

        if save_path:
            plt.savefig(self.output_dir / save_path, dpi=300, bbox_inches='tight')

        return fig

    def plot_volatility_surface(self, vol_surface: np.ndarray,
                              strikes: np.ndarray,
                              expiries: np.ndarray,
                              title: str = 'Volatility Surface',
                              save_path: Optional[str] = None) -> plt.Figure:
        """
        Plot 3D volatility surface.

        Args:
            vol_surface: 2D array of volatilities
            strikes: Array of strike prices
            expiries: Array of expiries
            title: Plot title
            save_path: Path to save plot (optional)

        Returns:
            Matplotlib figure
        """
        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_subplot(111, projection='3d')

        X, Y = np.meshgrid(strikes, expiries)
        surf = ax.plot_surface(X, Y, vol_surface, cmap='viridis', alpha=0.8)

        ax.set_xlabel('Strike')
        ax.set_ylabel('Expiry')
        ax.set_zlabel('Volatility')
        ax.set_title(title)

        fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5)

        if save_path:
            plt.savefig(self.output_dir / save_path, dpi=300, bbox_inches='tight')

        return fig

    def plot_option_payoffs(self, spot_prices: np.ndarray,
                          strikes: Union[float, np.ndarray],
                          option_type: str = 'call',
                          title: str = 'Option Payoff',
                          save_path: Optional[str] = None) -> plt.Figure:
        """
        Plot option payoff diagram.

        Args:
            spot_prices: Array of spot prices
            strikes: Strike price(s)
            option_type: 'call' or 'put'
            title: Plot title
            save_path: Path to save plot (optional)

        Returns:
            Matplotlib figure
        """
        fig, ax = plt.subplots(figsize=(10, 6))

        if np.isscalar(strikes):
            strikes = [strikes]

        for strike in strikes:
            if option_type.lower() == 'call':
                payoff = np.maximum(spot_prices - strike, 0)
                label = f'Call (K={strike})'
            else:
                payoff = np.maximum(strike - spot_prices, 0)
                label = f'Put (K={strike})'

            ax.plot(spot_prices, payoff, label=label, linewidth=2)

        ax.axhline(y=0, color='black', linestyle='--', alpha=0.5)
        ax.axvline(x=strikes[0], color='red', linestyle='--', alpha=0.5, label='Strike')

        ax.set_xlabel('Spot Price')
        ax.set_ylabel('Payoff')
        ax.set_title(title)
        ax.legend()
        ax.grid(True, alpha=0.3)

        if save_path:
            plt.savefig(self.output_dir / save_path, dpi=300, bbox_inches='tight')

        return fig

    def plot_model_performance(self, y_true: np.ndarray,
                             y_pred: np.ndarray,
                             title: str = 'Model Performance',
                             save_path: Optional[str] = None) -> plt.Figure:
        """
        Plot model predictions vs actual values.

        Args:
            y_true: True values
            y_pred: Predicted values
            title: Plot title
            save_path: Path to save plot (optional)

        Returns:
            Matplotlib figure
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

        # Scatter plot
        ax1.scatter(y_true, y_pred, alpha=0.6, edgecolors='w', s=50)
        ax1.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()],
                'r--', linewidth=2, label='Perfect Prediction')
        ax1.set_xlabel('Actual Values')
        ax1.set_ylabel('Predicted Values')
        ax1.set_title('Predicted vs Actual')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Residual plot
        residuals = y_pred - y_true
        ax2.scatter(y_pred, residuals, alpha=0.6, edgecolors='w', s=50)
        ax2.axhline(y=0, color='r', linestyle='--', linewidth=2)
        ax2.set_xlabel('Predicted Values')
        ax2.set_ylabel('Residuals')
        ax2.set_title('Residual Plot')
        ax2.grid(True, alpha=0.3)

        fig.suptitle(title)

        if save_path:
            plt.savefig(self.output_dir / save_path, dpi=300, bbox_inches='tight')

        return fig

    def plot_feature_importance(self, feature_names: List[str],
                              importance_scores: np.ndarray,
                              title: str = 'Feature Importance',
                              top_n: int = 20,
                              save_path: Optional[str] = None) -> plt.Figure:
        """
        Plot feature importance scores.

        Args:
            feature_names: List of feature names
            importance_scores: Array of importance scores
            title: Plot title
            top_n: Number of top features to show
            save_path: Path to save plot (optional)

        Returns:
            Matplotlib figure
        """
        # Sort features by importance
        indices = np.argsort(importance_scores)[-top_n:]
        top_features = [feature_names[i] for i in indices]
        top_scores = importance_scores[indices]

        fig, ax = plt.subplots(figsize=(12, 8))

        bars = ax.barh(range(len(top_features)), top_scores)
        ax.set_yticks(range(len(top_features)))
        ax.set_yticklabels(top_features)
        ax.set_xlabel('Importance Score')
        ax.set_title(title)

        # Add value labels on bars
        for i, bar in enumerate(bars):
            width = bar.get_width()
            ax.text(width + 0.001, bar.get_y() + bar.get_height()/2,
                   '.3f', ha='left', va='center')

        plt.tight_layout()

        if save_path:
            plt.savefig(self.output_dir / save_path, dpi=300, bbox_inches='tight')

        return fig

    def plot_training_history(self, history: Dict[str, List[float]],
                            title: str = 'Training History',
                            save_path: Optional[str] = None) -> plt.Figure:
        """
        Plot training history for neural networks.

        Args:
            history: Dictionary with training metrics
            title: Plot title
            save_path: Path to save plot (optional)

        Returns:
            Matplotlib figure
        """
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))

        # Loss plot
        if 'loss' in history:
            axes[0].plot(history['loss'], label='Training Loss')
        if 'val_loss' in history:
            axes[0].plot(history['val_loss'], label='Validation Loss')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Loss')
        axes[0].set_title('Loss Over Time')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        # Metrics plot
        metrics_plotted = False
        for metric in ['accuracy', 'mae', 'mse']:
            if metric in history:
                axes[1].plot(history[metric], label=f'Training {metric.capitalize()}')
                metrics_plotted = True
            val_metric = f'val_{metric}'
            if val_metric in history:
                axes[1].plot(history[val_metric], label=f'Validation {metric.capitalize()}')
                metrics_plotted = True

        if metrics_plotted:
            axes[1].set_xlabel('Epoch')
            axes[1].set_ylabel('Metric Value')
            axes[1].set_title('Metrics Over Time')
            axes[1].legend()
            axes[1].grid(True, alpha=0.3)
        else:
            axes[1].text(0.5, 0.5, 'No metrics available',
                        ha='center', va='center', transform=axes[1].transAxes)

        fig.suptitle(title)

        if save_path:
            plt.savefig(self.output_dir / save_path, dpi=300, bbox_inches='tight')

        return fig


class RiskVisualizer(FinancialVisualizer):
    """
    Visualization class for risk metrics and analysis.
    """

    def plot_var_distribution(self, returns: pd.Series,
                            var_95: float,
                            cvar_95: float,
                            title: str = 'VaR Distribution',
                            save_path: Optional[str] = None) -> plt.Figure:
        """
        Plot return distribution with VaR and CVaR.

        Args:
            returns: Historical returns
            var_95: 95% VaR value
            cvar_95: 95% CVaR value
            title: Plot title
            save_path: Path to save plot (optional)

        Returns:
            Matplotlib figure
        """
        fig, ax = plt.subplots(figsize=(12, 8))

        # Plot return distribution
        sns.histplot(returns, bins=50, alpha=0.7, ax=ax, stat='density')

        # Plot normal distribution fit
        mu, sigma = returns.mean(), returns.std()
        x = np.linspace(returns.min(), returns.max(), 100)
        ax.plot(x, 1/(sigma * np.sqrt(2 * np.pi)) * np.exp(- (x - mu)**2 / (2 * sigma**2)),
               'r-', linewidth=2, label='Normal Fit')

        # Add VaR and CVaR lines
        ax.axvline(x=-var_95, color='orange', linestyle='--', linewidth=2,
                  label=f'VaR 95% (${var_95:.2f})')
        ax.axvline(x=-cvar_95, color='red', linestyle='--', linewidth=2,
                  label=f'CVaR 95% (${cvar_95:.2f})')

        # Shade CVaR area
        x_shade = np.linspace(returns.min(), -cvar_95, 100)
        y_shade = 1/(sigma * np.sqrt(2 * np.pi)) * np.exp(- (x_shade - mu)**2 / (2 * sigma**2))
        ax.fill_between(x_shade, 0, y_shade, alpha=0.3, color='red', label='CVaR Region')

        ax.set_xlabel('Portfolio Return')
        ax.set_ylabel('Density')
        ax.set_title(title)
        ax.legend()
        ax.grid(True, alpha=0.3)

        if save_path:
            plt.savefig(self.output_dir / save_path, dpi=300, bbox_inches='tight')

        return fig

    def plot_drawdown_chart(self, portfolio_values: pd.Series,
                          title: str = 'Portfolio Drawdown',
                          save_path: Optional[str] = None) -> plt.Figure:
        """
        Plot portfolio drawdown chart.

        Args:
            portfolio_values: Time series of portfolio values
            title: Plot title
            save_path: Path to save plot (optional)

        Returns:
            Matplotlib figure
        """
        # Calculate drawdown
        cumulative = portfolio_values / portfolio_values.iloc[0]
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max

        fig, ax = plt.subplots(figsize=(15, 8))

        # Plot portfolio value
        ax2 = ax.twinx()
        ax.plot(portfolio_values.index, portfolio_values, 'b-', alpha=0.7, label='Portfolio Value')
        ax.set_ylabel('Portfolio Value', color='b')
        ax.tick_params(axis='y', labelcolor='b')

        # Plot drawdown
        ax2.fill_between(drawdown.index, 0, drawdown, color='red', alpha=0.3, label='Drawdown')
        ax2.set_ylabel('Drawdown', color='r')
        ax2.tick_params(axis='y', labelcolor='r')

        ax.set_xlabel('Date')
        ax.set_title(title)
        ax.grid(True, alpha=0.3)

        # Combine legends
        lines1, labels1 = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(lines1 + lines2, labels1 + labels2, loc='upper left')

        if save_path:
            plt.savefig(self.output_dir / save_path, dpi=300, bbox_inches='tight')

        return fig

    def plot_risk_decomposition(self, risk_contributions: Dict[str, float],
                              title: str = 'Risk Decomposition',
                              save_path: Optional[str] = None) -> plt.Figure:
        """
        Plot risk decomposition by asset/component.

        Args:
            risk_contributions: Dictionary of risk contributions
            title: Plot title
            save_path: Path to save plot (optional)

        Returns:
            Matplotlib figure
        """
        fig, ax = plt.subplots(figsize=(12, 8))

        assets = list(risk_contributions.keys())
        contributions = list(risk_contributions.values())

        bars = ax.bar(assets, contributions, alpha=0.7)
        ax.set_xlabel('Asset/Component')
        ax.set_ylabel('Risk Contribution')
        ax.set_title(title)
        ax.grid(True, alpha=0.3)

        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.001,
                   '.3f', ha='center', va='bottom')

        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()

        if save_path:
            plt.savefig(self.output_dir / save_path, dpi=300, bbox_inches='tight')

        return fig

    def plot_stress_test_results(self, stress_results: Dict[str, Dict[str, float]],
                               title: str = 'Stress Test Results',
                               save_path: Optional[str] = None) -> plt.Figure:
        """
        Plot stress test results.

        Args:
            stress_results: Dictionary of stress test results
            title: Plot title
            save_path: Path to save plot (optional)

        Returns:
            Matplotlib figure
        """
        scenarios = list(stress_results.keys())
        losses = [result['loss_percentage'] for result in stress_results.values()]

        fig, ax = plt.subplots(figsize=(12, 8))

        bars = ax.bar(scenarios, losses, alpha=0.7, color='coral')
        ax.set_xlabel('Stress Scenario')
        ax.set_ylabel('Loss (%)')
        ax.set_title(title)
        ax.grid(True, alpha=0.3)

        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                   '.1f', ha='center', va='bottom')

        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()

        if save_path:
            plt.savefig(self.output_dir / save_path, dpi=300, bbox_inches='tight')

        return fig


class InteractiveVisualizer:
    """
    Interactive visualizations using Plotly.
    """

    def create_interactive_yield_curve(self, yields: pd.DataFrame,
                                     title: str = 'Interactive Yield Curve') -> go.Figure:
        """
        Create interactive yield curve plot.

        Args:
            yields: DataFrame with yield data
            title: Plot title

        Returns:
            Plotly figure
        """
        fig = go.Figure()

        tenors = [col.replace('maturity_', '') for col in yields.columns if col.startswith('maturity_')]
        tenors = [float(t) for t in tenors]

        for i, (_, row) in enumerate(yields.iterrows()):
            fig.add_trace(go.Scatter(
                x=tenors,
                y=row.values,
                mode='lines+markers',
                name=f'Curve {i+1}',
                hovertemplate='Tenor: %{x:.1f}Y<br>Yield: %{y:.3f}%<extra></extra>'
            ))

        fig.update_layout(
            title=title,
            xaxis_title='Tenor (Years)',
            yaxis_title='Yield (%)',
            hovermode='x unified'
        )

        return fig

    def create_risk_dashboard(self, returns: pd.Series,
                            var_95: float,
                            cvar_95: float) -> go.Figure:
        """
        Create interactive risk dashboard.

        Args:
            returns: Historical returns
            var_95: 95% VaR
            cvar_95: 95% CVaR

        Returns:
            Plotly figure
        """
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Return Distribution', 'VaR Analysis', 'Drawdown', 'Rolling Volatility'),
            specs=[[{'type': 'histogram'}, {'type': 'indicator'}],
                   [{'type': 'scatter'}, {'type': 'scatter'}]]
        )

        # Return distribution
        fig.add_trace(
            go.Histogram(x=returns, nbinsx=50, name='Returns'),
            row=1, col=1
        )

        # VaR indicators
        fig.add_trace(
            go.Indicator(
                mode="number+gauge",
                value=var_95,
                title={'text': "VaR 95%"},
                gauge={'axis': {'range': [0, returns.std() * 3]}}
            ),
            row=1, col=2
        )

        # Drawdown
        cumulative = (1 + returns).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max

        fig.add_trace(
            go.Scatter(x=returns.index, y=drawdown, fill='tozeroy', name='Drawdown'),
            row=2, col=1
        )

        # Rolling volatility
        rolling_vol = returns.rolling(window=30).std() * np.sqrt(252)
        fig.add_trace(
            go.Scatter(x=returns.index, y=rolling_vol, name='Rolling Vol'),
            row=2, col=2
        )

        fig.update_layout(height=800, title_text="Risk Dashboard")
        return fig


def create_model_comparison_plot(models_metrics: Dict[str, Dict[str, float]],
                               title: str = 'Model Comparison',
                               save_path: Optional[str] = None) -> plt.Figure:
    """
    Create model comparison bar plot.

    Args:
        models_metrics: Dictionary of model metrics
        title: Plot title
        save_path: Path to save plot

    Returns:
        Matplotlib figure
    """
    models = list(models_metrics.keys())
    metrics = list(models_metrics[models[0]].keys())

    fig, axes = plt.subplots(len(metrics), 1, figsize=(12, 6*len(metrics)))
    if len(metrics) == 1:
        axes = [axes]

    for i, metric in enumerate(metrics):
        values = [models_metrics[model][metric] for model in models]
        bars = axes[i].bar(models, values, alpha=0.7)
        axes[i].set_title(f'{metric.upper()} Comparison')
        axes[i].set_ylabel(metric.upper())
        axes[i].grid(True, alpha=0.3)

        # Add value labels
        for bar in bars:
            height = bar.get_height()
            axes[i].text(bar.get_x() + bar.get_width()/2., height + max(values)*0.01,
                        '.4f', ha='center', va='bottom')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

    return fig


if __name__ == "__main__":
    # Example usage
    import numpy as np

    # Create sample data
    visualizer = FinancialVisualizer()

    # Sample yield curve data
    tenors = np.linspace(0.25, 30, 10)
    yields = 0.03 + 0.001 * tenors + np.random.normal(0, 0.002, 10)
    yield_df = pd.DataFrame([yields], columns=[f'maturity_{t}' for t in tenors])

    # Plot yield curve
    fig = visualizer.plot_yield_curve(yield_df, save_path='yield_curve.png')
    plt.show()

    # Sample option data
    spots = np.linspace(80, 120, 100)
    fig = visualizer.plot_option_payoffs(spots, [100, 105], 'call', save_path='option_payoff.png')
    plt.show()

    print("Visualization examples completed. Check results/plots directory for saved images.")