"""
Plot utilities for PyOpenChannel examples
Author: Alexius Academia
"""

import os
import matplotlib.pyplot as plt
from datetime import datetime


def save_plot(filename_prefix="plot", plots_dir="plots", dpi=300, show_plot=True):
    """
    Save the current matplotlib figure to the plots directory with timestamp
    
    Parameters:
    -----------
    filename_prefix : str
        Prefix for the filename (default: "plot")
    plots_dir : str
        Directory to save plots (default: "plots")
    dpi : int
        Resolution for saved plot (default: 300)
    show_plot : bool
        Whether to display the plot after saving (default: True)
    
    Returns:
    --------
    str
        Full path to the saved plot file
    """
    
    # Create plots directory if it doesn't exist
    if not os.path.exists(plots_dir):
        os.makedirs(plots_dir)
    
    # Generate timestamp for unique filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{filename_prefix}_{timestamp}.png"
    filepath = os.path.join(plots_dir, filename)
    
    # Save the plot
    plt.savefig(filepath, dpi=dpi, bbox_inches='tight', facecolor='white')
    print(f"   💾 Plot saved to: {filepath}")
    
    if show_plot:
        plt.show()
    
    return filepath


def save_and_show_plot(filename_prefix="plot", plots_dir="plots", dpi=300):
    """
    Convenience function to save and show plot
    
    Parameters:
    -----------
    filename_prefix : str
        Prefix for the filename (default: "plot")
    plots_dir : str
        Directory to save plots (default: "plots")  
    dpi : int
        Resolution for saved plot (default: 300)
    
    Returns:
    --------
    str
        Full path to the saved plot file
    """
    return save_plot(filename_prefix, plots_dir, dpi, show_plot=True)


def save_plot_only(filename_prefix="plot", plots_dir="plots", dpi=300):
    """
    Save plot without showing it
    
    Parameters:
    -----------
    filename_prefix : str
        Prefix for the filename (default: "plot")
    plots_dir : str
        Directory to save plots (default: "plots")
    dpi : int
        Resolution for saved plot (default: 300)
    
    Returns:
    --------
    str
        Full path to the saved plot file
    """
    return save_plot(filename_prefix, plots_dir, dpi, show_plot=False)
