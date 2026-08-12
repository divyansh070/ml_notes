import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

plt.style.use('seaborn-v0_8-whitegrid')

def plot_svm_by_hand():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Coordinates
    A = np.array([1, 1])
    B = np.array([3, 3])
    midpoint = np.array([2, 2])
    
    # PLOT 1: Before Training
    ax1.scatter(*A, color='red', s=200, edgecolors='k', label='Class -1 (Point A)')
    ax1.scatter(*B, color='green', s=200, edgecolors='k', label='Class +1 (Point B)')
    
    ax1.text(A[0]-0.2, A[1]+0.2, 'A (1,1)', fontsize=12, color='red', weight='bold')
    ax1.text(B[0]-0.2, B[1]+0.2, 'B (3,3)', fontsize=12, color='green', weight='bold')
    
    ax1.set_xlim(0, 4.5)
    ax1.set_ylim(0, 4.5)
    ax1.set_title('Step 0: Before Training (The Raw Data)', fontsize=14)
    ax1.legend(loc='lower right')
    
    # PLOT 2: After Training
    ax2.scatter(*A, color='red', s=200, edgecolors='k', zorder=5)
    ax2.scatter(*B, color='green', s=200, edgecolors='k', zorder=5)
    
    ax2.text(A[0]-0.2, A[1]+0.2, 'A (1,1)', fontsize=12, color='red', weight='bold')
    ax2.text(B[0]-0.2, B[1]+0.2, 'B (3,3)', fontsize=12, color='green', weight='bold')
    
    # Plot midpoint
    ax2.scatter(*midpoint, color='black', s=100, marker='x', label='Midpoint (2,2)', zorder=5)
    
    # Decision boundary: x + y = 4 -> y = 4 - x
    x_vals = np.linspace(0, 4.5, 100)
    y_decision = 4 - x_vals
    ax2.plot(x_vals, y_decision, 'k-', linewidth=2, label='Hyperplane: x + y - 4 = 0')
    
    # Margins
    # Positive margin: y = 6 - x
    # Negative margin: y = 2 - x
    y_pos_margin = 6 - x_vals
    y_neg_margin = 2 - x_vals
    
    ax2.plot(x_vals, y_pos_margin, 'g--', linewidth=2, alpha=0.7, label='Positive Margin')
    ax2.plot(x_vals, y_neg_margin, 'r--', linewidth=2, alpha=0.7, label='Negative Margin')
    
    # Draw vector from A to B to show margin width
    ax2.annotate('', xy=B, xytext=A, arrowprops=dict(arrowstyle='<->', color='blue', linestyle=':', lw=2))
    
    # Text to show total margin
    ax2.text(1.2, 2.7, 'Margin Width = 2√2', color='blue', rotation=45, fontsize=12, weight='bold', 
             bbox=dict(facecolor='white', alpha=0.8, edgecolor='none'))
    
    ax2.set_xlim(0, 4.5)
    ax2.set_ylim(0, 4.5)
    ax2.set_title('Step 3: After Training (Geometric Solution)', fontsize=14)
    
    # Fix legend location
    ax2.legend(loc='upper right', bbox_to_anchor=(1.0, 1.0))
    
    plt.tight_layout()
    plt.savefig('/Users/divyanshverma/Desktop/ml_interview_questions/assets/svm_by_hand.png', dpi=300)
    plt.close()

if __name__ == "__main__":
    plot_svm_by_hand()
    print("SVM by hand plots successfully generated.")
