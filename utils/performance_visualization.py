import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import precision_recall_curve, roc_curve, auc

def plot_tar_far_frr(scores_same, scores_diff, thresholds=None):
    """
    Plot True Acceptance Rate (TAR), False Acceptance Rate (FAR), and 
    False Rejection Rate (FRR) as functions of threshold.
    
    Args:
        scores_same: Numpy array of similarity scores for same identity pairs
        scores_diff: Numpy array of similarity scores for different identity pairs
        thresholds: Optional array of thresholds to use
        
    Returns:
        fig: Matplotlib figure
        eer: Equal Error Rate value
        eer_threshold: Threshold at which EER occurs
    """
    if thresholds is None:
        thresholds = np.linspace(0, 1, 100)
    
    tar = np.zeros(len(thresholds))
    far = np.zeros(len(thresholds))
    frr = np.zeros(len(thresholds))
    
    # Calculate rates for each threshold
    for i, threshold in enumerate(thresholds):
        # True Acceptance Rate (TAR) - same identity correctly matched
        tar[i] = np.mean(scores_same >= threshold)
        
        # False Acceptance Rate (FAR) - different identities incorrectly matched
        far[i] = np.mean(scores_diff >= threshold)
        
        # False Rejection Rate (FRR) - same identity incorrectly rejected
        frr[i] = np.mean(scores_same < threshold)
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Plot rates
    ax.plot(thresholds, tar, 'g-', label='True Acceptance Rate (TAR)')
    ax.plot(thresholds, far, 'r-', label='False Acceptance Rate (FAR)')
    ax.plot(thresholds, frr, 'b-', label='False Rejection Rate (FRR)')
    
    # Find the Equal Error Rate (EER) point
    eer_index = np.argmin(np.abs(far - frr))
    eer = (far[eer_index] + frr[eer_index]) / 2
    eer_threshold = thresholds[eer_index]
    
    # Mark the EER point
    ax.plot(eer_threshold, eer, 'ko', markersize=8, label=f'EER = {eer:.3f} @ {eer_threshold:.3f}')
    ax.axvline(x=eer_threshold, color='k', linestyle='--', alpha=0.5)
    
    # Add labels and legend
    ax.set_xlabel('Threshold')
    ax.set_ylabel('Rate')
    ax.set_title('TAR, FAR, and FRR vs. Threshold')
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)
    
    # Set axis limits
    ax.set_xlim([0, 1])
    ax.set_ylim([-0.05, 1.05])
    
    return fig, eer, eer_threshold

def plot_roc_with_metrics(scores_same, scores_diff):
    """
    Plot ROC curve and calculate AUC
    
    Args:
        scores_same: Numpy array of similarity scores for same identity pairs
        scores_diff: Numpy array of similarity scores for different identity pairs
        
    Returns:
        fig: Matplotlib figure
        auc_value: Area Under Curve value
        fpr: False Positive Rate values
        tpr: True Positive Rate values
        thresholds: Threshold values for the ROC curve
    """
    # Create labels (1 for same, 0 for different)
    y_true = np.hstack([np.ones(len(scores_same)), np.zeros(len(scores_diff))])
    y_scores = np.hstack([scores_same, scores_diff])
    
    # Calculate ROC curve
    fpr, tpr, thresholds = roc_curve(y_true, y_scores)
    
    # Calculate area under curve
    auc_value = auc(fpr, tpr)
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Plot ROC curve
    ax.plot(fpr, tpr, 'b-', linewidth=2, label=f'ROC curve (AUC = {auc_value:.3f})')
    
    # Plot diagonal line (random classifier)
    ax.plot([0, 1], [0, 1], 'k--', label='Random classifier')
    
    # Add labels and legend
    ax.set_xlabel('False Positive Rate (FPR)')
    ax.set_ylabel('True Positive Rate (TPR)')
    ax.set_title('Receiver Operating Characteristic (ROC)')
    ax.legend(loc='lower right')
    ax.grid(True, alpha=0.3)
    
    # Set axis limits
    ax.set_xlim([-0.05, 1.05])
    ax.set_ylim([-0.05, 1.05])
    
    return fig, auc_value, fpr, tpr, thresholds

def plot_precision_recall_f1(scores_same, scores_diff):
    """
    Plot Precision, Recall, and F1-Score as functions of threshold
    
    Args:
        scores_same: Numpy array of similarity scores for same identity pairs
        scores_diff: Numpy array of similarity scores for different identity pairs
        
    Returns:
        fig: Matplotlib figure
        best_f1: Best F1-Score value
        best_threshold: Threshold at which best F1-Score occurs
        precision: Precision values
        recall: Recall values
        f1_scores: F1-Score values
    """
    # Create labels (1 for same, 0 for different)
    y_true = np.hstack([np.ones(len(scores_same)), np.zeros(len(scores_diff))])
    y_scores = np.hstack([scores_same, scores_diff])
    
    # Calculate precision-recall curve
    precision, recall, thresholds = precision_recall_curve(y_true, y_scores)
    
    # Calculate F1 score for each threshold
    f1_scores = np.zeros(len(thresholds))
    for i, threshold in enumerate(thresholds):
        # Get predictions at this threshold
        y_pred = (y_scores >= threshold).astype(int)
        
        # Calculate precision and recall at this threshold
        true_positives = np.sum((y_true == 1) & (y_pred == 1))
        false_positives = np.sum((y_true == 0) & (y_pred == 1))
        false_negatives = np.sum((y_true == 1) & (y_pred == 0))
        
        if true_positives == 0:
            f1_scores[i] = 0
        else:
            prec = true_positives / (true_positives + false_positives)
            rec = true_positives / (true_positives + false_negatives)
            f1_scores[i] = 2 * (prec * rec) / (prec + rec) if (prec + rec) > 0 else 0
    
    # Find optimal F1 score and threshold
    best_f1_idx = np.argmax(f1_scores)
    best_f1 = f1_scores[best_f1_idx]
    best_threshold = thresholds[best_f1_idx]
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Plot precision, recall, and F1
    ax.plot(thresholds, precision[:-1], 'b-', label='Precision')
    ax.plot(thresholds, recall[:-1], 'g-', label='Recall')
    ax.plot(thresholds, f1_scores, 'r-', label='F1-Score')
    
    # Mark the best F1 score
    ax.plot(best_threshold, best_f1, 'ko', markersize=8, 
            label=f'Best F1 = {best_f1:.3f} @ {best_threshold:.3f}')
    ax.axvline(x=best_threshold, color='k', linestyle='--', alpha=0.5)
    
    # Add labels and legend
    ax.set_xlabel('Threshold')
    ax.set_ylabel('Score')
    ax.set_title('Precision, Recall, and F1-Score vs. Threshold')
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)
    
    # Set axis limits
    ax.set_xlim([0, 1])
    ax.set_ylim([-0.05, 1.05])
    
    return fig, best_f1, best_threshold, precision, recall, f1_scores

def create_performance_dashboard(scores_same, scores_diff, current_threshold=0.5):
    """
    Create a comprehensive dashboard of face recognition performance metrics
    
    Args:
        scores_same: Numpy array of similarity scores for same identity pairs
        scores_diff: Numpy array of similarity scores for different identity pairs
        current_threshold: Current similarity threshold
        
    Returns:
        figs: List of matplotlib figures
        metrics: Dictionary of computed metrics
    """
    # Create uniform thresholds for all plots
    thresholds = np.linspace(0, 1, 100)
    
    # Plot TAR, FAR, FRR and get EER
    fig_rates, eer, eer_threshold = plot_tar_far_frr(scores_same, scores_diff, thresholds)
    
    # Plot ROC curve and get AUC
    fig_roc, auc_value, fpr, tpr, roc_thresholds = plot_roc_with_metrics(scores_same, scores_diff)
    
    # Plot Precision, Recall, F1 and get optimal F1
    fig_prf, best_f1, best_f1_threshold, precision, recall, f1_scores = plot_precision_recall_f1(
        scores_same, scores_diff)
    
    # Calculate metrics at current threshold
    current_idx = np.abs(thresholds - current_threshold).argmin()
    tar_current = np.mean(scores_same >= current_threshold)
    far_current = np.mean(scores_diff >= current_threshold)
    frr_current = np.mean(scores_same < current_threshold)
    
    # Calculate precision, recall, F1 at current threshold
    y_true = np.hstack([np.ones(len(scores_same)), np.zeros(len(scores_diff))])
    y_scores = np.hstack([scores_same, scores_diff])
    y_pred = (y_scores >= current_threshold).astype(int)
    
    true_positives = np.sum((y_true == 1) & (y_pred == 1))
    false_positives = np.sum((y_true == 0) & (y_pred == 1))
    false_negatives = np.sum((y_true == 1) & (y_pred == 0))
    
    if true_positives == 0:
        precision_current = 0
        recall_current = 0
        f1_current = 0
    else:
        precision_current = true_positives / (true_positives + false_positives)
        recall_current = true_positives / (true_positives + false_negatives)
        f1_current = 2 * (precision_current * recall_current) / (precision_current + recall_current) if (precision_current + recall_current) > 0 else 0
    
    # Compile metrics
    metrics = {
        'current_threshold': current_threshold,
        'tar_current': tar_current,
        'far_current': far_current,
        'frr_current': frr_current,
        'precision_current': precision_current,
        'recall_current': recall_current,
        'f1_current': f1_current,
        'eer': eer,
        'eer_threshold': eer_threshold,
        'auc': auc_value,
        'best_f1': best_f1,
        'best_f1_threshold': best_f1_threshold
    }
    
    return [fig_rates, fig_roc, fig_prf], metrics

def plot_performance_summary(metrics):
    """
    Create a summary visualization of key performance metrics
    
    Args:
        metrics: Dictionary of computed metrics
        
    Returns:
        fig: Matplotlib figure
    """
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Define metrics to display
    metric_names = [
        'True Acceptance Rate (TAR)', 
        'False Acceptance Rate (FAR)',
        'False Rejection Rate (FRR)',
        'Precision',
        'Recall',
        'F1-Score',
        'Area Under Curve (AUC)'
    ]
    
    metric_values = [
        metrics['tar_current'],
        metrics['far_current'],
        metrics['frr_current'],
        metrics['precision_current'],
        metrics['recall_current'],
        metrics['f1_current'],
        metrics['auc']
    ]
    
    # Define colors
    colors = ['green', 'red', 'blue', 'purple', 'orange', 'brown', 'cyan']
    
    # Create horizontal bar chart
    y_pos = np.arange(len(metric_names))
    ax.barh(y_pos, metric_values, color=colors, alpha=0.7)
    
    # Add metric values as text
    for i, value in enumerate(metric_values):
        ax.text(max(value + 0.02, 0.15), i, f'{value:.3f}', va='center')
    
    # Add labels
    ax.set_yticks(y_pos)
    ax.set_yticklabels(metric_names)
    ax.set_xlabel('Score')
    ax.set_title(f'Face Recognition Performance Metrics (Threshold = {metrics["current_threshold"]:.2f})')
    
    # Add threshold information
    text_str = (
        f'Current Threshold: {metrics["current_threshold"]:.3f}\n'
        f'EER Threshold: {metrics["eer_threshold"]:.3f} (EER = {metrics["eer"]:.3f})\n'
        f'Optimal F1 Threshold: {metrics["best_f1_threshold"]:.3f} (F1 = {metrics["best_f1"]:.3f})'
    )
    
    # Position text
    ax.text(0.5, -0.2, text_str, transform=ax.transAxes, ha='center', 
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # Set axis limits
    ax.set_xlim([0, 1.1])
    
    plt.tight_layout()
    return fig