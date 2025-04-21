import numpy as np
from sklearn.metrics import (
    precision_recall_fscore_support, 
    roc_curve, 
    auc, 
    confusion_matrix
)

class FaceRecognitionMetrics:
    """
    Class to compute various face recognition performance metrics
    """
    @staticmethod
    def compute_rates(y_true, y_pred, y_scores):
        """
        Compute key performance metrics
        
        Args:
            y_true: Ground truth labels
            y_pred: Predicted labels
            y_scores: Similarity scores
        
        Returns:
            dict: Performance metrics
        """
        # Compute ROC curve and AUC
        fpr, tpr, thresholds = roc_curve(y_true, y_scores)
        roc_auc = auc(fpr, tpr)
        
        # Find Equal Error Rate (EER)
        eer_idx = np.argmin(np.abs(fpr - (1 - tpr)))
        eer = fpr[eer_idx]
        
        # Compute Precision, Recall, F1
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_true, y_pred, average='binary'
        )
        
        # Confusion Matrix
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        
        # Compute Rates
        tar = tp / (tp + fn)  # True Acceptance Rate (Recall)
        far = fp / (fp + tn)  # False Acceptance Rate
        frr = fn / (fn + tp)  # False Rejection Rate
        
        return {
            'True Acceptance Rate (TAR)': tar,
            'False Acceptance Rate (FAR)': far,
            'False Rejection Rate (FRR)': frr,
            'Equal Error Rate (EER)': eer,
            'Area Under Curve (AUC)': roc_auc,
            'Precision': precision,
            'Recall': recall,
            'F1-Score': f1,
            'ROC Curve': {
                'False Positive Rates': fpr,
                'True Positive Rates': tpr,
                'Thresholds': thresholds
            }
        }
    
    @staticmethod
    def plot_roc_curve(fpr, tpr, roc_auc):
        """
        Plot ROC Curve
        
        Args:
            fpr: False Positive Rates
            tpr: True Positive Rates
            roc_auc: Area Under Curve
        
        Returns:
            matplotlib.figure.Figure
        """
        import matplotlib.pyplot as plt
        
        plt.figure(figsize=(10, 7))
        plt.plot(fpr, tpr, color='darkorange', lw=2, 
                 label=f'ROC curve (AUC = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic (ROC) Curve')
        plt.legend(loc="lower right")
        
        return plt.gcf()
    
    @staticmethod
    def generate_detailed_report(metrics):
        """
        Generate a detailed text report of metrics
        
        Args:
            metrics: Dictionary of computed metrics
        
        Returns:
            str: Formatted report
        """
        report = "Face Recognition Performance Metrics Report\n"
        report += "=" * 50 + "\n\n"
        
        # Add key metrics
        key_metrics = [
            'True Acceptance Rate (TAR)',
            'False Acceptance Rate (FAR)', 
            'False Rejection Rate (FRR)',
            'Equal Error Rate (EER)',
            'Area Under Curve (AUC)',
            'Precision',
            'Recall', 
            'F1-Score'
        ]
        
        for metric in key_metrics:
            report += f"{metric}: {metrics[metric]:.4f}\n"
        
        return report