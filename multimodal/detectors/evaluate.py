import torch
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
from torch.utils.data import DataLoader

def evaluate_model(model, dataloader, device):
    model.eval()  # Set the model to evaluation mode
    all_labels = []
    all_predictions = []
    all_probs = []
    
    with torch.no_grad():
        for batch in dataloader:
            # Unpack the batch data
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            graph_data = batch['graph'].to(device)
            dataset_info = batch['dataset_info']  # Assuming additional data is here
            
            # Forward pass
            probs = model(input_ids=input_ids, attention_mask=attention_mask, g=graph_data, dataset=dataset_info)
            
            # Convert probabilities to binary predictions
            predictions = (probs >= 0.5).long()
            
            # Store labels and predictions
            all_labels.extend(labels.cpu().numpy())
            all_predictions.extend(predictions.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
    
    # Convert lists to numpy arrays for metrics calculation
    all_labels = np.array(all_labels)
    all_predictions = np.array(all_predictions)
    all_probs = np.array(all_probs)
    
    # Calculate metrics
    accuracy = accuracy_score(all_labels, all_predictions)
    precision = precision_score(all_labels, all_predictions)
    recall = recall_score(all_labels, all_predictions)
    f1 = f1_score(all_labels, all_predictions)
    auc_roc = roc_auc_score(all_labels, all_probs)
    cm = confusion_matrix(all_labels, all_predictions)
    
    metrics = {
        'Accuracy': accuracy,
        'Precision': precision,
        'Recall': recall,
        'F1 Score': f1,
        'AUC-ROC': auc_roc,
        'Confusion Matrix': cm
    }
    
    return metrics

# Assuming you have a dataloader for the validation or test set
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

# Evaluate the model
val_metrics = evaluate_model(combined_model, validation_dataloader, device)

# Print the results
for metric_name, value in val_metrics.items():
    if metric_name == 'Confusion Matrix':
        print(f"{metric_name}:\n{value}")
    else:
        print(f"{metric_name}: {value:.4f}")


import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve

def plot_roc_curve(labels, probs):
    fpr, tpr, thresholds = roc_curve(labels, probs)
    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % auc_roc)
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")
    plt.show()

# Plot ROC curve
plot_roc_curve(all_labels, all_probs)
