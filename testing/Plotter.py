import csv
import os
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.ticker as ticker
import seaborn as sns
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# Function to plot data from a CSV file
def plot_csv_data(file_path, ylabel: str = "Value"):
    with open(file_path, 'r') as file:
        reader = csv.reader(file)
        headers = next(reader)  # Get the headers from the first row
        
        # Read the data, skipping rows that don't have the correct number of columns
        data_rows = [row for row in reader if len(row) == len(headers)]
        data = np.array(data_rows, dtype=float)  # Convert the remaining rows to a numpy array

    # Plot each set of values
    for i in range(1, len(headers)):
        plt.plot(data[:, 0], data[:, i], label=headers[i])

    plt.xlabel(headers[0])  # Set x-axis label
    plt.ylabel(ylabel) # Set y-axis label

    # Format x-axis values as whole numbers
    plt.gca().xaxis.set_major_formatter(ticker.FormatStrFormatter('%0.0f'))

    plt.legend()  # Show legend
    plt.show()  # Display the plot


def plot_cm(cm, title='Confusion matrix', epoch: int = None, invert: bool = False):
    """
    Plot the confusion matrix
    """
    # Visualize the confusion matrix
    xlabels = ['Predicted Normal', 'Predicted Anomaly'] if not invert else ['Predicted Anomaly', 'Predicted Normal']
    ylabels = ['True Normal', 'True Anomaly'] if not invert else ['True Anomaly', 'True Normal']

    if epoch is not None:
        title += f" (Epoch {epoch})"
    else:
        title += " (Last Epoch)"

    plt.figure(figsize=(10, 7))
    plt.title(title)
    sns.heatmap(cm, annot=True, xticklabels=xlabels, yticklabels=ylabels, fmt='g')
    # plt.xlabel('Predicted')
    # plt.ylabel('Truth')
    plt.gca().xaxis.tick_top()  # Move x-axis labels to top
    plt.show()


def plot_cm_sklearn(cm, title='Confusion matrix', epoch: int = None):
    """
    Plot the confusion matrix using sklearn
    """

    if epoch is not None:
        title += f" (Epoch {epoch})"
    else:
        title += " (Last Epoch)"

    skdisp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Normal', 'Anomaly'])
    skdisp.plot(include_values=True, cmap='viridis', ax=None, xticks_rotation='horizontal', values_format=None)
    plt.title(title)
    plt.show()


def import_cm(cm_file_path, epoch: int = None, invert: bool = False):
    """
    Import confusion matrix from file,
    if epoch is None, import the last epoch
    invert: if True, invert the confusion matrix for anomaly centered metrics
    """

    if epoch is not None:
        # Import the confusion matrix for the given epoch, -1 to account for the header
        cm = np.loadtxt(cm_file_path, delimiter=',', usecols=range(1, 5), skiprows=1)[epoch-1]
    else:
        # Import the confusion matrix for the last epoch
        cm = np.loadtxt(cm_file_path, delimiter=',', usecols=range(1, 5), skiprows=1)[-1]
        for entry in cm:
            print("Entry: ", entry, "Type: ", type(entry))
    if invert:
        # Invert the confusion matrix
        cm = np.array([[cm[3], cm[2]], [cm[1], cm[0]]])

    cm = cm.reshape(2, 2)
    return cm
        


# Example usage
script_dir = os.path.dirname(__file__)  # Get the directory containing the script
file_path = os.path.join(script_dir, 'Generated Sheets/Training_metrics_100_anomaly_samples.csv')  # Construct the file path
cm_file_path = os.path.join(script_dir, 'Generated Sheets/Confusion_matrix_100_anomaly_samples.csv')  # CM file path
ylabel = "Metrics score"

# Import confusion matrix
cm = import_cm(cm_file_path)
cm2 = import_cm(cm_file_path, 4, True)

# plot_csv_data(file_path, ylabel)
plot_cm(cm)
plot_cm(cm2, "Inverted Confusion Matrix", 4, True)

# for comparison plot cm using sklearn:
plot_cm_sklearn(cm)



