from numpy import arange
import matplotlib.pyplot as plt
import os

def get_latest_version(model_dir="../../models"):
    """
    Gets the latest version of the model.
    If no previous version exists, it returns -1.
    """
    version = -1
    for folder in os.listdir(model_dir):
        if folder.startswith('v'):
            curr_version = int(folder[1:])
            version = max(version, curr_version)

    return version

def visualize_metrics(e, print_every,
                    train_steps, test_steps, 
                    train_losses, train_accuracies, 
                    test_losses, test_accuracies, 
                    model_dir="../../models", vis_dir='../../reports/figures'):
    """
    Visualizes and saves the figure of the loss and accuracy over time of the trained model.
    """

    version = get_latest_version(model_dir)
    vis_version_dir = os.path.join(vis_dir, 'v{}'.format(version))

    fig, (ax1, ax2) = plt.subplots(2)

    ax1.plot(train_steps, train_losses, label="Train")
    ax1.plot(test_steps, test_losses, "-*", label="Validation")
    ax1.set_ylabel("NLL Loss")
    ax1.set_title("Loss")
    ax1.legend()

    ax2.plot(train_steps, train_accuracies, label="Train")
    ax2.plot(test_steps, test_accuracies, "-*", label="Validation")
    ax2.set_ylabel("Accuracy")
    ax2.set_title("Accuracy")
    ax2.legend()

    fig.supxlabel("step")

    fig.suptitle(f"Model: v{version} , Epoch: {e} , Step: {train_steps[-1]}")

    if not os.path.exists(vis_version_dir):
        os.makedirs(vis_version_dir)
    figdir = vis_version_dir + "/metrics.png"
    print(f"Saving figure in: {figdir}")
    plt.savefig(figdir)
    plt.close()

