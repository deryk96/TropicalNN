import matplotlib.pyplot as plt

def plot_images_in_grid(list_of_xs, row_labels, col_labels, save_path, input_elements):
    num_rows = len(row_labels)
    num_cols = len(col_labels)
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(num_cols * 2, num_rows * 2))
    if input_elements == 784:
        cmap = 'gray'
    else:
        cmap = None
    for i in range(num_rows):
        for j in range(num_cols):
            ax = axes[i, j]
            ax.imshow(list_of_xs[j][i,:,:,:], cmap=cmap)
            ax.axis('off')
            if i == 0:
                ax.set_title(col_labels[j], size='large')
    
    #plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()
