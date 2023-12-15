import csv
import numpy as np
import matplotlib.pyplot as plt
import cv2

def read_csv_colors(filename):
    colors = []
    with open(filename, 'r') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            L = float(row['L'])
            a = float(row['a'])
            b = float(row['b'])
            colors.append((L, a, b))
    return colors

def compute_deltaE(color1, color2):
    l1 = color1[0] * 100 / 255
    l2 = color2[0] * 100 / 255
    dL = l2 - l1
    da = color2[1] - color1[1]
    db = color2[2] - color1[2]
    return np.sqrt(dL**2 + da**2 + db**2)


def lab_to_rgb(l, a, b):
    """Convert LAB value to RGB"""
    # Rescale LAB components to the expected range
    # l = l * 255 / 100
    # a = a + 128
    # b = b + 128

    # Convert the LAB value to RGB
    lab = np.array([[[l, a, b]]], dtype=np.uint8)
    rgb = cv2.cvtColor(lab, cv2.COLOR_Lab2BGR)
    return tuple(map(int, rgb[0][0]))

def plot_colors(ax, rgb_colors):
    # Assume ax is an AxesSubplot object that has been created elsewhere
    for x, color in enumerate(rgb_colors):
        color_norm = [c / 255 for c in color]
        # Change order from BGR to RGB
        color_norm = color_norm[::-1]


        ax.add_patch(plt.Rectangle((x, 0), 1, 1, color=color_norm))
    ax.set_xlim(0, len(rgb_colors))
    ax.set_ylim(0, 1)
    # ax.axis('off')  # Disable axis lines and labels

def main(csv1, csv2):

    # Reading in CSVs
    colors1 = read_csv_colors(csv1)
    colors2 = read_csv_colors(csv2)

    # Get just the color
    colors1 = np.array(colors1)
    colors2 = np.array(colors2) 

    # Get number of samples
    num_samples = len(colors1)

    deltaE_values = []

    for i in range(num_samples):
        color1 = colors1[i]
        color2 = colors2[i]

        deltaE_values.append( compute_deltaE( color1,  color2 ) )
   
    # # Force the program to wait here until a key is pressed
    # cv2.waitKey(0)

    # Convert colors from Lab to RGB
    rgb_colors_1 = [lab_to_rgb(*color) for color in colors1]
    rgb_colors_2 = [lab_to_rgb(*color) for color in colors2]

 
    # Plot the colors on a matrix with the first row colors 1 and second colors 2
     # Create a figure with two subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 4), subplot_kw=dict(xticks=[], yticks=[]))

    # Adjust the spacing between subplots if needed
    fig.subplots_adjust(hspace=0.1)  # Adjust this value to change spacing, hspace is height space between subplots


    # Plot the colors in the existing subplots
    plot_colors(ax1, rgb_colors_1)
    plot_colors(ax2, rgb_colors_2)

    # Add y-axis labels to the subplots
    ax1.set_ylabel('Img 1', rotation=0, ha='right', va='center')
    ax2.set_ylabel('Img 2', rotation=0, ha='right', va='center')


    # Save the plot
    plt.savefig('CharsColors.png')

    plt.show()




csv1_path = './colors_1.csv'
csv2_path = './colors_2.csv'
main(csv1_path, csv2_path)