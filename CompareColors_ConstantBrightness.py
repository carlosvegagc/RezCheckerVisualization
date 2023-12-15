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

def main(csv1, csv2):

    # Reading in CSVs
    colors1 = read_csv_colors(csv1)
    colors2 = read_csv_colors(csv2)

    # Get just the color
    colors1 = np.array(colors1)
    colors2 = np.array(colors2) 

    # Set the brightness to 128
    colors1[:, 0] = 128
    colors2[:, 0] = 128

    # Remove the lines 9, 10, 11, 12, 15, 16, 17, 18, 21, 22, 23, 24
    colors1 = np.delete(colors1, [9, 10, 11, 12, 15, 16, 17, 18, 21, 22, 23, 24], axis=0)
    colors2 = np.delete(colors2, [9, 10, 11, 12, 15, 16, 17, 18, 21, 22, 23, 24], axis=0)
    print(colors1.shape)

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
    rgb_colors = [lab_to_rgb(*color) for color in colors1]

    
    # Print mean deltaE
    deltaE_values = np.array(deltaE_values)
    print('Mean Delta E: ')
    print(deltaE_values.mean(axis=0))

    print('Max Delta E: ')
    print(deltaE_values.max(axis=0))

    print('Min Delta E: ')
    print(deltaE_values.min(axis=0))

    # Plotting
    plt.bar(range(len(deltaE_values)), deltaE_values,  color=[(b/255.0, g/255.0, r/255.0) for r, g, b in rgb_colors])
    plt.xlabel('Sample Number')
    plt.ylabel('Delta E Value')
    plt.title('Delta E Values for Color Samples')

    # # Print the deltaE values on the plot
    # for index, value in enumerate(deltaE_values):
    #     plt.text(index, value, str(round(value, 2)))
    
    # Show the deltaE max, avg and min outside the plot
    plt.text(0, 29, 'Max: ' + str(round(deltaE_values.max(), 2)))
    plt.text(0, 27, 'Avg: ' + str(round(deltaE_values.mean(), 2)))
    plt.text(0, 25, 'Min: ' + str(round(deltaE_values.min(), 2)))

    # Set the y axis limits
    plt.ylim([0, 30])

    
    # Save the plot
    plt.savefig('deltaE_plot_ConstantBrightness.png')

    plt.show()



    # Saving to CSV
    with open('deltaE_results.csv', 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['L1', 'a1', 'b1', 'L2', 'a2', 'b2', 'Delta E'])  # column headers
        for color1,color2, deltaE in zip(colors1,colors2, deltaE_values):
            writer.writerow([color1[0], color1[1], color1[2],color2[0], color2[1], color2[2], deltaE])

    # Export it to a csv the deltaE values
     

csv1_path = './colors_1.csv'
csv2_path = './colors_2.csv'

main(csv1_path, csv2_path)