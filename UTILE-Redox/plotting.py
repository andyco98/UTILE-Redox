import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.stats import norm
import matplotlib.ticker as ticker
import tifffile

## TO DO ##

def plot_properties(csv_file, case_name):
    #Read the csv
    df = pd.read_csv(csv_file)

    # Plot the results
    # Create a figure with a 2x2 grid for the bottom plots and individual plots for the top row
    fig = plt.figure(figsize=(17, 8))
    gs = fig.add_gridspec(2, 3)

    ax1 = fig.add_subplot(gs[0, 0])  # Top left plot
    ax2 = fig.add_subplot(gs[0, 1])  # Top middle plot
    ax3 = fig.add_subplot(gs[0, 2])  # Top right plot (polar plot)

    ax4 = fig.add_subplot(gs[1, 0])  # Bottom left plot (spanning two columns)
    ax5 = fig.add_subplot(gs[1, 1])  # Bottom right plot (spanning two columns)
    ax6 = fig.add_subplot(gs[1, 2])
    #Volume ############################################
    #Read the data
    # # Manually compute the histogram using numpy

    volumes = df["volume"][df["volume"] > 0]

    # Compute the histogram on a logarithmic scale
    hist_counts, bin_edges = np.histogram(np.log10(volumes), bins=30)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    # Normalize the histogram counts to get percentages
    hist_percentages = 100 * hist_counts / np.sum(hist_counts)

    # Plot the histogram
    ax1.bar(bin_centers, hist_percentages, width=np.diff(bin_edges), edgecolor='black', alpha=0.7)

    # Set axis labels and title
    ax1.set_title('Individual Volume Distribution (Log Scale)', fontsize=18)
    ax1.set_xlabel('Log10(Volume) [voxels]', fontsize=18)
    ax1.set_ylabel('Frequency [%]', fontsize=18)

    # Set x-axis to logarithmic scale
    ax1.set_xlim(bin_edges[0], bin_edges[-1])
    ax1.set_ylim(0, max(hist_percentages) + 5)  # Adding a bit of padding on top
    ax1.tick_params(axis='x', labelsize=16)
    ax1.tick_params(axis='y', labelsize=16)
    # Plot the Gaussian distribution on log scale
    mu, std = np.mean(np.log10(volumes)), np.std(np.log10(volumes))
    x = np.linspace(bin_edges[0], bin_edges[-1], 1000)
    pdf = norm.pdf(x, mu, std)
    #ax1.plot(x, pdf * np.sum(hist_percentages) * np.diff(bin_edges)[0], '-', color='red', label=f'Mean={mu:.2f}, SD={std:.2f}')

    # Add legend
    ax1.legend([f"Mean={mu:.2f}, SD={std:.2f}"],loc='upper right', fontsize=14)

    ### VOlume computation wihtout log10
    # hist_counts, bin_edges = np.histogram(df["volume"], bins=30)

    # # Compute bin centers for plotting
    # bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    # # Normalize the histogram counts to get percentages
    # hist_percentages = 100 * hist_counts / np.sum(hist_counts)

    # mu, std = np.mean(df["volume"]), np.std(df["volume"])
    
    # ax1.bar(bin_centers, hist_percentages, width=np.diff(bin_edges), edgecolor='black', alpha=0.7)
    # ax1.set_title(f'Individual volume distribution', fontsize = 18)
    # ax1.set_xlabel("Volume [voxels]", fontsize = 18)
    # ax1.set_ylabel('Frequency [%]', fontsize = 18)
    # ax1.set_ylim(0, max(hist_percentages) + 5)  # Adding a bit of padding on top
    # ax1.tick_params(axis='x', labelsize=16)
    # ax1.tick_params(axis='y', labelsize=16)
    # # Plot the Gaussian distribution
    # x = np.linspace(min(df["volume"]), max(df["volume"]), 1000)
    # pdf = norm.pdf(x, mu, std)
    # ax1.plot(x, pdf * np.sum(hist_percentages) * np.diff(bin_edges)[0], '-', color='red', label=f'Mean={mu:.2f}, SD={std:.2f}')
            
    # ax1.legend(loc='upper right', fontsize = 14)

    #Sphericity ################################################
    #Read the data
    # Manually compute the histogram using numpy
    hist_counts, bin_edges = np.histogram(df["sphericity"], bins=30)

    # Compute bin centers for plotting
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    # Normalize the histogram counts to get percentages
    hist_percentages = 100 * hist_counts / np.sum(hist_counts)

    mu, std = np.mean(df["sphericity"]), np.std(df["sphericity"])
    
    ax2.bar(bin_centers, hist_percentages, width=np.diff(bin_edges), edgecolor='black', alpha=0.7)
    ax2.set_title(f'Individual sphericity distribution', fontsize = 18)
    ax2.set_xlabel(f'Sphericity', fontsize = 18)
    ax2.set_ylabel('Frequency [%]', fontsize = 18)
    ax2.set_ylim(0, max(hist_percentages) + 5)  # Adding a bit of padding on top
    ax2.tick_params(axis='x', labelsize=16)
    ax2.tick_params(axis='y', labelsize=16)
    # Plot the Gaussian distribution
    x = np.linspace(min(df["sphericity"]), max(df["sphericity"]), 1000)
    pdf = norm.pdf(x, mu, std)
    #ax2.plot(x, pdf * np.sum(hist_percentages) * np.diff(bin_edges)[0], '-', color='red', label=f'Mean={mu:.2f}, SD={std:.2f}')
            
    ax2.legend([f"Mean={mu:.2f}, SD={std:.2f}"],loc='upper right', fontsize = 14)

    # # "orientation" #####################################################
    # #Read the data
    # Create a 2D histogram
    theta_radians = np.radians(df["theta"])
    phi_radians = np.radians(df["phi"])

    hist, xedges, yedges = np.histogram2d(theta_radians, phi_radians, bins=36)
    ax_hist = fig.add_subplot(2, 3, 3, projection='polar')

    ax3.spines['left'].set_visible(False)
    ax3.spines['top'].set_visible(False)
    ax3.spines['bottom'].set_visible(False)
    ax3.spines['right'].set_visible(False)
    ax3.xaxis.set_ticks_position('none') 
    ax3.yaxis.set_ticks_position('none')
    ax3.xaxis.set_ticklabels([])
    ax3.yaxis.set_ticklabels([])

    
    # Increase font size for ticks and labels
    ax_hist.tick_params(axis='x', labelsize=14)  # Increase font size for theta ticks
    ax_hist.tick_params(axis='y', labelsize=14, labelcolor= "white")  # Increase font size for radial ticks

    # Plot
    x, y = np.meshgrid(xedges, yedges)
    pc = ax_hist.pcolormesh(x, y, hist.T,cmap='jet')

    # Customizing the plot
    ax_hist.set_theta_zero_location('N')
    ax_hist.set_theta_direction(-1)
    ax_hist.yaxis.set_major_locator(ticker.FixedLocator([np.radians(30), np.radians(60), np.radians(90)]))
    ax_hist.yaxis.set_major_formatter(ticker.FixedFormatter(['30°', '60°', '90°']))

    cbar = plt.colorbar(pc, ax=ax_hist, label='Number of Bubbles')
    cbar.set_label('Number of Bubbles', size=18)
    ax_hist.set_title('Bubble Orientation Distribution', fontsize = 18)



    #Elongation ###################################################
    #Read the data
    # Ensure no zero or negative values for log scale
    elongations = df["elongation"][df["elongation"] > 0]

    # Compute the histogram on a logarithmic scale
    hist_counts, bin_edges = np.histogram(np.log10(elongations), bins=30)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    # Normalize the histogram counts to get percentages
    hist_percentages = 100 * hist_counts / np.sum(hist_counts)

    # Plot the histogram
    ax4.bar(bin_centers, hist_percentages, width=np.diff(bin_edges), edgecolor='black', alpha=0.7)

    # Set axis labels and title
    ax4.set_title('Individual Elongation Distribution (Log Scale)', fontsize=18)
    ax4.set_xlabel('Log10(Elongation)', fontsize=18)
    ax4.set_ylabel('Frequency [%]', fontsize=18)

    # Set x-axis to logarithmic scale
    ax4.set_xlim(bin_edges[0], bin_edges[-1])
    ax4.set_ylim(0, max(hist_percentages) + 5)  # Adding a bit of padding on top
    ax4.tick_params(axis='x', labelsize=16)
    ax4.tick_params(axis='y', labelsize=16)
   # Plot the Gaussian distribution on log scale
    mu, std = np.mean(np.log10(elongations)), np.std(np.log10(elongations))
    x = np.linspace(bin_edges[0], bin_edges[-1], 1000)
    pdf = norm.pdf(x, mu, std)
    #ax4.plot(x, pdf * np.sum(hist_percentages) * np.diff(bin_edges)[0], '-', color='red', label=f'Mean={mu:.2f}, SD={std:.2f}')

    # Add legend
    ax4.legend([f"Mean={mu:.2f}, SD={std:.2f}"],loc='upper right', fontsize=14)


    #Flatness ###################################################
    #Read the data
    # Ensure no zero or negative values for log scale
    flatnesses = df["flatness"][df["flatness"] > 0]

    # Compute the histogram on a logarithmic scale
    hist_counts, bin_edges = np.histogram(np.log10(flatnesses), bins=30)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    # Normalize the histogram counts to get percentages
    hist_percentages = 100 * hist_counts / np.sum(hist_counts)

    # Plot the histogram
    ax5.bar(bin_centers, hist_percentages, width=np.diff(bin_edges), edgecolor='black', alpha=0.7)

    # Set axis labels and title
    ax5.set_title('Individual Flatness Distribution (Log Scale)', fontsize=18)
    ax5.set_xlabel('Log10(Flatness)', fontsize=18)
    ax5.set_ylabel('Frequency [%]', fontsize=18)

    # Set x-axis to logarithmic scale
    ax5.set_xlim(bin_edges[0], bin_edges[-1])
    ax5.set_ylim(0, max(hist_percentages) + 5)  # Adding a bit of padding on top
    ax5.tick_params(axis='x', labelsize=16)
    ax5.tick_params(axis='y', labelsize=16)
   # Plot the Gaussian distribution on log scale
    mu, std = np.mean(np.log10(flatnesses)), np.std(np.log10(flatnesses))
    x = np.linspace(bin_edges[0], bin_edges[-1], 1000)
    pdf = norm.pdf(x, mu, std)
    #ax5.plot(x, pdf * np.sum(hist_percentages) * np.diff(bin_edges)[0], '-', color='red', label=f'Mean={mu:.2f}, SD={std:.2f}')

    # Add legend
    ax5.legend([f"Mean={mu:.2f}, SD={std:.2f}"],loc='upper right', fontsize=14)
    # # Manually compute the histogram using numpy
    # hist_counts, bin_edges = np.histogram(df["elongation"], bins=30)

    # # Compute bin centers for plotting
    # bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    # # Normalize the histogram counts to get percentages
    # hist_percentages = 100 * hist_counts / np.sum(hist_counts)

    # mu, std = np.mean(df["elongation"]), np.std(df["elongation"])
    
    # ax4.bar(bin_centers, hist_percentages, width=np.diff(bin_edges), edgecolor='black', alpha=0.7)
    # ax4.set_title(f'Individual elongation distribution', fontsize = 18)
    # ax4.set_xlabel("Elongation", fontsize = 18)
    # ax4.set_ylabel('Frequency [%]', fontsize = 18)
    # ax4.set_ylim(0, max(hist_percentages) + 5)  # Adding a bit of padding on top
    # ax4.tick_params(axis='x', labelsize=16)
    # ax4.tick_params(axis='y', labelsize=16)
    # # Plot the Gaussian distribution
    # x = np.linspace(min(df["elongation"]), max(df["elongation"]), 1000)
    # pdf = norm.pdf(x, mu, std)
    # ax4.plot(x, pdf * np.sum(hist_percentages) * np.diff(bin_edges)[0], '-', color='red', label=f'Mean={mu:.2f}, SD={std:.2f}')
            
    # ax4.legend(loc='upper right', fontsize = 14)

    # #Wall proximity ##################################################
    # #Volume
    # #Read the data
    # # Manually compute the histogram using numpy
    hist_counts, bin_edges = np.histogram(df["closest_distance"], bins=30)

    # Compute bin centers for plotting
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    # Normalize the histogram counts to get percentages
    hist_percentages = 100 * hist_counts / np.sum(hist_counts)

    mu, std = np.mean(df["closest_distance"]), np.std(df["closest_distance"])
    
    ax6.bar(bin_centers, hist_percentages, width=np.diff(bin_edges), edgecolor='black', alpha=0.7)
    ax6.set_title(f'Individual wall proximity distribution', fontsize = 18)
    ax6.set_xlabel(f'Wall proximity [voxels]', fontsize = 18)
    ax6.set_ylabel('Frequency [%]', fontsize = 18)
    ax6.set_ylim(0, max(hist_percentages) + 5)  # Adding a bit of padding on top
    ax6.tick_params(axis='x', labelsize=16)
    ax6.tick_params(axis='y', labelsize=16)
    # Plot he Gaussian distribution
    x = np.linspace(min(df["closest_distance"]), max(df["closest_distance"]), 1000)
    pdf = norm.pdf(x, mu, std)
    #ax6.plot(x, pdf * np.sum(hist_percentages) * np.diff(bin_edges)[0], '-', color='red', label=f'Mean={mu:.2f}, SD={std:.2f}')
            
    ax6.legend([f"Mean={mu:.2f}, SD={std:.2f}"],loc='upper right', fontsize = 14)

    # Hide the 6th subplot as we only need 5
    #ax5.set_visible(False)


    plt.tight_layout()
    plt.savefig(f"./{case_name}/{case_name}_plots.png")
    plt.show()


def plot_densities(stack_path,case_name, target_class=1):
    with tifffile.TiffFile(stack_path) as tif:
        images = tif.asarray()

    # Assuming images are binary (0s and 1s)
    # Isolate the target class
    binary_target = (images == target_class)
    # Summation across all frames for XY plane
    xy_density = np.sum(binary_target, axis=0)

    # Summation for XZ plane
    xz_density = np.sum(binary_target, axis=1)

    # Summation for ZY plane
    zy_density = np.sum(binary_target, axis=2)

    # Visualization
    plt.figure(figsize=(15, 5))
 

    plt.subplot(1, 3, 1)
    plt.imshow(xy_density, cmap='jet')
    plt.title('XY Plane Density')
    plt.tick_params(left = False, right = False , labelleft = False , 
                labelbottom = False, bottom = False)
    plt.subplot(1, 3, 2)
    plt.imshow(xz_density, cmap='jet')
    plt.title('XZ Plane Density')
    plt.tick_params(left = False, right = False , labelleft = False , 
                labelbottom = False, bottom = False)
    plt.subplot(1, 3, 3)
    plt.imshow(zy_density, cmap='jet')
    plt.tick_params(left = False, right = False , labelleft = False , 
                labelbottom = False, bottom = False)
    plt.title('ZY Plane Density')
    plt.savefig(f"./{case_name}/densities_{case_name}.png")
    plt.show()

    

# csv_file = "C:/Users/a.colliard/Desktop/zeis_imgs/outputS9.csv"

# stack_path = 'C:/Users/a.colliard/Desktop/zeis_imgs/maskS9.tif'

#plot_properties(csv_file)

#plot_densities(stack_path)

