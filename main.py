import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage
from skimage.util import random_noise
from PIL import Image
import imageio
from Perona_Malik import *

# Paths to Images
# Original Image
IMAGE_PATH = "images/lena.png"
# Output files after processing
# GrayScale Image
IMAGE_GRAY = "images/lena_bw.png"
# Salt & Pepper Image
IMAGE_SP_PATH = "images/lena_salt_and_pepper.png"
# Possion Noise Image
IMAGE_POISSON_PATH = "images/lena_poisson.png"
# Median_Filtered_Poisson Image
MEDIAN_FILTERED_POISSON_PATH = "images/lena_poisson_median_filtered.png"
# Median Filtered S&P image
MEDIAN_FILTERED_SP_PATH = "images/lena_salt_pepper_median_filtered.png"

# # Gaussian Noise Filtered Image
# GAUSSIAN_FILTERED_IMAGE = "images/gaussian_noise_filtered_dog.jpg"
# # Salt & Pepper Noise Filtered Image
# SALT_PEPPER__FILTERED_IMAGE = "images/salt_pepper_filtered_dog.jpg"
# # Gaussian Noise Filtered Image - Edge
# GAUSSIAN_FILTERED_IMAGE_EDGE = "images/gaussian_noise_filtered_dog_edge.jpg"
# # Salt & Pepper Noise Filtered Image - Edge
# SALT_PEPPER__FILTERED_IMAGE_EDGE = "images/salt_pepper_filtered_dog_edge.jpg"

# Read Image
def read_image(path):
  img = Image.open(path)
  img = img.convert('L') # Convert to Gray Scale
  img_array = np.array(img)
  return img_array

def add_noise():
    """
        Load the Image 
        and then add noise with salt & pepper and Possion Distribution
    """

    print("adding noise...")

    figure = plt.figure()
    plt.gray()

    image = read_image(IMAGE_PATH)
    possion_image = random_noise(image=image, mode='poisson', clip=True)
    salt_pepper_image = random_noise(image=image, mode='s&p', clip=True)

    ax1 = figure.add_subplot(131)
    plt.title("Original Image")
    plt.axis('off')
    ax2 = figure.add_subplot(132)
    plt.title("Poission Noise Image")
    plt.axis('off')
    ax3 = figure.add_subplot(133)
    plt.title("Salt & Pepper Noise Image")
    plt.axis('off')

    ax1.imshow(image)
    ax2.imshow(possion_image)
    ax3.imshow(salt_pepper_image)

    figure.tight_layout()
    plt.show()

    imageio.imwrite(IMAGE_GRAY, image)
    print(f"\tGray Scale Image saved in: {IMAGE_GRAY}")
    imageio.imwrite(IMAGE_POISSON_PATH, np.uint8(possion_image * 255))
    print(f"\tImage with Poisson noise saved in: {IMAGE_POISSON_PATH}")
    imageio.imwrite(IMAGE_SP_PATH, np.uint8(salt_pepper_image * 255))
    print(f"\tImage with Salt & Pepper noise saved in: {IMAGE_SP_PATH}")
    
    print("")

    # done with adding noise to images

# Apply median Filtering to remove the noise (Standard Baseline)
def meidan_filtering():
    """
    Load the noisy images
        1. Salt and Pepper Image
        2. Poission Image
    and apply simple Median Filter to denoise the image
    """
    print("\n Applying Median Filtering to the noisy Images")

    figure = plt.figure()

    original_image = read_image(IMAGE_PATH)

    possion_image = read_image(IMAGE_POISSON_PATH)
    salt_pepper_image = read_image(IMAGE_SP_PATH)

    possion_filtered_image = ndimage.median_filter(possion_image, size = 3)
    salt_pepper_filtered_image = ndimage.median_filter(salt_pepper_image, size = 3)

    psnr_poisson = PSNR(possion_filtered_image, original_image)
    psnr_snp = PSNR(salt_pepper_filtered_image, original_image)

    print(f"\tPSNR for Poisson Noise Image: {psnr_poisson}")
    print(f"\tPSNR for Salt & Pepper Noise Image: {psnr_snp}")

    mse_poisson = nmse(possion_filtered_image, original_image)
    mse_snp = nmse(salt_pepper_filtered_image, original_image)

    print(f"\tNMSE for Poisson Noise Image: {mse_poisson}")
    print(f"\tNMSE for Salt & Pepper Noise Image: {mse_snp}")

    imageio.imwrite(MEDIAN_FILTERED_POISSON_PATH, possion_filtered_image)
    print(f"\tImage with Poisson noise with median filter saved in: {MEDIAN_FILTERED_POISSON_PATH}")
    imageio.imwrite(MEDIAN_FILTERED_SP_PATH, salt_pepper_filtered_image)
    print(f"\tImage with S&P noise with Median filter saved in: {MEDIAN_FILTERED_SP_PATH}")

    ax1 = figure.add_subplot(321)
    plt.title("Poisson Noise Image")
    plt.axis('off')
    ax2 = figure.add_subplot(322)
    plt.title("Salt & Pepper Noise Image")
    plt.axis('off')
    ax3 = figure.add_subplot(323)
    plt.title("Poisson Noise Image Median Filtered")
    plt.axis('off')
    ax4 = figure.add_subplot(324)
    plt.title("Salt and Pepper Noise Image Median Filtered")
    plt.axis('off')


    log_pde_p = anisotropic_diffusion(
        image = possion_image,
        function = f2,
        iterations = 100,
        k = 0.01,
        log_time = 100 // (5 - 1),
        lamb = 0.1
    )

    log_pde_s = anisotropic_diffusion(
        image = salt_pepper_image,
        function = f2,
        iterations = 80,
        k = 0.15,
        log_time = 100 // (5 - 1),
        lamb = 0.1
    )

    ax5 = figure.add_subplot(325)
    plt.title("Poisson Noise Image PDE")
    plt.axis('off')
    ax6 = figure.add_subplot(326)
    plt.title("Salt & Pepper Noise Image PDE")
    plt.axis('off')


    ax1.imshow(possion_image)
    ax2.imshow(salt_pepper_image)
    ax3.imshow(possion_filtered_image)
    ax4.imshow(salt_pepper_filtered_image)
    ax5.imshow(log_pde_p[-1])
    ax6.imshow(log_pde_s[-1])

    figure.tight_layout()
    plt.show()

    print("")

# def plot_edge_functions():
    
#     # Generate Values for gradI
#     gradI_values = np.linspace(0, 10, 100)

#     # Compute the Values for both the functions
#     k = 2
#     f1_values = f1(gradI_values, k)
#     f2_values = f2(gradI_values, k)

#     figure = plt.figure(1)

#     axes = figure.add_subplot(111)

#     # Plot the graph
#     axes.plot(gradI_values, f1_values, label = 'f1')
#     axes.plot(gradI_values, f2_values, label = 'f2')

#     axes.set_xlim(0, None)
#     axes.set_xticks(np.arange(0, 11, 1))
#     axes.set_yticks(np.arange(0, 1.25, 0.25))
#     axes.set_xlabel('gradI')
#     axes.set_ylabel('f(gradI)')
#     axes.set_title(f'Edge Stopping Function for k = {k}')
#     axes.legend()

#     figure.tight_layout()
#     plt.show()

#     print("")

def plot_edge_functions(k_values, f, const):
    # Generate Values for gradI
    gradI_values = np.linspace(0, 10, 100)

    fig_f1 = plt.figure()
    ax_f1 = fig_f1.add_subplot(111)

    for k in k_values:
        # Compute the Values for both the functions
        f1_values = f(gradI_values, k)

        # Plot f1 in a separate figure
        
        
        ax_f1.plot(gradI_values, f1_values, label=f'{const} (k={k})')
        

        # Plot f2 in a separate figure
        
        
        
    ax_f1.set_xlim(0, None)
    ax_f1.set_xticks(np.arange(0, 11, 1))
    ax_f1.set_yticks(np.arange(0, 1.25, 0.25))
    ax_f1.set_xlabel('gradI')
    ax_f1.set_ylabel('f(gradI)')
    ax_f1.set_title(f'Edge Stopping Function {const} for varying K values')
    ax_f1.legend()
    fig_f1.tight_layout()
    plt.show()

    
    print("")

def PDE(image_path: str, noise: str, iterations: int, k: int, lamb: float, num_col: int = 5):

    print("Applying PDE for the noisy Image")

    image = read_image(image_path)
    # def anisotropic_diffusion(image, function, iterations, k, log_time, lamb=0.01):
    log_pde = anisotropic_diffusion(
        image = image,
        function = f1,
        iterations = iterations,
        k = k,
        log_time = iterations // (num_col - 1),
        lamb = lamb
    )
    
    figure = plt.figure()
    plt.title(f"PDE on {noise} noise Image with k = {k}")
    plt.axis('off')
    plt.gray()

    for i in range(len(log_pde)):
        ax1 = figure.add_subplot(1, num_col, i + 1)
        plt.title(f"t = {iterations // (num_col - 1) * i}")
        plt.axis('off')
        ax1.imshow(log_pde[i])

    plt.savefig(f"images/PDE_{image_path.split('/')[-1]}")
    plt.show()
    print(f"\tPDE with {noise} noise image saved in: images/PDE_{image_path.split('/')[-1]}\n")

    print(f"Analyzing the PSNR of {noise} Noisy Image")

    original_image = read_image(IMAGE_PATH)
    original_image = original_image / 255

    PSNR_values = []
    for target in log_pde:
        PSNR_values.append(PSNR(target, original_image))
    
    NMSE_values = []
    for target in log_pde:
        NMSE_values.append(nmse(target, original_image))
    
    # print(f"\tPSNR for Gaussian with k = {k} in iterations {iterations} is: ")

    figure = plt.figure()
    x = np.arange(0, iterations + iterations / (num_col - 1), iterations / (num_col - 1))
    ax1 = figure.add_subplot(2, 1, 1)
    ax1.plot(x, PSNR_values, ".-")
    plt.xlabel("Iterations")
    plt.ylabel("PSNR/dB")
    plt.title(f"PSNR for {noise} Noise image with k = {k}")

    ax2 = figure.add_subplot(2, 1, 2)
    ax2.plot(x, NMSE_values, ".-")
    plt.xlabel("Iterations")
    plt.ylabel("NMSE")
    plt.title(f"NMSE for {noise} Noise image with k = {k}")
    plt.tight_layout()
    plt.savefig(f"images/PSNR_{noise}.jpg")
    print(f"\tPSNR result save in: images/PSNR_{noise}.jpg")
    print(f"\tPSNR for {noise} with k = {k} in iterations {iterations} is: ")
    print(f"\t\t{PSNR_values}")
    plt.show()

    print("")

def compare_k(
        image_path: str,
        noise: str,
        iterations: int,
        k: np.ndarray,
        lamb: float,

):
    """
    Compare the effects of different k values on the image
    """
    print("comparing the effects of different k...")
    image = read_image(image_path)
    result_image = []
    result_psnr = []

    original_image = read_image(IMAGE_PATH)

    for value_k in k:
        log = anisotropic_diffusion(
            image = image, 
            function = f2, 
            iterations = iterations, 
            log_time = iterations,
            k = value_k, 
            lamb = lamb
        )
        result_image.append(log[-1])
        result_psnr.append(PSNR(log[-1], original_image))
    
    print(f"\tPSNR for {noise} nosie image : ", result_psnr)

    figure = plt.figure()
    plt.axis('off')
    plt.gray()

    ax1 = figure.add_subplot(1, 1, 1)
    ax1.plot(k, result_psnr, ".-")
    plt.xlabel("k")
    plt.ylabel("PSNR/dB")
    plt.title(f"PSNR for {noise} noise image with {iterations} iterations")

    figure.tight_layout()
    plt.savefig(f"images/PDE_k_{noise}.jpg")
    plt.show()

    print("")


    iterations_gaussian = 40
    iterations_sp = 100
    k = np.arange(0.05, 0.61, 0.01)
    lamb = 0.1


def main():
    print("\nECE6560 course project")
    help(add_noise)

    add_noise()
    meidan_filtering()
    plot_edge_functions([0.01, 0.03, 0.05, 0.1, 0.3, 0.7, 1, 2, 5, 10], f1, "F1")
    plot_edge_functions([0.01, 0.03, 0.05, 0.1, 0.3, 0.7, 1, 2, 5, 10], f2, "F2")
    # Poisson Noise Image
    PDE(IMAGE_POISSON_PATH, "Poisson", 100, 0.1, 0.1)

    # Salt & Pepper Noise Image
    PDE(IMAGE_SP_PATH, "Salt & Pepper", 80, 0.3, 0.1)

    # Compare the effects of different k values
    # For Poisson Noise Image
    compare_k(IMAGE_POISSON_PATH, "Poisson", 80, np.arange(0.05, 0.61, 0.01), 0.1)

    # For Salt & Pepper Noise Image
    compare_k(IMAGE_SP_PATH, "Salt & Pepper", 80, np.arange(0.05, 0.61, 0.01), 0.1)

    print("\ndone...")


if __name__ == "__main__":
    main()
