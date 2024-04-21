import numpy as np
from PIL import Image
import math



# Define the Edge Stopping Functions
def f1(gradI, k):
  """
  param gradI: Gradient of the Image
  param k: K parameter
  return: function value
  """
  return np.exp(-1 * (np.power(gradI, 2)) / (np.power(k, 2)))

def f2(gradI, k):
  """
  param gradI: Gradient of the Image
  param k: K parameter
  return: function value
  """
  f = 1 / (1 + ((gradI / k)** 2))
  return f



def anisotropic_diffusion(image, function, iterations, k, log_time, lamb=0.01):
    """
    param image: numpy array of the image
    param iterations: number of iterations
    param k: parameter k
    param log_time: time when to log 
    param lamb: lambda value
    return:
    """
    image = image / 255
    new_image = np.zeros(image.shape, dtype=image.dtype)

    result = [image]

    for t in range(iterations):
        dn = image[:-2, 1:-1] - image[1:-1, 1:-1]
        ds = image[2:, 1:-1] - image[1:-1, 1:-1]
        de = image[1:-1, 2:] - image[1:-1, 1:-1]
        dw = image[1:-1, :-2] - image[1:-1, 1:-1]

        new_image[1:-1, 1:-1] = image[1:-1, 1:-1] + lamb * (
            function(dn, k) * dn +
            function(ds, k) * ds +
            function(de, k) * de +
            function(dw, k) * dw
        )

        image = new_image

        if (t+1) % log_time == 0:
            result.append(image.copy())

    return result


def PSNR(target, reference):
    """
    compute the PSNR between result image and original image
    :param target: Image numpy array
    :param reference: Image numpy array
    :return:
    """
    mse = np.mean((target - reference) ** 2)

    if mse == 0:
        return 100

    max_val = 1.0

    return 20 * math.log10(max_val / math.sqrt(mse))


def nmse(target, reference):
    """
    Compute the NMSE between the denoised image and the original image (normalized by L2 norm)
    param target: Image numpy array
    param reference: Image numpy array
    return: NMSE
    """
    # Calculate the L2 norm of the error
    l2_norm = np.linalg.norm(target - reference)

    # Calculate the NMSE
    nmse = l2_norm ** 2 / np.mean(reference ** 2)

    return nmse


def PSNR_split(target, ref, original_pde_result):
    """
    compute the PSNR between result image and original image
    :param target: type float64
    :param ref: type float64
    :param original_pde_result:
    :return:
    """
    h = target.shape[0]
    w = target.shape[1]

    target_copy = target.copy()

    for i in range(h):
        for j in range(w):
            if target_copy[i, j] == 0:
                target_copy[i, j] = original_pde_result[i, j]

    mse = np.mean((target_copy - ref) ** 2)

    if mse == 0:
        return 100

    max_val = 1.0

    return 20 * math.log10(max_val / math.sqrt(mse))

def read_image(path):
  img = Image.open(path)
  img = img.convert('L') # Convert to Gray Scale
  img_array = np.array(img)
  return img_array
def test():
    image = read_image("images/lena_poisson.png")
    res = anisotropic_diffusion(
        image = image, 
        function = f1, 
        log_time=1, 
        iterations=10, 
        k=0.1, 
        lamb=0.1
    )


if __name__ == "__main__":
    test()
