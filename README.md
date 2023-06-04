# CUDA Grayscale Conversion

## Aim:
The aim of this project is to demonstrate how to convert an image to grayscale using CUDA programming without relying on the OpenCV library. It serves as an example of GPU-accelerated image processing using CUDA.

## Procedure:
1. Load the input image using the `stb_image` library.
2. Allocate memory on the GPU for the input and output image buffers.
3. Copy the input image data from the CPU to the GPU.
4. Define a CUDA kernel function that performs the grayscale conversion on each pixel of the image.
5. Launch the CUDA kernel with appropriate grid and block dimensions.
6. Copy the resulting grayscale image data from the GPU back to the CPU.
7. Save the grayscale image using the `stb_image_write` library.
8. Clean up allocated memory.

## Output:

### Input file:
![gr](https://github.com/vignesh0011/PCA---Mini-Project-Mini-Project---Face-Detection-or-Convert-an-image-into-gray-scale-image-using-CUD/assets/53014593/3f3ce9ea-0884-46f8-b6d7-599fba23b455)


### Output file:
![output_image](https://github.com/vignesh0011/PCA---Mini-Project-Mini-Project---Face-Detection-or-Convert-an-image-into-gray-scale-image-using-CUD/assets/53014593/bf8c2aad-6149-44ca-aabc-ab185e6254f8)

## Result:
The CUDA program successfully converts the input image to grayscale using the GPU. The resulting grayscale image is saved as an output file. This example demonstrates the power of GPU parallelism in accelerating image processing tasks.
