#include <iostream>
#include <cstdlib>
#include <cstdio>
#include <cmath>
#include <cuda_runtime.h>
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

__global__ void grayscaleConversion(unsigned char* input, unsigned char* output, int width, int height, int channels) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < width && y < height) {
        int grayOffset = y * width + x;
        int rgbOffset = grayOffset * channels;
        unsigned char r = input[rgbOffset];
        unsigned char g = input[rgbOffset + 1];
        unsigned char b = input[rgbOffset + 2];

        unsigned char grayValue = static_cast<unsigned char>(0.299f * r + 0.587f * g + 0.114f * b);

        output[grayOffset] = grayValue;
    }
}

int main() {
    const char* inputImagePath = "C:\\Users\\CHARAN\\Downloads\\gr.jpg";
    const char* outputImagePath = "output_image.jpg";

    int width, height, channels;
    unsigned char* inputImage = stbi_load(inputImagePath, &width, &height, &channels, 0);

    if (inputImage == nullptr) {
        std::cerr << "Error loading image: " << inputImagePath << std::endl;
        return -1;
    }

    int imageSize = width * height * channels;

    // Allocate memory on GPU
    unsigned char* d_inputImage;
    unsigned char* d_outputImage;
    cudaMalloc((void**)&d_inputImage, imageSize * sizeof(unsigned char));
    cudaMalloc((void**)&d_outputImage, width * height * sizeof(unsigned char));

    // Copy input image data from CPU to GPU
    cudaMemcpy(d_inputImage, inputImage, imageSize * sizeof(unsigned char), cudaMemcpyHostToDevice);

    // Define CUDA grid and block dimensions
    dim3 blockDim(16, 16);
    dim3 gridDim((width + blockDim.x - 1) / blockDim.x, (height + blockDim.y - 1) / blockDim.y);

    // Launch CUDA kernel for grayscale conversion
    grayscaleConversion << <gridDim, blockDim >> > (d_inputImage, d_outputImage, width, height, channels);

    // Copy output image data from GPU to CPU
    unsigned char* outputImage = new unsigned char[width * height];
    cudaMemcpy(outputImage, d_outputImage, width * height * sizeof(unsigned char), cudaMemcpyDeviceToHost);

    // Save grayscale image
    stbi_write_jpg(outputImagePath, width, height, 1, outputImage, 100);

    // Cleanup
    stbi_image_free(inputImage);
    delete[] outputImage;
    cudaFree(d_inputImage);
    cudaFree(d_outputImage);

    return 0;
}
