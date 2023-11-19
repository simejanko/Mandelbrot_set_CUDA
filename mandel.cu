#include <iostream>
#include <complex>
#include <vector>
#include <chrono>

#include <cuda/std/complex>
#include <thrust/system_error.h>
#include <thrust/system/cuda/error.h>

#include "window.h"
#include "save_image.h"
#include "utils.h"

// Use an alias to simplify the use of complex type
using Complex = cuda::std::complex<float>;

#define cuda_err_chk(ans) { cuda_throw((ans), __FILE__, __LINE__); }

inline void cuda_throw(cudaError_t code, const char *file, int line) {
    if (code != cudaSuccess) {
        std::stringstream ss;
        ss << file << "(" << line << ")";
        auto file_and_line = ss.str();
        throw thrust::system_error(code, thrust::cuda_category(), file_and_line);
    }
}

// Convert a pixel coordinate to the complex domain
__device__
Complex scale(const window<int> &scr, const window<float> &fr, Complex c) {
    Complex aux(c.real() / (float) scr.width() * fr.width() + fr.x_min(),
                c.imag() / (float) scr.height() * fr.height() + fr.y_min());
    return aux;
}

// Check if a point is in the set or escapes to infinity, return the number if iterations
__device__
int escape_mandelbrot(Complex c, int iter_max) {
    Complex z(0);
    int iter = 0;

    while (cuda::std::abs(z) < 2.0 && iter < iter_max) {
        z = z * z + c;
        ++iter;
    }

    return iter;
}

// Loop over each pixel from our image and check if the points associated with this pixel escape to infinity
__global__
void get_number_iterations(const window<int> *scr, const window<float> *fract, int iter_max, int *colors) {
    int row_idx = blockIdx.y * blockDim.y + threadIdx.y;
    int col_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int row_stride = blockDim.y * gridDim.y;
    int col_stride = blockDim.x * gridDim.x;

    for (int i = row_idx; i < scr->height(); i += row_stride) {
        for (int j = col_idx; j < scr->width(); j += col_stride) {
            Complex c((float) (scr->x_min() + j), (float) (scr->y_min() + i));
            c = scale(*scr, *fract, c);
            colors[row_idx * scr->width() + col_idx] = escape_mandelbrot(c, iter_max);
        }
    }
}

void fractal(window<int> &scr, window<float> &fract, int iter_max, const char *fname, bool smooth_color) {
    auto start = std::chrono::steady_clock::now();
    window<int> *scr_uni;
    window<float> *fract_uni;
    int *colors_gpu;

    cuda_err_chk(cudaMallocManaged(&scr_uni, sizeof(window<int>)));
    cuda_err_chk(cudaMallocManaged(&fract_uni, sizeof(window<float>)));
    cuda_err_chk(cudaMalloc(&colors_gpu, sizeof(int) * scr.size()));
    *scr_uni = scr;
    *fract_uni = fract;

    dim3 threads_per_block(16, 16);
    dim3 n_blocks((scr.width() + threads_per_block.x - 1) / threads_per_block.x,
                  (scr.height() + threads_per_block.y - 1) / threads_per_block.y);
    get_number_iterations<<<n_blocks, threads_per_block>>>(scr_uni, fract_uni, iter_max, colors_gpu);
    cuda_err_chk(cudaGetLastError());
    cuda_err_chk(cudaDeviceSynchronize());

    std::vector<int> colors(scr.size());
    cuda_err_chk(cudaMemcpy(colors.data(), colors_gpu, sizeof(int) * scr.size(), cudaMemcpyDeviceToHost));

    auto end = std::chrono::steady_clock::now();
    std::cout << "Time to generate " << fname << " = " << std::chrono::duration<float, std::milli>(end - start).count()
              << " [ms]" << std::endl;

    cuda_err_chk(cudaFree(scr_uni));
    cuda_err_chk(cudaFree(fract_uni));
    cuda_err_chk(cudaFree(colors_gpu));

    // Save (show) the result as an image
    plot(scr, colors, iter_max, fname, smooth_color);
}

void mandelbrot() {
    // Define the size of the image
    window<int> scr(0, 2400, 0, 2400);
    // The domain in which we test for points
    window<float> fract(-2.2, 1.2, -1.7, 1.7);

    int iter_max = 500;
    const char *fname = "mandelbrot.png";
    bool smooth_color = true;

    // Experimental zoom (bugs ?). This will modify the fract window (the domain in which we calculate the fractal function)
    //zoom(1.0, -1.225, -1.22, 0.15, 0.16, fract); //Z2
    fractal(scr, fract, iter_max, fname, smooth_color);
}

int main() {
    mandelbrot();
    return 0;
}
