#ifndef WINDOW__H
#define WINDOW__H

#ifdef __CUDACC__
#define CUDA_HOSTDEV __host__ __device__
#else
#define CUDA_HOSTDEV
#endif

template <typename T>
class window {
	T _x_min, _x_max, _y_min, _y_max;

public:
	window(T x_min, T x_max, T y_min, T y_max)
	: _x_min(x_min), _x_max(x_max), _y_min(y_min), _y_max(y_max)
	{}

// Utility functions for getting the size, width and height of the window
    CUDA_HOSTDEV
    T size() const {
		return (width() * height());
	}

    CUDA_HOSTDEV
	T width() const {
		return (_x_max - _x_min);
	}

    CUDA_HOSTDEV
	T height() const {
		return (_y_max - _y_min);
	}

// Getters and setters for the window elements
    CUDA_HOSTDEV
	T x_min() const {
		return _x_min;
	}

    CUDA_HOSTDEV
	void x_min(T x_min) {
		_x_min = x_min;
	}

    CUDA_HOSTDEV
	T x_max() const {
		return _x_max;
	}

    CUDA_HOSTDEV
	void x_max(T x_max) {
		_x_max = x_max;
	}

    CUDA_HOSTDEV
    T y_min() const {
		return _y_min;
	}

    CUDA_HOSTDEV
    void y_min(T y_min) {
		_y_min = y_min;
	}

    CUDA_HOSTDEV
    T y_max() const {
		return _y_max;
	}

    CUDA_HOSTDEV
    void y_max(T y_max) {
		_y_max = y_max;
	}

// Reset all values
    CUDA_HOSTDEV
	void reset(T x_min, T x_max, T y_min, T y_max) {
		_x_min = x_min;
		_x_max = x_max;
		_y_min = y_min;
		_y_max = y_max;
	}
};
#endif

