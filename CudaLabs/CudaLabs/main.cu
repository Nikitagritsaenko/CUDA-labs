#include <new>
#include <algorithm>
#include <random>
#include <iostream>
#include <windows.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#define DIM 1000
#define MATRIX_MIN_VALUE -100
#define MATRIX_MAX_VALUE 100
#define BLOCK_SIZE 16

double* generateRandomMatrix(size_t dim)
{
	std::random_device device;
	std::mt19937 generator(device());
	std::uniform_real_distribution<double> distribution(MATRIX_MIN_VALUE, MATRIX_MAX_VALUE);

	size_t elementsTotal = dim * dim;
	double* randomMatrix = new double[elementsTotal];

	for (size_t i = 0; i < elementsTotal; ++i) 
	{
		randomMatrix[i] = distribution(generator);
	}

	return randomMatrix;
}

float cpuMatrixMultiplication(double* A, double* B, double* C, size_t n)
{
	LARGE_INTEGER frequency;
	QueryPerformanceFrequency(&frequency);

	LARGE_INTEGER start;
	QueryPerformanceCounter(&start);

	for (size_t i = 0; i < n; ++i)
	{
		for (size_t j = 0; j < n; ++j)
		{
			for (size_t k = 0; k < n; ++k)
			{
				C[i * n + j] += A[i * n + k] * B[k * n + j];
			}
		}
	}

	LARGE_INTEGER end;
	QueryPerformanceCounter(&end);

	return (float)(end.QuadPart - start.QuadPart) / frequency.QuadPart;
}

__global__ void matrixMul(double* A, double* B, double* C, size_t n)
{
	size_t row = blockIdx.y * blockDim.y + threadIdx.y;
	size_t column = blockIdx.x * blockDim.x + threadIdx.x;

	double cellValue = 0;
	if (row < n && column < n)
	{
		for (int i = 0; i < n; ++i)
		{
			cellValue += A[row * n + i] * B[i * n + column];
		}
		C[row * n + column] = cellValue;
	}
}

__global__ void matrixMulShared(double* A, double* B, double* C, size_t n)
{
	size_t row = blockDim.y * blockIdx.y + threadIdx.y;
	size_t column = blockDim.x * blockIdx.x + threadIdx.x;

	double cellValue = 0;

	__shared__ double sA[BLOCK_SIZE][BLOCK_SIZE];
	__shared__ double sB[BLOCK_SIZE][BLOCK_SIZE];

	for (size_t k = 0; k * BLOCK_SIZE < n; k++) {

		if (row < n && k * BLOCK_SIZE + threadIdx.x < n) {
			sA[threadIdx.y][threadIdx.x] = A[row * n + k * BLOCK_SIZE + threadIdx.x];
		}
		else {
			sA[threadIdx.y][threadIdx.x] = 0;
		}

		if (column < n && k * BLOCK_SIZE + threadIdx.y < n) {
			sB[threadIdx.y][threadIdx.x] = B[(k * BLOCK_SIZE + threadIdx.y) * n + column];
		}
		else {
			sB[threadIdx.y][threadIdx.x] = 0;
		}

		__syncthreads();

		for (size_t i = 0; i < BLOCK_SIZE; i++) {
			cellValue += sA[threadIdx.y][i] * sB[i][threadIdx.x];
		}

		__syncthreads();
	}

	if (row < n && column < n) {
		C[row * n + column] = cellValue;
	}
}

float gpuMatrixMultiplication(double* A, double* B, double* C, size_t n, bool isSharedMemory)
{
	double* dA;
	double* dB;
	double* dC;
	size_t matrixSizeInBytes = n * n * sizeof(double);

	cudaMalloc(&dA, matrixSizeInBytes);
	cudaMalloc(&dB, matrixSizeInBytes);
	cudaMalloc(&dC, matrixSizeInBytes);

	cudaMemcpy(dA, A, matrixSizeInBytes, cudaMemcpyHostToDevice);
	cudaMemcpy(dB, B, matrixSizeInBytes, cudaMemcpyHostToDevice);

	int gridSize = n / BLOCK_SIZE + 1;

	dim3 grid(gridSize, gridSize);
	dim3 threads(BLOCK_SIZE, BLOCK_SIZE);

	cudaEvent_t start, end;
	float time;
	cudaEventCreate(&start);
	cudaEventCreate(&end);

	cudaEventRecord(start, 0);

	if (isSharedMemory) {
		matrixMulShared << <grid, threads >> > (dA, dB, dC, n);
	}
	else {
		matrixMul << <grid, threads >> > (dA, dB, dC, n);
	}

	cudaEventRecord(end, 0);
	cudaEventSynchronize(end);
	cudaEventElapsedTime(&time, start, end);

	cudaEventDestroy(start);
	cudaEventDestroy(end);

	cudaMemcpy(C, dC, matrixSizeInBytes, cudaMemcpyDeviceToHost);

	cudaFree(dA);
	cudaFree(dB);
	cudaFree(dC);

	return time / 1000.0f;
}

float gpuMatrixMultiplicationWithoutSharedMemory(double* A, double* B, double* C, size_t n)
{
	return gpuMatrixMultiplication(A, B, C, n, false);
}

float gpuMatrixMultiplicationWithSharedMemory(double* A, double* B, double* C, size_t n)
{
	return gpuMatrixMultiplication(A, B, C, n, true);
}

double getMaximumDeviation(double* A, double* B, size_t n)
{
	double maxDeviation = 0.0;

	for (size_t i = 0; i < n * n; ++i) 
	{
		maxDeviation = std::max(maxDeviation, std::abs(A[i] - B[i]));
	}

	return maxDeviation;
}

int main()
{
	size_t n = DIM;

	double* mA = generateRandomMatrix(n);
	double* mB = generateRandomMatrix(n);

	// CPU
	double* resultCPU = new double[n * n];
	std::fill_n(resultCPU, n * n, 0);
	float cpuTime = cpuMatrixMultiplication(mA, mB, resultCPU, n);

	// GPU (without shared memory)
	double* resultGPU = new double[n * n];
	std::fill_n(resultGPU, n * n, 0);
	float gpuTime = gpuMatrixMultiplicationWithoutSharedMemory(mA, mB, resultGPU, n);
	double maxDeviationWithoutSharedMemory = getMaximumDeviation(resultCPU, resultGPU, n);

	// GPU (with shared memory)
	double* resultGPUWithSharedMemory = new double[n * n];
	std::fill_n(resultGPUWithSharedMemory, n * n, 0);
	float gpuTimeWithSharedMemory = gpuMatrixMultiplicationWithSharedMemory(mA, mB, resultGPUWithSharedMemory, n);
	double maxDeviationWithSharedMemory = getMaximumDeviation(resultCPU, resultGPUWithSharedMemory, n);

	// Results
	std::cout << "CPU time = " << cpuTime << std::endl;
	std::cout << "GPU time (without shared memory) = " << gpuTime << std::endl;
	std::cout << "GPU time (with shared memory) = " << gpuTimeWithSharedMemory << std::endl;
	std::cout << "Maximum deviation (without shared memory) = " << maxDeviationWithoutSharedMemory << std::endl;
	std::cout << "Maximum deviation (with shared memory) = " << maxDeviationWithSharedMemory << std::endl;

	delete[] mA;
	delete[] mB;
	delete[] resultCPU;
	delete[] resultGPU;
	delete[] resultGPUWithSharedMemory;

	return 0;
}
