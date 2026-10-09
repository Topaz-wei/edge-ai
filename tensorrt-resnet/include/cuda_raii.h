#pragma once

#include <cuda_runtime_api.h>

#include <cstddef>
#include <stdexcept>
#include <string>

class DeviceBuffer {
public:
    explicit DeviceBuffer(std::size_t bytes)
        : bytes_(bytes)
    {
        if (bytes_ == 0) {
            return;
        }

        const cudaError_t err =
            cudaMalloc(&ptr_, bytes_);

        if (err != cudaSuccess) {
            throw std::runtime_error(
                std::string("cudaMalloc failed: ") +
                cudaGetErrorString(err)
            );
        }
    }

    ~DeviceBuffer()
    {
        if (ptr_) {
            cudaFree(ptr_);
        }
    }

    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;

    DeviceBuffer(DeviceBuffer&& other) noexcept
        : ptr_(other.ptr_),
          bytes_(other.bytes_)
    {
        other.ptr_ = nullptr;
        other.bytes_ = 0;
    }

    DeviceBuffer& operator=(DeviceBuffer&& other) noexcept
    {
        if (this != &other) {
            if (ptr_) {
                cudaFree(ptr_);
            }

            ptr_ = other.ptr_;
            bytes_ = other.bytes_;

            other.ptr_ = nullptr;
            other.bytes_ = 0;
        }

        return *this;
    }

    void* get() const
    {
        return ptr_;
    }

    std::size_t size() const
    {
        return bytes_;
    }

private:
    void* ptr_{nullptr};
    std::size_t bytes_{0};
};


class CudaStream {
public:
    CudaStream()
    {
        const cudaError_t err =
            cudaStreamCreate(&stream_);

        if (err != cudaSuccess) {
            throw std::runtime_error(
                std::string("cudaStreamCreate failed: ") +
                cudaGetErrorString(err)
            );
        }
    }

    ~CudaStream()
    {
        if (stream_) {
            cudaStreamDestroy(stream_);
        }
    }

    CudaStream(const CudaStream&) = delete;
    CudaStream& operator=(const CudaStream&) = delete;

    CudaStream(CudaStream&& other) noexcept
        : stream_(other.stream_)
    {
        other.stream_ = nullptr;
    }

    CudaStream& operator=(CudaStream&& other) noexcept
    {
        if (this != &other) {
            if (stream_) {
                cudaStreamDestroy(stream_);
            }

            stream_ = other.stream_;
            other.stream_ = nullptr;
        }

        return *this;
    }

    cudaStream_t get() const
    {
        return stream_;
    }

    void synchronize() const
    {
        const cudaError_t err =
            cudaStreamSynchronize(stream_);

        if (err != cudaSuccess) {
            throw std::runtime_error(
                std::string(
                    "cudaStreamSynchronize failed: "
                ) +
                cudaGetErrorString(err)
            );
        }
    }

private:
    cudaStream_t stream_{nullptr};
};