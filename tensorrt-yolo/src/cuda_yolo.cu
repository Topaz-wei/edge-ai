#include "include/cuda_yolo.hpp"
#include "include/infer_raw.hpp"

#include <cub/cub.cuh>
#include <opencv2/imgproc.hpp>

#include <cstring>
#include <stdexcept>

namespace {
constexpr int width = 360, height = 640, pixels = width * height;
constexpr int anchors = 8400, threads = 256;
constexpr std::size_t imageBytes = pixels * 3;

__global__ void sumPixels(
    const unsigned char* image,
    unsigned int* total
) {
    __shared__ unsigned int sums[3][threads];
    const int i = blockIdx.x * blockDim.x + threadIdx.x;

    for (int c = 0; c < 3; ++c)
        sums[c][threadIdx.x] = i < pixels ? image[3 * i + c] : 0;
    __syncthreads();

    for (int stride = threads / 2; stride; stride /= 2) {
        if (threadIdx.x < stride)
            for (int c = 0; c < 3; ++c)
                sums[c][threadIdx.x] += sums[c][threadIdx.x + stride];
        __syncthreads();
    }

    if (threadIdx.x == 0)
        for (int c = 0; c < 3; ++c)
            atomicAdd(
                total + c,
                sums[c][0]
            );
}

__global__ void applyMean(
    unsigned char* image,
    const unsigned char* mask,
    const unsigned int* sums
) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i >= pixels || !mask[i])
        return;

    for (int c = 0; c < 3; ++c) {
        // Match OpenCV mean (double reciprocal multiply), then truncate to uint8.
        image[3 * i + c] = static_cast<unsigned char>(__double2uint_rz(__dmul_rn(
            static_cast<double>(sums[c]),
            1.0 / pixels
        )));
    }
}

__global__ void makeTensor(
    const unsigned char* image,
    float* tensor
) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i >= 640 * 640)
        return;

    const int x = i % 640, y = i / 640;

    for (int c = 0; c < 3; ++c) {
        const unsigned char value =
            x < 140 || x >= 500 ? 114 : image[(y * width + x - 140) * 3 + 2 - c];
        tensor[c * 640 * 640 + i] = __fdiv_rn(
            static_cast<float>(value),
            255.0f
        );
    }
}

__global__ void decode(
    const float* raw,
    Detection* decoded,
    unsigned long long* keys,
    int* indices
) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i >= anchors)
        return;

    float score = -1.f;
    int label = -1;

    for (int c = 0; c < 80; ++c) {
        const float value = raw[(4 + c) * anchors + i];

        if (value > score) {
            score = value;
            label = c;
        }
    }

    keys[i] = 0;
    indices[i] = i;

    if (label != 0 || score < 0.7f)
        return;

    const float x = raw[i], y = raw[anchors + i];
    const float w = raw[2 * anchors + i], h = raw[3 * anchors + i];
    Detection d{};
    d.candidate = i;
    d.classId = label;
    d.confidence = score;
    d.modelBox = {
        x - w / 2.f,
        y - h / 2.f,
        x + w / 2.f,
        y + h / 2.f
    };
    decoded[i] = d;
    // Positive float scores sort by bits; ties prefer the lower candidate index.
    keys[i] = (static_cast<unsigned long long>(__float_as_uint(score)) << 32) |
              static_cast<unsigned int>(0xffffffffu - i);
}

__device__ float area(
    const Box& b
) {
    return fmaxf(
               0.f,
               b.x2 - b.x1
           ) *
           fmaxf(
               0.f,
               b.y2 - b.y1
           );
}

__device__ float iou(
    const Box& a,
    const Box& b
) {
    const float intersection = fmaxf(
                                   0.f,
                                   fminf(
                                       a.x2,
                                       b.x2
                                   ) -
                                       fmaxf(
                                           a.x1,
                                           b.x1
                                       )
                               ) *
                               fmaxf(
                                   0.f,
                                   fminf(
                                       a.y2,
                                       b.y2
                                   ) -
                                       fmaxf(
                                           a.y1,
                                           b.y1
                                       )
                               );

    const float total = area(a) + area(b) - intersection;

    return total <= 0.f ? 0.f
                        : __fdiv_rn(
                              intersection,
                              total
                          );
}

// Exact greedy NMS. Few survivors in the current person detector (9 on test.jpg).
// This intentionally prioritizes reference semantics over approximate parallel NMS.
__global__ void nms(
    const Detection* decoded,
    const unsigned long long* sortedKeys,
    const int* order,
    Detection* kept,
    int* count
) {
    if (threadIdx.x || blockIdx.x)
        return;

    int n = 0;

    for (int j = 0; j < anchors && sortedKeys[j]; ++j) {
        Detection d = decoded[order[j]];
        bool suppressed = false;

        for (int k = 0; k < n; ++k) {
            if (iou(d.modelBox,
                    kept[k].modelBox) > 0.7f) {
                suppressed = true;
                break;
            }
        }

        if (!suppressed) {
            d.businessBox = {
                fminf(
                    360.f,
                    fmaxf(
                        0.f,
                        d.modelBox.x1 - 140.f
                    )
                ),
                fminf(
                    640.f,
                    fmaxf(
                        0.f,
                        d.modelBox.y1
                    )
                ),
                fminf(
                    360.f,
                    fmaxf(
                        0.f,
                        d.modelBox.x2 - 140.f
                    )
                ),
                fminf(
                    640.f,
                    fmaxf(
                        0.f,
                        d.modelBox.y2
                    )
                )
            };
            kept[n++] = d;
        }
    }

    *count = n;
}
}

struct CudaYoloOps::Impl {
    cudaStream_t stream;
    unsigned char *image{}, *masks{}, *hostImage{};
    unsigned int* sums{};
    unsigned long long *keys{}, *sortedKeys{};
    int *indices{}, *sortedIndices{}, *count{}, *hostCount{};
    Detection *decoded{}, *kept{}, *hostKept{};
    void* scratch{};
    std::size_t scratchBytes{};

    explicit Impl(
        cudaStream_t s
    )
        : stream(s) {
    }

    ~Impl() {
        cudaStreamSynchronize(stream);

        cudaFree(image);

        cudaFree(masks);

        cudaFree(sums);

        cudaFree(keys);

        cudaFree(sortedKeys);

        cudaFree(indices);

        cudaFree(sortedIndices);

        cudaFree(count);

        cudaFree(decoded);

        cudaFree(kept);

        cudaFree(scratch);

        cudaFreeHost(hostImage);

        cudaFreeHost(hostCount);

        cudaFreeHost(hostKept);
    }

    void initialize() {
        checkCuda(
            cudaMalloc(
                &image,
                imageBytes
            ),
            "allocate BGR"
        );

        checkCuda(
            cudaMalloc(
                &masks,
                2 * pixels
            ),
            "allocate masks"
        );

        checkCuda(
            cudaMalloc(
                &sums,
                3 * sizeof(unsigned int)
            ),
            "allocate sums"
        );

        checkCuda(
            cudaMalloc(
                &keys,
                anchors * sizeof(*keys)
            ),
            "allocate keys"
        );

        checkCuda(
            cudaMalloc(
                &sortedKeys,
                anchors * sizeof(*keys)
            ),
            "allocate sorted keys"
        );

        checkCuda(
            cudaMalloc(
                &indices,
                anchors * sizeof(int)
            ),
            "allocate indices"
        );

        checkCuda(
            cudaMalloc(
                &sortedIndices,
                anchors * sizeof(int)
            ),
            "allocate sorted indices"
        );

        checkCuda(
            cudaMalloc(
                &decoded,
                anchors * sizeof(Detection)
            ),
            "allocate decoded"
        );

        checkCuda(
            cudaMalloc(
                &kept,
                anchors * sizeof(Detection)
            ),
            "allocate kept"
        );

        checkCuda(
            cudaMalloc(
                &count,
                sizeof(int)
            ),
            "allocate count"
        );

        checkCuda(
            cudaMallocHost(
                &hostImage,
                imageBytes
            ),
            "allocate pinned image"
        );

        checkCuda(
            cudaMallocHost(
                &hostCount,
                sizeof(int)
            ),
            "allocate pinned count"
        );

        checkCuda(
            cudaMallocHost(
                &hostKept,
                anchors * sizeof(Detection)
            ),
            "allocate pinned detections"
        );

        checkCuda(
            cub::DeviceRadixSort::SortPairsDescending(
                scratch,
                scratchBytes,
                keys,
                sortedKeys,
                indices,
                sortedIndices,
                anchors,
                0,
                64,
                stream
            ),
            "size sort scratch"
        );

        checkCuda(
            cudaMalloc(
                &scratch,
                scratchBytes
            ),
            "allocate sort scratch"
        );

        // Rasterize exact OpenCV fillPoly boundaries once, not per frame.
        cv::Mat mask = cv::Mat::zeros(
            2 * height,
            width,
            CV_8UC1
        );

        auto head = mask.rowRange(
                 0,
                 height
             ),
             window = mask.rowRange(
                 height,
                 2 * height
             );

        cv::fillPoly(
            head,
            std::vector<std::vector<cv::Point>>{
                {{240,
                     0},
                    {360,
                        0},
                    {360,
                        200},
                    {300,
                        100}}
            },
            cv::Scalar(1)
        );

        cv::fillPoly(
            window,
            std::vector<std::vector<cv::Point>>{
                {{0,
                     150},
                    {65,
                        106},
                    {95,
                        237},
                    {6,
                        327}}
            },
            cv::Scalar(1)
        );

        checkCuda(
            cudaMemcpyAsync(
                masks,
                mask.data,
                2 * pixels,
                cudaMemcpyHostToDevice,
                stream
            ),
            "upload fixed masks"
        );

        checkCuda(
            cudaStreamSynchronize(stream),
            "masks sync"
        );
    }
};

CudaYoloOps::CudaYoloOps(
    cudaStream_t stream
)
    : impl_(new Impl(stream)) {
    impl_->initialize();
}

CudaYoloOps::~CudaYoloOps() = default;

void CudaYoloOps::upload(
    const cv::Mat& bgr
) {
    if (bgr.rows != height || bgr.cols != width || bgr.type() != CV_8UC3)
        throw std::runtime_error("Expected 360x640 uint8 BGR business image");

    auto& p = *impl_;
    // Handles camera images and non-contiguous Mat ROIs without mutating input.
    for (int y = 0; y < height; ++y)
        std::memcpy(
            p.hostImage + y * width * 3,
            bgr.ptr(y),
            width * 3
        );

    checkCuda(
        cudaMemcpyAsync(
            p.image,
            p.hostImage,
            imageBytes,
            cudaMemcpyHostToDevice,
            p.stream
        ),
        "upload BGR"
    );
}

void CudaYoloOps::preprocess(
    float* input
) {
    auto& p = *impl_;

    for (int mask = 0; mask < 2; ++mask) {
        checkCuda(
            cudaMemsetAsync(
                p.sums,
                0,
                3 * sizeof(unsigned int),
                p.stream
            ),
            "reset means"
        );

        sumPixels<<<(pixels + threads - 1) / threads, threads, 0, p.stream>>>(
            p.image,
            p.sums
        );

        checkCuda(
            cudaGetLastError(),
            "sum pixels"
        );

        applyMean<<<(pixels + threads - 1) / threads, threads, 0, p.stream>>>(
            p.image,
            p.masks + mask * pixels,
            p.sums
        );

        checkCuda(
            cudaGetLastError(),
            "apply mean mask"
        );
    }

    makeTensor<<<(640 * 640 + threads - 1) / threads, threads, 0, p.stream>>>(
        p.image,
        input
    );

    checkCuda(
        cudaGetLastError(),
        "make tensor"
    );
}

void CudaYoloOps::postprocess(
    const float* output
) {
    auto& p = *impl_;
    decode<<<(anchors + threads - 1) / threads, threads, 0, p.stream>>>(
        output,
        p.decoded,
        p.keys,
        p.indices
    );

    checkCuda(
        cudaGetLastError(),
        "decode candidates"
    );

    checkCuda(
        cub::DeviceRadixSort::SortPairsDescending(
            p.scratch,
            p.scratchBytes,
            p.keys,
            p.sortedKeys,
            p.indices,
            p.sortedIndices,
            anchors,
            0,
            64,
            p.stream
        ),
        "sort candidates"
    );

    nms<<<1, 1, 0, p.stream>>>(
        p.decoded,
        p.sortedKeys,
        p.sortedIndices,
        p.kept,
        p.count
    );

    checkCuda(
        cudaGetLastError(),
        "greedy NMS"
    );
}

std::vector<Detection> CudaYoloOps::download() {
    auto& p = *impl_;
    checkCuda(
        cudaMemcpyAsync(
            p.hostCount,
            p.count,
            sizeof(int),
            cudaMemcpyDeviceToHost,
            p.stream
        ),
        "download count"
    );

    checkCuda(
        cudaStreamSynchronize(p.stream),
        "count sync"
    );

    const int n = *p.hostCount;

    if (n < 0 || n > anchors)
        throw std::runtime_error("Invalid CUDA detection count");

    if (n) {
        checkCuda(
            cudaMemcpyAsync(
                p.hostKept,
                p.kept,
                n * sizeof(Detection),
                cudaMemcpyDeviceToHost,
                p.stream
            ),
            "download detections"
        );

        checkCuda(
            cudaStreamSynchronize(p.stream),
            "detections sync"
        );
    }

    return {
        p.hostKept,
        p.hostKept + n
    };
}
