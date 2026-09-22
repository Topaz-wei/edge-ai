#include <NvInfer.h>
#include <cuda_runtime_api.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>


class Logger : public nvinfer1::ILogger {
public:
    void log(Severity severity, const char* msg) noexcept override {
        if (severity <= Severity::kWARNING) {
            std::cout << "[TensorRT] " << msg << '\n';
        }
    }
};

static Logger gLogger;


void checkCuda(cudaError_t error, const char* op) {
    if (error != cudaSuccess) {
        throw std::runtime_error(
            std::string(op) + ": " +
            cudaGetErrorString(error)
        );
    }
}


std::vector<char> loadEngineFile(const std::string& path) {
    std::ifstream file(path, std::ios::binary);

    if (!file) {
        throw std::runtime_error(
            "Failed to open engine: " + path
        );
    }

    file.seekg(0, std::ios::end);
    const std::size_t size = file.tellg();
    file.seekg(0, std::ios::beg);

    std::vector<char> data(size);
    file.read(data.data(), size);

    return data;
}


std::size_t volume(const nvinfer1::Dims& dims) {
    std::size_t v = 1;

    for (int i = 0; i < dims.nbDims; ++i) {
        if (dims.d[i] <= 0) {
            throw std::runtime_error(
                "Unresolved dynamic dimension"
            );
        }

        v *= dims.d[i];
    }

    return v;
}


double percentile(
    std::vector<float> values,
    double p
) {
    if (values.empty()) {
        return 0.0;
    }

    std::sort(values.begin(), values.end());

    const double pos =
        p * (values.size() - 1);

    const std::size_t lower =
        static_cast<std::size_t>(
            std::floor(pos)
        );

    const std::size_t upper =
        static_cast<std::size_t>(
            std::ceil(pos)
        );

    if (lower == upper) {
        return values[lower];
    }

    const double weight =
        pos - lower;

    return values[lower] * (1.0 - weight)
         + values[upper] * weight;
}


void printStats(
    const std::string& name,
    const std::vector<float>& values
) {
    const double sum =
        std::accumulate(
            values.begin(),
            values.end(),
            0.0
        );

    const double mean =
        sum / values.size();

    std::cout
        << std::left
        << std::setw(18)
        << name
        << " mean="
        << std::setw(9)
        << mean
        << " ms"
        << " p50="
        << std::setw(9)
        << percentile(values, 0.50)
        << " p95="
        << std::setw(9)
        << percentile(values, 0.95)
        << " p99="
        << percentile(values, 0.99)
        << '\n';
}


int main(int argc, char** argv) {
    if (argc != 5) {
        std::cerr
            << "Usage:\n  "
            << argv[0]
            << " <engine.plan>"
            << " <batch>"
            << " <warmup>"
            << " <iterations>\n";

        return 1;
    }

    try {
        const std::string enginePath = argv[1];

        const int batch =
            std::stoi(argv[2]);

        const int warmup =
            std::stoi(argv[3]);

        const int iterations =
            std::stoi(argv[4]);

        if (
            batch <= 0 ||
            warmup < 0 ||
            iterations <= 0
        ) {
            throw std::runtime_error(
                "Invalid benchmark arguments"
            );
        }

        // ---------------------------------
        // Load TensorRT engine
        // ---------------------------------

        auto engineData =
            loadEngineFile(enginePath);

        auto* runtime =
            nvinfer1::createInferRuntime(
                gLogger
            );

        auto* engine =
            runtime->deserializeCudaEngine(
                engineData.data(),
                engineData.size()
            );

        if (!engine) {
            throw std::runtime_error(
                "Failed to deserialize engine"
            );
        }

        auto* context =
            engine->createExecutionContext();

        if (!context) {
            throw std::runtime_error(
                "Failed to create context"
            );
        }

        // ---------------------------------
        // Find bindings
        // ---------------------------------

        int inputIndex = -1;
        int outputIndex = -1;

        const int nbBindings =
            engine->getNbBindings();

        for (int i = 0; i < nbBindings; ++i) {
            if (engine->bindingIsInput(i)) {
                inputIndex = i;
            } else {
                outputIndex = i;
            }
        }

        if (inputIndex < 0 || outputIndex < 0) {
            throw std::runtime_error(
                "Bindings not found"
            );
        }

        // ---------------------------------
        // CUDA stream
        // ---------------------------------

        cudaStream_t stream;

        checkCuda(
            cudaStreamCreate(&stream),
            "cudaStreamCreate"
        );

        // We currently only have profile 0.
        constexpr int profileIndex = 0;

        if (!context->setOptimizationProfileAsync(
                profileIndex,
                stream)) {

            throw std::runtime_error(
                "Failed to select profile"
            );
        }

        // ---------------------------------
        // Runtime input shape
        // ---------------------------------

        const nvinfer1::Dims4 inputDims{
            batch,
            3,
            224,
            224
        };

        if (!context->setBindingDimensions(
                inputIndex,
                inputDims)) {

            throw std::runtime_error(
                "setBindingDimensions failed"
            );
        }

        if (!context->allInputDimensionsSpecified()) {
            throw std::runtime_error(
                "Dynamic dimensions unresolved"
            );
        }

        const auto actualInputDims =
            context->getBindingDimensions(
                inputIndex
            );

        const auto outputDims =
            context->getBindingDimensions(
                outputIndex
            );

        const std::size_t inputElements =
            volume(actualInputDims);

        const std::size_t outputElements =
            volume(outputDims);

        const std::size_t inputBytes =
            inputElements * sizeof(float);

        const std::size_t outputBytes =
            outputElements * sizeof(float);

        // ---------------------------------
        // Pinned host memory
        // ---------------------------------

        float* hostInput = nullptr;
        float* hostOutput = nullptr;

        checkCuda(
            cudaMallocHost(
                reinterpret_cast<void**>(&hostInput),
                inputBytes
            ),
            "cudaMallocHost input"
        );

        checkCuda(
            cudaMallocHost(
                reinterpret_cast<void**>(&hostOutput),
                outputBytes
            ),
            "cudaMallocHost output"
        );

        // Fixed dummy input.
        for (std::size_t i = 0;
             i < inputElements;
             ++i) {

            hostInput[i] =
                static_cast<float>(
                    (i % 255) / 255.0
                );
        }

        // ---------------------------------
        // Device memory
        // ---------------------------------

        void* deviceInput = nullptr;
        void* deviceOutput = nullptr;

        checkCuda(
            cudaMalloc(
                &deviceInput,
                inputBytes
            ),
            "cudaMalloc input"
        );

        checkCuda(
            cudaMalloc(
                &deviceOutput,
                outputBytes
            ),
            "cudaMalloc output"
        );

        std::vector<void*> bindings(
            nbBindings,
            nullptr
        );

        bindings[inputIndex] =
            deviceInput;

        bindings[outputIndex] =
            deviceOutput;

        // ---------------------------------
        // Warmup
        // ---------------------------------

        checkCuda(
            cudaMemcpyAsync(
                deviceInput,
                hostInput,
                inputBytes,
                cudaMemcpyHostToDevice,
                stream
            ),
            "warmup H2D"
        );

        for (int i = 0; i < warmup; ++i) {
            if (!context->enqueueV2(
                    bindings.data(),
                    stream,
                    nullptr)) {

                throw std::runtime_error(
                    "Warmup enqueue failed"
                );
            }
        }

        checkCuda(
            cudaStreamSynchronize(stream),
            "warmup synchronize"
        );

        // =================================
        // Benchmark 1: GPU compute only
        // =================================

        cudaEvent_t computeStart;
        cudaEvent_t computeEnd;

        checkCuda(
            cudaEventCreate(&computeStart),
            "cudaEventCreate"
        );

        checkCuda(
            cudaEventCreate(&computeEnd),
            "cudaEventCreate"
        );

        std::vector<float> computeTimes;
        computeTimes.reserve(iterations);

        for (int i = 0;
             i < iterations;
             ++i) {

            cudaEventRecord(
                computeStart,
                stream
            );

            if (!context->enqueueV2(
                    bindings.data(),
                    stream,
                    nullptr)) {

                throw std::runtime_error(
                    "enqueueV2 failed"
                );
            }

            cudaEventRecord(
                computeEnd,
                stream
            );

            cudaEventSynchronize(
                computeEnd
            );

            float ms = 0.0f;

            cudaEventElapsedTime(
                &ms,
                computeStart,
                computeEnd
            );

            computeTimes.push_back(ms);
        }

        // =================================
        // Benchmark 2: End-to-end
        // H2D + inference + D2H
        // =================================

        cudaEvent_t totalStart;
        cudaEvent_t afterH2D;
        cudaEvent_t afterCompute;
        cudaEvent_t totalEnd;

        cudaEventCreate(&totalStart);
        cudaEventCreate(&afterH2D);
        cudaEventCreate(&afterCompute);
        cudaEventCreate(&totalEnd);

        std::vector<float> h2dTimes;
        std::vector<float> inferenceTimes;
        std::vector<float> d2hTimes;
        std::vector<float> totalTimes;

        h2dTimes.reserve(iterations);
        inferenceTimes.reserve(iterations);
        d2hTimes.reserve(iterations);
        totalTimes.reserve(iterations);

        const auto hostStart =
            std::chrono::steady_clock::now();

        for (int i = 0;
             i < iterations;
             ++i) {

            cudaEventRecord(
                totalStart,
                stream
            );

            cudaMemcpyAsync(
                deviceInput,
                hostInput,
                inputBytes,
                cudaMemcpyHostToDevice,
                stream
            );

            cudaEventRecord(
                afterH2D,
                stream
            );

            if (!context->enqueueV2(
                    bindings.data(),
                    stream,
                    nullptr)) {

                throw std::runtime_error(
                    "enqueueV2 failed"
                );
            }

            cudaEventRecord(
                afterCompute,
                stream
            );

            cudaMemcpyAsync(
                hostOutput,
                deviceOutput,
                outputBytes,
                cudaMemcpyDeviceToHost,
                stream
            );

            cudaEventRecord(
                totalEnd,
                stream
            );

            cudaEventSynchronize(
                totalEnd
            );

            float h2d = 0.0f;
            float infer = 0.0f;
            float d2h = 0.0f;
            float total = 0.0f;

            cudaEventElapsedTime(
                &h2d,
                totalStart,
                afterH2D
            );

            cudaEventElapsedTime(
                &infer,
                afterH2D,
                afterCompute
            );

            cudaEventElapsedTime(
                &d2h,
                afterCompute,
                totalEnd
            );

            cudaEventElapsedTime(
                &total,
                totalStart,
                totalEnd
            );

            h2dTimes.push_back(h2d);
            inferenceTimes.push_back(infer);
            d2hTimes.push_back(d2h);
            totalTimes.push_back(total);
        }

        const auto hostEnd =
            std::chrono::steady_clock::now();

        const double wallSeconds =
            std::chrono::duration<double>(
                hostEnd - hostStart
            ).count();

        // ---------------------------------
        // Statistics
        // ---------------------------------

        const double meanCompute =
            std::accumulate(
                computeTimes.begin(),
                computeTimes.end(),
                0.0
            ) / computeTimes.size();

        const double computeInferPerSec =
            1000.0 / meanCompute;

        const double computeSamplesPerSec =
            batch * computeInferPerSec;

        const double wallInferPerSec =
            iterations / wallSeconds;

        const double wallSamplesPerSec =
            batch * wallInferPerSec;

        std::cout
            << "\n====================================\n"
            << "TensorRT C++ Benchmark\n"
            << "====================================\n";

        std::cout
            << "Batch       : "
            << batch << '\n';

        std::cout
            << "Warmup      : "
            << warmup << '\n';

        std::cout
            << "Iterations  : "
            << iterations << '\n';

        std::cout
            << "Input elems : "
            << inputElements << '\n';

        std::cout
            << "Output elems: "
            << outputElements << "\n\n";

        std::cout
            << "----- Compute only -----\n";

        printStats(
            "GPU Compute",
            computeTimes
        );

        std::cout
            << "Inference/s : "
            << computeInferPerSec
            << '\n';

        std::cout
            << "Samples/s   : "
            << computeSamplesPerSec
            << "\n\n";

        std::cout
            << "----- End-to-end -----\n";

        printStats("H2D", h2dTimes);
        printStats(
            "GPU Compute",
            inferenceTimes
        );
        printStats("D2H", d2hTimes);
        printStats("Total", totalTimes);

        std::cout
            << "\nHost wall time : "
            << wallSeconds
            << " s\n";

        std::cout
            << "Inference/s    : "
            << wallInferPerSec
            << '\n';

        std::cout
            << "Samples/s      : "
            << wallSamplesPerSec
            << '\n';

        // ---------------------------------
        // Cleanup
        // ---------------------------------

        cudaEventDestroy(computeStart);
        cudaEventDestroy(computeEnd);

        cudaEventDestroy(totalStart);
        cudaEventDestroy(afterH2D);
        cudaEventDestroy(afterCompute);
        cudaEventDestroy(totalEnd);

        cudaFree(deviceInput);
        cudaFree(deviceOutput);

        cudaFreeHost(hostInput);
        cudaFreeHost(hostOutput);

        cudaStreamDestroy(stream);

        context->destroy();
        engine->destroy();
        runtime->destroy();

    } catch (const std::exception& e) {
        std::cerr
            << "ERROR: "
            << e.what()
            << '\n';

        return 1;
    }

    return 0;
}