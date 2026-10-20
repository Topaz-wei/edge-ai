#include "include/precision_options.hpp"

#include <iostream>
#include <fstream>
#include <functional>
#include <filesystem>

void require(bool ok) {
    if (!ok)
        throw std::runtime_error("assertion failed");
}

void rejects(std::function<void()> fn) {
    bool caught = false;
    try {
        fn();
    } catch (const std::exception&) {
        caught = true;
    }
    require(caught);
}

int main() {
    // Positional calibration arguments remain compatible with named options.
    auto o = precision::parse({
        "model.onnx", "engine.plan", "int8", "images", "cal.cache",
        "--timing-cache", "timing.cache", "--fp16-layers", "layers.txt"
    });
    require(o.calibration == "images" && o.cache == "cal.cache" && o.timing == "timing.cache");

    auto base = precision::parse({"model.onnx", "engine.plan", "int8", "images"});
    require(base.layers.empty() && base.cache == "engine.plan.cache");

    // Invalid flags and aliases must fail before any output can be written.
    rejects([] {
        precision::parse({"m", "e", "fp32", "--fp16-layers", "l"});
    });
    rejects([] {
        precision::parse({"m", "e", "fp16", "--timing-cache"});
    });
    rejects([] {
        precision::parse({"m", "e", "fp16", "--typo", "x"});
    });
    rejects([] {
        precision::parse({"m", "e", "fp16", "--inspect-only"});
    });
    rejects([] {
        precision::parse({"m", "e", "fp16", "--timing-cache", "m"});
    });
    rejects([] {
        precision::parse({
            "m", "e", "fp16", "--timing-cache", "t", "--timing-cache", "u"
        });
    });
    rejects([] {
        precision::parse({
            "timing.tmp", "engine", "fp16", "--timing-cache", "timing"
        });
    });
    rejects([] {
        precision::parse({"cache.meta", "engine", "int8", "images", "cache"});
    });

    // Layer-list parsing trims whitespace and comments, but rejects duplicates.
    auto path = std::filesystem::temp_directory_path() / "yolo-precision-options-test.txt";
    {
        std::ofstream f(path);
        f << "# comment\n /model.1/Conv \r\n\n/model.2/Conv\n";
    }
    require(precision::readNames(path.string()).size() == 2);

    {
        std::ofstream f(path);
        f << "same\nsame\n";
    }
    rejects([&] {
        precision::readNames(path.string());
    });

    std::filesystem::remove(path);
    std::cout << "precision options passed\n";
}
