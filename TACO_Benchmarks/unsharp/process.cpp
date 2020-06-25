#include <cstdio>
#include <chrono>

#include "unsharp.h"
#include "unsharp_auto_schedule.h"

#include "halide_benchmark.h"
#include "HalideBuffer.h"

using namespace Halide::Tools;
using namespace Halide::Runtime;

int main(int argc, char **argv) {


    int W = 1536;
    int H = 2560;

    Buffer<float> in(W, H, 4);

    // 8-bit RGBA image (per apps/images/rgba.png)
    for (int y = 0; y < in.height(); y++) {
        for (int x = 0; x < in.width(); x++) {
            for (int c = 0; c < 4; c++) {
                in(x, y, c) = rand() & 0xff;
            }
        }
    }

    Buffer<float> output(W, H,3);

    unsharp(in, output);

    // Timing code

    // Manually-tuned version
    double min_t_manual = benchmark(10, 10, [&]() {
        unsharp(in, output);
    });
    printf("Manually-tuned time: %gms\n", min_t_manual * 1e3);

    // Auto-scheduled version
    double min_t_auto = benchmark(10, 10, [&]() {
        unsharp_auto_schedule(in, output);
    });
    printf("Auto-scheduled time: %gms\n", min_t_auto * 1e3);

    return 0;
}
