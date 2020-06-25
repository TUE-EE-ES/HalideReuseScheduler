#include "Halide.h"

namespace {

using namespace Halide;

using std::vector;
class Unsharp : public Halide::Generator<Unsharp> {
public:
    Input<Buffer<float>>  in{"input", 3};
    Output<Buffer<float>> unsharp{"unsharp", 3};



void generate(){
    const float PI = 3.14159265358979323846f;
   // Define a 7x7 Gaussian Blur with a repeat-edge boundary condition.
    float sigma = 1.5f;

    Var x, y, c;
    Func kernel("kernel");
    kernel(x) = exp(-x*x/(2*sigma*sigma)) / (sqrtf(2*PI)*sigma);

    Func in_bounded = BoundaryConditions::repeat_edge(in);

    Func gray("gray");
    gray(x, y) = 0.299f * in_bounded(x, y, 0) + 0.587f * in_bounded(x, y, 1) +
                 0.114f * in_bounded(x, y, 2);

    Func blur_y("blur_y");
    blur_y(x, y) = (kernel(0) * gray(x, y) +
                    kernel(1) * (gray(x, y-1) + gray(x, y+1)) +
                    kernel(2) * (gray(x, y-2) + gray(x, y+2)) +
                    kernel(3) * (gray(x, y-3) + gray(x, y+3)));

    Func blur_x("blur_x");
    blur_x(x, y) = (kernel(0) * blur_y(x, y) +
                    kernel(1) * (blur_y(x-1, y) + blur_y(x+1, y)) +
                    kernel(2) * (blur_y(x-2, y) + blur_y(x+2, y)) +
                    kernel(3) * (blur_y(x-3, y) + blur_y(x+3, y)));

    Func sharpen("sharpen");
    sharpen(x, y) = 2 * gray(x, y) - blur_x(x, y);

    Func ratio("ratio");
    ratio(x, y) = sharpen(x, y) / gray(x, y);

    
    unsharp(x, y, c) = ratio(x, y) * in(x, y, c);




    


    if (!auto_schedule) {
        blur_y.compute_at(unsharp, y).vectorize(x, 8);
        ratio.compute_at(unsharp, y).vectorize(x, 8);
        unsharp.vectorize(x, 8).parallel(y).reorder(x, c, y);
    } else {
        // Auto-schedule the pipeline
        unsharp.estimate(x,0,1920)
                 .estimate(y,0,1024)
                 .estimate(c,0,3);
        in.dim(0).set_bounds_estimate(0,1920);
        in.dim(1).set_bounds_estimate(0,1024);
        in.dim(2).set_bounds_estimate(0,3);
    
    }
    
  
  }

 
};

}  // namespace

HALIDE_REGISTER_GENERATOR(Unsharp, unsharp)

