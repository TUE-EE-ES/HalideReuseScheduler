#include "Halide.h"

namespace {

using namespace Halide;

using std::vector;
class Interpolate : public Halide::Generator<Interpolate> {
public:
    Input<Buffer<float>>  in{"input", 3};
    Output<Buffer<float>> normalize{"normalize", 3};



void generate(){
    Var x("x"), y("y"), c("c");
    const int levels = 10;
     Func downsampled[levels];
    Func downx[levels];
    Func interpolated[levels];
    Func upsampled[levels];
    Func upsampledx[levels];
  Func clamped = BoundaryConditions::repeat_edge(in);
   
   // This triggers a bug in llvm 3.3 (3.2 and trunk are fine), so we
    // rewrite it in a way that doesn't trigger the bug. The rewritten
    // form assumes the in alpha is zero or one.
    // downsampled[0](x, y, c) = select(c < 3, clamped(x, y, c) * clamped(x, y, 3), clamped(x, y, 3));
    downsampled[0](x, y, c) = clamped(x, y, c) * clamped(x, y, 3);

    for (int l = 1; l < levels; ++l) {
        Func prev = downsampled[l-1];

        if (l == 4) {
            // Also add a boundary condition at a middle pyramid level
            // to prevent the footprint of the downsamplings to extend
            // too far off the base image. Otherwise we look 512
            // pixels off each edge.
            Expr w = in.width()/(1 << l);
            Expr h = in.height()/(1 << l);
            prev = lambda(x, y, c, prev(clamp(x, 0, w), clamp(y, 0, h), c));
        }

        downx[l](x, y, c) = (prev(x*2-1, y, c) +
                             2.0f * prev(x*2, y, c) +
                             prev(x*2+1, y, c)) * 0.25f;
        downsampled[l](x, y, c) = (downx[l](x, y*2-1, c) +
                                   2.0f * downx[l](x, y*2, c) +
                                   downx[l](x, y*2+1, c)) * 0.25f;
    }
    interpolated[levels-1](x, y, c) = downsampled[levels-1](x, y, c);
    for (int l = levels-2; l >= 0; --l) {
        upsampledx[l](x, y, c) = (interpolated[l+1](x/2, y, c) +
                                  interpolated[l+1]((x+1)/2, y, c)) / 2.0f;
        upsampled[l](x, y, c) =  (upsampledx[l](x, y/2, c) +
                                  upsampledx[l](x, (y+1)/2, c)) / 2.0f;
        interpolated[l](x, y, c) = downsampled[l](x, y, c) + (1.0f - downsampled[l](x, y, 3)) * upsampled[l](x, y, c);
    }

    
    normalize(x, y, c) = interpolated[0](x, y, c) / interpolated[0](x, y, 3);
    normalize
        .estimate(c, 0, 4)
        .estimate(x, 0, in.width())
        .estimate(y, 0, in.height());

    std::cout << "Finished function setup." << std::endl;



    


    if (!auto_schedule) {
Var xi, yi;
            std::cout << "Flat schedule with parallelization + vectorization." << std::endl;
            for (int l = 1; l < levels-1; ++l) {
                downsampled[l]
                    .compute_root()
                    .parallel(y, 8)
                    .vectorize(x, 4);
                interpolated[l]
                    .compute_root()
                    .parallel(y, 8)
                    .unroll(x, 2)
                    .unroll(y, 2)
                    .vectorize(x, 8);
            }
            normalize
                .reorder(c, x, y)
                .bound(c, 0, 3)
                .unroll(c)
                .tile(x, y, xi, yi, 2, 2)
                .unroll(xi)
                .unroll(yi)
                .parallel(y, 8)
                .vectorize(x, 8)
                .bound(x, 0, in.width())
                .bound(y, 0, in.height());
    } else {
        // Auto-schedule the pipeline
        normalize.estimate(x,0,1536)
                 .estimate(y,0,2560)
                 .estimate(c,0,4);
        in.dim(0).set_bounds_estimate(0,1536);
        in.dim(1).set_bounds_estimate(0,2560);
        in.dim(2).set_bounds_estimate(0,3);
    
    }
    
  
  }

 
};

}  // namespace

HALIDE_REGISTER_GENERATOR(Interpolate, interpolate)

