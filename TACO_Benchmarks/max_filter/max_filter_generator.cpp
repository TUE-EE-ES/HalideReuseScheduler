#include "Halide.h"

namespace {

using namespace Halide;

using std::vector;
class MaxFilter : public Halide::Generator<MaxFilter> {
public:
    Input<Buffer<float>>  in{"input", 3};

    Output<Buffer<float>> final{"final", 3};



void generate(){
   
      const int radius = 26;

    Func input = BoundaryConditions::repeat_edge(in);

    Var x, y, c, t;

    const int slices = (int)(ceilf(logf(radius) / logf(2))) + 1;

    // A sequence of vertically-max-filtered versions of the input,
    // each filtered twice as tall as the previous slice. All filters
    // are downward-looking.
    Func vert_log;
    vert_log(x, y, c, t) = input(x, y, c);
    RDom r(-radius, in.height() + radius, 1, slices-1);
    vert_log(x, r.x, c, r.y) = max(vert_log(x, r.x, c, r.y - 1),
                                   vert_log(x, r.x + clamp((1<<(r.y-1)), 0, radius*2), c, r.y - 1));

    // We're going to take a max filter of arbitrary diameter
    // by maxing two samples from its floor log 2 (e.g. maxing two
    // 8-high overlapping samples). This next Func tells us which
    // slice to draw from for a given radius:
    Func slice_for_radius;
    slice_for_radius(t) = cast<int>(floor(log(2*t+1) / logf(2)));

    // Produce every possible vertically-max-filtered version of the image:
    Func vert;
    // t is the blur radius
    Expr slice = clamp(slice_for_radius(t), 0, slices);
    Expr first_sample = vert_log(x, y - t, c, slice);
    Expr second_sample = vert_log(x, y + t + 1 - clamp(1 << slice, 0, 2*radius), c, slice);
    vert(x, y, c, t) = max(first_sample, second_sample);

    Func filter_height;
    RDom dy(0, radius+1);
    filter_height(x) = sum(select(x*x + dy*dy < (radius+0.25f)*(radius+0.25f), 1, 0));

    // Now take an appropriate horizontal max of them at each output pixel
    
    RDom dx(-radius, 2*radius+1);
    final(x, y, c) = maximum(vert(x + dx, y, c, clamp(filter_height(dx), 0, radius+1)));



    


    if (!auto_schedule) {
        Var tx;
     slice_for_radius.compute_root();
        filter_height.compute_root();

        // vert_log.update(1) doesn't have enough parallelism, but I
        // can't figure out how to give it more... Split whole image
        // into slices.

        final.compute_root().split(x, tx, x, 256).reorder(x, y, c, tx).fuse(c, tx, t).parallel(t).vectorize(x, 8);
        vert_log.compute_at(final, t);
        vert_log.vectorize(x, 8);
        vert_log.update().reorder(x, r.x, r.y, c).vectorize(x, 8);
        vert.compute_at(final, y).vectorize(x, 8);
    } else {
        // Auto-schedule the pipeline
      final.estimate(x, 0, 1920)
            .estimate(y, 0, 1024)
            .estimate(c, 0, 3);
        in.dim(0).set_bounds_estimate(0,1920);
        in.dim(1).set_bounds_estimate(0,1024);
        in.dim(2).set_bounds_estimate(0,3);
    
    }
    
  
  }

 
};

}  // namespace

HALIDE_REGISTER_GENERATOR(MaxFilter, maxfilter)

