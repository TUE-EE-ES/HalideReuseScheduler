#include "Halide.h"

namespace {

using namespace Halide;
Expr sum3x3(Func f, Var x, Var y) {
    return f(x-1, y-1) + f(x-1, y) + f(x-1, y+1) +
           f(x, y-1)   + f(x, y)   + f(x, y+1) +
           f(x+1, y-1) + f(x+1, y) + f(x+1, y+1);
}  


class Harris : public Halide::Generator<Harris> {
public:
    Input<Buffer<float>>  in{"input", 3};
    Output<Buffer<float>> shifted{"shifted", 2};



void generate(){
    Var x("x"), y("y"), c("c");
Func in_b = BoundaryConditions::repeat_edge(in);
    Func gray("gray");
    gray(x, y) = 0.299f * in_b(x, y, 0) + 0.587f * in_b(x, y, 1) + 0.114f * in_b(x, y, 2);

    Func Iy("Iy");
    Iy(x, y) = gray(x-1, y-1)*(-1.0f/12) + gray(x-1, y+1)*(1.0f/12) +
               gray(x, y-1)*(-2.0f/12) + gray(x, y+1)*(2.0f/12) +
               gray(x+1, y-1)*(-1.0f/12) + gray(x+1, y+1)*(1.0f/12);

    Func Ix("Ix");
    Ix(x, y) = gray(x-1, y-1)*(-1.0f/12) + gray(x+1, y-1)*(1.0f/12) +
               gray(x-1, y)*(-2.0f/12) + gray(x+1, y)*(2.0f/12) +
               gray(x-1, y+1)*(-1.0f/12) + gray(x+1, y+1)*(1.0f/12);

    Func Ixx("Ixx");
    Ixx(x, y) = Ix(x, y) * Ix(x, y);

    Func Iyy("Iyy");
    Iyy(x, y) = Iy(x, y) * Iy(x, y);

    Func Ixy("Ixy");
    Ixy(x, y) = Ix(x, y) * Iy(x, y);

    Func Sxx("Sxx");

    Sxx(x, y) = sum3x3(Ixx, x, y);

    Func Syy("Syy");
    Syy(x, y) = sum3x3(Iyy, x, y);

    Func Sxy("Sxy");
    Sxy(x, y) = sum3x3(Ixy, x, y);

    Func det("det");
    det(x, y) = Sxx(x, y) * Syy(x, y) - Sxy(x, y) * Sxy(x, y);

    Func trace("trace");
    trace(x, y) = Sxx(x, y) + Syy(x, y);

    Func harris("harris");
    harris(x, y) = det(x, y) - 0.04f * trace(x, y) * trace(x, y);

    
    shifted(x, y) = harris(x + 2, y + 2);



    


    if (!auto_schedule) {
        Var xi("xi"), yi("yi");
            shifted.tile(x, y, xi, yi, 128, 128)
                   .vectorize(xi, 8).parallel(y);
            Ix.compute_at(shifted, x).vectorize(x, 8);
            Iy.compute_at(shifted, x).vectorize(x, 8);
    } else {
        // Auto-schedule the pipeline
        shifted.estimate(x,0,1920)
               .estimate(y,0,1024);
        in.dim(0).set_bounds_estimate(0,1920);
        in.dim(1).set_bounds_estimate(0,1024);
        in.dim(2).set_bounds_estimate(0,3);
    shifted.print_loop_nest();    
    }
    
  
  }

 
};

}  // namespace

HALIDE_REGISTER_GENERATOR(Harris, harris)

