#include "Halide.h"

namespace {

using namespace Halide;

using std::vector;
class VDSR_first_layer : public Halide::Generator<VDSR_first_layer> {
public:
    Input<Buffer<float>>  convert2{"convert2", 3};
Input<Buffer<float>>  W0{"W0", 4};
Input<Buffer<float>>  b0{"b0", 1};
Output<Buffer<float>> f_conv0{"f_conv0", 3};
void generate(){
Var n,m,o;
//ImageParam in_layer(type_of<float>(),3);

RDom r(0,3,0,3);

int pad=1;


Func bin;
Func in_layert;
in_layert(n,m)=max(16.0f/255,min(235.0f/255,convert2(n,m,0)));
bin(n,m) = BoundaryConditions::constant_exterior(in_layert,0.0f,0,convert2.width(),0,convert2.height())(n,m);
  f_conv0(n, m, o) = (b0(o));
f_conv0(n, m, o) += W0(r.x, r.y,0, o) *
        bin(n + r.x-pad,
                   m + r.y-pad
                   );
f_conv0(n, m, o) = max(0.0f, f_conv0(n, m, o));

}
};

}  // namespace

HALIDE_REGISTER_GENERATOR(VDSR_first_layer, vdsr_first_layer)

