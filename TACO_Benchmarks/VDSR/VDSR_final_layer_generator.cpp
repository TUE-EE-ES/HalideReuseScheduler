#include "Halide.h"

namespace {

using namespace Halide;

using std::vector;
class VDSR_Final_layer : public Halide::Generator<VDSR_Final_layer> {
public:
	Input<Buffer<float>>  in_layer{"in_layer", 3};
	Input<Buffer<float>>  convert2{"convert2", 3};
	Input<Buffer<float>>  W19{"W19",4};
	Input<Buffer<float>>  b19{"b19",1};

    
    Output<Buffer<float>> final{"final", 2};



void generate(){
	int pad=1;
	Var n,m,o;
	RDom r(0,3,0,3,0,64);
	Func in_layert;
in_layert(n,m)=max(16.0f/255,min(235.0f/255,convert2(n,m,0)));
Func b_conv19("b_conv18");
b_conv19(n,m,o) = BoundaryConditions::constant_exterior(in_layer,0.0f,0,convert2.width(),0,convert2.height())(n,m,o);
Func f_conv19("f_conv19");
f_conv19=Func("convf");
f_conv19(n, m) = (b19(0));
f_conv19(n, m) += W19(r.x, r.y,r.z,0) *
        b_conv19(n + r.x-pad,
                   m + r.y-pad,r.z
                   );



final(n,m)=((f_conv19(n,m))+in_layert(n,m));

}
};

}  // namespace

HALIDE_REGISTER_GENERATOR(VDSR_Final_layer, vdsr_final_layer)

