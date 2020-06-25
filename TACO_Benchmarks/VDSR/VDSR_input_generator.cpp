#include "Halide.h"

namespace {

using namespace Halide;

using std::vector;
class VDSR_input : public Halide::Generator<VDSR_input> {
public:
    Input<Buffer<uint8_t>>  in_layer{"in_layer", 3};
    Output<Buffer<float>> convert2{"convert2", 3};



void generate(){
Func in_layerd = BoundaryConditions::constant_exterior(in_layer,0,0,in_layer.width(),0,in_layer.height());       
Func in_layerb,in_layert;
Var n,m,o;

in_layerb(n,m,o)=(in_layerd(n,m,o));
convert2(n,m,o)=cast<float>(0);
convert2(n,m,0)=16+0.257f*in_layerb(n,m,0)+0.504f*in_layerb(n,m,1)+0.098f*in_layerb(n,m,2);
convert2(n,m,1)=128+(-0.148f)*in_layerb(n,m,0)+(-0.291f)*in_layerb(n,m,1)+0.439f*in_layerb(n,m,2);
convert2(n,m,2)=128+(0.439f)*in_layerb(n,m,0)+(-0.368f)*in_layerb(n,m,1)+(-0.071f)*in_layerb(n,m,2);
convert2(n,m,o)=convert2(n,m,o)/255;

}
};

}  // namespace

HALIDE_REGISTER_GENERATOR(VDSR_input, vdsr_input)

