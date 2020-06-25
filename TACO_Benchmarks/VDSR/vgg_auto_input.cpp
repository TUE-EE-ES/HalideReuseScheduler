#include "Halide.h"
#include "clock.h"
#include <stdio.h>
#include "halide_image_io.h"

using namespace Halide;
using namespace Halide::Tools;
using namespace std;

int main( int argc, char **argv)
{


ImageParam in_layer(type_of<uint8_t>(), 3);
Func   in_layerd = BoundaryConditions::constant_exterior(in_layer,0,0,in_layer.width(),0,in_layer.height());       
Func in_layerb,in_layert;
Var n,m,o;
Func convert2("convert");
in_layerb(n,m,o)=(in_layerd(n,m,o));
convert2(n,m,o)=cast<float>(0);
convert2(n,m,0)=16+0.257f*in_layerb(n,m,0)+0.504f*in_layerb(n,m,1)+0.098f*in_layerb(n,m,2);
convert2(n,m,1)=128+(-0.148f)*in_layerb(n,m,0)+(-0.291f)*in_layerb(n,m,1)+0.439f*in_layerb(n,m,2);
convert2(n,m,2)=128+(0.439f)*in_layerb(n,m,0)+(-0.368f)*in_layerb(n,m,1)+(-0.071f)*in_layerb(n,m,2);
convert2(n,m,o)=convert2(n,m,o)/255;





 //RDom r(0, 3, 0, 3, 0, O);
 //r.where(((n>=0)&&(m<in_layer.width())&&(m>=0)&&(m<in_layer.height())));

//int pad=1;

//RDom ri(0,3,0,3);


convert2.compile_to_static_library("vgg_ao_input",{in_layer},"vgg_ao_input");
}