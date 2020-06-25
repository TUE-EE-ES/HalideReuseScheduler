#include "Halide.h"
#include "clock.h"
#include <stdio.h>
#include "halide_image_io.h"
#define XL 1024
#define YL 1024
#define XIN 128
#define YIN 128
#define Tm 16
#define Tn 64
#define Ti 16
#define To 16
#define d 20
#define M 128
#define N 128
#define O 64
#define s1 3*3*64*64
#define s2 64
#define s3 3*3*64
#define auto





using namespace Halide;
using namespace Halide::Tools;
using namespace std;

  


int main( int argc, char **argv)
{
	ImageParam in_layer(type_of<float>(),3);
	ImageParam b19(type_of<float>(),1);
	ImageParam W19(type_of<float>(),4);
	ImageParam convert2(type_of<float>(), 3);
	int pad=1;
	Var n,m,o;
	RDom r(0,3,0,3,0,64);
	Func in_layert;
in_layert(n,m)=max(16.0f/255,min(235.0f/255,convert2(n,m,0)));
Func b_conv19("b_conv18");
b_conv19(n,m,o) = BoundaryConditions::constant_exterior(in_layer,0.0f,0,in_layer.width(),0,in_layer.height())(n,m,o);
Func f_conv19("f_conv19");
f_conv19=Func("convf");
f_conv19(n, m) = (b19(0));
f_conv19(n, m) += W19(r.x, r.y,r.z,0) *
        b_conv19(n + r.x-pad,
                   m + r.y-pad,r.z
                   );


Func final;
//final(n,m)=((f_conv[d-1](n,m))+in_layert(n,m))*255.0f;
final(n,m)=((f_conv19(n,m))+in_layert(n,m));


final.compile_to_static_library("vgg_ao_auto_final_layer",{in_layer,convert2,W19,b19},"vgg_ao_auto_final_layer");

}