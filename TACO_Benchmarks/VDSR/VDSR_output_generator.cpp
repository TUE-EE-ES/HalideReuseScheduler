#include "Halide.h"

namespace {

using namespace Halide;

using std::vector;
class VDSR_Output : public Halide::Generator<VDSR_Output> {
public:

    Input<Buffer<float>>  convert2{"convert2", 3};
    Input<Buffer<float>> final{"final", 2};
    Output<Buffer<uint8_t>> output{"output", 3};


void generate(){
Var n,m,o;
Func convertb("convertb");
Func newconvert("nconv");
Func mconv("mconv");

newconvert(n,m,o)=convert2(n,m,o);
newconvert(n,m,0)=final(n,m);
mconv(n,m,o)=(newconvert(n,m,o))*255;
convertb(n,m,o)=cast<float>(0);
convertb(n,m,0)=1.164f*(mconv(n,m,0)-16)+0.00f*(mconv(n,m,1)-128)+1.596f*(mconv(n,m,2)-128);
convertb(n,m,1)=(1.164f)*(mconv(n,m,0)-16)+(-0.392f)*(mconv(n,m,1)-128)+(-0.813f)*(mconv(n,m,2)-128);
convertb(n,m,2)=(1.164f)*(mconv(n,m,0)-16)+(2.017f)*(mconv(n,m,1)-128)+(-0.00f)*(mconv(n,m,2)-128);
output(n,m,o)=cast<uint8_t>(convertb(n,m,o));
}
};

}  // namespace

HALIDE_REGISTER_GENERATOR(VDSR_Output, vdsr_output)

