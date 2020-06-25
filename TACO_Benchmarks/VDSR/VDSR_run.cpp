
#include <cstdio>
#include <chrono>


#include "vdsr_input.h"
#include "vdsr_first_layer.h"
#include "vdsr_final_layer.h"
#include "vdsr_convnet.h"
#include "vdsr_output.h"
#include "halide_benchmark.h"
#include "halide_image_io.h"
#include "HalideBuffer.h"
#define s1 3*3*64*64
#define s2 64
#define s3 3*3*64
using namespace Halide::Runtime;
using namespace Halide::Tools;






void input_f(char *fn,int size,float *output)
{
  FILE *fp = fopen(fn,"r");

  int i=0;
  char k;
  do{
    fscanf(fp,"%f",&output[i++]);

    k=fgetc(fp);
}while(k!=EOF);

fclose(fp);

}





int main(int argc, char **argv) {





  static float bias_0[s2];
  static float bias_1[s2];
  static float bias_2[s2];
  static float bias_3[s2];
  static float bias_4[s2];
  static float bias_5[s2];
  static float bias_6[s2];
  static float bias_7[s2];
  static float bias_8[s2];
  static float bias_9[s2];
  static float bias_10[s2];
  static float bias_11[s2];
  static float bias_12[s2];
  static float bias_13[s2];
  static float bias_14[s2];
  static float bias_15[s2];
  static float bias_16[s2];
  static float bias_17[s2];
  static float bias_18[s2];
  static float bias_19[1];

  static float weight_0[s3];
  static float weight_1[s1];
  static float weight_2[s1];
  static float weight_3[s1];
  static float weight_4[s1];
  static float weight_5[s1];
  static float weight_6[s1];
  static float weight_7[s1];
  static float weight_8[s1];
  static float weight_9[s1];
  static float weight_10[s1];
  static float weight_11[s1];
  static float weight_12[s1];
  static float weight_13[s1];
  static float weight_14[s1];
  static float weight_15[s1];
  static float weight_16[s1];
  static float weight_17[s1];
  static float weight_18[s1];
  static float weight_19[s3];


  input_f("DATA/bias_0.dat",s2,bias_0);
  input_f("DATA/bias_1.dat",s2,bias_1);
  input_f("DATA/bias_2.dat",s2,bias_2);
  input_f("DATA/bias_3.dat",s2,bias_3);
  input_f("DATA/bias_4.dat",s2,bias_4);
  input_f("DATA/bias_5.dat",s2,bias_5);
  input_f("DATA/bias_6.dat",s2,bias_6);
  input_f("DATA/bias_7.dat",s2,bias_7);
  input_f("DATA/bias_8.dat",s2,bias_8);
  input_f("DATA/bias_9.dat",s2,bias_9);
  input_f("DATA/bias_10.dat",s2,bias_10);
  input_f("DATA/bias_11.dat",s2,bias_11);
  input_f("DATA/bias_12.dat",s2,bias_12);
  input_f("DATA/bias_13.dat",s2,bias_13);
  input_f("DATA/bias_14.dat",s2,bias_14);
  input_f("DATA/bias_15.dat",s2,bias_15);
  input_f("DATA/bias_16.dat",s2,bias_16);
  input_f("DATA/bias_17.dat",s2,bias_17);
  input_f("DATA/bias_18.dat",s2,bias_18);
  input_f("DATA/bias_19.dat",1,bias_19);




  input_f("DATA/weight_0.dat",s3,weight_0);

  input_f("DATA/weight_1.dat",s1,weight_1);

  input_f("DATA/weight_2.dat",s1,weight_2);
  input_f("DATA/weight_3.dat",s1,weight_3);
  input_f("DATA/weight_4.dat",s1,weight_4);
  input_f("DATA/weight_5.dat",s1,weight_5);
  input_f("DATA/weight_6.dat",s1,weight_6);
  input_f("DATA/weight_7.dat",s1,weight_7);
  input_f("DATA/weight_8.dat",s1,weight_8);
  input_f("DATA/weight_9.dat",s1,weight_9);
  input_f("DATA/weight_10.dat",s1,weight_10);
  input_f("DATA/weight_11.dat",s1,weight_11);
  input_f("DATA/weight_12.dat",s1,weight_12);
  input_f("DATA/weight_13.dat",s1,weight_13);
  input_f("DATA/weight_14.dat",s1,weight_14);
  input_f("DATA/weight_15.dat",s1,weight_15);
  input_f("DATA/weight_16.dat",s1,weight_16);
  input_f("DATA/weight_17.dat",s1,weight_17);
  input_f("DATA/weight_18.dat",s1,weight_18);
  input_f("DATA/weight_19.dat",s3,weight_19);

  Halide::Runtime::Buffer<uint8_t> in_layer = load_and_convert_image("../images/blr.png");
  Halide::Runtime::Buffer<float> W0(weight_0,3, 3,1, 64), b0(bias_0,64);
  Halide::Runtime::Buffer<float> W1(weight_1,3, 3, 64,64), b1(bias_1,64);
  Halide::Runtime::Buffer<float> W2(weight_2,3, 3, 64,64), b2(bias_2,64);
  Halide::Runtime::Buffer<float> W3(weight_3,3, 3, 64,64), b3(bias_3,64);
  Halide::Runtime::Buffer<float> W4(weight_4,3, 3, 64,64), b4(bias_4,64);
  Halide::Runtime::Buffer<float> W5(weight_5,3, 3, 64,64), b5(bias_5,64);
  Halide::Runtime::Buffer<float> W6(weight_6,3, 3, 64,64), b6(bias_6,64);
  Halide::Runtime::Buffer<float> W7(weight_7,3, 3, 64,64), b7(bias_7,64);
  Halide::Runtime::Buffer<float> W8(weight_8,3, 3, 64,64), b8(bias_8,64);
  Halide::Runtime::Buffer<float> W9(weight_9,3, 3, 64,64), b9(bias_9,64);
  Halide::Runtime::Buffer<float> W10(weight_10,3, 3, 64,64), b10(bias_10,64);
  Halide::Runtime::Buffer<float> W11(weight_11,3, 3, 64,64), b11(bias_11,64);
  Halide::Runtime::Buffer<float> W12(weight_12,3, 3, 64,64), b12(bias_12,64);
  Halide::Runtime::Buffer<float> W13(weight_13,3, 3, 64,64), b13(bias_13,64);
  Halide::Runtime::Buffer<float> W14(weight_14,3, 3, 64,64), b14(bias_14,64);
  Halide::Runtime::Buffer<float> W15(weight_15,3, 3, 64,64), b15(bias_15,64);
  Halide::Runtime::Buffer<float> W16(weight_16,3, 3, 64,64), b16(bias_16,64);
  Halide::Runtime::Buffer<float> W17(weight_17,3, 3, 64,64), b17(bias_17,64);
  Halide::Runtime::Buffer<float> W18(weight_18,3, 3, 64,64), b18(bias_18,64);
  Halide::Runtime::Buffer<float> W19(weight_19,3, 3, 64,1), b19(bias_19,1);
  Halide::Runtime::Buffer<uint8_t>output_im(in_layer.width(),in_layer.height(),3);
  Halide::Runtime::Buffer<float>net_input(in_layer.width(),in_layer.height(),3);
  Halide::Runtime::Buffer<float>first_layer(in_layer.width(),in_layer.height(),64);
  Halide::Runtime::Buffer<float>dnn(in_layer.width(),in_layer.height(),64);
  Halide::Runtime::Buffer<float>last_layer(in_layer.width(),in_layer.height());
  
  vdsr_input(in_layer,net_input);
  vdsr_first_layer(net_input,W0,b0,first_layer);    
// Timing code
  double min_t_auto = benchmark(2, 2, [&]() {
    vdsr_convnet(first_layer,W1,W2,W3,W4,W5,W6,W7,W8,W9,W10,W11,W12,W13,W14,W15,W16,W17,W18,b1,b2,b3,b4,b5,b6,b7,b8,b9,b10,b11,b12,b13,b14,b15,b16,b17,b18,dnn);
});
  vdsr_final_layer(dnn,net_input,W19,b19,last_layer);
  vdsr_output(net_input,last_layer,output_im);
  printf("Auto-scheduled time: %gms\n", min_t_auto * 1e3);
  convert_and_save_image(output_im,"output.png");
  return 0;
}
