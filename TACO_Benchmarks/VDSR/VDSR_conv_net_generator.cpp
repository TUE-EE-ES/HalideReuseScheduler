#include "Halide.h"

namespace {

	using namespace Halide;

	using std::vector;
	class VDSR_ConvNet : public Halide::Generator<VDSR_ConvNet> {
	public:

		Input<Buffer<float>> data{"data",3};

		Input<Buffer<float>> W1{"W1",4};
		Input<Buffer<float>> W2{"W2",4};
		Input<Buffer<float>> W3{"W3",4};
		Input<Buffer<float>> W4{"W4",4};
		Input<Buffer<float>> W5{"W5",4};
		Input<Buffer<float>> W6{"W6",4};
		Input<Buffer<float>> W7{"W7",4};
		Input<Buffer<float>> W8{"W8",4};
		Input<Buffer<float>> W9{"W9",4};
		Input<Buffer<float>> W10{"W10",4};
		Input<Buffer<float>> W11{"W11",4};
		Input<Buffer<float>> W12{"W12",4};
		Input<Buffer<float>> W13{"W13",4};
		Input<Buffer<float>> W14{"W14",4};
		Input<Buffer<float>> W15{"W15",4};
		Input<Buffer<float>> W16{"W16",4};
		Input<Buffer<float>> W17{"W17",4};
		Input<Buffer<float>> W18{"W18",4};

		Input<Buffer<float>> b1{"b1",1};
		Input<Buffer<float>> b2{"b2",1};
		Input<Buffer<float>> b3{"b3",1};
		Input<Buffer<float>> b4{"b4",1};
		Input<Buffer<float>> b5{"b5",1};
		Input<Buffer<float>> b6{"b6",1};
		Input<Buffer<float>> b7{"b7",1};
		Input<Buffer<float>> b8{"b8",1};
		Input<Buffer<float>> b9{"b9",1};
		Input<Buffer<float>> b10{"b10",1};
		Input<Buffer<float>> b11{"b11",1};
		Input<Buffer<float>> b12{"b12",1};
		Input<Buffer<float>> b13{"b13",1};
		Input<Buffer<float>> b14{"b14",1};
		Input<Buffer<float>> b15{"b15",1};
		Input<Buffer<float>> b16{"b16",1};
		Input<Buffer<float>> b17{"b17",1};
		Input<Buffer<float>> b18{"b18",1};
		Output<Buffer<float>> f_conv18{"f_conv18",3};


		void generate(){
			int pad=1;
			Var n("n"),m("m"),i("i"),o("o"),c("c");
			RDom r(0, 3, 0, 3, 0, 64);
			Func bconv1("b_conv1");
			Func max_layer1("max_layer1");
			Func f_conv1("f_conv1");
			Func data_in("data_in");

			bconv1(n,m,o) = BoundaryConditions::constant_exterior(data,0.0f,0,data.width(),0,data.height())(n,m,o);
			f_conv1(n,m,o)=b1(o);
			f_conv1(n, m, o) +=(W1(r.x, r.y, r.z, o) *
				bconv1(n + r.x - pad, m + r.y - pad, r.z));
			f_conv1(n, m, o) = (max(0.0f, f_conv1(n, m, o)));
			f_conv1(n,m,o)=select(((n>=0)&&(n<data.width())&&(m>=0)&&(m<data.height())),likely(f_conv1(n,m,o)),0.0f); 
// max_layer3(n, m, o) = (max(0.0f, f_conv3(n, m, o)));


////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
			Func f_conv2("f_conv2");


			
			f_conv2(n,m,o)=b2(o);
			f_conv2(n, m, o) +=(W2(r.x, r.y, r.z, o) *
				f_conv1(n + r.x - pad, m + r.y - pad, r.z));
			f_conv2(n, m, o) = (max(0.0f, f_conv2(n, m, o)));
			f_conv2(n,m,o)=select(((n>=0)&&(n<data.width())&&(m>=0)&&(m<data.height())),likely(f_conv2(n,m,o)),0.0f); 

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
			Func f_conv3("f_conv3");


			
			f_conv3(n,m,o)=b3(o);
			f_conv3(n, m, o) +=(W3(r.x, r.y, r.z, o) *
				f_conv2(n + r.x - pad, m + r.y - pad, r.z));
			f_conv3(n, m, o) = (max(0.0f, f_conv3(n, m, o)));
			f_conv3(n,m,o)=select(((n>=0)&&(n<data.width())&&(m>=0)&&(m<data.height())),likely(f_conv3(n,m,o)),0.0f); 

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
			Func f_conv4("f_conv4");


			f_conv4(n,m,o)=b4(o);
			f_conv4(n, m, o) +=(W4(r.x, r.y, r.z, o) *
				f_conv3(n + r.x - pad, m + r.y - pad, r.z));
			f_conv4(n, m, o) = (max(0.0f, f_conv4(n, m, o)));
			f_conv4(n,m,o)=select(((n>=0)&&(n<data.width())&&(m>=0)&&(m<data.height())),likely(f_conv4(n,m,o)),0.0f); 

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
			Func f_conv5("f_conv5");


			f_conv5(n,m,o)=b5(o);
			f_conv5(n, m, o) +=(W5(r.x, r.y, r.z, o) *
				f_conv4(n + r.x - pad, m + r.y - pad, r.z));
			f_conv5(n, m, o) = (max(0.0f, f_conv5(n, m, o)));
			f_conv5(n,m,o)=select(((n>=0)&&(n<data.width())&&(m>=0)&&(m<data.height())),likely(f_conv5(n,m,o)),0.0f); 

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
			Func f_conv6("f_conv6");


			f_conv6(n,m,o)=b6(o);
			f_conv6(n, m, o) +=(W6(r.x, r.y, r.z, o) *
				f_conv5(n + r.x - pad, m + r.y - pad, r.z));
			f_conv6(n, m, o) = (max(0.0f, f_conv6(n, m, o)));
			f_conv6(n,m,o)=select(((n>=0)&&(n<data.width())&&(m>=0)&&(m<data.height())),likely(f_conv6(n,m,o)),0.0f); 

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
			Func f_conv7("f_conv7");


			f_conv7(n,m,o)=b7(o);
			f_conv7(n, m, o) +=(W7(r.x, r.y, r.z, o) *
				f_conv6(n + r.x - pad, m + r.y - pad, r.z));
			f_conv7(n, m, o) = (max(0.0f, f_conv7(n, m, o)));
			f_conv7(n,m,o)=select(((n>=0)&&(n<data.width())&&(m>=0)&&(m<data.height())),likely(f_conv7(n,m,o)),0.0f); 

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
			Func f_conv8("f_conv8");


			f_conv8(n,m,o)=b8(o);
			f_conv8(n, m, o) +=(W8(r.x, r.y, r.z, o) *
				f_conv7(n + r.x - pad, m + r.y - pad, r.z));
			f_conv8(n, m, o) = (max(0.0f, f_conv8(n, m, o)));
			f_conv8(n,m,o)=select(((n>=0)&&(n<data.width())&&(m>=0)&&(m<data.height())),likely(f_conv8(n,m,o)),0.0f); 

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
			Func f_conv9("f_conv9");


			f_conv9(n,m,o)=b9(o);
			f_conv9(n, m, o) +=(W9(r.x, r.y, r.z, o) *
				f_conv8(n + r.x - pad, m + r.y - pad, r.z));
			f_conv9(n, m, o) = (max(0.0f, f_conv9(n, m, o)));
			f_conv9(n,m,o)=select(((n>=0)&&(n<data.width())&&(m>=0)&&(m<data.height())),likely(f_conv9(n,m,o)),0.0f); 


////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
			Func f_conv10("f_conv10");

			f_conv10(n,m,o)=b10(o);
			f_conv10(n, m, o) +=(W10(r.x, r.y, r.z, o) *
				f_conv9(n + r.x - pad, m + r.y - pad, r.z));
			f_conv10(n, m, o) = (max(0.0f, f_conv10(n, m, o)));
			f_conv10(n,m,o)=select(((n>=0)&&(n<data.width())&&(m>=0)&&(m<data.height())),likely(f_conv10(n,m,o)),0.0f); 


////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
			Func f_conv11("f_conv11");


			
			f_conv11(n,m,o)=b11(o);
			f_conv11(n, m, o) +=(W11(r.x, r.y, r.z, o) *
				f_conv10(n + r.x - pad, m + r.y - pad, r.z));
			f_conv11(n, m, o) = (max(0.0f, f_conv11(n, m, o)));
			f_conv11(n,m,o)=select(((n>=0)&&(n<data.width())&&(m>=0)&&(m<data.height())),likely(f_conv11(n,m,o)),0.0f); 


////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
			Func f_conv12("f_conv12");


			f_conv12(n,m,o)=b12(o);
			f_conv12(n, m, o) +=(W12(r.x, r.y, r.z, o) *
				f_conv11(n + r.x - pad, m + r.y - pad, r.z));
			f_conv12(n, m, o) = (max(0.0f, f_conv12(n, m, o)));
			f_conv12(n,m,o)=select(((n>=0)&&(n<data.width())&&(m>=0)&&(m<data.height())),likely(f_conv12(n,m,o)),0.0f); 


////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
			Func f_conv13("f_conv13");

			f_conv13(n,m,o)=b13(o);
			f_conv13(n, m, o) +=(W13(r.x, r.y, r.z, o) *
				f_conv12(n + r.x - pad, m + r.y - pad, r.z));
			f_conv13(n, m, o) = (max(0.0f, f_conv13(n, m, o)));
			f_conv13(n,m,o)=select(((n>=0)&&(n<data.width())&&(m>=0)&&(m<data.height())),likely(f_conv13(n,m,o)),0.0f); 


////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
			Func f_conv14("f_conv14");


			f_conv14(n,m,o)=b14(o);
			f_conv14(n, m, o) +=(W14(r.x, r.y, r.z, o) *
				f_conv13(n + r.x - pad, m + r.y - pad, r.z));
			f_conv14(n, m, o) = (max(0.0f, f_conv14(n, m, o)));
			f_conv14(n,m,o)=select(((n>=0)&&(n<data.width())&&(m>=0)&&(m<data.height())),likely(f_conv14(n,m,o)),0.0f); 

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
			Func f_conv15("f_conv15");

			f_conv15(n,m,o)=b15(o);
			f_conv15(n, m, o) +=(W15(r.x, r.y, r.z, o) *
				f_conv14(n + r.x - pad, m + r.y - pad, r.z));
			f_conv15(n, m, o) = (max(0.0f, f_conv15(n, m, o)));
			f_conv15(n,m,o)=select(((n>=0)&&(n<data.width())&&(m>=0)&&(m<data.height())),likely(f_conv15(n,m,o)),0.0f); 


////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

			Func f_conv16("f_conv16");


			f_conv16(n,m,o)=b16(o);
			f_conv16(n, m, o) +=(W16(r.x, r.y, r.z, o) *
				f_conv15(n + r.x - pad, m + r.y - pad, r.z));
			f_conv16(n, m, o) = (max(0.0f, f_conv16(n, m, o)));
			f_conv16(n,m,o)=select(((n>=0)&&(n<data.width())&&(m>=0)&&(m<data.height())),likely(f_conv16(n,m,o)),0.0f); 

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

			Func f_conv17("f_conv17");


			f_conv17(n,m,o)=b17(o);
			f_conv17(n, m, o) +=(W17(r.x, r.y, r.z, o) *
				f_conv16(n + r.x - pad, m + r.y - pad, r.z));
			f_conv17(n, m, o) = (max(0.0f, f_conv17(n, m, o)));
			f_conv17(n,m,o)=select(((n>=0)&&(n<data.width())&&(m>=0)&&(m<data.height())),likely(f_conv17(n,m,o)),0.0f); 


////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////




			f_conv18(n,m,o)=b18(o);
			f_conv18(n, m, o) +=(W18(r.x, r.y, r.z, o) *
				f_conv17(n + r.x - pad, m + r.y - pad, r.z));
			f_conv18(n, m, o) = (max(0.0f, f_conv18(n, m, o)));
			f_conv18(n,m,o)=select(((n>=0)&&(n<data.width())&&(m>=0)&&(m<data.height())),likely(f_conv18(n,m,o)),0.0f); 




//estimates
			if(auto_schedule){


				data.dim(0).set_bounds_estimate(0, 500);
				data.dim(1).set_bounds_estimate(0,480);
				data.dim(2).set_bounds_estimate(0, 64);
				



				W1.dim(0).set_bounds_estimate(0, 3);
				W1.dim(1).set_bounds_estimate(0, 3);
				W1.dim(2).set_bounds_estimate(0, 3);
				W1.dim(3).set_bounds_estimate(0, 64);

				W2.dim(0).set_bounds_estimate(0, 3);
				W2.dim(1).set_bounds_estimate(0, 3);
				W2.dim(2).set_bounds_estimate(0, 64);
				W2.dim(3).set_bounds_estimate(0, 64);

				W3.dim(0).set_bounds_estimate(0, 3);
				W3.dim(1).set_bounds_estimate(0, 3);
				W3.dim(2).set_bounds_estimate(0, 64);
				W3.dim(3).set_bounds_estimate(0, 64);

				W3.dim(0).set_bounds_estimate(0, 3);
				W3.dim(1).set_bounds_estimate(0, 3);
				W3.dim(2).set_bounds_estimate(0, 64);
				W3.dim(3).set_bounds_estimate(0, 64);

				W4.dim(0).set_bounds_estimate(0, 3);
				W4.dim(1).set_bounds_estimate(0, 3);
				W4.dim(2).set_bounds_estimate(0, 64);
				W4.dim(3).set_bounds_estimate(0, 64);

				W5.dim(0).set_bounds_estimate(0, 3);
				W5.dim(1).set_bounds_estimate(0, 3);
				W5.dim(2).set_bounds_estimate(0, 64);
				W5.dim(3).set_bounds_estimate(0, 64);

				W6.dim(0).set_bounds_estimate(0, 3);
				W6.dim(1).set_bounds_estimate(0, 3);
				W6.dim(2).set_bounds_estimate(0, 64);
				W6.dim(3).set_bounds_estimate(0, 64);

				W7.dim(0).set_bounds_estimate(0, 3);
				W7.dim(1).set_bounds_estimate(0, 3);
				W7.dim(2).set_bounds_estimate(0, 64);
				W7.dim(3).set_bounds_estimate(0, 64);

				W8.dim(0).set_bounds_estimate(0, 3);
				W8.dim(1).set_bounds_estimate(0, 3);
				W8.dim(2).set_bounds_estimate(0, 64);
				W8.dim(3).set_bounds_estimate(0, 64);

				W9.dim(0).set_bounds_estimate(0, 3);
				W9.dim(1).set_bounds_estimate(0, 3);
				W9.dim(2).set_bounds_estimate(0, 64);
				W9.dim(3).set_bounds_estimate(0, 64);

				W10.dim(0).set_bounds_estimate(0, 3);
				W10.dim(1).set_bounds_estimate(0, 3);
				W10.dim(2).set_bounds_estimate(0, 64);
				W10.dim(3).set_bounds_estimate(0, 64);

				W11.dim(0).set_bounds_estimate(0, 3);
				W11.dim(1).set_bounds_estimate(0, 3);
				W11.dim(2).set_bounds_estimate(0, 64);
				W11.dim(3).set_bounds_estimate(0, 64);

				W12.dim(0).set_bounds_estimate(0, 3);
				W12.dim(1).set_bounds_estimate(0, 3);
				W12.dim(2).set_bounds_estimate(0, 64);
				W12.dim(3).set_bounds_estimate(0, 64);

				W13.dim(0).set_bounds_estimate(0, 3);
				W13.dim(1).set_bounds_estimate(0, 3);
				W13.dim(2).set_bounds_estimate(0, 64);
				W13.dim(3).set_bounds_estimate(0, 64);

				W14.dim(0).set_bounds_estimate(0, 3);
				W14.dim(1).set_bounds_estimate(0, 3);
				W14.dim(2).set_bounds_estimate(0, 64);
				W14.dim(3).set_bounds_estimate(0, 64);

				W15.dim(0).set_bounds_estimate(0, 3);
				W15.dim(1).set_bounds_estimate(0, 3);
				W15.dim(2).set_bounds_estimate(0, 64);
				W15.dim(3).set_bounds_estimate(0, 64);

				W16.dim(0).set_bounds_estimate(0, 3);
				W16.dim(1).set_bounds_estimate(0, 3);
				W16.dim(2).set_bounds_estimate(0, 64);
				W16.dim(3).set_bounds_estimate(0, 64);


				W17.dim(0).set_bounds_estimate(0, 3);
				W17.dim(1).set_bounds_estimate(0, 3);
				W17.dim(2).set_bounds_estimate(0, 64);
				W17.dim(3).set_bounds_estimate(0, 64);

				W18.dim(0).set_bounds_estimate(0, 3);
				W18.dim(1).set_bounds_estimate(0, 3);
				W18.dim(2).set_bounds_estimate(0, 64);
				W18.dim(3).set_bounds_estimate(0, 64);




				
				b1.dim(0).set_bounds_estimate(0, 64);
				b2.dim(0).set_bounds_estimate(0, 64);
				b3.dim(0).set_bounds_estimate(0, 64);
				b4.dim(0).set_bounds_estimate(0, 64);
				b5.dim(0).set_bounds_estimate(0, 64);
				b6.dim(0).set_bounds_estimate(0, 64);
				b7.dim(0).set_bounds_estimate(0, 64);
				b8.dim(0).set_bounds_estimate(0, 64);
				b9.dim(0).set_bounds_estimate(0, 64);
				b10.dim(0).set_bounds_estimate(0, 64);
				b11.dim(0).set_bounds_estimate(0, 64);
				b12.dim(0).set_bounds_estimate(0, 64);
				b13.dim(0).set_bounds_estimate(0, 64);
				b14.dim(0).set_bounds_estimate(0, 64);
				b15.dim(0).set_bounds_estimate(0, 64);
				b16.dim(0).set_bounds_estimate(0, 64);
				b17.dim(0).set_bounds_estimate(0, 64);
				b18.dim(0).set_bounds_estimate(0, 64);
				
				f_conv18.estimate(n,0,500).estimate(m,0,480).estimate(o,0,64);    
			}

		}
	};

}  // namespace

HALIDE_REGISTER_GENERATOR(VDSR_ConvNet, vdsr_convnet)

