// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "gtest/gtest.h"
#include "test/providers/provider_test_utils.h"
using namespace std;
namespace onnxruntime {
namespace test {

namespace {

struct ConvOpAndTestAttributes {
  string auto_pad;
  vector<int64_t> dilations;
  int64_t group;
  vector<int64_t> kernel_shape;
  vector<int64_t> pads;
  vector<int64_t> strides;
  std::unordered_set<std::string> excluded_providers;
};

void TestConvOp(const ConvOpAndTestAttributes& attributes,
                const vector<vector<float>>& inputs,
                const vector<vector<int64_t>>& input_shapes,
                const std::initializer_list<float>& expected_output,
                const vector<int64_t>& expected_output_shape,
                bool weight_is_initializer = true,
                OpTester::ExpectResult expect_result = OpTester::ExpectResult::kExpectSuccess,
                const std::string& err_str = "",
                int opset = 7) {
  OpTester test("Conv", opset);
  test.AddAttribute("group", attributes.group);
  test.AddAttribute("kernel_shape", attributes.kernel_shape);

  if (!attributes.dilations.empty()) {
    test.AddAttribute("dilations", attributes.dilations);
  }

  // Only one of pads / auto_pad can be present
  if (!attributes.pads.empty()) {
    test.AddAttribute("pads", attributes.pads);
  } else {
    test.AddAttribute("auto_pad", attributes.auto_pad);
  }

  if (!attributes.strides.empty()) {
    test.AddAttribute("strides", attributes.strides);
  }

  ORT_ENFORCE(inputs.size() <= 3, "Our name array is only setup to handle 3 inputs");
  const char* szNames[] = {"X", "W", "B"};
  test.AddInput<float>(szNames[0], input_shapes[0], inputs[0]);
  test.AddInput<float>(szNames[1], input_shapes[1], inputs[1], weight_is_initializer);
  if (inputs.size() == 3)
    test.AddInput<float>(szNames[2], input_shapes[2], inputs[2]);

  test.AddOutput<float>("Y", expected_output_shape, expected_output);

  std::unordered_set<std::string> excluded_providers(attributes.excluded_providers);
  // Disable TensorRT because weight as input is not supported
  excluded_providers.insert(kTensorrtExecutionProvider);

  test.Run(expect_result, err_str, excluded_providers);
}

}  // namespace

// Conv
TEST(ConvTest, Conv1D_1) {
  ConvOpAndTestAttributes attrs = {
      "",                     // auto_pad
      vector<int64_t>{1},     // dilations
      1,                      // group
      vector<int64_t>{1},     // kernel_shape
      vector<int64_t>{0, 0},  // pads
      vector<int64_t>{1},     // strides
      {}                      // excluded EPs
  };

  vector<float> X = {-0.21559301018714905f, 0.4691687822341919f, 0.4426700472831726f, -0.4517466723918915f,
                     -0.05216419696807861f, 0.29067182540893555f, 0.251010000705719f};
  vector<int64_t> X_shape = {1, 1, 7};
  vector<float> W = {0.24472862482070923f};
  vector<int64_t> W_shape = {1, 1, 1};
  vector<int64_t> Y_shape = {1, 1, 7};
  auto expected_vals = {-0.052761781960725784f, 0.11481902748346329f, 0.10833403468132019f, -0.11055534332990646f,
                        -0.012766072526574135f, 0.07113571465015411f, 0.061429332941770554f};

  TestConvOp(attrs, {X, W}, {X_shape, W_shape}, expected_vals, Y_shape);
}

TEST(ConvTest, Conv1D_1_DefaultStridesAndDilations) {
  ConvOpAndTestAttributes attrs = {
      "",                     // auto_pad
      vector<int64_t>{},      // dilations
      1,                      // group
      vector<int64_t>{1},     // kernel_shape
      vector<int64_t>{0, 0},  // pads
      vector<int64_t>{},      // strides
      {}                      // excluded EPs
  };

  vector<float> X = {-0.21559301018714905f, 0.4691687822341919f, 0.4426700472831726f, -0.4517466723918915f,
                     -0.05216419696807861f, 0.29067182540893555f, 0.251010000705719f};
  vector<int64_t> X_shape = {1, 1, 7};
  vector<float> W = {0.24472862482070923f};
  vector<int64_t> W_shape = {1, 1, 1};
  vector<int64_t> Y_shape = {1, 1, 7};
  auto expected_vals = {-0.052761781960725784f, 0.11481902748346329f, 0.10833403468132019f, -0.11055534332990646f,
                        -0.012766072526574135f, 0.07113571465015411f, 0.061429332941770554f};

  TestConvOp(attrs, {X, W}, {X_shape, W_shape}, expected_vals, Y_shape);
}

// Conv3
TEST(ConvTest, Conv1D_2) {
  ConvOpAndTestAttributes attrs = {
      "",                     // auto_pad
      vector<int64_t>{2},     // dilations
      1,                      // group
      vector<int64_t>{2},     // kernel_shape
      vector<int64_t>{2, 2},  // pads
      vector<int64_t>{2},     // strides
      {}                      // excluded EPs
  };

  vector<float> X = {0.11094123125076294f, -0.0038032233715057373f, 0.3896123170852661f, 0.33259105682373047f,
                     0.02794349193572998f, -0.08360505104064941f, -0.4100455045700073f, -0.09502679109573364f,
                     -0.11361867189407349f, -0.025495320558547974f, 0.3696536421775818f, 0.3529144525527954f,
                     -0.34991076588630676f, -0.22024285793304443f, 0.23085933923721313f, -0.4575521945953369f,
                     -0.17685726284980774f, -0.06030535697937012f, -0.3996139168739319f, -0.19385704398155212f,
                     -0.10454908013343811f, -0.14503943920135498f, -0.31941986083984375f, -0.15372398495674133f};
  vector<int64_t> X_shape = {3, 1, 8};
  vector<float> W = {0.13225573301315308f, 0.09750443696975708f, 0.3469849228858948f, 0.4743430018424988f};
  vector<int64_t> W_shape = {2, 1, 2};
  vector<int64_t> Y_shape = {3, 2, 5};
  auto expected_vals = {0.010817262344062328f, 0.05266154557466507f, 0.054253075271844864f, -0.03628557175397873f,
                        -0.05423086881637573f, 0.05262419581413269f, 0.22330480813980103f, 0.14844439923763275f,
                        -0.1848062425851822f, -0.14227961003780365f, -0.011078324168920517f, 0.02101614698767662f,
                        0.014770962297916412f, -0.023767895996570587f, 0.03053247183561325f, -0.053894221782684326f,
                        0.13591864705085754f, -0.03771348297595978f, -0.011907249689102173f, 0.08010470867156982f,
                        -0.01724436692893505f, -0.06235451623797417f, -0.06304522603750229f, -0.044972069561481476f,
                        -0.042245108634233475f, -0.08389100432395935f, -0.2509208619594574f, -0.18825212121009827f,
                        -0.18779152631759644f, -0.11083387583494186f};

  TestConvOp(attrs, {X, W}, {X_shape, W_shape}, expected_vals, Y_shape);
}

// Conv1
TEST(ConvTest, Conv1D_Bias) {
  ConvOpAndTestAttributes attrs = {
      "",                     // auto_pad
      vector<int64_t>{2},     // dilations
      1,                      // group
      vector<int64_t>{1},     // kernel_shape
      vector<int64_t>{1, 1},  // pads
      vector<int64_t>{3},     // strides
      {}                      // excluded EPs
  };

  vector<float> X = {0.4582272171974182f, 0.3877705931663513f, -0.05413919687271118f, -0.3013981878757477f,
                     0.19299334287643433f, -0.4758569598197937f, 0.4670986533164978f, 0.4078403115272522f,
                     0.24010121822357178f, 0.41645896434783936f, -0.038333237171173096f, 0.22969317436218262f,
                     0.3565492033958435f, 0.12812334299087524f, 0.10096627473831177f, 0.25682520866394043f,
                     0.41700226068496704f, 0.34114283323287964f, -0.429997980594635f, 0.3545404076576233f,
                     0.40339237451553345f, 0.10174298286437988f, 0.45713120698928833f, 0.08574831485748291f,
                     0.38086581230163574f, 0.16378509998321533f, 0.12321442365646362f, -0.19936135411262512f,
                     0.26019394397735596f, -0.18406429886817932f, 0.3110783100128174f, 0.15553230047225952f,
                     -0.14629846811294556f, -0.1779327094554901f, -0.01390346884727478f, -0.09264758229255676f};
  vector<int64_t> X_shape = {2, 2, 9};
  vector<float> W = {-0.17206084728240967f, 0.3236315846443176f};
  vector<int64_t> W_shape = {1, 2, 1};
  vector<float> B = {0.37892162799835205f};
  vector<int64_t> B_shape = {1};
  vector<int64_t> Y_shape = {2, 1, 4};
  auto expected_vals = {0.37892162799835205f, 0.4625728130340576f, 0.4934738576412201f, 0.44801419973373413f,
                        0.37892162799835205f, 0.2499445676803589f, 0.31682088971138f, 0.32773756980895996f};
  TestConvOp(attrs, {X, W, B}, {X_shape, W_shape, B_shape}, expected_vals, Y_shape);
}

// Conv47
TEST(ConvTest, Conv2D_1) {
  ConvOpAndTestAttributes attrs = {
      "",                           // auto_pad
      vector<int64_t>{1, 1},        // dilations
      1,                            // group
      vector<int64_t>{3, 3},        // kernel_shape
      vector<int64_t>{1, 1, 1, 2},  // pads
      vector<int64_t>{3, 1},        // strides
      {}                            // excluded EPs
  };

  vector<float> X = {-0.09103918075561523f, -0.32513630390167236f};
  vector<int64_t> X_shape = {2, 1, 1, 1};
  vector<float> W = {0.4312484860420227f, -0.12559029459953308f, 0.44889551401138306f, -0.3100617825984955f,
                     0.13522827625274658f, -0.06791308522224426f, 0.22671669721603394f, -0.17391827702522278f,
                     -0.31299442052841187f, -0.31545522809028625f, 0.06560015678405762f, 0.2656586766242981f,
                     0.41363757848739624f, 0.31231558322906494f, -0.376018226146698f, -0.005708813667297363f,
                     0.34922850131988525f, 0.45095211267471313f};
  vector<int64_t> W_shape = {2, 1, 3, 3};
  vector<int64_t> Y_shape = {2, 2, 1, 2};
  auto expected_vals = {-0.012311071157455444f, 0.02822777070105076f, -0.028432954102754593f, -0.037657227367162704f,
                        -0.04396762326359749f, 0.10081233829259872f, -0.10154513269662857f, -0.13448859751224518f};

  TestConvOp(attrs, {X, W}, {X_shape, W_shape}, expected_vals, Y_shape);

  // NNAPI/CoreML EP requires weight to be an initializer
  TestConvOp(attrs, {X, W}, {X_shape, W_shape}, expected_vals, Y_shape, true);
}

TEST(ConvTest, Conv1D_Invalid_Input_Shape) {
  ConvOpAndTestAttributes attrs = {
      "",                     // auto_pad
      vector<int64_t>{1},     // dilations
      1,                      // group
      vector<int64_t>{2},     // kernel_shape
      vector<int64_t>{0, 0},  // pads
      vector<int64_t>{1},     // strides
      {}                      // excluded EPs
  };

  vector<float> X = vector<float>(1, 1.0f);
  vector<int64_t> X_shape = {1, 1, 1};
  vector<int64_t> dummy_shape = {1, 1, 2};
  auto dummy_vals = {0.0f, 0.0f};
  TestConvOp(attrs, {X, dummy_vals}, {X_shape, dummy_shape}, dummy_vals, dummy_shape, false,
             OpTester::ExpectResult::kExpectFailure,
             "Node:node1 Output:Y [ShapeInferenceError] Can't merge shape info. "
             "Both source and target dimension have values but they differ. Source=0 Target=2 Dimension=2",
             -1);  // use latest opset for shape inferencing errors
}

TEST(ConvTest, Conv2D_Invalid_Input_Shape) {
  ConvOpAndTestAttributes attrs = {
      "",                           // auto_pad
      vector<int64_t>{1, 1},        // dilations
      1,                            // group
      vector<int64_t>{3, 3},        // kernel_shape
      vector<int64_t>{0, 0, 0, 0},  // pads
      vector<int64_t>{1, 1},        // strides
      {}                            // excluded EPs
  };

  vector<float> X = vector<float>(1 * 3 * 1 * 111, 1.0f);
  vector<int64_t> X_shape = {1, 3, 1, 111};
  vector<int64_t> dummy_shape = {2, 2, 1, 2};
  auto dummy_vals = {-0.0f, 0.0f, -0.0f, -0.0f,
                     -0.0f, 0.0f, -0.0f, -0.0f};
  TestConvOp(attrs, {X, dummy_vals}, {X_shape, dummy_shape}, dummy_vals, dummy_shape, false,
             OpTester::ExpectResult::kExpectFailure,
             "Node:node1 Output:Y [ShapeInferenceError] Can't merge shape info. "
             "Both source and target dimension have values but they differ. Source=1 Target=2 Dimension=0",
             -1);  // use latest opset for shape inferencing errors
}

// Conv30
TEST(ConvTest, Conv2D_2) {
  ConvOpAndTestAttributes attrs = {
      "",                           // auto_pad
      vector<int64_t>{1, 1},        // dilations
      1,                            // group
      vector<int64_t>{1, 1},        // kernel_shape
      vector<int64_t>{0, 0, 0, 0},  // pads
      vector<int64_t>{1, 1},        // strides
      {}                            // excluded EPs
  };

  vector<float> X = {0.45246148109436035f, 0.15498268604278564f, 0.11199361085891724f, -0.39421093463897705f,
                     0.2626858949661255f, 0.13414543867111206f, -0.27184486389160156f, -0.43028733134269714f,
                     -0.26825493574142456f, 0.3893144130706787f, -0.13631996512413025f, -0.009590476751327515f,
                     -0.48771554231643677f, -0.25256502628326416f, -0.2812897562980652f, 0.4043201804161072f,
                     0.07795023918151855f, 0.326981782913208f, 0.13114392757415771f, -0.4416425824165344f,
                     0.12446999549865723f, 0.36739975214004517f, 0.1698915958404541f, 0.2008744478225708f,
                     0.23339951038360596f, 0.38613730669021606f, 0.11117297410964966f, 0.3877097964286804f,
                     0.20812749862670898f, -0.34297940135002136f, -0.029246658086776733f, -0.20483523607254028f,
                     -0.19244328141212463f, -0.11104947328567505f, -0.32830488681793213f, -0.01800677180290222f,
                     0.3618946671485901f, -0.40949052572250366f, -0.18248388171195984f, -0.3349453806877136f,
                     -0.34091079235076904f, 0.006497859954833984f, 0.4537564516067505f, 0.08006560802459717f,
                     -0.14788749814033508f, 0.034442365169525146f, -0.33322954177856445f, 0.06049239635467529f,
                     0.42619407176971436f};
  vector<int64_t> X_shape = {1, 1, 7, 7};
  vector<float> W = {-0.4406261742115021f};
  vector<int64_t> W_shape = {1, 1, 1, 1};
  vector<int64_t> Y_shape = {1, 1, 7, 7};
  auto expected_vals = {-0.19936637580394745f, -0.06828942894935608f, -0.04934731498360634f, 0.17369966208934784f,
                        -0.11574628204107285f, -0.05910799279808998f, 0.1197819635272026f, 0.18959586322307587f,
                        0.1182001456618309f, -0.17154212296009064f, 0.06006614491343498f, 0.0042258151806890965f,
                        0.21490024030208588f, 0.11128675937652588f, 0.12394362688064575f, -0.17815405130386353f,
                        -0.034346915781497955f, -0.14407673478126526f, -0.05778544768691063f, 0.19459928572177887f,
                        -0.05484473705291748f, -0.16188594698905945f, -0.07485868036746979f, -0.08851054310798645f,
                        -0.10284193605184555f, -0.17014220356941223f, -0.04898572340607643f, -0.17083507776260376f,
                        -0.09170642495155334f, 0.1511256992816925f, 0.012886842712759972f, 0.09025576710700989f,
                        0.08479554951190948f, 0.0489313043653965f, 0.14465972781181335f, 0.007934254594147205f,
                        -0.15946026146411896f, 0.1804322451353073f, 0.08040717244148254f, 0.1475857049226761f,
                        0.15021422505378723f, -0.0028631272725760937f, -0.19993697106838226f, -0.03527900204062462f,
                        0.06516310572624207f, -0.015176207758486271f, 0.14682966470718384f, -0.02665453404188156f,
                        -0.18779225647449493f};
  TestConvOp(attrs, {X, W}, {X_shape, W_shape}, expected_vals, Y_shape);

  // NNAPI/CoreML EP requires weight to be an initializer
  TestConvOp(attrs, {X, W}, {X_shape, W_shape}, expected_vals, Y_shape, true);
}

TEST(ConvTest, Conv2D_Bias_1) {
  ConvOpAndTestAttributes attrs = {
      "",                           // auto_pad
      vector<int64_t>{1, 1},        // dilations
      1,                            // group
      vector<int64_t>{2, 2},        // kernel_shape
      vector<int64_t>{0, 0, 0, 0},  // pads
      vector<int64_t>{1, 1},        // strides
      {}                            // excluded EPs
  };

  vector<float> X = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f};
  vector<int64_t> X_shape = {1, 1, 3, 3};
  vector<float> W = {1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f};
  vector<int64_t> W_shape = {2, 1, 2, 2};
  vector<int64_t> Y_shape = {1, 2, 2, 2};
  vector<float> B = {1.0f, -1.0f};
  vector<int64_t> B_shape = {2};
  auto expected_vals = {13.0f, 17.0f, 25.0f, 29.0f, 11.0f, 15.0f, 23.0f, 27.0f};

  TestConvOp(attrs, {X, W, B}, {X_shape, W_shape, B_shape}, expected_vals, Y_shape);

  // NNAPI/CoreML EP requires weight to be an initializer
  TestConvOp(attrs, {X, W, B}, {X_shape, W_shape, B_shape}, expected_vals, Y_shape, true);
}

// Conv48
TEST(ConvTest, Conv2D_Bias_2) {
  ConvOpAndTestAttributes attrs = {
      "",                           // auto_pad
      vector<int64_t>{1, 1},        // dilations
      1,                            // group
      vector<int64_t>{4, 4},        // kernel_shape
      vector<int64_t>{1, 2, 3, 1},  // pads
      vector<int64_t>{2, 3},        // strides
      {}                            // excluded EPs
  };

  vector<float> X = {-0.22904816269874573f, -0.20278319716453552f, -0.4723144471645355f, 0.027880489826202393f,
                     0.2685856819152832f, -0.19361668825149536f, -0.39857280254364014f, 0.40285515785217285f,
                     0.20966708660125732f, -0.39234158396720886f, -0.07502302527427673f, 0.4662899374961853f,
                     -0.2567148208618164f, -0.1186269223690033f, -0.1897754967212677f, -0.3967694342136383f,
                     -0.4268943667411804f, -0.344584584236145f, -0.4483465552330017f, -0.41608482599258423f,
                     -0.23649904131889343f, -0.4195239543914795f, 0.3277903199195862f, -0.11628741025924683f,
                     0.2873995900154114f, 0.21717703342437744f, -0.26514798402786255f, 0.08272713422775269f,
                     0.0050997138023376465f, -0.41409194469451904f, 0.2826550006866455f, 0.4891064763069153f,
                     -0.1522480845451355f, -0.2554396986961365f, 0.04099029302597046f, -0.35793858766555786f,
                     0.2557554841041565f, 0.41162675619125366f, -0.06953108310699463f, 0.029517710208892822f,
                     0.32956594228744507f, 0.4615175127983093f, -0.3216847777366638f, 0.15545696020126343f,
                     -0.3779126703739166f, -0.01712372899055481f, 0.07461833953857422f, 0.38875824213027954f,
                     0.1980893611907959f, -0.19913813471794128f, -0.011296629905700684f, 0.30053526163101196f,
                     0.4461088180541992f, 0.025034189224243164f, -0.3370230793952942f, -0.21012544631958008f,
                     -0.41627752780914307f, -0.43801137804985046f, 0.13566172122955322f, -0.47898364067077637f,
                     -0.45526939630508423f, -0.3007912039756775f, 0.06994932889938354f, -0.0749855637550354f,
                     -0.22754916548728943f, -0.469131737947464f, 0.08644282817840576f, 0.06157493591308594f,
                     -0.3920745849609375f, 0.458797812461853f, 0.18890488147735596f, 0.40145808458328247f};
  vector<int64_t> X_shape = {1, 2, 6, 6};
  vector<float> W = {-0.48007914423942566f, -0.21048793196678162f, 0.2505034804344177f, 0.1610567569732666f,
                     -0.24951639771461487f, 0.1918455958366394f, 0.44247758388519287f, 0.06943017244338989f,
                     -0.10510382056236267f, -0.41663575172424316f, -0.3053555488586426f, -0.19126328825950623f,
                     -0.42332321405410767f, 0.498790979385376f, 0.081226646900177f, -0.21777048707008362f,
                     0.46603143215179443f, -0.43488776683807373f, -0.3080252408981323f, -0.3844330906867981f,
                     -0.17214277386665344f, -0.3650006353855133f, 0.21724021434783936f, 0.1636529564857483f,
                     -0.22924479842185974f, 0.044009625911712646f, 0.274614155292511f, -0.06811442971229553f,
                     0.450619637966156f, 0.4611729383468628f, 0.20782196521759033f, -0.3136714696884155f};
  vector<int64_t> W_shape = {1, 2, 4, 4};
  vector<float> B = {-0.40378910303115845f};
  vector<int64_t> B_shape = {1};
  vector<int64_t> Y_shape = {1, 1, 4, 2};
  auto expected_vals = {-0.3419531583786011f, -0.6116723418235779f, -0.39677709341049194f, -0.7316848039627075f,
                        -0.5647197365760803f, 0.02788025140762329f, -0.30450713634490967f, -0.6786775588989258f};

  TestConvOp(attrs, {X, W, B}, {X_shape, W_shape, B_shape}, expected_vals, Y_shape);

  TestConvOp(attrs, {X, W, B}, {X_shape, W_shape, B_shape}, expected_vals, Y_shape, true);
}

TEST(ConvTest, Conv2D_AutoPad1) {
  ConvOpAndTestAttributes attrs = {
      "SAME_UPPER",           // auto_pad
      vector<int64_t>{1, 1},  // dilations
      1,                      // group
      vector<int64_t>{3, 3},  // kernel_shape
      {},                     // pads
      vector<int64_t>{1, 1},  // strides
      {}                      // excluded EPs
  };

  vector<float> X = vector<float>(25, 1.0f);
  vector<int64_t> X_shape = {1, 1, 5, 5};
  vector<float> W = {0.0f, 1.0f, 2.0f,
                     3.0f, 4.0f, 5.0f,
                     6.0f, 7.0f, 8.0f};

  vector<int64_t> W_shape = {1, 1, 3, 3};
  vector<int64_t> Y_shape = {1, 1, 5, 5};
  auto expected_vals = {24.0f, 33.0f, 33.0f, 33.0f, 20.0f,
                        27.0f, 36.0f, 36.0f, 36.0f, 21.0f,
                        27.0f, 36.0f, 36.0f, 36.0f, 21.0f,
                        27.0f, 36.0f, 36.0f, 36.0f, 21.0f,
                        12.0f, 15.0f, 15.0f, 15.0f, 8.0f};
  TestConvOp(attrs, {X, W}, {X_shape, W_shape}, expected_vals, Y_shape);

  // NNAPI/CoreML EP requires weight to be an initializer
  TestConvOp(attrs, {X, W}, {X_shape, W_shape}, expected_vals, Y_shape, true);
}

TEST(ConvTest, Conv2D_AutoPad2) {
  ConvOpAndTestAttributes attrs = {
      "SAME_LOWER",           // auto_pad
      vector<int64_t>{1, 1},  // dilations
      1,                      // group
      vector<int64_t>{3, 3},  // kernel_shape
      {},                     // pads
      vector<int64_t>{1, 1},  // strides
      {}                      // excluded EPs
  };

  vector<float> X = {1.0f, 0.0f, 1.0f, 0.0f, 1.0f,
                     1.0f, 0.0f, 1.0f, 0.0f, 1.0f,
                     1.0f, 0.0f, 1.0f, 0.0f, 1.0f,
                     1.0f, 0.0f, 1.0f, 0.0f, 1.0f,
                     1.0f, 0.0f, 1.0f, 0.0f, 1.0f};
  vector<int64_t> X_shape = {1, 1, 5, 5};
  vector<float> W = {0.0f, 1.0f, 2.0f,
                     3.0f, 4.0f, 5.0f,
                     6.0f, 7.0f, 8.0f};

  vector<int64_t> W_shape = {1, 1, 3, 3};
  vector<int64_t> Y_shape = {1, 1, 5, 5};
  auto expected_vals = {11.0f, 22.0f, 11.0f, 22.0f, 11.0f,
                        12.0f, 24.0f, 12.0f, 24.0f, 12.0f,
                        12.0f, 24.0f, 12.0f, 24.0f, 12.0f,
                        12.0f, 24.0f, 12.0f, 24.0f, 12.0f,
                        5.0f, 10.0f, 5.0f, 10.0f, 5.0f};
  TestConvOp(attrs, {X, W}, {X_shape, W_shape}, expected_vals, Y_shape);

  // NNAPI/CoreML EP requires weight to be an initializer
  TestConvOp(attrs, {X, W}, {X_shape, W_shape}, expected_vals, Y_shape, true);
}

// Conv10
TEST(ConvTest, Conv3D_1) {
  ConvOpAndTestAttributes attrs = {
      "",                                 // auto_pad
      vector<int64_t>{1, 1, 1},           // dilations
      1,                                  // group
      vector<int64_t>{1, 1, 1},           // kernel_shape
      vector<int64_t>{0, 0, 0, 0, 0, 0},  // pads
      vector<int64_t>{1, 1, 1},           // strides
      {}                                  // excluded EPs
  };

  vector<float> X = {-0.43337246775627136f, -0.48385289311408997f, -0.30954962968826294f,
                     0.16074687242507935f, -0.46670910716056824f, 0.46576786041259766f,
                     -0.37056273221969604f, 0.40604978799819946f, -0.035478413105010986f,
                     -0.3125576674938202f, 0.42677170038223267f, 0.39851123094558716f,
                     -0.3906140625476837f, 0.2590462565422058f, -0.20646807551383972f,
                     0.1382436752319336f, -0.20149192214012146f, 0.10030072927474976f,
                     -0.2413364052772522f, 0.1231224536895752f, 0.032734215259552f,
                     0.29610633850097656f, -0.23117440938949585f, 0.3345826268196106f,
                     0.02567422389984131f, 0.24579226970672607f, 0.11724984645843506f};
  vector<int64_t> X_shape = {1, 1, 3, 3, 3};
  vector<float> W = {-0.44214117527008057f};
  vector<int64_t> W_shape = {1, 1, 1, 1, 1};
  vector<int64_t> Y_shape = {1, 1, 3, 3, 3};
  auto expected_vals = {0.19161181151866913f, 0.21393129229545593f, 0.13686463236808777f,
                        -0.07107280939817429f, 0.20635131001472473f, -0.20593515038490295f,
                        0.16384103894233704f, -0.17953133583068848f, 0.01568646728992462f,
                        0.13819462060928345f, -0.1886933445930481f, -0.17619822919368744f,
                        0.17270655930042267f, -0.11453501880168915f, 0.09128803759813309f,
                        -0.06112322211265564f, 0.08908787369728088f, -0.04434708133339882f,
                        0.10670476406812668f, -0.054437506943941116f, -0.014473143965005875f,
                        -0.13092079758644104f, 0.10221172869205475f, -0.1479327529668808f,
                        -0.011351631954312325f, -0.10867488384246826f, -0.05184098333120346f};
  TestConvOp(attrs, {X, W}, {X_shape, W_shape}, expected_vals, Y_shape);
}

// Conv22
TEST(ConvTest, Conv3D_2) {
  ConvOpAndTestAttributes attrs = {
      "",                                 // auto_pad
      vector<int64_t>{1, 1, 1},           // dilations
      1,                                  // group
      vector<int64_t>{1, 1, 1},           // kernel_shape
      vector<int64_t>{2, 2, 2, 2, 2, 2},  // pads
      vector<int64_t>{2, 2, 2},           // strides
      {}                                  // excluded EPs
  };

  vector<float> X = {0.010772407054901123f, -0.43806642293930054f, 0.455391526222229f, -0.28657248616218567f,
                     0.45676887035369873f, -0.0320507287979126f, 0.4229400157928467f, -0.18730869889259338f,
                     -0.45851585268974304f, 0.042054951190948486f, -0.13332295417785645f, -0.25374430418014526f,
                     -0.23845627903938293f, 0.12214112281799316f, -0.1778157651424408f, 0.1891845464706421f,
                     0.37962496280670166f, -0.033982306718826294f, 0.12737131118774414f, -0.040284961462020874f,
                     0.46427029371261597f, -0.22687292098999023f, 0.17398333549499512f, -0.3014046251773834f,
                     -0.4043419063091278f, -0.33206477761268616f, 0.04655301570892334f, -0.4947906732559204f,
                     0.0755157470703125f, 0.1173025369644165f, 0.47043120861053467f, 0.4824737310409546f,
                     -0.37734976410865784f, -0.056491583585739136f, -0.10790631175041199f, 0.043476223945617676f,
                     0.24469023942947388f, -0.4100031852722168f, 0.0616222620010376f, 0.2296960949897766f,
                     0.27883386611938477f, 0.08150351047515869f, 0.2453773021697998f, 0.08250969648361206f,
                     -0.1471814215183258f, -0.43011274933815f, 0.027180075645446777f, 0.3605625033378601f,
                     0.24954384565353394f, -0.22505927085876465f, -0.36272895336151123f, -0.47674262523651123f,
                     0.11275297403335571f, 0.49773406982421875f, 0.2686365246772766f, 0.025525271892547607f,
                     -0.3037869930267334f, 0.41126757860183716f, 0.36149072647094727f, 0.00883406400680542f,
                     -0.07959523797035217f, 0.3601323366165161f, 0.17322391271591187f, -0.012007325887680054f};
  vector<int64_t> X_shape = {1, 1, 4, 4, 4};
  vector<float> W = {0.32824617624282837f};
  vector<int64_t> W_shape = {1, 1, 1, 1, 1};
  vector<int64_t> Y_shape = {1, 1, 4, 4, 4};
  auto expected_vals = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
                        0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
                        0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0035360013134777546f, 0.14948052167892456f, 0.0f,
                        0.0f, -0.15050607919692993f, -0.043762750923633575f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
                        0.0f, 0.0f, 0.0f, 0.0f, 0.0f, -0.12386361509561539f, -0.03541983291506767f, 0.0f,
                        0.0f, 0.09152615070343018f, 0.08054415881633759f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
                        0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
                        0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
  TestConvOp(attrs, {X, W}, {X_shape, W_shape}, expected_vals, Y_shape);
}

// Conv23
TEST(ConvTest, Conv3D_Bias) {
  ConvOpAndTestAttributes attrs = {
      "",                                 // auto_pad
      vector<int64_t>{2, 2, 2},           // dilations
      1,                                  // group
      vector<int64_t>{2, 2, 2},           // kernel_shape
      vector<int64_t>{2, 2, 2, 2, 2, 2},  // pads
      vector<int64_t>{2, 2, 2},           // strides
      {}                                  // excluded EPs
  };

  vector<float> X = {0.46796226501464844f, -0.4613912105560303f, 0.33512794971466064f, -0.4010460674762726f,
                     0.41722816228866577f, -0.048133403062820435f, 0.20415884256362915f, 0.03189706802368164f,
                     -0.04779183864593506f, -0.0795503556728363f, 0.4987630844116211f, 0.3506373167037964f,
                     0.48065757751464844f, 0.269855260848999f, -0.2463444471359253f, 0.19044137001037598f,
                     -0.11830493807792664f, -0.2576887905597687f, -0.33940935134887695f, -0.257951021194458f,
                     -0.08279827237129211f, 0.3513314127922058f, -0.29122066497802734f, -0.43358397483825684f,
                     -0.13429927825927734f, 0.44032156467437744f, 0.05308258533477783f, -0.3499870300292969f,
                     -0.28474611043930054f, -0.44209951162338257f, -0.07418054342269897f, -0.10919415950775146f,
                     0.2845439314842224f, 0.3498746156692505f, -0.19313520193099976f, 0.32609254121780396f,
                     0.4880145788192749f, 0.05574071407318115f, -0.46457427740097046f, -0.02524462342262268f,
                     -0.18780940771102905f, -0.14720159769058228f, 0.207585871219635f, 0.47157740592956543f,
                     -0.05567386746406555f, -0.49871665239334106f, 0.2274145483970642f, 0.4589425325393677f,
                     -0.4725189805030823f, -0.4358765780925751f, 0.2841453552246094f, -0.27037882804870605f,
                     0.34227508306503296f, 0.33575427532196045f, -0.19485199451446533f, -0.27679920196533203f,
                     -0.4238079786300659f, -0.4385119676589966f, 0.43724071979522705f, 0.3065117597579956f,
                     0.45696544647216797f, 0.05291992425918579f, -0.023618370294570923f, -0.1860884726047516f,
                     0.08669537305831909f, 0.32541000843048096f, 0.1846179962158203f, -0.1984834372997284f,
                     -0.2754465937614441f, 0.32004624605178833f, -0.34846532344818115f, 0.0999596118927002f,
                     -0.11374691128730774f, 0.21225297451019287f, -0.02315312623977661f, 0.1671370267868042f,
                     0.22319108247756958f, 0.03609824180603027f, -0.1587022840976715f, 0.059984564781188965f,
                     -0.03951650857925415f, -0.4841443598270416f, 0.32919085025787354f, -0.23115816712379456f,
                     0.39441078901290894f, -0.3554944396018982f, -0.17022761702537537f, -0.055081307888031006f,
                     0.15856128931045532f, -0.4183449149131775f, -0.2474445104598999f, 0.03603637218475342f,
                     -0.2836887538433075f, 0.4602506160736084f, 0.29092925786972046f, -0.199321448802948f,
                     0.380856454372406f, -0.13847029209136963f, -0.238397479057312f, -0.1907123327255249f,
                     -0.11061936616897583f, -0.08717870712280273f, 0.24449139833450317f, -0.14727482199668884f,
                     0.1437196135520935f, 0.3955056071281433f, -0.12538021802902222f, 0.11590522527694702f,
                     0.4598066806793213f, -0.30005723237991333f, -0.46578651666641235f, -0.33955082297325134f,
                     -0.2671887278556824f, 0.3611910939216614f, -0.11423084139823914f, -0.08382436633110046f,
                     -0.31819307804107666f, 0.14515334367752075f, 0.3157258629798889f, 0.33179205656051636f,
                     -0.2558857202529907f, 0.11888682842254639f, 0.12824326753616333f, -0.33106181025505066f,
                     0.2549159526824951f, -0.46760573983192444f, -0.11983257532119751f, 0.1834418773651123f};
  vector<int64_t> X_shape = {2, 1, 4, 4, 4};
  vector<float> W = {0.388077974319458f, -0.16366064548492432f, -0.42871910333633423f, 0.4276432394981384f,
                     0.21517693996429443f, 0.007908165454864502f, 0.33897721767425537f, 0.21843165159225464f,
                     0.34095364809036255f, -0.17043980956077576f, -0.013571739196777344f, -0.26793742179870605f,
                     -0.34863436222076416f, -0.2672275900840759f, -0.36691007018089294f, 0.37296557426452637f};
  vector<int64_t> W_shape = {2, 1, 2, 2, 2};
  vector<float> B = {0.4310183525085449f, -0.4564093053340912f};
  vector<int64_t> B_shape = {2};
  vector<int64_t> Y_shape = {2, 2, 3, 3, 3};

  auto expected_vals = {0.5332361459732056f, 0.6628494262695312f, 0.544619083404541f, 0.4242798388004303f,
                        0.6271085739135742f, 0.6721994876861572f, 0.43064039945602417f, 0.4246789515018463f,
                        0.53834068775177f, 0.6932926177978516f, 0.42797625064849854f, 0.2218741625547409f,
                        0.29522019624710083f, 0.8329390287399292f, 0.37605351209640503f, 0.43735477328300476f,
                        0.2920728623867035f, 0.6692450046539307f, 0.5527016520500183f, 0.22643595933914185f,
                        0.5138190984725952f, 0.3041342794895172f, 0.7423423528671265f, 0.26707080006599426f,
                        0.4617553651332855f, 0.32416003942489624f, 0.511577844619751f, -0.28187549114227295f,
                        -0.5031181573867798f, -0.5793710947036743f, -0.5992864370346069f, -0.5055556893348694f,
                        -0.7562476396560669f, -0.44363799691200256f, -0.5730307102203369f, -0.6302952766418457f,
                        -0.4756688177585602f, -0.728988528251648f, -0.3900943398475647f, -0.6694478988647461f,
                        -0.38822290301322937f, -0.35774707794189453f, -0.39807581901550293f, -0.547709047794342f,
                        -0.35872578620910645f, -0.5326492786407471f, -0.40852290391921997f, -0.4537881314754486f,
                        -0.4545857608318329f, -0.379546195268631f, -0.5250767469406128f, -0.42439910769462585f,
                        -0.5558245182037354f, -0.38563215732574463f, 0.44995537400245667f, 0.5007325410842896f,
                        0.49359965324401855f, 0.40685802698135376f, 0.407518208026886f, 0.4628955125808716f,
                        0.4301188290119171f, 0.40635955333709717f, 0.4260363280773163f, 0.55128413438797f,
                        0.5498291254043579f, 0.27105778455734253f, 0.40259143710136414f, 0.5747092962265015f,
                        0.4187920391559601f, 0.4507707953453064f, 0.420598566532135f, 0.3950541913509369f,
                        0.593889057636261f, 0.16578882932662964f, 0.5332239270210266f, 0.43014785647392273f,
                        0.50260329246521f, 0.39225444197654724f, 0.4074971079826355f, 0.5073125958442688f,
                        0.3823610544204712f, -0.4240749180316925f, -0.41936254501342773f, -0.5241475105285645f,
                        -0.5220003724098206f, -0.502869725227356f, -0.5122783780097961f, -0.4260129928588867f,
                        -0.4105660617351532f, -0.4483373165130615f, -0.33759188652038574f, -0.735706090927124f,
                        -0.3714444637298584f, -0.4888814687728882f, -0.6191370487213135f, -0.2640320658683777f,
                        -0.47542816400527954f, -0.5078460574150085f, -0.4205915927886963f, -0.5584549903869629f,
                        -0.39770257472991943f, -0.45317384600639343f, -0.5598302483558655f, -0.2542789578437805f,
                        -0.5359901785850525f, -0.48090484738349915f, -0.38603779673576355f, -0.4991581439971924f};
  TestConvOp(attrs, {X, W, B}, {X_shape, W_shape, B_shape}, expected_vals, Y_shape);
}

TEST(ConvTest, Conv2D_group) {
  ConvOpAndTestAttributes attrs = {
      "",                           // auto_pad
      vector<int64_t>{1, 1},        // dilations
      2,                            // group
      vector<int64_t>{1, 1},        // kernel_shape
      vector<int64_t>{0, 0, 0, 0},  // pads
      vector<int64_t>{1, 1},        // strides
      {}                            // excluded EPs
  };

  vector<float> X = {0.0f, 1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f, 10.0f, 11.0f, 12.0f, 13.0f, 14.0f, 15.0f, 16.0f, 17.0f};
  vector<int64_t> X_shape = {1, 2, 3, 3};
  vector<float> W = {1.0f, 2.0f};
  vector<int64_t> W_shape = {2, 1, 1, 1};
  vector<int64_t> Y_shape = {1, 2, 3, 3};
  auto expected_vals = {0.0f, 1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 18.0f, 20.0f, 22.0f, 24.0f, 26.0f, 28.0f, 30.0f, 32.0f, 34.0f};

  TestConvOp(attrs, {X, W}, {X_shape, W_shape}, expected_vals, Y_shape);

  // NNAPI/CoreML EP requires weight to be an initializer
  TestConvOp(attrs, {X, W}, {X_shape, W_shape}, expected_vals, Y_shape, true);
}

TEST(ConvTest, ConvDimWithZero) {
  ConvOpAndTestAttributes attrs = {
      "",                           // auto_pad
      vector<int64_t>{1, 1},        // dilations
      1,                            // group
      vector<int64_t>{1, 1},        // kernel_shape
      vector<int64_t>{0, 0, 0, 0},  // pads
      vector<int64_t>{1, 1},        // strides
      {}                            // excluded EPs
  };

  vector<float> X = vector<float>();
  vector<int64_t> X_shape = {0, 2, 4, 4};  // N of 0 should be handled
  vector<float> W = {1.0f, 2.0f, 1.0f, 2.0f};
  vector<int64_t> W_shape = {2, 2, 1, 1};
  vector<int64_t> out_shape = {0, 2, 4, 4};

  // not handled by ACL
  attrs.excluded_providers.insert(kAclExecutionProvider);

  TestConvOp(attrs, {X, W}, {X_shape, W_shape}, {}, out_shape, false, OpTester::ExpectResult::kExpectSuccess, "", 10);
}

TEST(ConvTest, Conv1D_asymmetric_padding) {
  ConvOpAndTestAttributes attrs = {
      "",                     // auto_pad
      vector<int64_t>{1},     // dilations
      1,                      // group
      vector<int64_t>{3},     // kernel_shape
      vector<int64_t>{1, 0},  // pads
      vector<int64_t>{1},     // strides
      {}                      // excluded EPs
  };

  vector<float> X = {1.f, 2.f, 3.f};
  vector<int64_t> X_shape = {1, 1, 3};
  vector<float> W = {1.f, 1.f, 1.f};
  vector<int64_t> W_shape = {1, 1, 3};
  vector<float> B = {0.f};
  vector<int64_t> B_shape = {1};
  vector<int64_t> Y_shape = {1, 1, 2};
  auto expected_vals = {3.f, 6.f};

  TestConvOp(attrs, {X, W, B}, {X_shape, W_shape, B_shape}, expected_vals, Y_shape);

  TestConvOp(attrs, {X, W, B}, {X_shape, W_shape, B_shape}, expected_vals, Y_shape, true);
}

TEST(ConvTest, Conv_AutoPad_with_non_default_strides) {
  ConvOpAndTestAttributes attrs = {
      "SAME_LOWER",           // auto_pad
      vector<int64_t>{1, 1},  // dilations
      1,                      // group
      vector<int64_t>{3, 3},  // kernel_shape
      vector<int64_t>{},      // pads
      vector<int64_t>{2, 2},  // strides
      {}                      // excluded EPs
  };

  vector<float> X = {0.0f, 1.0f, 2.0f, 3.0f, 4.0f,
                     5.0f, 6.0f, 7.0f, 8.0f, 9.0f,
                     10.0f, 11.0f, 12.0f, 13.0f, 14.0f,
                     15.0f, 16.0f, 17.0f, 18.0f,
                     19.0f, 20.0f, 21.0, 22.0f, 23.0f, 24.0f};
  vector<int64_t> X_shape = {1, 1, 5, 5};

  vector<float> W = {1.0f, 1.0f, 1.0f,
                     1.0f, 1.0f, 1.0f,
                     1.0f, 1.0f, 1.0f};
  vector<int64_t> W_shape = {1, 1, 3, 3};

  auto expected_vals = {12.0f, 27.0f, 24.0f,
                        63.0f, 108.0f, 81.0f,
                        72.0f, 117.0f, 84.0f};
  vector<int64_t> Y_shape = {1, 1, 3, 3};

  // Test with weight as initializer
  TestConvOp(attrs, {X, W}, {X_shape, W_shape}, expected_vals, Y_shape);

  // Test with weight as initializer
  TestConvOp(attrs, {X, W}, {X_shape, W_shape}, expected_vals, Y_shape, true);
}

}  // namespace test
}  // namespace onnxruntime
