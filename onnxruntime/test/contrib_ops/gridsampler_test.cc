// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "gtest/gtest.h"
#include "test/providers/provider_test_utils.h"
#include "core/util/math.h"

namespace onnxruntime {
namespace test {

TEST(GridsamplerContribOpTest, gridsampler_default) {
  OpTester test("GridSampler", 1, kMSDomain);
  test.AddInput<float>("X", {1, 1, 4, 4}, {0.0f, 1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f, 10.0f, 11.0f, 12.0f, 13.0f, 14.0f, 15.0f});
  test.AddInput<float>("Grid", {1, 6, 6, 2},
    {-1.0000f, -1.0000f, -0.6000f, -1.0000f, -0.2000f, -1.0000f,  0.2000f, -1.0000f,
     0.6000f, -1.0000f,  1.0000f, -1.0000f, -1.0000f, -0.6000f, -0.6000f, -0.6000f,
     -0.2000f, -0.6000f,  0.2000f, -0.6000f,  0.6000f, -0.6000f,  1.0000f, -0.6000f,
     -1.0000f, -0.2000f, -0.6000f, -0.2000f, -0.2000f, -0.2000f,  0.2000f, -0.2000f,
     0.6000f, -0.2000f,  1.0000f, -0.2000f, -1.0000f,  0.2000f, -0.6000f,  0.2000f,
     -0.2000f,  0.2000f,  0.2000f,  0.2000f,  0.6000f,  0.2000f,  1.0000f,  0.2000f,
     -1.0000f,  0.6000f, -0.6000f,  0.6000f, -0.2000f,  0.6000f,  0.2000f,  0.6000f,
     0.6000f,  0.6000f,  1.0000f,  0.6000f, -1.0000f,  1.0000f, -0.6000f,  1.0000f,
     -0.2000f,  1.0000f,  0.2000f,  1.0000f,  0.6000f,  1.0000f,  1.0000f,  1.0000f
    });
  int64_t align_corners = 0;
  test.AddAttribute("mode", "bilinear");
  test.AddAttribute("padding_mode", "zeros");
  test.AddAttribute("align_corners", align_corners);
  test.AddOutput<float>("Y", {1, 1, 6, 6},
    {0.0000f, 0.1500f, 0.5500f, 0.9500f, 1.3500f, 0.7500f,
     0.6000f, 1.5000f, 2.3000f, 3.1000f, 3.9000f, 2.1000f,
     2.2000f, 4.7000f, 5.5000f, 6.3000f, 7.1000f, 3.7000f,
     3.8000f, 7.9000f, 8.7000f, 9.5000f, 10.3000f, 5.3000f,
     5.4000f, 11.1000f, 11.9000f, 12.7000f, 13.5000f, 6.9000f,
     3.0000f, 6.1500f, 6.5500f, 6.9500f, 7.3500f, 3.7500f
    });
  test.Run();
}

TEST(GridsamplerContribOpTest, gridsampler_paddingmode_zeros) {
  OpTester test("GridSampler", 1, kMSDomain);
  test.AddInput<float>("X", {1, 1, 3, 2}, {0.0f, 1.0f, 2.0f, 3.0f, 4.0f, 5.0f});
  test.AddInput<float>("Grid", {1, 2, 4, 2},
    {-10.0000f, -10.0000f, -5.0000f, -5.0000f,
     -0.2000f, -0.2000f, 10.0000f, 10.0000f,
     10.0000f, 10.0000f, -0.2000f, -0.2000f,
     5.0000f, 5.0000f, 10.0000f, 10.0000f
    });
  test.AddAttribute("padding_mode", "zeros");
  test.AddOutput<float>("Y", {1, 1, 2, 4}, {0.0000f, 0.0000f, 1.7000f, 0.0000f, 0.0000f, 1.7000f, 0.0000f, 0.0000f});
  test.Run();
}

TEST(GridsamplerContribOpTest, gridsampler_paddingmode_border) {
  OpTester test("GridSampler", 1, kMSDomain);
  test.AddInput<float>("X", {1, 1, 3, 2}, {0.0f, 1.0f, 2.0f, 3.0f, 4.0f, 5.0f});
  test.AddInput<float>("Grid", {1, 2, 4, 2},
    {-10.0000f, -10.0000f, -5.0000f, -5.0000f,
     -0.2000f, -0.2000f, 10.0000f, 10.0000f,
     10.0000f, 10.0000f, -0.2000f, -0.2000f,
     5.0000f, 5.0000f, 10.0000f, 10.0000f
    });
  test.AddAttribute("padding_mode", "border");
  test.AddOutput<float>("Y", {1, 1, 2, 4}, {5.0000f, 0.0000f, 1.7000f, 5.0000f, 5.0000f, 1.7000f, 5.0000f, 5.0000f});
  test.Run();
}

TEST(GridsamplerContribOpTest, gridsampler_paddingmode_reflection) {
  OpTester test("GridSampler", 1, kMSDomain);
  test.AddInput<float>("X", {1, 1, 3, 2}, {0.0f, 1.0f, 2.0f, 3.0f, 4.0f, 5.0f});
  test.AddInput<float>("Grid", {1, 2, 4, 2},
    {-10.0000f, -10.0000f, -5.0000f, -5.0000f,
     -0.2000f, -0.2000f, 10.0000f, 10.0000f,
     10.0000f, 10.0000f, -0.2000f, -0.2000f,
     5.0000f, 5.0000f, 10.0000f, 10.0000f
    });
  test.AddAttribute("padding_mode", "reflection");
  test.AddOutput<float>("Y", {1, 1, 2, 4}, {2.5000f, 0.0000f, 1.7000f, 2.5000f, 2.5000f, 1.7000f, 5.0000f, 2.5000f});
  test.Run();
}

TEST(GridsamplerContribOpTest, gridsampler_aligncorners_true) {
  OpTester test("GridSampler", 1, kMSDomain);
  test.AddInput<float>("X", {1, 1, 3, 2}, {0.0f, 1.0f, 2.0f, 3.0f, 4.0f, 5.0f});
  test.AddInput<float>("Grid", {1, 2, 4, 2},
    {-1.0000f, -1.0000f, -0.5000f, -0.5000f,
     -0.2000f, -0.2000f, 0.0000f, 0.0000f,
     0.0000f, 0.0000f, -0.2000f, -0.2000f,
     0.5000f, 0.5000f, 1.0000f, 1.0000f
    });
  int64_t align_corners = 1;
  test.AddAttribute("mode", "bilinear");
  test.AddAttribute("align_corners", align_corners);
  test.AddOutput<float>("Y", {1, 1, 2, 4}, {0.0000f, 1.2500f, 2.0000f, 2.5000f, 2.5000f, 2.0000f, 3.7500f, 5.0000f});
  test.Run();
}

TEST(GridsamplerContribOpTest, gridsampler_mode_bilinear) {
  OpTester test("GridSampler", 1, kMSDomain);
  test.AddInput<float>("X", {1, 1, 3, 2}, {0.0f, 1.0f, 2.0f, 3.0f, 4.0f, 5.0f});
  test.AddInput<float>("Grid", {1, 2, 4, 2},
    {-1.0000f, -1.0000f, -0.5000f, -0.5000f,
     -0.2000f, -0.2000f, 0.0000f, 0.0000f,
     0.0000f, 0.0000f, -0.2000f, -0.2000f,
     0.5000f, 0.5000f, 1.0000f, 1.0000f
    });
  test.AddAttribute("mode", "bilinear");
  test.AddOutput<float>("Y", {1, 1, 2, 4}, {0.0000f, 0.5000f, 1.7000f, 2.5000f, 2.5000f, 1.7000f, 4.5000f, 1.2500f});
  test.Run();
}

TEST(GridsamplerContribOpTest, gridsampler_mode_nearest) {
  OpTester test("GridSampler", 1, kMSDomain);
  test.AddInput<float>("X", {1, 1, 3, 2}, {0.0f, 1.0f, 2.0f, 3.0f, 4.0f, 5.0f});
  test.AddInput<float>("Grid", {1, 2, 4, 2},
    {-1.0000f, -1.0000f, -0.5000f, -0.5000f,
     -0.2000f, -0.2000f, 0.0000f, 0.0000f,
     0.0000f, 0.0000f, -0.2000f, -0.2000f,
     0.5000f, 0.5000f, 1.0000f, 1.0000f
    });
  test.AddAttribute("mode", "nearest");
  test.AddOutput<float>("Y", {1, 1, 2, 4}, {0.f, 0.f, 2.f, 2.f, 2.f, 2.f, 5.f, 0.f});
  test.Run();
}

TEST(GridsamplerContribOpTest, gridsampler_mode_bicubic) {
  OpTester test("GridSampler", 1, kMSDomain);
  test.AddInput<float>("X", {1, 1, 3, 2}, {0.0f, 1.0f, 2.0f, 3.0f, 4.0f, 5.0f});
  test.AddInput<float>("Grid", {1, 2, 4, 2},
    {-1.0000f, -1.0000f, -0.5000f, -0.5000f,
     -0.2000f, -0.2000f, 0.0000f, 0.0000f,
     0.0000f, 0.0000f, -0.2000f, -0.2000f,
     0.5000f, 0.5000f, 1.0000f, 1.0000f
    });
  test.AddAttribute("mode", "bicubic");
  test.AddOutput<float>("Y", {1, 1, 2, 4}, {-0.1406f, 0.3828f, 1.7556f, 2.9688f, 2.9688f, 1.7556f, 5.1445f, 1.3906f});
  test.Run();
}

}  // namespace test
}  // namespace onnxruntime
