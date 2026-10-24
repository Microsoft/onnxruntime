// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/framework/kernel_type_str_resolver_utils.h"

#include <iostream>
#include <sstream>
#include <iomanip>
#include "gtest/gtest.h"

#include "core/flatbuffers/schema/ort.fbs.h"
#include "core/graph/constants.h"
#include "core/graph/model.h"
#include "core/graph/schema_registry.h"
#include "core/optimizer/layout_transformation/layout_transformation_potentially_added_ops.h"
#include "core/session/onnxruntime_session_options_config_keys.h"
#include "test/test_environment.h"
#include "test/util/include/asserts.h"
#include "test/util/include/inference_session_wrapper.h"

#include <filesystem>
#include <fstream>

namespace onnxruntime::test {

static Status LoadLayoutTransformationRequiredOpsFromOpSchemas(KernelTypeStrResolver& kernel_type_str_resolver) {
  const auto schema_registry = SchemaRegistryManager{};
  for (const auto& op_id : kLayoutTransformationPotentiallyAddedOps) {
    const auto* op_schema = schema_registry.GetSchema(std::string{op_id.op_type}, op_id.since_version,
                                                      std::string{op_id.domain});
    ORT_RETURN_IF(op_schema == nullptr,
                  "Failed to get op schema for domain='", op_id.domain,
                  "', op_type='", op_id.op_type,
                  "', since_version=", op_id.since_version, ".");
    ORT_RETURN_IF_ERROR(kernel_type_str_resolver.RegisterOpSchema(*op_schema));
  }
  return Status::OK();
}

TEST(KernelTypeStrResolverUtilsTest, VerifyLayoutTransformationRequiredOpsResolver) {
  KernelTypeStrResolver expected_resolver;
  ASSERT_STATUS_OK(LoadLayoutTransformationRequiredOpsFromOpSchemas(expected_resolver));

  KernelTypeStrResolver actual_resolver;
  ASSERT_STATUS_OK(
      kernel_type_str_resolver_utils::AddLayoutTransformationRequiredOpsToKernelTypeStrResolver(actual_resolver));

  for (const auto* op_type : {"QuantizeLinear", "DequantizeLinear"}) {
    const OpIdentifier op_id{kOnnxDomain, op_type, 28};
    ASSERT_TRUE(expected_resolver.GetOpKernelTypeStrMap().contains(op_id));
    ASSERT_TRUE(actual_resolver.GetOpKernelTypeStrMap().contains(op_id));
  }

#if !defined(DISABLE_CONTRIB_OPS)
  ASSERT_EQ(actual_resolver.GetOpKernelTypeStrMap(), expected_resolver.GetOpKernelTypeStrMap());
#else   // !defined(DISABLE_CONTRIB_OPS)
  // check that each element of expected_resolver is present and equivalent in actual_resolver
  const auto& expected_op_kernel_type_str_map = expected_resolver.GetOpKernelTypeStrMap();
  const auto& actual_op_kernel_type_str_map = actual_resolver.GetOpKernelTypeStrMap();

  for (const auto& [expected_op_id, expected_kernel_type_str_map] : expected_op_kernel_type_str_map) {
    const auto actual_op_kernel_type_str_map_it = actual_op_kernel_type_str_map.find(expected_op_id);
    ASSERT_NE(actual_op_kernel_type_str_map_it, actual_op_kernel_type_str_map.end());
    ASSERT_EQ(actual_op_kernel_type_str_map_it->second, expected_kernel_type_str_map);
  }
#endif  // !defined(DISABLE_CONTRIB_OPS)
}

#if !defined(ORT_MINIMAL_BUILD) && !defined(DISABLE_CONTRIB_OPS)
TEST(KernelTypeStrResolverUtilsTest, ResolveNhwcFusedConvFromLayoutTransformationRequiredOps) {
  KernelTypeStrResolver resolver;
  ASSERT_STATUS_OK(kernel_type_str_resolver_utils::AddLayoutTransformationRequiredOpsToKernelTypeStrResolver(resolver));

  Model model("nhwc_fused_conv_layout_transform_resolver_test", false, DefaultLoggingManager().DefaultLogger());
  auto& graph = model.MainGraph();

  ONNX_NAMESPACE::TypeProto float_tensor;
  auto* tensor_type = float_tensor.mutable_tensor_type();
  tensor_type->set_elem_type(ONNX_NAMESPACE::TensorProto_DataType_FLOAT);
  tensor_type->mutable_shape()->add_dim()->set_dim_value(1);

  auto& x = graph.GetOrCreateNodeArg("x", &float_tensor);
  auto& w = graph.GetOrCreateNodeArg("w", &float_tensor);
  auto& y = graph.GetOrCreateNodeArg("y", &float_tensor);

  auto& nhwc_fused_conv = graph.AddNode(
      "nhwc_fused_conv", "NhwcFusedConv", "test node", {&x, &w}, {&y}, nullptr, kMSDomain);
  nhwc_fused_conv.SetSinceVersion(1);

  gsl::span<const ArgTypeAndIndex> resolved_args;
  ASSERT_STATUS_OK(resolver.ResolveKernelTypeStr(nhwc_fused_conv, "T", resolved_args));
  ASSERT_FALSE(resolved_args.empty());
}

#endif  // !defined(ORT_MINIMAL_BUILD) && !defined(DISABLE_CONTRIB_OPS)

// run this test manually to output a hard-coded byte array.
// update AddLayoutTransformationRequiredOpsToKernelTypeStrResolver in
// onnxruntime/core/framework/kernel_type_str_resolver_utils.cc
TEST(KernelTypeStrResolverUtilsTest, DISABLED_PrintExpectedLayoutTransformationRequiredOpsResolverByteArray) {
#if defined(DISABLE_CONTRIB_OPS)
  FAIL() << "Contrib ops must be enabled.";
#else   // defined(DISABLE_CONTRIB_OPS)
  KernelTypeStrResolver expected_resolver;
  ASSERT_STATUS_OK(LoadLayoutTransformationRequiredOpsFromOpSchemas(expected_resolver));

  flatbuffers::DetachedBuffer buffer;
  gsl::span<const uint8_t> buffer_span;
  ASSERT_STATUS_OK(kernel_type_str_resolver_utils::SaveKernelTypeStrResolverToBuffer(expected_resolver,
                                                                                     buffer, buffer_span));

  constexpr size_t kBytesPerLine = 16;
  std::ostringstream os;
  os << std::hex << std::setfill('0')
     << "  constexpr uint8_t kLayoutTransformationRequiredOpsKernelTypeStrResolverBytes[] = {\n      ";
  for (size_t i = 0; i < buffer_span.size(); ++i) {
    os << "0x" << std::setw(2) << static_cast<int32_t>(buffer_span[i]) << ",";
    if (i < buffer_span.size() - 1) {
      os << ((i % kBytesPerLine == kBytesPerLine - 1) ? "\n      " : " ");
    }
  }
  os << "\n  };\n";

  std::cout << os.str();
#endif  // defined(DISABLE_CONTRIB_OPS)
}

}  // namespace onnxruntime::test
