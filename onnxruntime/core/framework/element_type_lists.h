// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <cstdint>
#include <string>

#include "boost/mp11.hpp"

#include "core/common/type_list.h"
#include "core/common/float8.h"
#include "core/common/float16.h"
#include "core/framework/int4.h"
#include "core/framework/int2.h"
#include "core/framework/float4.h"
#include "core/framework/float6.h"

namespace onnxruntime {

// Contains type lists grouping various ORT element types.
// Element type refers to the element type of a Tensor, Sequence, etc.
namespace element_type_lists {

using AllFixedSizeExceptHalfIRv4 =
    TypeList<
        float,
        double,
        int64_t,
        uint64_t,
        int32_t,
        uint32_t,
        int16_t,
        uint16_t,
        int8_t,
        uint8_t,
        bool>;

using AllFixedSizeExceptHalf = AllFixedSizeExceptHalfIRv4;

using AllFixedSizeIRv4 =
    TypeList<
        float,
        double,
        int64_t,
        uint64_t,
        int32_t,
        uint32_t,
        int16_t,
        uint16_t,
        int8_t,
        uint8_t,
        MLFloat16,
        BFloat16,
        bool>;

#if !defined(DISABLE_FLOAT8_TYPES)
using AllFixedSizeIRv9 =
    boost::mp11::mp_push_back<
        AllFixedSizeIRv4,
        Float8E4M3FN,
        Float8E4M3FNUZ,
        Float8E5M2,
        Float8E5M2FNUZ>;
#else
using AllFixedSizeIRv9 = AllFixedSizeIRv4;
#endif

using AllFixedSize = AllFixedSizeIRv4;

using AllIRv4 =
    boost::mp11::mp_push_back<
        AllFixedSizeIRv4,
        std::string>;

#if !defined(DISABLE_FLOAT8_TYPES)
using AllIRv9 =
    boost::mp11::mp_push_back<
        AllIRv4,
        Float8E4M3FN,
        Float8E4M3FNUZ,
        Float8E5M2,
        Float8E5M2FNUZ>;

#else
using AllIRv9 = AllIRv4;
#endif

using AllIRv10 =
    boost::mp11::mp_push_back<
        AllIRv9,
        UInt4x2,
        Int4x2>;

#if !defined(DISABLE_FLOAT4_TYPES)
using AllIRv11 =
    boost::mp11::mp_push_back<
        AllIRv10,
        Float4E2M1x2>;
#else
using AllIRv11 = AllIRv10;
#endif

// IR v12 adds FLOAT8E8M0 (8-bit exponent-only float for MX scaling)
#if !defined(DISABLE_FLOAT8_TYPES)
using AllIRv12 =
    boost::mp11::mp_push_back<
        AllIRv11,
        Float8E8M0>;
#else
using AllIRv12 = AllIRv11;
#endif

// IR v13 adds INT2/UINT2 (2-bit integer types)
using AllIRv13 =
    boost::mp11::mp_push_back<
        AllIRv12,
        UInt2x4,
        Int2x4>;

// IR v14 adds FLOAT6E2M3 and FLOAT6E3M2 (6-bit floating-point types).
using AllIRv14 =
    boost::mp11::mp_push_back<
        AllIRv13,
        Float6E2M3,
        Float6E3M2>;

// TODO: This needs upgrade to some newer version ,buit it has been
// at this version for a while and it needs changes at the use sites
// where-in the types in the newer IR versions are not supported.
// This may need a sweep across multiple EPs as this is mostly used
// for kernel registration.
using All = AllIRv4;

#if !defined(DISABLE_FLOAT8_TYPES)
using AllFloat8 =
    TypeList<
        Float8E4M3FN,
        Float8E4M3FNUZ,
        Float8E5M2,
        Float8E5M2FNUZ,
        Float8E8M0>;
#endif

#if !defined(DISABLE_FLOAT4_TYPES)
using AllFloat4 = TypeList<Float4E2M1x2>;
#endif

using AllIeeeFloat =
    TypeList<
        float,
        double,
        MLFloat16>;

using AllNumericIRv4 =
    TypeList<
        float,
        double,
        int64_t,
        uint64_t,
        int32_t,
        uint32_t,
        int16_t,
        uint16_t,
        int8_t,
        uint8_t,
        MLFloat16,
        BFloat16>;

using AllNumeric = AllNumericIRv4;

}  // namespace element_type_lists

}  // namespace onnxruntime
