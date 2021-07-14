// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#define optional_CONFIG_SELECT_OPTIONAL optional_OPTIONAL_NONSTD

#include <nonstd/optional.hpp>


namespace onnxruntime {

using nonstd::optional;

#ifndef ORT_NO_EXCEPTIONS
using nonstd::bad_optional_access;
#endif

using nonstd::nullopt;
using nonstd::nullopt_t;

using nonstd::in_place;
using nonstd::in_place_t;

using nonstd::make_optional;

}  // namespace onnxruntime
