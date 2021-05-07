// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <sstream>

// Struct to represent a physical device.
struct OrtDevice {
  using DeviceType = int8_t;
  using MemoryType = int8_t;
  using DeviceId = int16_t;

  // Pre-defined device types.
  static const DeviceType CPU = 0;
  static const DeviceType GPU = 1;  // Nvidia or AMD
  static const DeviceType FPGA = 2;

  struct MemType {
    // Pre-defined memory types.
    static const MemoryType DEFAULT = 0;
    static const MemoryType GPU_PINNED = 1;
    static const MemoryType HIP_PINNED = 2;
  };

  constexpr OrtDevice(DeviceType device_type_, MemoryType memory_type_, DeviceId device_id_)
      : device_type(device_type_),
        memory_type(memory_type_),
        device_id(device_id_) {}

  constexpr OrtDevice() : OrtDevice(CPU, MemType::DEFAULT, 0) {}

  DeviceType Type() const {
    return device_type;
  }

  MemoryType MemType() const {
    return memory_type;
  }

  DeviceId Id() const {
    return device_id;
  }

  std::string ToString() const {
    std::ostringstream ostr;
    ostr << "Device:["
         << "DeviceType:" << static_cast<int>(device_type)
         << " MemoryType:" << static_cast<int>(memory_type)
         << " DeviceId:" << device_id
         << "]";
    return ostr.str();
  }

 private:
  // Device type.
  DeviceType device_type;

  // Memory type.
  MemoryType memory_type;

  // Device index.
  DeviceId device_id;
};

inline bool operator==(const OrtDevice& left, const OrtDevice& other) {
  return left.Id() == other.Id() && left.MemType() == other.MemType() && left.Type() == other.Type();
}

inline bool operator!=(const OrtDevice& left, const OrtDevice& other) {
  return !(left == other);
}
