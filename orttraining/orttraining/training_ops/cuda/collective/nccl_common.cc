// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "orttraining/training_ops/cuda/collective/nccl_common.h"
#include <mpi.h>

#include "orttraining/core/framework/communication/mpi/mpi_context.h"

namespace onnxruntime {
namespace cuda {

ncclDataType_t GetNcclDataType(onnxruntime::MLDataType type) {
  if (type == DataTypeImpl::GetType<uint8_t>()) {
    return ncclUint8;
  } else if (type == DataTypeImpl::GetType<int8_t>()) {
    return ncclInt8;
  } else if (type == DataTypeImpl::GetType<int32_t>()) {
    return ncclInt32;
  } else if (type == DataTypeImpl::GetType<int64_t>()) {
    return ncclInt64;
  } else if (type == DataTypeImpl::GetType<MLFloat16>()) {
    return ncclFloat16;
  } else if (type == DataTypeImpl::GetType<float>()) {
    return ncclFloat32;
  } else if (type == DataTypeImpl::GetType<double>()) {
    return ncclFloat64;
  } else {
    throw std::logic_error("Tensor type not supported in NCCL.");
  }
}

#ifdef USE_MPI
static Status CreateNcclCommunicator(MPI_Group* mpi_world_group,
                                     const training::WorkerGroupType worker_group_type,
                                     ncclComm_t* group_comm) {
  auto worker_group = training::DistributedRunContext::GetInstance().GetWorkerGroup(worker_group_type);

  // Create new group
  MPI_Group mpi_group;
  MPI_CHECK(MPI_Group_incl(*mpi_world_group, worker_group.ranks.size(), worker_group.ranks.data(), &mpi_group));
  // Create new MPI communicator
  MPI_Comm mpi_comm;
  static int32_t mpi_group_id = 0;
  MPI_CHECK(MPI_Comm_create_group(MPI_COMM_WORLD, mpi_group, ++mpi_group_id, &(mpi_comm)));
  ORT_ENFORCE(mpi_comm != MPI_COMM_NULL, "MPI communicator creation failed.");

  // Create new NCCL communicator
  ncclUniqueId nccl_id;
  if (worker_group.rank_in_group == 0) {
    NCCL_RETURN_IF_ERROR(ncclGetUniqueId(&nccl_id));
  }
  MPI_CHECK(MPI_Bcast(&nccl_id, sizeof(nccl_id), MPI_BYTE, 0, mpi_comm));
  NCCL_RETURN_IF_ERROR(ncclCommInitRank(group_comm, worker_group.ranks.size(), nccl_id, worker_group.rank_in_group));

  // Clean up
  MPI_CHECK(MPI_Group_free(&mpi_group));
  MPI_CHECK(MPI_Comm_free(&mpi_comm));
  return Status::OK();
}
#endif

NcclContext::NcclContext() {
#ifdef USE_MPI
  int is_mpi_initialized = 0;
  MPI_Initialized(&is_mpi_initialized);
  if (!is_mpi_initialized) {
    int mpi_threads_provided = 0;
    MPI_Init_thread(nullptr, nullptr, MPI_THREAD_MULTIPLE, &mpi_threads_provided);
  }

  // Get the group under MPI_COMM_WORLD
  MPI_Group mpi_world_group;
  MPI_Comm_group(MPI_COMM_WORLD, &mpi_world_group);

  // Initialize global Parallel Group NCCL Communicator
  auto ret = CreateNcclCommunicator(&mpi_world_group, training::WorkerGroupType::GlobalParallel,
                                    &global_group_comm_);
  ORT_ENFORCE(ret.IsOK());

  // Initialize Data Parallel Group NCCL Communicator
  ret = CreateNcclCommunicator(&mpi_world_group, training::WorkerGroupType::DataParallel,
                               &data_group_comm_);
  ORT_ENFORCE(ret.IsOK());

  // Initialize Horizontal Model Parallel Group NCCL Communicator
  ret = CreateNcclCommunicator(&mpi_world_group, training::WorkerGroupType::HorizontalParallel,
                               &horizontal_group_comm_);
  ORT_ENFORCE(ret.IsOK());

  // Initialize node local Parallel Group NCCL Communicator
  ret = CreateNcclCommunicator(&mpi_world_group, training::WorkerGroupType::NodeLocalDataParallel,
                               &node_local_comm_);
  ORT_ENFORCE(ret.IsOK());

  // Initialize cross node Parallel Group NCCL Communicator
  ret = CreateNcclCommunicator(&mpi_world_group, training::WorkerGroupType::CrossNodeDataParallel,
                               &cross_node_comm_);
  ORT_ENFORCE(ret.IsOK());

  MPI_Group_free(&mpi_world_group);
#else
  ORT_THROW("ORT must be built with MPI to use NCCL.");
#endif
}

ncclComm_t NcclContext::Comm(training::WorkerGroupType group_type) {
  if (training::WorkerGroupType::GlobalParallel == group_type) {
    return global_group_comm_;
  } else if (training::WorkerGroupType::DataParallel == group_type) {
    return data_group_comm_;
  } else if (training::WorkerGroupType::HorizontalParallel == group_type) {
    return horizontal_group_comm_;
  } else if (training::WorkerGroupType::NodeLocalDataParallel == group_type) {
    return node_local_comm_;
  } else if (training::WorkerGroupType::CrossNodeDataParallel == group_type) {
    return cross_node_comm_;
  }

  return nullptr;
}

NcclContext::~NcclContext() {
  if (data_group_comm_ != nullptr) {
    ncclCommDestroy(data_group_comm_);
  }

  if (horizontal_group_comm_ != nullptr) {
    ncclCommDestroy(horizontal_group_comm_);
  }

  if (node_local_comm_ != nullptr) {
    ncclCommDestroy(node_local_comm_);
  }

  if (cross_node_comm_ != nullptr) {
    ncclCommDestroy(cross_node_comm_);
  }

#ifdef USE_MPI
  int is_mpi_finalized = 0;
  MPI_Finalized(&is_mpi_finalized);
  if (!is_mpi_finalized) {
    MPI_Finalize();
  }
#endif
}

NcclKernel::NcclKernel(const OpKernelInfo& info) : CudaKernel(info) {
  static NcclContext context;
  nccl_ = &context;
  int64_t group_type;
  info.GetAttrOrDefault("group_type", &group_type, static_cast<int64_t>(0));
  group_type_ = static_cast<training::WorkerGroupType>(group_type);
}

}  // namespace cuda
}  // namespace onnxruntime
