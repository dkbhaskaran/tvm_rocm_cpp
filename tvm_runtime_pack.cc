/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

#include <dlpack/dlpack.h>
#include <tvm/runtime/module.h>
#include <tvm/runtime/packed_func.h>
#include <tvm/runtime/registry.h>

#include "c_runtime_api.cc"
#include "container.cc"
#include "cpu_device_api.cc"
#include "file_utils.cc"
#include "library_module.cc"
#include "logging.cc"
#include "module.cc"
#include "ndarray.cc"
#include "object.cc"
#include "registry.cc"
#include "thread_pool.cc"
#include "threading_backend.cc"
#include "workspace_pool.cc"

#include "dso_library.cc"
#include "system_library.cc"

// Graph executor
#include "graph_executor/graph_executor.cc"
#include "graph_executor/graph_executor_factory.cc"

#define TVM_ROCM_RUNTIME 1
#define TVM_USE_MIOPEN 1
#define __HIP_PLATFORM_HCC__ 1

#include "contrib/miopen/conv_forward.cc"
#include "contrib/miopen/miopen_utils.cc"
#include "contrib/rocblas/rocblas.cc"
#include "rocm/rocm_device_api.cc"
#include "rocm/rocm_module.cc"
