#include <cassert>
#include <chrono>
#include <fstream>
#include <iostream>
#include <iterator>
#include <string>
#include <thread>

#include <hip/hip_runtime_api.h>
#include <tvm/runtime/module.h>
#include <tvm/runtime/packed_func.h>
#include <tvm/runtime/registry.h>

class TvmGraphModule {
public:
  explicit TvmGraphModule(std::string lib_path, std::string graph_path,
                          std::string params_path, int64_t device_type,
                          int64_t device_id) {
    LOG(INFO) << "[TvmGraphModule] loading module at path: [" << lib_path
              << "] on device [" << (device_type == kDLROCM ? "rocm:" : "cpu:")
              << device_id << "]..." << std::endl;

    auto status = hipSetDevice(device_id);
    if (status != hipSuccess) {
      LOG(ERROR) << "hipSetDevice failed";
      exit(1);
    }

    // load graph
    std::ifstream graph_in(graph_path);
    std::string graph_data((std::istreambuf_iterator<char>(graph_in)),
                           std::istreambuf_iterator<char>());
    graph_in.close();

    // load mod syslib
    tvm::runtime::Module lib = tvm::runtime::Module::LoadFromFile(lib_path);
    const auto runtime_create =
        *tvm::runtime::Registry::Get("tvm.graph_executor.create");

    // read params data
    std::ifstream params_in(params_path, std::ios::binary);
    std::string params_data((std::istreambuf_iterator<char>(params_in)),
                            std::istreambuf_iterator<char>());
    params_in.close();

    TVMByteArray params_arr;
    params_arr.data = params_data.c_str();
    params_arr.size = params_data.length();

    // set devices
    module_ = runtime_create(graph_data, lib, device_type, device_id);
    const tvm::runtime::PackedFunc load_params =
        module_.GetFunction("load_params");
    load_params(params_arr);

    get_input = module_.GetFunction("get_input");
    get_num_inputs = module_.GetFunction("get_num_inputs");
    set_input = module_.GetFunction("set_input_zero_copy");
    run = module_.GetFunction("run");
    get_output = module_.GetFunction("get_output");
    set_output = module_.GetFunction("set_output_zero_copy");
    num_outputs_ = module_.GetFunction("get_num_outputs")();

    LOG(INFO) << "[TvmGraphModule] loading module complete";
  }

  const int64_t num_outputs() const { return num_outputs_; }

  tvm::runtime::PackedFunc get_input;
  tvm::runtime::PackedFunc get_num_inputs;
  tvm::runtime::PackedFunc set_input;
  tvm::runtime::PackedFunc run;
  tvm::runtime::PackedFunc get_output;
  tvm::runtime::PackedFunc set_output;

private:
  tvm::runtime::Module module_;
  int64_t num_outputs_;
};

bool run_inference(const TvmGraphModule &mod, std::string in_path,
                   int64_t device_type, int64_t device_id) {

  std::ifstream input(in_path, std::ios::binary);
  std::string in_data((std::istreambuf_iterator<char>(input)),
                      std::istreambuf_iterator<char>());

  int64_t in_shape[] = {128, 3, 224, 224};
  constexpr int64_t out_shape[] = {128, 1000};
  constexpr int in_dim = 4;
  constexpr int dtype_code = kDLFloat;
  constexpr int dtype_bits = 32;
  constexpr int dtype_lanes = 1;
  constexpr int out_dim = 2;

  uint64_t num_inputs = mod.get_num_inputs();

  DLTensor *in_tensor, *out_tensor;

  auto status = hipSetDevice(device_id);
  if (status != hipSuccess) {
    LOG(ERROR) << "hipSetDevice failed";
    return false;
  }

  int ret = TVMArrayAlloc(in_shape, in_dim, dtype_code, dtype_bits, dtype_lanes,
                          device_type, device_id, &in_tensor);
  if (ret) {
    LOG(ERROR) << "TVMArrayAlloc failed for input tensor : "
               << TVMGetLastError();
    return false;
  }
  ret = TVMArrayAlloc(out_shape, out_dim, dtype_code, dtype_bits, dtype_lanes,
                      device_type, device_id, &out_tensor);
  if (ret) {
    LOG(ERROR) << "TVMArrayAlloc failed for output tensor : "
               << TVMGetLastError();
    return false;
  }

  size_t totalSize = in_data.length() * 128;
  char *batchData = (char *)malloc(totalSize);

  if (!batchData) {
    LOG(ERROR) << "Memory allocation failed\n";
    return false;
  }

  for (int i = 0; i < 128; i++) {
    char *dst = batchData + i * in_data.length();
    memcpy(dst, in_data.c_str(), in_data.length());
  }

  void *deviceMem;
  auto err = hipMalloc(&deviceMem, totalSize);
  ret = TVMArrayCopyFromBytes(in_tensor, batchData, totalSize);
  if (ret) {
    LOG(ERROR) << "TVMArrayCopyFromBytes failed for input tensor : "
               << TVMGetLastError();
    return false;
  }

  for (int i = 0; i < 10; i++) {
    mod.set_input("data", in_tensor);
    auto start = std::chrono::high_resolution_clock::now();
    mod.run();
    auto end = std::chrono::high_resolution_clock::now();
    mod.get_output(0, out_tensor);

    auto output_iter = static_cast<float *>(out_tensor->data);
    auto max_iter = std::max_element(output_iter, output_iter + 1000);
    auto max_index = std::distance(output_iter, max_iter);

    auto elapsed = std::chrono::duration<double>(end - start).count();

    LOG(INFO) << "[run_inference] The maximum position in output vector is: "
              << max_index;
    LOG(INFO) << "[run_inference] time taken : " << elapsed;
  }

  TVMArrayFree(in_tensor);
  TVMArrayFree(out_tensor);

  return true;
}

class thread_obj {
public:
  void operator()(std::string lib_path, std::string graph_path,
                  std::string params_path, std::string image_path,
                  const int64_t device_id) {
    constexpr int64_t device_type = kDLROCM;

    TvmGraphModule mod(lib_path, graph_path, params_path, device_type,
                       device_id);
    run_inference(mod, image_path, device_type, device_id);
  }
};

int main(int argc, char **argv) {
  if (argc != 6) {
    LOG(ERROR) << "Usage: inference <model.so> <model.json> <model.params> "
                  "<image_path>";
    exit(1);
  }

  const std::string lib(argv[1]);
  const std::string graph(argv[2]);
  const std::string params(argv[3]);
  const std::string image1(argv[4]);
  const std::string image2(argv[5]);

  std::thread th1(thread_obj(), lib, graph, params, image1, 0 /* Device ID */);
  std::thread th2(thread_obj(), lib, graph, params, image2, 1 /* Device ID */);

  th1.join();
  th2.join();

  return 0;
}
