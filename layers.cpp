#include <TH/TH.h>
#include <sstream>
#include <iostream>
#include <string>
#include <cstdarg>

#include "caffe.pb.h"
#include "layers.h"

#define LayerInit(NAME)                                                 \
  NAME ## Layer::NAME ## Layer(const caffe::LayerParameter& params,     \
      std::vector<Layer*> inputs)                                       \
    : Layer(params, inputs)

using google::protobuf::Message;
using google::protobuf::RepeatedPtrField;

// DEBUGGING
using std::cout;
using std::endl;

template <typename T>
void RepVec(std::vector<T>& vec, int len) {
  T fill = vec[0];
  while(vec.size() < len)
    vec.push_back(fill);
}

void THCopy(const caffe::BlobProto& src, THFloatTensor* dest) {
  auto& blob_shape = src.shape();
  int num_cpy = 1;
  for(int dim : blob_shape.dim()) num_cpy *= dim;

  THFloatTensor_newContiguous(dest);
  assert(THFloatTensor_numel(dest) == num_cpy);
  memcpy(THFloatTensor_data(dest), src.data().data(), sizeof(float)*num_cpy);
}

Layer* Layer::MakeLayer(const caffe::LayerParameter& params,
                        std::vector<Layer*> inputs) {
  if(params.type() == "Data")
    return new DataLayer(params, inputs);
  else if(params.type() == "Convolution")
    return new ConvolutionLayer(params, inputs);
  else if(params.type() == "Pooling")
    return new PoolingLayer(params, inputs);
  else if(params.type() == "BatchNorm")
    return new BatchNormLayer(params, inputs);
  else if(params.type() == "InnerProduct")
    return new InnerProductLayer(params, inputs);
  else if(params.type() == "Eltwise")
    return new EltwiseLayer(params, inputs);
  else if(params.type() == "Scale")
    return new ScaleLayer(params, inputs);
  else if(params.type() == "ReLU")
    return new ReLULayer(params, inputs);
  else if(params.type() == "Sigmoid")
    return new SigmoidLayer(params, inputs);
  else if(params.type() == "Tanh")
    return new TanhLayer(params, inputs);
  else if(params.type() == "Dropout")
    return new DropoutLayer(params, inputs);
  else if(params.type() == "Softmax" || params.type() == "SoftmaxWithLoss")
    return new SoftmaxLayer(params, inputs);
  else if(params.type() == "Input")
    return new InputLayer(params, inputs);
  else
    return new Layer(params, inputs);
}

Layer::Layer(const caffe::LayerParameter& params, std::vector<Layer*> inputs)
   : params(params), inputs(inputs) {
  name = params.name();
}

std::vector<modstrs> Layer::layer_strs() {
  if(lua_layers.size() == 0) {
    std::string missing = params.type();
    missing.append(" (");
    missing.append(name);
    missing.append(")");
    lua_layers.emplace_back("-- Missing: ", missing, "");
  }
  return lua_layers;
}

void Layer::Parameterize(THFloatTensor** params) {}

LayerInit(Data) {
  if(params.has_input_param()) {
    auto& input_param = params.input_param();
    for(auto& shape : input_param.shape()) {
      auto& dims = shape.dim();
      std::vector<int> inp_dims(dims.begin(), dims.end());
      output_sizes.push_back(inp_dims);
    }
  }
  lua_layers.emplace_back(name, "nn.Identity()", "");
}

std::vector<std::vector<int>> Layer::GetOutputSizes() {
  if(output_sizes.size() == 0)
    output_sizes.push_back(inputs[0]->GetOutputSizes()[0]);
  return output_sizes;
}

LayerInit(Convolution) {
  auto& conv_params = params.convolution_param();
  int groups = conv_params.group() == 0 ? 1 : conv_params.group();
  auto& weight = params.blobs(0);
  nInputPlane = weight.shape().dim(1) * groups;
  nOutputPlane = conv_params.num_output();

  if(conv_params.has_kernel_w()) {
    k = {conv_params.kernel_w(), conv_params.kernel_h()};
    p = {conv_params.kernel_w(), conv_params.kernel_h()};
    k = {conv_params.kernel_w(), conv_params.kernel_h()};
  } else {
    auto& ks = conv_params.kernel_size();
    k = std::vector<unsigned int>(ks.begin(), ks.end());
    auto& ps = conv_params.pad();
    p = std::vector<unsigned int>(ps.begin(), ps.end());
    auto& ds = conv_params.stride();
    d = std::vector<unsigned int>(ds.begin(), ds.end());

    int dim = weight.shape().dim_size() - 1;
    RepVec(k, dim-1);
    RepVec(p, dim-1);
    RepVec(d, dim-1);
  }

  std::ostringstream module_os;
  if(k.size() == 1) {
    module_os << "nn.TemporalConvolution(";
  } else if(k.size() == 2) {
    module_os << "nn.SpatialConvolution(";
  } else {
    module_os << "nn.VolumetricConvolution(";
  }
  module_os << nInputPlane << ", " << nOutputPlane;
  for(int ks : k) module_os << ", " << ks;
  for(int ds : d) module_os << ", " << ds;
  for(int ps : p) module_os << ", " << ps;
  module_os << ")";

  lua_layers.emplace_back(name, module_os.str(), inputs[0]->name);

  std::vector<int> input_size = inputs[0]->GetOutputSizes()[0];
  std::vector<int> output_size(input_size.size());
  output_size[0] = nOutputPlane;
  for(int i = 0; i < k.size(); ++i)
    output_size[i+1] = (input_size[i+1] + 2*p[i] - k[i]) / d[i] + 1;
  output_sizes.push_back(output_size);
}

void ConvolutionLayer::Parameterize(THFloatTensor** tensors) {
  auto& conv_params = params.convolution_param();
  for(int i = 0; i < params.blobs_size(); ++i)
    THCopy(params.blobs(i), tensors[i]);
  if(!conv_params.bias_term())
    THFloatTensor_zero(tensors[1]);
}

LayerInit(Pooling) {
  auto& pooling_params = params.pooling_param();

  if(pooling_params.kernel_size() > 0) {
    k.push_back(pooling_params.kernel_size());
    k.push_back(pooling_params.kernel_size());
  } else {
    k.push_back(pooling_params.kernel_w());
    k.push_back(pooling_params.kernel_h());
  }

  if(pooling_params.stride() > 0) {
    d.push_back(pooling_params.stride());
    d.push_back(pooling_params.stride());
  } else {
    d.push_back(pooling_params.stride_w());
    d.push_back(pooling_params.stride_h());
  }

  if(pooling_params.pad() > 0) {
    p.push_back(pooling_params.pad());
    p.push_back(pooling_params.pad());
  } else {
    p.push_back(pooling_params.pad_w());
    p.push_back(pooling_params.pad_h());
  }

  std::string pool_type = pooling_params.pool() == caffe::PoolingParameter::MAX ?
    "Max" : "Average";

  std::ostringstream module_os;
  module_os << "nn.Spatial" << pool_type << "Pooling(" << k[0] << ", " << k[1] << ", ";
  module_os << d[0] << ", " << d[1] << ", " << p[0] << ", " << p[1] << "):ceil()";
  lua_layers.emplace_back(name, module_os.str(), inputs[0]->name);

  std::vector<int> input_size = inputs[0]->GetOutputSizes()[0];
  std::vector<int> output_size(input_size.size());
  output_size[0] = input_size[0];
  for(int i = 0; i < k.size(); ++i)
    output_size[i+1] = ceil((double)(input_size[i+1] + 2*p[i] - k[i]) / d[i] + 1);
  output_sizes.push_back(output_size);
}

LayerInit(BatchNorm) {
  auto& bn_params = params.batch_norm_param();
  float eps = bn_params.eps();
  float momentum = bn_params.moving_average_fraction();

  std::vector<int> input_size = inputs[0]->GetOutputSizes()[0];

  std::ostringstream module_os;
  module_os << "nn.";
  if(input_size.size() == 3)
    module_os << "Spatial";
  else if(input_size.size() == 4)
    module_os << "Volumetric";
  module_os << "BatchNormalization(" << input_size[0] << ", " << eps << ", " << momentum;
  module_os << ")";
  lua_layers.emplace_back(name, module_os.str(), inputs[0]->name);
}

void BatchNormLayer::Parameterize(THFloatTensor** tensors) {
  THCopy(params.blobs(0), tensors[0]); // mean
  THCopy(params.blobs(1), tensors[1]); // var

  float runningScale = 1 / params.blobs(2).data(0);
  THFloatTensor_mul(tensors[0], tensors[0], runningScale);
  THFloatTensor_mul(tensors[1], tensors[1], runningScale);
}

LayerInit(InnerProduct) {
  auto input_size = inputs[0]->GetOutputSizes()[0];

  std::ostringstream view_module_os;
  view_module_os << "nn.View(-1):setNumInputDims(" << input_size.size() << ")";
  lua_layers.emplace_back("collapse", view_module_os.str(), inputs[0]->name);

  int nInputs = 1;
  for(int size : input_size) nInputs *= size;

  int nOutputs = params.inner_product_param().num_output();

  std::ostringstream module_os;
  module_os << "nn.Linear(" << nInputs << ", " << nOutputs << ")";
  lua_layers.emplace_back(name, module_os.str(), "collapse");
}

void InnerProductLayer::Parameterize(THFloatTensor** tensors) {
  for(int i = 0; i < params.blobs_size(); ++i)
    THCopy(params.blobs(i), tensors[i+2]); // +2 because view has no params
}

LayerInit(Eltwise) {
  auto op = params.eltwise_param().operation();

  std::string module = "nn.CAddTable()";
  if(op == caffe::EltwiseParameter_EltwiseOp_PROD)
    module = "nn.CMulTable()";
  else if(op == caffe::EltwiseParameter_EltwiseOp_MAX)
    module = "nn.CMaxTable()";

  std::ostringstream graph_args_os;
  graph_args_os << "{";
  for(int i = 0; i < inputs.size(); i++) {
    graph_args_os << inputs[i]->name;
    if(i < inputs.size()-1)
      graph_args_os << ", ";
  }
  graph_args_os << "}";

  lua_layers.emplace_back(name, module, graph_args_os.str());
}

LayerInit(Scale) {
  if(inputs.size() > 1)
    std::cerr << "Multi-input Scale not yet supported!" << std::endl;

  auto& scale_params = params.scale_param();

  std::vector<int> input_size = inputs[0]->GetOutputSizes()[0];
  int input_dim = input_size.size();

  std::string input_name = inputs[0]->name;
  if(scale_params.bias_term()) {
    std::string scale_modname = name;
    scale_modname.append("_scale");
    lua_layers.emplace_back(scale_modname, "nn.CMul()", input_name);

    lua_layers.emplace_back(name, "nn.Add(1)", scale_modname);
  } else {
    lua_layers.emplace_back(name, "nn.CMul()", input_name);
  }
}

void THCopyAxis(const caffe::BlobProto& src, THFloatTensor* dest,
                std::vector<int> size, int axis) {
  int ndim = size.size();
  std::vector<long int> resize(size.begin(), size.end());
  THLongStorage* szst = THLongStorage_newWithData(resize.data(), ndim);

  std::vector<long int> vec_sz(ndim, 1);
  vec_sz[axis] = src.shape().dim(0);

  std::vector<long int> expand_stride(ndim, 0);
  expand_stride[axis] = 1;

  THLongStorage* vec_szst = THLongStorage_newWithData(vec_sz.data(), ndim);
  THFloatTensor* vec = THFloatTensor_newWithSize1d(src.shape().dim(0));
  THCopy(src, vec);


  THFloatStorage* vec_storage = THFloatTensor_storage(vec);
  THLongStorage* expand_stridest = THLongStorage_newWithData(expand_stride.data(), ndim);
  THFloatTensor_setStorage(vec, vec_storage, 0, szst, expand_stridest);

  THFloatTensor_resize(dest, szst, NULL);
  THFloatTensor_copy(dest, vec);
}

void ScaleLayer::Parameterize(THFloatTensor** tensors) {
  // tensors: scale_weight, scale_bias, add_weight, add_bias
  auto& scale_params = params.scale_param();
  std::vector<int> input_size = inputs[0]->GetOutputSizes()[0];
  int axis = scale_params.axis() - 1;

  if(scale_params.bias_term())
    THCopyAxis(params.blobs(1), tensors[3], input_size, axis);

  THCopyAxis(params.blobs(0), tensors[0], input_size, axis);
}

LayerInit(Softmax) {
  lua_layers.emplace_back(name, "nn.SoftMax()", inputs[0]->name);
}

LayerInit(ReLU) {
  lua_layers.emplace_back(name, "nn.ReLU(true)", inputs[0]->name);
}

LayerInit(Sigmoid) {
  lua_layers.emplace_back(name, "nn.Sigmoid()", inputs[0]->name);
}

LayerInit(Tanh) {
  lua_layers.emplace_back(name, "nn.Tanh(true)", inputs[0]->name);
}

LayerInit(Dropout) {
  std::ostringstream module_os;
  module_os << "nn.Dropout(" << params.dropout_param().dropout_ratio() << ")";
  lua_layers.emplace_back(name, module_os.str(), inputs[0]->name);
}

LayerInit(Input) {}
std::vector<modstrs> InputLayer::layer_strs() {
  return std::vector<modstrs>(0);
}
