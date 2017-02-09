#include <TH/TH.h>
#include <algorithm>
#include <fcntl.h>
#include <fstream>
#include <iostream>
#include <limits>
#include <locale>
#include <memory>
#include <sstream>
#include <string>
#include <unistd.h>
#include <unordered_map>
#include <unordered_set>

#include <google/protobuf/io/coded_stream.h>
#include <google/protobuf/io/zero_copy_stream_impl.h>
#include <google/protobuf/text_format.h>

#include "caffe.pb.h"
#include "layers.h"

using google::protobuf::io::FileInputStream;
using google::protobuf::io::ZeroCopyInputStream;
using google::protobuf::io::CodedInputStream;

#define print(VAL) std::cout << VAL << std::endl; // DEBUGGING

extern "C" {
  void loadModel(void** handle, const char* prototxt, const char* caffemodel);
  void buildModel(const void** handle, const char* luafile);
  void getParams(const void** handle, THFloatTensor*** params);
  void freeModel(void** handle);
}

class Model {
  public:
    Model(caffe::NetParameter* net_params) : net_params(net_params) {
      int num_layers = net_params->layer_size();
      std::unordered_map<std::string, Layer*> modmap(num_layers);

      for(int i = -1; i < num_layers-1; ++i) {
        auto& layer_params = net_params->layer(i == -1 ? num_layers - 1 : i);

        if(layer_params.type() == "Split") {
          auto& bottom = layer_params.bottom(0);
          for(std::string top : layer_params.top())
            modmap[top] = modmap[bottom];
          tips.erase(layer_params.bottom(0));
          continue;
        }

        if(layer_params.type().find("Data") != std::string::npos) {
          if(i > -1)
            continue;
          roots.push_back(layer_params.name());
        }

        int num_inputs = layer_params.bottom_size();
        std::vector<Layer*> inputs(0);

        // pre-flight check for bottoms
        bool skip_layer = false;
        for(std::string bottom : layer_params.bottom()) {
          if(modmap.count(bottom) > 0)
            continue;
          std::cerr << "[WARN] Missing bottom \"" << bottom << "\" for layer \""
            << layer_params.name() << "\"" << std::endl;
          skip_layer = true;
        }
        if(skip_layer)
          continue;

        for(std::string bottom : layer_params.bottom()) {
          inputs.push_back(modmap[bottom]);
          tips.erase(bottom);
        }

        Layer* layer = Layer::MakeLayer(layer_params, inputs);
        for(std::string top : layer_params.top()) {
          modmap[top] = layer;
          tips.insert(top);
        }
        layers.push_back(layer);
      }
    }

    void Serialize(std::ostream& out) {
      bool as_graph = true; // graph optimization should probably be in nngraph, itself

      out << "require 'nngraph'\n\n";

      out << "modmap = {}\n\n";

      for(Layer* layer : layers) {
        auto lua_layers = layer->layer_strs();
        if(lua_layers.size() == 0)
          continue;

        std::string modmap = "modmap[#modmap+1] = {";
        std::ostringstream modmap_os;
        for(int i = 0; i < lua_layers.size(); ++i) {
          modstrs ll = lua_layers[i];
          std::ostringstream module_os;
          module_os << std::get<0>(ll) << " = " << std::get<1>(ll);
          if(as_graph)
            module_os << "(" << std::get<2>(ll) << ")";

          modmap.append(std::get<0>(ll));
          if(i < lua_layers.size()-1) modmap.append(", ");

          out << module_os.str() << "\n";
        }
        modmap.append("}");
        out << modmap << "\n\n";
      }

      out << "model = nn.gModule({";
      for(int i = 0; i < roots.size(); ++i) {
        out << roots[i];
        if(i < roots.size()-1) out << ", ";
      }
      out << "}, {";
      int i = 0;
      for(std::string tip : tips) {
        std::replace(tip.begin(), tip.end(), '/', '_');
        out << tip;
        if(i < tips.size()-1) out << ", ";
        ++i;
      }
      out << "})\n\n";

      out << "return model, modmap" << std::endl;
    }

    void Parameterize(THFloatTensor*** tensors) {
      for(int i = 0; i < layers.size(); ++i)
        layers[i]->Parameterize(tensors[i]);
    }

    ~Model() {
      delete net_params;
      for(Layer* layer : layers)
        delete layer;
    }
  private:
    caffe::NetParameter* net_params;
    std::vector<Layer*> layers;
    std::unordered_set<std::string> tips;
    std::vector<std::string> roots;
};

void loadModel(void** handle, const char* prototxt, const char* caffemodel) {
  int fd = open(caffemodel, O_RDONLY);
  if(fd < 0) return;

  ZeroCopyInputStream* raw_input = new FileInputStream(fd);
  CodedInputStream* coded_input = new CodedInputStream(raw_input);
  coded_input->SetTotalBytesLimit(std::numeric_limits<int>::max(), -1);

  caffe::NetParameter* net_params = new caffe::NetParameter();
  bool loaded_caffemodel = net_params->ParseFromCodedStream(coded_input);

  delete raw_input;
  delete coded_input;
  close(fd);
  if(!loaded_caffemodel) return;

  // find and canonicalize input shape
  bool has_input_shape = false;
  auto* l0 = net_params->mutable_layer(0);
  auto* canon_input_param = new caffe::InputParameter();
  auto input_shape_ins = RepeatedFieldBackInserter(canon_input_param->mutable_shape());
  std::string data_layer_name;

  // check if the first layer has an input_param and, if so, remove the batch size
  if(l0->has_input_param()) {
    auto& ip_shape = l0->input_param().shape();
    std::copy(ip_shape.begin()+1, ip_shape.end(), input_shape_ins);
    data_layer_name = l0->top(0);
    has_input_shape = true;
  }

  // otherwise, check if any of the later layers has an input_param and move it
  // to the first layer (again, removing the batch size)
  if(!has_input_shape) {
    for(auto& layer : net_params->layer()) {
      if(layer.has_input_param()) {
        auto& ip_shape = layer.input_param().shape();
        std::copy(ip_shape.begin()+1, ip_shape.end(), input_shape_ins);
        data_layer_name = layer.top(0);
        has_input_shape = true;
        break;
      }
    }
  }

  // there was no input_param, so we have to check the prototxt
  if(!has_input_shape) {
    int fd = open(prototxt, O_RDONLY);
    if(fd < 0) return;

    caffe::NetParameter* proto_params = new caffe::NetParameter();
    FileInputStream* input = new FileInputStream(fd);
    bool loaded_proto = google::protobuf::TextFormat::Parse(input, proto_params);

    delete input;
    close(fd);
    if(!loaded_proto) return;

    if(proto_params->input_dim_size() > 0) {
      auto* shape = canon_input_param->add_shape();
      for(int i = 1; i < 4; ++i) // assume first input is data
        shape->add_dim(proto_params->input_dim(i));
    } else {
      auto& ip_shape = proto_params->input_shape();
      std::copy(ip_shape.begin()+1, ip_shape.end(), input_shape_ins);
    }
    data_layer_name = proto_params->input(0);

    delete proto_params;
  }

  // canonicalize the datalayer with the canonical input shape
  auto* canon_data_layer = net_params->add_layer();
  canon_data_layer->set_type("Data");
  canon_data_layer->add_top(data_layer_name);
  canon_data_layer->set_name(data_layer_name);
  canon_data_layer->set_allocated_input_param(canon_input_param);

  Model* model = new Model(net_params);

  handle[1] = model;
}

void buildModel(const void** handle, const char* luafile) {
  Model* model = (Model*)handle[1];
  std::ofstream out(luafile);
  model->Serialize(out);
  out.close();
}

void getParams(const void** handle, THFloatTensor*** params) {
  Model* model = (Model*)handle[1];
  model->Parameterize(params);
}

void freeModel(void** handle) {
  Model* model = (Model*)handle[1];
  delete model;
}
