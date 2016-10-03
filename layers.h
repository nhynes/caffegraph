#ifndef LAYERS_H_
#define LAYERS_H_

#define LayerBase(NAME)                                         \
  class NAME ## Layer: public Layer {                           \
    friend class Layer;                                         \
    protected:                                                  \
      NAME ## Layer(const caffe::LayerParameter& params,        \
                    const std::vector<Layer*> inputs);

#define LayerDef(NAME)  \
  LayerBase(NAME)       \
};

#define LayerParamDef(NAME)                             \
  LayerBase(NAME)                                       \
    public:                                             \
      void Parameterize(THFloatTensor** tensors);       \
};

#define LayerExtDef(NAME, FIELDS)       \
  LayerBase(NAME)                       \
    private:                            \
      FIELDS;                           \
};

#define LayerExtParamDef(NAME, FIELDS)                  \
  LayerBase(NAME)                                       \
    public:                                             \
      void Parameterize(THFloatTensor** tensors);       \
    private:                                            \
      FIELDS;                                           \
};

typedef std::tuple<std::string, std::string, std::string> modstrs;

class Layer {
  public:
    static Layer* MakeLayer(const caffe::LayerParameter& params,
                            const std::vector<Layer*> inputs);
    virtual std::vector<std::vector<int>> GetOutputSizes();
    virtual void Parameterize(THFloatTensor** tensors);
    virtual std::vector<modstrs> layer_strs();
    std::string name;
  protected:
    Layer(const caffe::LayerParameter& params, const std::vector<Layer*> inputs);
    const caffe::LayerParameter& params;
    std::vector<Layer*> inputs;
    std::vector<modstrs> lua_layers;
    std::vector<std::vector<int>> output_sizes;
};

class InputLayer: public Layer {
  friend class Layer;
  public:
    std::vector<modstrs> layer_strs();
  protected:
    InputLayer(const caffe::LayerParameter& params, const std::vector<Layer*> inputs);
};

LayerDef(Data);
LayerDef(Dropout);
LayerDef(Eltwise);
LayerDef(Concat);
LayerDef(ReLU);
LayerDef(Sigmoid);
LayerDef(Softmax);
LayerDef(Tanh);
LayerParamDef(BatchNorm);
LayerParamDef(InnerProduct);
LayerParamDef(Scale);
LayerExtParamDef(Convolution, int nInputPlane; int nOutputPlane;
                              std::vector<unsigned int> k;
                              std::vector<unsigned int> p;
                              std::vector<unsigned int> d);
LayerExtDef(Pooling, std::vector<unsigned int> k;
                     std::vector<unsigned int> p;
                     std::vector<unsigned int> d);

#endif
