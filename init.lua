require 'nn'
local ffi = require 'ffi'

caffegraph = {}

ffi.cdef[[
struct params { int num_params; THFloatTensor** params; };
void loadModel(void** handle, const char* prototxt, const char* caffemodel);
void buildModel(void** handle, const char* lua_path);
void getParams(void** handle, THFloatTensor*** params);
void freeModel(void** handle);
]]

caffegraph.C = ffi.load(package.searchpath('libcaffegraph', package.cpath))

caffegraph.load = function(prototxt, caffemodel)
  local handle = ffi.new('void*[1]')

  -- load the caffemodel into a graph structure
  local initHandle = handle[1]
  caffegraph.C.loadModel(handle, prototxt, caffemodel)
  if handle[1] == initHandle then
    error('Unable to load model.')
  end

  -- serialize the graph and write it out
  local luaModel = path.splitext(caffemodel)
  luaModel = luaModel..'.lua'
  caffegraph.C.buildModel(handle, luaModel)

  -- -- bring the model into lua world
  local model, modmap = dofile(luaModel)

  -- transfer the parameters
  local noData = torch.FloatTensor():zero():cdata()
  local module_params = {}
  for i,nodes in ipairs(modmap) do
    local params = {}
    for i=1,#nodes do
      local module = nodes[i].data.module
      module:float()
      if torch.isTypeOf(module, nn.BatchNormalization) then
        module.weight:fill(1)
        module.bias:zero()
        params[(i-1)*2+1] = module.running_mean:cdata()
        params[i*2] = module.running_var:cdata()
      else
        params[(i-1)*2+1] = module.weight and module.weight:cdata() or noData
        params[i*2] = module.bias and module.bias:cdata() or noData
      end
    end
    module_params[i] = ffi.new('THFloatTensor*['..#params..']', params)
  end
  local cParams = ffi.new('THFloatTensor**['..#module_params..']', module_params)
  caffegraph.C.getParams(handle, cParams)

  caffegraph.C.freeModel(handle)

  return model
end

return caffegraph
