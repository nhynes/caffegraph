caffegraph
==========

Load [Caffe](http://caffe.berkeleyvision.org/) networks in [Torch7](http://torch.ch/) using [nngraph](https://github.com/torch/nngraph).

After installing Torch, ensure that you have the `protobuf` libraries.
On Ubuntu, this might look like

```sh
sudo apt-get install libprotobuf-dev protobuf-compiler
```

or, on Macintosh OperatingSystem 10:

```sh
brew install protobuf
```

You can then install this package using

```sh
luarocks install caffegraph
```

Then, similarly to [loadcaffe](https://github.com/szagoruyko/loadcaffe),

```lua
caffegraph = require 'caffegraph'
model = caffegraph.load('deploy_resnet152.prototxt', 'resnet152.caffemodel')
```

Note that some modules that are loadable using loadcaffe are not yet implemented in caffegraph. You are welcome to submit a PR with any that you feel are missing!

[`caffe.proto`](https://github.com/BVLC/caffe/blob/master/src/caffe/proto/caffe.proto) is used under [license](https://github.com/BVLC/caffe/blob/master/LICENSE) from the University of California.
