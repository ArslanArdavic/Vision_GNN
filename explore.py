import torch
from PIL import Image

from vig import Stem, DeepGCN
from visualizer_utils import pre_transforms

class OptInit:
        def __init__(self, num_classes=1000, drop_path_rate=0.0, drop_rate=0.0, num_knn=9, **kwargs):
            self.k = num_knn # neighbor num (default:9)
            self.conv = 'mr' # graph conv layer {edge, mr}
            self.act = 'gelu' # activation layer {relu, prelu, leakyrelu, gelu, hswish}
            self.norm = 'batch' # batch or instance normalization {batch, instance}
            self.bias = True # bias of conv layer True or False
            self.n_blocks = 12 # number of basic blocks in the backbone
            self.n_filters = 192 # number of channels of deep features
            self.n_classes = num_classes # Dimension of out_channels
            self.dropout = drop_rate # dropout rate
            self.use_dilation = False # use dilated knn or not
            self.epsilon = 0.2 # stochastic epsilon for gcn
            self.use_stochastic = False # stochastic for gcn, True or False
            self.drop_path = drop_path_rate

if __name__ == "__main__":
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    img_path = 'fig/ILSVRC2012_val_00000075.JPEG'
    img = Image.open(img_path).resize((224, 224))
    print(img.s)
    img_data = torch.unsqueeze(pre_transforms(img), 0).to(device)

    #img_data = torch.rand(3, 3, 224, 224)
    #stem = Stem()
    #a = stem(img_data)
    #print(a.shape)

    opt = OptInit()
    model = DeepGCN(opt=opt)
    model.to(device)

    output = model(img_data)
    print("Output shape:", output.shape)
    index = torch.argmax(output)
    print('Predicted class:', index)
