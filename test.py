import torch
from utils import *
# from src.hgcn_modules import HGCN_Network
import time

##更新测试


import torch
import torch.nn as nn
from torch import Tensor
from torch.nn import Parameter
from torch_scatter import scatter
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.dense.linear import Linear
class HGCN(MessagePassing):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        W:Tensor,
        bias: bool = True,
        W_learn = True,
        **kwargs,#**kwargs 会收集 所有未被显式定义的关键字参数
    ):
        kwargs.setdefault('aggr', 'add')
        super().__init__(flow='source_to_target', node_dim=0, **kwargs)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.W_e = W
        if W_learn == True:
            self.W_e = Parameter(torch.ones(W.shape[0]),requires_grad=True)
        self.lin = Linear(in_channels, out_channels, bias=False,
                              weight_initializer='glorot')
        self.bias = bias
        self.W_learn = W_learn
        if self.bias == True:
            self.b = Parameter(torch.empty(out_channels))

    def forward(self, x: Tensor, hyperedge_index: Tensor):
        num_nodes = x.size(0)
        num_edges = int(hyperedge_index[1].max()) + 1
        hyperedge_weight = x.new_ones(num_edges)
        x = self.lin(x)
        D_v = scatter(hyperedge_weight[hyperedge_index[1]], hyperedge_index[0],
                    dim=0, dim_size=num_nodes, reduce='sum')
        D_v = D_v ** -0.5
        D_v[D_v == float("inf")] = 0
        # print('D_v',D_v)
        D_e = scatter(x.new_ones(hyperedge_index.size(1)), hyperedge_index[1],
                    dim=0, dim_size=num_edges, reduce='sum')
        D_e = 1.0 / D_e
        D_e[D_e == float("inf")] = 0

        out = self.propagate(hyperedge_index, x=x, norm=D_v, size=(num_nodes, num_edges))
        out = D_e.view(-1,1) * out
        if self.W_learn == True:
            out = torch.mul(self.W_e.view(-1,1),out)
        else:
            out = self.W_e.mm(out)        
        
        out = self.propagate(hyperedge_index.flip([0]), x=out,  size=(num_edges, num_nodes))
        # print('out1',out)
        out = D_v.view(-1,1) * out
        # print('out2',out)
        if self.bias == True:
            out = out + self.b
        return out

    def message(self, x_j: Tensor, norm_j: Tensor) -> Tensor:
        if isinstance(norm_j, Tensor):
            out = norm_j.view(-1,1) * x_j
        else:
            out = x_j
        return out

class WMF(nn.Module):
    """
    The WMF class implements a simple neural network module that includes batch normalization, a 1x1 convolution, and a LeakyReLU activation function.

    Args:
        in_dim (int): The number of input channels of the feature map.
        out_dim (int): The number of output channels of the feature map.
    """
    def __init__(self, in_dim: int, out_dim: int):
        # Call the constructor of the parent class nn.Module
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        
        # Define a 2D batch normalization layer with in_dim input channels
        self.BN = nn.BatchNorm2d(in_dim)
        # Define a 1x1 2D convolution layer with in_dim input channels and out_dim output channels
        self.conv = nn.Conv2d(in_dim, out_dim, kernel_size=(1, 1))
        # Fix the spelling error and define a LeakyReLU activation function layer
        self.activation = nn.LeakyReLU()

    def forward(self, X):
        """
        Forward propagation method that defines the flow of data through the module.

        Args:
            X (torch.Tensor): The input feature map.

        Returns:
            torch.Tensor: The processed feature map.
        """
        # Apply batch normalization to the input data
        X = self.BN(X)
        # Apply a 1x1 convolution operation to the normalized data
        X = self.conv(X)
        # Apply the LeakyReLU activation function to the convolution result
        X = self.activation(X)
        return X

class SSConv(nn.Module):
    '''
    Spectral-Spatial Convolution
    '''
    def __init__(self, in_ch, out_ch, kernel_size=3):
        super(SSConv, self).__init__()
        # Point-wise convolution to transform input channels to output channels
        self.point_conv = nn.Conv2d(
            in_channels=in_ch,
            out_channels=out_ch,
            kernel_size=1,
            stride=1,
            padding=0,
            groups=1,
            bias=False
        )
        # Initialize point-wise convolution weights
        nn.init.kaiming_normal_(self.point_conv.weight, mode='fan_out', nonlinearity='leaky_relu')

        # Depth-wise convolution to extract spatial features
        self.depth_conv = nn.Conv2d(
            in_channels=out_ch,
            out_channels=out_ch,
            kernel_size=kernel_size,
            stride=1,
            padding=kernel_size//2,
            groups=out_ch
        )
        # Initialize depth-wise convolution weights
        nn.init.kaiming_normal_(self.depth_conv.weight, mode='fan_out', nonlinearity='leaky_relu')

        # Activation functions for point-wise and depth-wise convolutions
        self.pointwise_activation = nn.LeakyReLU()
        self.depthwise_activation = nn.LeakyReLU()

        # Batch normalization layer for input
        self.BN = nn.BatchNorm2d(in_ch)

    def forward(self, input):
        # Apply batch normalization to the input
        normalized_input = self.BN(input)
        # Apply point-wise convolution
        pointwise_output = self.point_conv(normalized_input)
        # Apply the first activation function
        activated_pointwise = self.pointwise_activation(pointwise_output)
        # Apply depth-wise convolution
        depthwise_output = self.depth_conv(activated_pointwise)
        # Apply the second activation function
        final_output = self.depthwise_activation(depthwise_output)
        return final_output

class HGCN_Network(torch.nn.Module):
    def __init__(self, height: int, width: int, in_dim: int,in_dim_Lidar:int, class_num: int,W_e: torch.tensor,lam: float,
                 output_dim=128, dropout=0.4):

        super(HGCN_Network, self).__init__()
        self.height = height
        self.width = width
        self.in_dim = in_dim
        self.in_dim_Lidar = in_dim_Lidar
        self.class_num = class_num
        self.output_dim = output_dim
        self.W_e = W_e
        self.dropout = dropout
        self.lam = lam

        self.WMF_branch = nn.Sequential()
        self.WMF_branch.add_module('WMF_branch1',WMF(in_dim, output_dim))
        self.WMF_branch.add_module('WMF_branch2',WMF(output_dim, output_dim))

        self.HGCN1 = HGCN(in_channels=output_dim, out_channels=output_dim, W=W_e, bias=False)
        self.HGCN2 = HGCN(in_channels=output_dim, out_channels=output_dim, W=W_e, bias=False)

        self.CNN_Branch = nn.Sequential()
        self.CNN_Branch.add_module('CNN_Branch1',SSConv(output_dim, output_dim,kernel_size=5))
        self.CNN_Branch.add_module('CNN_Branch2',SSConv(output_dim, output_dim,kernel_size=5))

        self.relu = nn.LeakyReLU()
        self.dropout = nn.Dropout(dropout)
        self.Softmax_linear = nn.Linear(output_dim, class_num)
        self.lam = lam
        
    def forward(self, X,H):

        Z = torch.unsqueeze(X.permute([2, 0, 1]), 0)        
        Z_WMF = self.WMF_branch(Z)
        Z_C = self.CNN_Branch(Z_WMF)
        Z_C = torch.squeeze(Z_C, 0).permute([1, 2, 0]).reshape([self.height * self.width, -1])

        Z_G = torch.squeeze(Z_WMF, 0).permute([1, 2, 0])
        Z_G = self.dropout(Z_G)
        Z_G = self.relu(Z_G)
        Z_G = torch.reshape(Z_G,[self.height*self.width,Z_G.shape[2]])
        # x = self.batch_normalzation1(x)
        Z_G = self.HGCN1(Z_G, H)
        Z_G = self.dropout(Z_G)
        Z_G = self.relu(Z_G)
        # x = self.batch_normalzation2(x)
        Z_G = self.HGCN2(Z_G, H)

        Z_G = self.dropout(Z_G)
        Z_G = self.relu(Z_G)
        # x = x*(torch.exp(self.lambda1)) + Z_CNN*(1-torch.exp(self.lambda1))
        Z = self.lam*Z_G + Z_C*(1-self.lam)
        # x = torch.cat((x, Z_CNN), 1)
        out = self.Softmax_linear(Z)
        return out
    
    
    
#main
    
    
FLAG = 3 # 1:MUUFL, 2: Houston2013, 3:Trento, 4: Augsburg
DataName = "MUUFL"
max_iters = 2
max_epochs = 2  #1000
learning_rate = 0.0001
weight_decay = 0.00001
scales = [30]    
k = 1 #如果k=0 使用计算权重
pca_n = 15
sigma1 = 0.1
sigma2 = 0.1
if FLAG == 3:
    DataName = "Trento"
    pca_n = 10 #15
    lambdas = [0.5] 
    scales = [100]
elif FLAG == 1:
    DataName = "MUUFL"
    pca_n = 10 #30      
    lambdas = [0.5]    
    scales = [100]  
elif FLAG == 2:
    DataName = "Houston2013"
    pca_n = 25   #30
    lambdas = [0.5]    
    scales = [300]  
elif FLAG == 4:
    pca_n = 10   
    lambdas = [0.5]    
    scales = [100] 

    
FLAG = 3 # 1:MUUFL, 2: Houston2013, 3:Trento, 4: Augsburg
DataName = "MUUFL"
max_iters = 2
max_epochs = 2  #1000
learning_rate = 0.0001
weight_decay = 0.00001
scales = [30]    
k = 1 #如果k=0 使用计算权重
pca_n = 15
sigma1 = 0.1
sigma2 = 0.1
if FLAG == 3:
    DataName = "Trento"
    pca_n = 10 #15
    lambdas = [0.5] 
    scales = [100]
elif FLAG == 1:
    DataName = "MUUFL"
    pca_n = 10 #30      
    lambdas = [0.5]    
    scales = [100]  
elif FLAG == 2:
    DataName = "Houston2013"
    pca_n = 25   #30
    lambdas = [0.5]    
    scales = [300]  
elif FLAG == 4:
    pca_n = 10   
    lambdas = [0.5]    
    scales = [100] 

(X_HSI,X_LiDAR, gt, class_num) = get_dataset(FLAG)  
#X_HSI (166 100 63)  X_LiDAR (166 100)       gt 地面真值标签（每个像素点代表的地物标签  整数表示）(166 100)   class_num 6（6类地物）
# no
X_HSI = (X_HSI - float(np.min(X_HSI)))     # 平移：减去最小值
X_HSI = X_HSI/(np.max(X_HSI)-np.min(X_HSI)) # 缩放：除以极差 所有数值在1以内 （min-max归一化）
X_LiDAR = (X_LiDAR - float(np.min(X_LiDAR)))
X_LiDAR = X_LiDAR/(np.max(X_LiDAR)-np.min(X_LiDAR))
X_LiDAR = X_LiDAR[:,:, np.newaxis]#为 LiDAR 数据增加一个维度，形状从 (height, width) 变为 (height, width, 1)
#
H = obtain_H_from_HSI_with_LiDAR(X_HSI,X_LiDAR,scales)
print(H.shape)
LiDAR_bands = 1
gt_flatten = np.reshape(gt, -1)
h,w = X_HSI.shape[0],X_HSI.shape[1]
X_HSI = np.reshape(X_HSI,(-1,X_HSI.shape[2]))

pca = PCA(n_components=pca_n)  # 指定要保留的主成分数量
X_HSI = pca.fit_transform(X_HSI)
X_HSI = np.reshape(X_HSI, (h, w, -1))
(height, width, bands_hsi) = X_HSI.shape
X = np.concatenate([X_HSI,X_LiDAR],axis=2)#166 100 11
bands = bands_hsi+1
print(X.shape)


# 将 NumPy 数组转换为 PyTorch 张量
X = torch.from_numpy(X).to(device)
X = X.type(torch.float32)

# 初始化边权矩阵，该参数现有系统为可学习参数，也可设置为固定参数
W_e = torch.diag(torch.ones(H.shape[1])).to(device)
W_e = W_e.to_sparse_coo()
# 将H矩阵转换为边索引的形式
idx = np.where(H.T == 1)
edge_index = np.stack([idx[1],idx[0]],axis=0)
H = torch.tensor(edge_index,dtype=torch.long).to(device)
print("H.shape:", H.shape)


TVT_sets = CTrainValTest_Sets()
TVT_sets.get_TrainValTest_Sets(4242,gt,class_num,0.1,0.1,'ratio')    

# 使用已存在的训练和测试样本为了保证和其它论文一致
if FLAG == 3:
    TRLabel = sio.loadmat('data\Trento\TrLabel.mat')['TRLabel']
    TELabel = sio.loadmat('data\Trento\TSLabel.mat')['TSLabel']
    
elif FLAG == 2:
    TRLabel = sio.loadmat('data\Houston2013\TrLabel.mat')['Data']
    TELabel = sio.loadmat('data\Houston2013\TeLabel.mat')['Data']
    
elif FLAG == 1:
    TRLabel = sio.loadmat('data\MUUFL\TrLabel.mat')['Data']
    TELabel = sio.loadmat('data\MUUFL\TeLabel.mat')['Data']
    
elif FLAG == 4:
    TRLabel = sio.loadmat('data\Augsburg\TrLabel.mat')['Data']
    TELabel = sio.loadmat('data\Augsburg\TeLabel.mat')['Data']

TELabel_flatten = np.reshape(TELabel,[-1])
TRLabel_flatten = np.reshape(TRLabel,[-1])
idx = np.where(TRLabel_flatten!=0)
TVT_sets.train_data_index = idx[0]
TVT_sets.train_gt = torch.tensor(gt_flatten[idx[0]], dtype=torch.long).to(device) - 1

TELabel_flatten = np.reshape(TELabel,[-1])
idx = np.where(TELabel_flatten!=0)
TVT_sets.test_data_index = idx[0]
TVT_sets.test_gt = torch.tensor(gt_flatten[idx[0]], dtype=torch.long).to(device) - 1
TVT_sets.val_data_index = TVT_sets.test_data_index
TVT_sets.val_gt = TVT_sets.test_gt







OA_ALL = []
AA_ALL = []
KPP_ALL = []
AVG_ALL = []
Train_Time_ALL=[]
Test_Time_ALL=[]

torch.cuda.empty_cache()
W_e = W_e.type(torch.float32)
X_torch = X.type(torch.float32)

print("X_torch.shape:", X_torch.shape)

net = HGCN_Network(height,width,bands,LiDAR_bands,class_num,W_e,0.5)
net.to(device)