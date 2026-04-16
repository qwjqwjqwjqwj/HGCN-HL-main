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
class HGCN(MessagePassing):#继承PyG 的消息传递基类，便于实现自定义图卷积
    def __init__(
        self,
        in_channels: int,#输入特征维度128
        out_channels: int,#输出特征维度128
        W:Tensor,
        bias: bool = True,#false
        W_learn = True,
        **kwargs,   #**kwargs 会收集所有未被显式定义的关键字参数，打包成一个字典
    ):
        kwargs.setdefault('aggr', 'add')#如果 kwargs 中没有 aggr 这个键，就设置默认值为 'add'
        super().__init__(flow='source_to_target', node_dim=0, **kwargs)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.W_e = W
        if W_learn == True:
            self.W_e = Parameter(torch.ones(W.shape[0]),requires_grad=True)#w_learn torch.Size([1062])创建一个全1的张量 requires_grad=True表示反向传播时会被更新
        
        self.lin = Linear(in_channels, out_channels, bias=False,weight_initializer='glorot')#output = input @ weight.T无偏置项，权重矩阵使用 Glorot 初始化方法进行初始化
        self.bias = bias
        self.W_learn = W_learn
        if self.bias == True:
            self.b = Parameter(torch.empty(out_channels))
    def forward(self, x: Tensor, hyperedge_index: Tensor ):#X(99600, 128) hyperedge_index.shape: torch.Size([2, 199200]) 
        print('x', x.shape , 'hyperedge_index', hyperedge_index.shape)
        num_nodes = x.size(0)#99600
        num_edges = int(hyperedge_index[1].max()) + 1#超边索引数量1062
        hyperedge_weight = x.new_ones(num_edges)#全1的张量，长度为超边数量1062 与x数据类型一致
        x = self.lin(x)#(99600, 128)线性变换
        
        D_v = scatter(hyperedge_weight[hyperedge_index[1]], hyperedge_index[0],##hyperedge_weight[hyperedge_index[1]]199200个1，hyperedge_index[0] 199200个节点索引，scatter函数根据节点索引将对应的权重值进行聚合，
                    dim=0, dim_size=num_nodes, reduce='sum')#99600个像素每个像素的度数 = 该像素在所有超边出现的数量
        D_v = D_v ** -0.5#归一化
        D_v[D_v == float("inf")] = 0#将无穷大值（inf）替换为0 处理节点度为0的情况
        

        D_e = scatter(x.new_ones(hyperedge_index.size(1)), hyperedge_index[1],
                    dim=0, dim_size=num_edges, reduce='sum')#每个超边的度数 = 该超边包含的像素数量
        D_e = 1.0 / D_e
        D_e[D_e == float("inf")] = 0#1

        out = self.propagate(hyperedge_index, x=x, norm=D_v, size=(num_nodes, num_edges))#每个超像素块（超边）里所有像素的特征“融合”起来，再“分发”回给每个像素
                                                                                            #把一个超像素里的所有像素的特征和求出来
        out = D_e.view(-1,1) * out#每个像素的特征乘以该像素的度数的-0.5次方进行归一化
        if self.W_learn == True:
            out = torch.mul(self.W_e.view(-1,1),out)#每个像素的特征乘以对应的可学习权重值进行调整
        else:
            out = self.W_e.mm(out)        
        
        out = self.propagate(hyperedge_index.flip([0]), x=out,  norm=D_e,size=(num_edges, num_nodes))#（99600）将超像素的特征进行“融合”后再“分发”回给每个像素 同一个超像素内的像素特征相同
        # print('out1',out)
        out = D_v.view(-1,1) * out#每个像素的特征乘以该像素的度数的-0.5次方进行归一化
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
                 output_dim=128, dropout=0.4):#net = HGCN_Network(height,width,bands=11,LiDAR_bands=1,class_num=6,W_e(1062 1062)的稀疏矩阵对角线全部为1,lam=0.5)

        super(HGCN_Network, self).__init__()
        self.height = height
        self.width = width
        self.in_dim = in_dim#11
        self.in_dim_Lidar = in_dim_Lidar#1
        self.class_num = class_num#6
        self.output_dim = output_dim#128
        self.W_e = W_e#(1062 1062)的稀疏矩阵对角线全部为1
        self.dropout = dropout#0.4
        self.lam = lam#0.5

        self.WMF_branch = nn.Sequential()
        self.WMF_branch.add_module('WMF_branch1',WMF(in_dim, output_dim))
        self.WMF_branch.add_module('WMF_branch2',WMF(output_dim, output_dim))

        self.HGCN1 = HGCN(in_channels=output_dim, out_channels=output_dim, W=W_e, bias=False)#128 128
        self.HGCN2 = HGCN(in_channels=output_dim, out_channels=output_dim, W=W_e, bias=False)

        self.CNN_Branch = nn.Sequential()
        self.CNN_Branch.add_module('CNN_Branch1',SSConv(output_dim, output_dim,kernel_size=5))
        self.CNN_Branch.add_module('CNN_Branch2',SSConv(output_dim, output_dim,kernel_size=5))

        self.relu = nn.LeakyReLU()
        self.dropout = nn.Dropout(dropout)
        self.Softmax_linear = nn.Linear(output_dim, class_num)
        self.lam = lam
        
    def forward(self, X,H):

        Z = torch.unsqueeze(X.permute([2, 0, 1]), 0)#       
        Z_WMF = self.WMF_branch(Z)#Z_WMF.shape: torch.Size([1, 128, 166, 600])
        #print("Z_WMF.shape:", Z_WMF.shape)
        
        Z_C = self.CNN_Branch(Z_WMF)
        Z_C = torch.squeeze(Z_C, 0).permute([1, 2, 0]).reshape([self.height * self.width, -1])#Z_C.shape: torch.Size([99600, 128])
        #print("Z_C.shape:", Z_C.shape)

        Z_G = torch.squeeze(Z_WMF, 0).permute([1, 2, 0])#Z_G.shape: torch.Size([166, 600, 128])
        Z_G = self.dropout(Z_G)#随机丢弃一些神经元的输出，以防止过拟合
        Z_G = self.relu(Z_G)#对输入进行非线性变换，增加模型的表达能力
        Z_G = torch.reshape(Z_G,[self.height*self.width,Z_G.shape[2]])#将Z_G的形状调整为 (99600, 128)，以便后续的图卷积操作
        #print("Z_G.shape:", Z_G.shape)
        # x = self.batch_normalzation1(x)
        Z_G = self.HGCN1(Z_G, H)
        Z_G = self.dropout(Z_G)
        Z_G = self.relu(Z_G)
        
        # x = self.batch_normalzation2(x)
        Z_G = self.HGCN2(Z_G, H)#Z_G.shape: torch.Size([99600, 128])
        Z_G = self.dropout(Z_G)
        Z_G = self.relu(Z_G)
        #print("Z_G.shape:", Z_G.shape)
        
        # x = x*(torch.exp(self.lambda1)) + Z_CNN*(1-torch.exp(self.lambda1))
        Z = self.lam*Z_G + Z_C*(1-self.lam)#特征融合相加
        # x = torch.cat((x, Z_CNN), 1)
        out = self.Softmax_linear(Z)#99600 6预测综合结果
        return out
    
    
    
#main
    
FLAG = 3 # 1:MUUFL, 2: Houston2013, 3:Trento, 4: Augsburg
DataName = "MUUFL"
max_iters = 2
max_epochs = 10  #1000
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
H = obtain_H_from_HSI_with_LiDAR(X_HSI,X_LiDAR,scales)#H.shape: (99600, 1062)
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

# 将 NumPy 数组转换为 PyTorch 张量
X = torch.from_numpy(X).to(device)
X = X.type(torch.float32)

# 初始化边权矩阵，该参数现有系统为可学习参数，也可设置为固定参数
W_e = torch.diag(torch.ones(H.shape[1])).to(device)#(1062 1062)的对角矩阵，初始值为1
W_e = W_e.to_sparse_coo()#将稠密矩阵转换为稀疏 COO 格式，非零元素的值和位置被存储在两个张量中：values 和 indices。values 张量包含非零元素的值，indices 张量包含这些元素在原始矩阵中的位置（行索引和列索引）
# 将H矩阵转换为边索引的形式
idx = np.where(H.T == 1)
edge_index = np.stack([idx[1],idx[0]],axis=0)
H = torch.tensor(edge_index,dtype=torch.long).to(device)

TVT_sets = CTrainValTest_Sets()
TVT_sets.get_TrainValTest_Sets(4242,gt,class_num,0.1,0.1,'ratio')    

# 使用已存在的训练和测试样本为了保证和其它论文一致
if FLAG == 3:
    TRLabel = sio.loadmat('data\Trento\TrLabel.mat')['TRLabel']#预定义的训练样本标签
    TELabel = sio.loadmat('data\Trento\TSLabel.mat')['TSLabel']##预定义的测试样本标签
    
elif FLAG == 2:
    TRLabel = sio.loadmat('data\Houston2013\TrLabel.mat')['Data']
    TELabel = sio.loadmat('data\Houston2013\TeLabel.mat')['Data']
    
elif FLAG == 1:
    TRLabel = sio.loadmat('data\MUUFL\TrLabel.mat')['Data']
    TELabel = sio.loadmat('data\MUUFL\TeLabel.mat')['Data']
    
elif FLAG == 4:
    TRLabel = sio.loadmat('data\Augsburg\TrLabel.mat')['Data']
    TELabel = sio.loadmat('data\Augsburg\TeLabel.mat')['Data']

TELabel_flatten = np.reshape(TELabel,[-1])#数据是二维的，所以将其展平为一维数组，方便后续处理
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
oa_lambda = []
oastd_lambda = []
print("节点数：",X.shape)
print("超边数：",H.shape[1])
for lam in lambdas:
    
    OA_ALL = []
    AA_ALL = []
    KPP_ALL = []
    AVG_ALL = []
    Train_Time_ALL=[]
    Test_Time_ALL=[]
    seed = 1
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    for id in range(max_iters):
        # print("实验：",id)
        resut_train = CResult()
        resut_val = CResult()
        resut_test = CResult()
        #######################
        net = HGCN_Network(height,width,bands,LiDAR_bands,class_num,W_e,lam)
        net.to(device)
        # loss object
        cal_loss = CLoss().to(device)
        learning_rate = learning_rate
        optimizer = torch.optim.Adam(net.parameters(),lr=learning_rate,weight_decay=weight_decay)
        net.train()    
        
        best_loss = 0.5
        tic_train = time.time()
        for i in range(max_epochs):
            # network forward
            optimizer.zero_grad()
            output= net(X_torch,H)# 预测结果（99600，6）        X_torch (166 100 11) H.shape: torch.Size([2, 199200])
            output_train = output[TVT_sets.train_data_index]#取出训练集的预测结果
            loss = cal_loss(output_train, TVT_sets.train_gt)#计算损失
            loss.backward()
            optimizer.step()  # Does the update
            if i%10 == 0:
                with torch.no_grad():
                    net.eval()
                    output= net(X_torch,H)
                    
                    output_train = output[TVT_sets.train_data_index]
                    loss_train = cal_loss(output_train, TVT_sets.train_gt)
                    resut_train.get_permance(TVT_sets.train_gt, output_train)
                    resut_train.lossvalue = loss_train.item()
                
                    output_val = output[TVT_sets.test_data_index]
                    loss_val = cal_loss(output_val, TVT_sets.test_gt)
                    resut_val.get_permance(TVT_sets.test_gt, output_val)
                    resut_val.lossvalue = loss_val.item()
                    if   resut_val.oa > best_loss:
                        best_loss = resut_val.oa
                        torch.save(net.state_dict(),"model\\"+str(DataName)+str(id)+"model.pt")
                torch.cuda.empty_cache()
                net.train()
                toc1 = time.time()
        
        toc_train = time.time()
        Train_Time_ALL.append(toc_train-tic_train)
        torch.cuda.empty_cache()
        
        with torch.no_grad():
            try:
                net.load_state_dict(torch.load("model\\"+str(DataName)+str(id)+"model.pt"))
                net.eval()
                tic_test = time.time()
                output= net(X_torch,H)
                toc_test = time.time()
                Test_Time_ALL.append(toc_test-tic_test)
                
                # performance for classification
                output_test = output[TVT_sets.test_data_index]
                # loss_val = 0 #cal_loss(output_test, TVT_sets.test_gt)
                loss_val = cal_loss(output_test, TVT_sets.test_gt)
                # testgtnumpy = TVT_sets.test_gt.detach().cpu().numpy()
                resut_test.get_permance(TVT_sets.test_gt, output_test)
                print("test_loss:{} \toa_test:{}".format(loss_val, resut_test.oa))
                
                # testOA, testAA, testkappa
                OA_ALL.append(resut_test.oa)
                AA_ALL.append(resut_test.aa)
                KPP_ALL.append(resut_test.kappa)
                AVG_ALL.append(resut_test.acc_perclass)
                # print(net.HGCN1.W_e)  
                
            except FileNotFoundError:
                print(f"模型文件不存在，跳过测试")          
        torch.cuda.empty_cache()
        # del net
    OA_ALL = np.array(OA_ALL)
    AA_ALL = np.array(AA_ALL)
    KPP_ALL = np.array(KPP_ALL)
    AVG_ALL = np.array(AVG_ALL)
    Train_Time_ALL=np.array(Train_Time_ALL)
    Test_Time_ALL=np.array(Test_Time_ALL)

    print("\ntrain_ratio={}".format(1),
            "\n==============================================")
    print('OA=', np.mean(OA_ALL), '+-', np.std(OA_ALL))
    print('AA=', np.mean(AA_ALL), '+-', np.std(AA_ALL))
    print('Kpp=', np.mean(KPP_ALL), '+-', np.std(KPP_ALL))
    print('AVG=', np.mean(AVG_ALL, 0), '+-', np.std(AVG_ALL, 0))
    print("Average training time:{}".format(np.mean(Train_Time_ALL)))
    print("Average testing time:{}".format(np.mean(Test_Time_ALL)))
    oa_lambda.append(np.mean(OA_ALL))
    oastd_lambda.append(np.std(OA_ALL))
    