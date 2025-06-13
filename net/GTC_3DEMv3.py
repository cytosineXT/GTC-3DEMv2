# GTC-3DEMv3.py (集成物理先验)
# 版本 3: 在v2的基础上，将亥姆霍兹方程和频域带限作为物理先验正则项，直接加入训练的损失函数中。
# 核心思想：借鉴PINN，让网络在学习数据拟合的同时，必须遵守基本的波动物理规律，
# 从而在稀疏数据下也能获得更好的泛化能力和物理自洽性。

from torch.nn import Module, ModuleList
import torch 
from torch import nn
from torch_geometric.nn.conv import SAGEConv
import torch.nn.functional as F
import numpy as np
from functools import partial
from net.utils import derive_face_edges_from_faces, transform_to_log_coordinates, psnr, batch_mse
from net.utils import ssim as myssim
from pytorch_msssim import ms_ssim, ssim
from einops import rearrange, pack
from math import pi

# --- 物理先验损失函数 ---

def calculate_helmholtz_loss(rcs_pred, freqs_ghz, device):
    """
    计算亥姆霍兹方程残差损失。
    要求生成的RCS图在物理上是自洽的。
    """
    # 1. 定义离散拉普拉斯卷积核
    laplacian_kernel = torch.tensor([[0, 1, 0], [1, -4, 1], [0, 1, 0]], 
                                    dtype=torch.float32, device=device).unsqueeze(0).unsqueeze(0)

    # 2. 对每个批次的RCS图计算拉普拉斯算子
    # rcs_pred 形状: [batch, height, width] -> [batch, 1, height, width]
    rcs_pred_batch = rcs_pred.unsqueeze(1)
    laplacian_of_rcs = F.conv2d(rcs_pred_batch, laplacian_kernel, padding=1)

    # 3. 计算波数 k
    c = 299792458.0  # 光速 m/s
    # freqs_ghz 形状: [batch] -> [batch, 1, 1, 1] 以便广播
    k = (2 * pi * (freqs_ghz * 1e9) / c).view(-1, 1, 1, 1).to(device)
    k_squared = k.pow(2)

    # 4. 计算亥姆霍兹残差
    helmholtz_residual = laplacian_of_rcs + k_squared * rcs_pred_batch
    
    # 5. 损失是残差的均方误差
    loss = torch.mean(helmholtz_residual.pow(2))
    return torch.log1p(loss) #tensor(18933.2702, dtype=torch.float64, grad_fn=<MeanBackward0>) log后tensor(9.8487, dtype=torch.float64, grad_fn=<Log1PBackward0>)

def calculate_bandlimit_loss(rcs_pred, freqs_ghz, device, alpha=10.0):
    """
    计算频域带限损失。
    惩罚那些与给定频率不匹配的、非物理的高频空间细节。
    """
    # 1. 对每个批次的RCS图计算2D FFT
    # rcs_pred 形状: [batch, height, width]
    fft_pred = torch.fft.fftshift(torch.fft.fft2(rcs_pred, norm='ortho'), dim=(-2, -1))
    
    # 2. 定义非物理区域的掩码
    h, w = rcs_pred.shape[-2:]
    center_h, center_w = h // 2, w // 2
    
    # 计算波数和截止半径
    c = 299792458.0
    k = (2 * pi * (freqs_ghz * 1e9) / c).to(device)
    cutoff_radii = (alpha * k).int() # 每个样本可能有不同的截止半径
    
    # 创建网格
    y, x = np.ogrid[-center_h:h-center_h, -center_w:w-center_w]
    radius_grid = torch.from_numpy(np.sqrt(x*x + y*y)).to(device)

    # 为批次中的每个样本创建其对应的掩码
    # cutoff_radii: [batch] -> [batch, 1, 1]
    # radius_grid: [h, w] -> [1, h, w]
    non_physical_mask = (radius_grid.unsqueeze(0) > cutoff_radii.view(-1, 1, 1)).float()
    
    # 3. 计算在非物理区的能量
    non_physical_energy = fft_pred.abs().pow(2) * non_physical_mask
    
    # 4. 损失是该区域的平均能量
    loss = non_physical_energy.mean()
    return loss


# --- 修改后的复合损失函数 ---
def loss_fn(decoded, GT, freqs_ghz, device, loss_type='L1', gama=0.001, lambda_helmholtz=0.1, lambda_bandlimit=0.1):
    """
    计算一个复合损失，包含监督损失和物理先验损失。
    """
    # 1. 计算监督损失 (与v2相同)
    maxloss = torch.mean(torch.abs(torch.amax(decoded, dim=(1, 2)) - torch.amax(GT, dim=(1, 2))))
    l1 = F.l1_loss(decoded, GT)
    mse = F.mse_loss(decoded, GT)
    
    if loss_type == 'L1':
        supervision_loss = l1
    elif loss_type == 'mse':
        supervision_loss = mse
    else: # 默认为L1
        supervision_loss = l1
        
    primary_loss = supervision_loss + gama * maxloss
    
    # 2. 计算物理先验损失
    helmholtz_loss = calculate_helmholtz_loss(decoded, freqs_ghz, device) #tensor(18933.2702, dtype=torch.float64, grad_fn=<MeanBackward0>)
    bandlimit_loss = calculate_bandlimit_loss(decoded, freqs_ghz, device) #tensor(0.1101, grad_fn=<MeanBackward0>)
    
    # 3. 加权求和得到总损失
    total_loss = primary_loss + lambda_helmholtz * helmholtz_loss + lambda_bandlimit * bandlimit_loss
    
    # 返回总损失以及各个子项，方便监控
    return total_loss, primary_loss, helmholtz_loss, bandlimit_loss
    # maxloss=1.8444 l1=0.4883 helLoss=9.8487 bandLoss=0.1101 
    # 0.001 1 0.001 0
    #         0.0001 0
    #         0 0.01

# --- 原有代码（保持不变）---
def l2norm(t):
    return F.normalize(t, dim = -1, p = 2)

def jxtget_face_coords(vertices, face_indices):
    batch_size, num_faces, num_vertices_per_face = face_indices.shape
    reshaped_face_indices = face_indices.reshape(batch_size, -1).to(dtype=torch.int64) 
    face_coords = torch.gather(vertices, 1, reshaped_face_indices.unsqueeze(-1).expand(-1, -1, vertices.shape[-1]))
    face_coords = face_coords.reshape(batch_size, num_faces, num_vertices_per_face, -1)
    return face_coords

def coords_interanglejxt2(x, y, eps=1e-5):
    edge_vector = x - y
    normv = l2norm(edge_vector)
    normdot = -(normv * torch.cat((normv[..., -1:], normv[..., :-1]), dim=3)).sum(dim=2)
    normdot = torch.clamp(normdot, -1 + eps, 1 - eps)
    radians = torch.acos(normdot)
    angle = torch.rad2deg(radians)
    return radians, angle

def polar_to_cartesian2(theta, phi):
    theta_rad = torch.deg2rad(theta)
    phi_rad = torch.deg2rad(phi)
    x = torch.sin(phi_rad) * torch.cos(theta_rad)
    y = torch.sin(phi_rad) * torch.sin(theta_rad)
    z = torch.cos(phi_rad)
    return torch.stack([x, y, z], dim=1)

def vector_anglejxt2(x, y, eps=1e-5):
    normdot = -(l2norm(x) * l2norm(y)).sum(dim=-1)
    normdot = torch.clamp(normdot, -1 + eps, 1 - eps)
    radians = normdot.acos()
    angle = torch.rad2deg(radians)
    return radians, angle

def get_derived_face_featuresjxt(face_coords, in_em, device):
    shifted_face_coords = torch.cat((face_coords[:, :, -1:], face_coords[:, :, :-1]), dim = 2).to(device)
    angles, _  = coords_interanglejxt2(face_coords, shifted_face_coords)
    edge1, edge2, *_ = (face_coords - shifted_face_coords).unbind(dim = 2)
    normals = l2norm(torch.cross(edge1, edge2, dim = -1))
    area = torch.cross(edge1, edge2, dim = -1).norm(dim = -1, keepdim = True) * 0.5
    incident_angle_vec = polar_to_cartesian2(in_em[1],in_em[2])
    incident_angle_mtx = incident_angle_vec.unsqueeze(1).repeat(1, area.shape[1], 1).to(device)
    incident_freq_mtx = in_em[3].float().unsqueeze(1).unsqueeze(2).repeat(1, area.shape[1], 1).to(device)
    incident_mesh_anglehudu, _ = vector_anglejxt2(normals, incident_angle_mtx)
    return dict(
        angles = angles, area = area, normals = normals,
        emnoangle = incident_mesh_anglehudu.unsqueeze(-1),
        emangle = incident_angle_mtx, emfreq = incident_freq_mtx
    )

class MeshCodec(Module):
    def __init__(
            self,
            device,
            attn_encoder_depth=0,
            middim=64,
            attn_dropout=0.,
            dim_coor_embed = 64,        
            dim_area_embed = 16,        
            dim_normal_embed = 64,      
            dim_angle_embed = 16,       
            dim_emnoangle_embed = 16,   
            dim_emangle_embed = 64,     
            dim_emfreq_embed = 16,      
            encoder_dims_through_depth = (64, 128, 256, 256, 576),
            **kwargs # 接收额外的损失权重参数
            ):
        super().__init__()
        
        # --- 存储物理损失的权重 ---
        self.lambda_helmholtz = kwargs.get('lambda_helmholtz', 0.1)
        self.lambda_bandlimit = kwargs.get('lambda_bandlimit', 0.1)

        #---Conditioning
        self.condfreqlayers = ModuleList([
            nn.Linear(1, 64), nn.Linear(1, 128), nn.Linear(1, 256), nn.Linear(1, 256),])
        self.condanglelayers = ModuleList([
            nn.Linear(2, 64), nn.Linear(2, 128), nn.Linear(2, 256), nn.Linear(2, 256),])
        self.incident_angle_linear1 = nn.Linear(2, 2250)
        self.emfreq_embed1 = nn.Linear(1, 2250)
        self.incident_angle_linear2 = nn.Linear(2, 4050)
        self.emfreq_embed2 = nn.Linear(1, 4050)
        self.incident_angle_linear3 = nn.Linear(2, 90*180)
        self.emfreq_embed3 = nn.Linear(1, 90*180)
        self.incident_angle_linear4 = nn.Linear(2, 180*360)
        self.emfreq_embed4 = nn.Linear(1, 180*360)
        self.incident_angle_linear5 = nn.Linear(2, 360*720)
        self.emfreq_embed5 = nn.Linear(1, 360*720)

        #---Encoder
        self.angle_embed = nn.Linear(3, 3*dim_angle_embed)
        self.area_embed = nn.Linear(1, dim_area_embed)
        self.normal_embed = nn.Linear(3, 3*dim_normal_embed)
        self.emnoangle_embed = nn.Linear(1, dim_emnoangle_embed)
        self.emangle_embed = nn.Linear(3, 3*dim_emangle_embed)
        self.emfreq_embed = nn.Linear(1, dim_emfreq_embed)
        self.coor_embed = nn.Linear(9, 9*dim_coor_embed) 

        init_dim = dim_coor_embed * 9 + dim_angle_embed * 3 + dim_normal_embed * 3 + dim_area_embed + dim_emangle_embed * 3 + dim_emnoangle_embed + dim_emfreq_embed
        sageconv_kwargs = dict(normalize=True, project=True)
        init_encoder_dim, *encoder_dims_through_depth = encoder_dims_through_depth
        curr_dim = init_encoder_dim
        self.init_sage_conv = SAGEConv(init_dim, init_encoder_dim, **sageconv_kwargs)
        self.init_encoder_act_and_norm = nn.Sequential(nn.SiLU(), nn.LayerNorm(init_encoder_dim))
        self.encoders = ModuleList([])
        self.encoder_act_and_norm = ModuleList([])
        for dim_layer in encoder_dims_through_depth:
            sage_conv = SAGEConv(curr_dim, dim_layer, **sageconv_kwargs)
            self.encoders.append(sage_conv)
            self.encoder_act_and_norm.append(nn.Sequential(nn.SiLU(), nn.LayerNorm(dim_layer)))
            curr_dim = dim_layer
        self.encoder_attn_blocks = nn.ModuleList([
            nn.TransformerEncoderLayer(d_model=curr_dim, nhead=8, dropout=attn_dropout)
            for _ in range(attn_encoder_depth)
        ])

        #---Adaptation Module
        self.conv1d1 = nn.Conv1d(576, middim, kernel_size=10, stride=10, dilation=1, padding=0)
        self.fc1d1 = nn.Linear(2250, 45*90)

        #---Decoder
        self.upconv1 = nn.ConvTranspose2d(middim, middim//2, kernel_size=2, stride=2)
        self.bn1 = nn.BatchNorm2d(middim//2); self.conv1_1 = nn.Conv2d(middim//2, middim//2, 3, 1, 1); self.conv1_2 = nn.Conv2d(middim//2, middim//2, 3, 1, 1)
        self.bn1_1 = nn.BatchNorm2d(middim//2); self.bn1_2 = nn.BatchNorm2d(middim//2)
        self.upconv2 = nn.ConvTranspose2d(middim//2, middim//4, kernel_size=2, stride=2)
        self.bn2 = nn.BatchNorm2d(middim//4); self.conv2_1 = nn.Conv2d(middim//4, middim//4, 3, 1, 1); self.conv2_2 = nn.Conv2d(middim//4, middim//4, 3, 1, 1)
        self.bn2_1 = nn.BatchNorm2d(middim//4); self.bn2_2 = nn.BatchNorm2d(middim//4)
        self.upconv3 = nn.ConvTranspose2d(middim//4, middim//8, kernel_size=2, stride=2)
        self.bn3 = nn.BatchNorm2d(middim//8); self.conv3_1 = nn.Conv2d(middim//8, middim//8, 3, 1, 1); self.conv3_2 = nn.Conv2d(middim//8, middim//8, 3, 1, 1)
        self.bn3_1 = nn.BatchNorm2d(middim//8); self.bn3_2 = nn.BatchNorm2d(middim//8)
        self.conv1x1 = nn.Conv2d(middim//8, 1, kernel_size=1, stride=1, padding=0)

    def encode(self, *, vertices, faces, face_edges, in_em):
        device = vertices.device 
        original_freqs = in_em[3].clone() # 保存原始频率
        in_em[3] = transform_to_log_coordinates(in_em[3])
        face_coords = jxtget_face_coords(vertices, faces)
        derived_features = get_derived_face_featuresjxt(face_coords, in_em, device)
        
        angle_embed = self.angle_embed(derived_features['angles'])
        area_embed = self.area_embed(derived_features['area'])
        normal_embed = self.normal_embed(derived_features['normals'])
        emnoangle_embed = self.emnoangle_embed(derived_features['emnoangle'])
        emangle_embed = self.emangle_embed(derived_features['emangle'])
        emfreq_embed = self.emfreq_embed(derived_features['emfreq'])
        face_coords_re = rearrange(face_coords, 'b nf nv c -> b nf (nv c)')
        face_coor_embed = self.coor_embed(face_coords_re)

        face_embed, _ = pack([face_coor_embed, angle_embed, area_embed, normal_embed, emnoangle_embed, emangle_embed, emfreq_embed], 'b nf *') 

        face_edges = face_edges.reshape(2, -1).to(device)
        orig_face_embed_shape = face_embed.shape[:2]
        face_embed = face_embed.reshape(-1, face_embed.shape[-1])
        face_embed = self.init_sage_conv(face_embed, face_edges)
        face_embed = self.init_encoder_act_and_norm(face_embed)
        face_embed = face_embed.reshape(orig_face_embed_shape[0], orig_face_embed_shape[1], -1)

        in_angle = torch.stack([in_em[1]/180, in_em[2]/360]).t().float().unsqueeze(1).to(device)
        in_freq = in_em[3].float().unsqueeze(1).unsqueeze(1).to(device)

        for i, (conv, act_norm) in enumerate(zip(self.encoders, self.encoder_act_and_norm)):
            condfreq = self.condfreqlayers[i](in_freq)
            condangle = self.condanglelayers[i](in_angle)
            face_embed = face_embed + condangle + condfreq
            face_embed = face_embed.reshape(-1, face_embed.shape[-1])
            face_embed = conv(face_embed, face_edges)
            face_embed = act_norm(face_embed)
            face_embed = face_embed.reshape(orig_face_embed_shape[0], orig_face_embed_shape[1], -1)
          
        for attn_layer in self.encoder_attn_blocks:
            face_embed = face_embed.permute(1, 0, 2)
            face_embed = attn_layer(face_embed) + face_embed
            face_embed = face_embed.permute(1, 0, 2)

        in_em[3] = original_freqs # 恢复原始频率
        return face_embed, in_angle, in_freq
    
    def decode(self, x, in_angle, in_freq, device):
        pad_size = 22500 - x.size(1)
        x = F.pad(x, (0, 0, 0, pad_size)) 
        x = x.view(x.size(0), -1, 22500) 
        x = self.conv1d1(x) 
        x = F.relu(x)
        x = self.fc1d1(x)
        x = x.view(x.size(0), -1, 45, 90) 
        x = F.relu(self.bn1(self.upconv1(x)))
        x = F.relu(self.bn1_1(self.conv1_1(x)))
        x = F.relu(self.bn1_2(self.conv1_2(x)))
        x = F.relu(self.bn2(self.upconv2(x)))
        x = F.relu(self.bn2_1(self.conv2_1(x)))
        x = F.relu(self.bn2_2(self.conv2_2(x)))
        x = F.relu(self.bn3(self.upconv3(x)))
        x = F.relu(self.bn3_1(self.conv3_1(x)))
        x = F.relu(self.bn3_2(self.conv3_2(x)))
        x = self.conv1x1(x)
        return x.squeeze(dim=1)

    def forward(self, *, vertices, faces, face_edges=None, in_em, GT=None, logger=None, device='cpu', gama=0.001, loss_type='L1', **kwargs):
        # 更新物理损失的权重
        self.lambda_helmholtz = kwargs.get('lambda_helmholtz', self.lambda_helmholtz)
        self.lambda_bandlimit = kwargs.get('lambda_bandlimit', self.lambda_bandlimit)
        
        # 保存原始频率，以供物理损失计算使用
        original_freqs = in_em[3].clone()
        
        if face_edges is None:
            # 这是一个简化的假设，实际应用中可能需要更复杂的pad_id处理
            face_edges = derive_face_edges_from_faces(faces, pad_id=-1)

        encoded, in_angle, in_freq = self.encode(
            vertices=vertices, faces=faces, face_edges=face_edges, in_em=in_em)

        decoded = self.decode(encoded, in_angle, in_freq, device)

        if GT is None:
            return decoded
        else:
            if GT.shape[1:] == (361, 720):
                GT = GT[:, :-1, :]
            
            # 计算复合损失
            total_loss, primary_loss, helmholtz_loss, bandlimit_loss = loss_fn(
                decoded=decoded, GT=GT, freqs_ghz=original_freqs, device=device,
                loss_type=loss_type, gama=gama, 
                lambda_helmholtz=self.lambda_helmholtz, 
                lambda_bandlimit=self.lambda_bandlimit
            )

            # 计算其他指标用于监控
            with torch.no_grad():
                psnr_list = psnr(decoded, GT)
                ssim_list = myssim(decoded, GT)
                mse_list = batch_mse(decoded, GT)
                mean_psnr = psnr_list.mean()
                mean_ssim = ssim_list.mean()
                minus = decoded - GT
                mse = ((minus) ** 2).mean()
                nmse = mse / torch.var(GT)
                rmse = torch.sqrt(mse)
                l1 = (decoded-GT).abs().mean()
                percentage_error = (minus / (GT + 1e-4)).abs().mean() * 100

            # 返回总损失和所有监控指标
            # 注意：返回值的顺序和数量需要与训练脚本的接收端匹配
            # 可以在这里返回一个字典，会更清晰
            metrics = {
                'total_loss': total_loss,
                'primary_loss': primary_loss,
                'helmholtz_loss': helmholtz_loss,
                'bandlimit_loss': bandlimit_loss,
                'decoded': decoded,
                'psnr': mean_psnr,
                'ssim': mean_ssim,
                'mse': mse,
                'nmse': nmse,
                'rmse': rmse,
                'l1': l1,
                'percentage_error': percentage_error
            }
            # 为了兼容你旧的训练脚本，我们还是按顺序返回
            return total_loss, decoded, mean_psnr, psnr_list, mean_ssim, ssim_list, mse, nmse, rmse, l1, percentage_error, mse_list
