import torch
import torch.nn.functional as F
import numpy as np
import os
import re
from tqdm import tqdm
from math import pi
import argparse
import sys
from collections import defaultdict

# 将 'net' 目录的父目录添加到系统路径，以便导入模块
from pathlib import Path
FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))

# 导入所需的工具函数和模型
from net.utils import EMRCSDataset, MultiEMRCSDataset, get_logger,increment_path
# 在此版本中，互易性指标不再依赖模型推理，因此 MeshCodec 不再是必需的。
# 但为了保持代码结构的完整性，如果您仍需要加载模型进行其他指标的计算，可以保留它。
# from net.GTC_3DEMv3_3 import MeshCodec # 如果只需要GT文件的物理指标，则不需要模型


# --- 辅助函数：角度到 RCS 矩阵索引的转换 ---
def angles_to_rcs_indices(theta_deg: float, phi_deg: float, rcs_map_shape: tuple = (360, 720)) -> tuple:
    """
    将出射角度 (theta, phi) 转换为 RCS 矩阵的索引。
    假定 theta 范围 [0, 179.5]，phi 范围 [0, 359.5]，均为 0.5 度采样。
    
    参数:
        theta_deg (float): 出射天顶角 (0-180度)。
        phi_deg (float): 出射方位角 (0-360度)。
        rcs_map_shape (tuple): RCS 矩阵的期望形状，默认为 (360, 720)。
    返回:
        tuple: (theta_index, phi_index)
    """
    # 由于采样是 0.5 度，直接除以 0.5 即可得到索引。
    # 确保索引在有效范围内，并转换为整数。
    theta_index = min(int(round(theta_deg / 0.5)), rcs_map_shape[0] - 1)
    phi_index = min(int(round(phi_deg / 0.5)), rcs_map_shape[1] - 1)
    return theta_index, phi_index

# --- 物理指标函数 (只基于 GT 数据) ---

def calculate_helmholtz_first_order_metric_gt(rcs_gt: torch.Tensor, freqs_ghz: torch.Tensor, device: torch.device) -> torch.Tensor:
    """
    计算亥姆霍兹方程一阶差分指标（基于梯度的平滑度），使用 GT 数据。
    此指标衡量 RCS 图的平滑度，与物理场的一阶导数相关。
    
    参数:
        rcs_gt (torch.Tensor): GT RCS 图 (批次, 高度, 宽度)。
        freqs_ghz (torch.Tensor): 频率，单位 GHz (批次, )。
        device (torch.device): CUDA 或 CPU 设备。
    返回:
        torch.Tensor: 梯度的 L2 范数均值。
    """
    # 使用 Sobel 算子近似梯度
    sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], 
                           dtype=torch.float32, device=device).unsqueeze(0).unsqueeze(0)
    sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], 
                           dtype=torch.float32, device=device).unsqueeze(0).unsqueeze(0)

    rcs_gt_batch = rcs_gt.unsqueeze(1) # 添加通道维度

    grad_x = F.conv2d(rcs_gt_batch, sobel_x, padding=1)
    grad_y = F.conv2d(rcs_gt_batch, sobel_y, padding=1)

    # 计算梯度幅值 (L2 范数)
    gradient_magnitude = torch.sqrt(grad_x**2 + grad_y**2)
    
    return torch.mean(gradient_magnitude) # 返回梯度的均值作为“一阶差分”指标

def calculate_helmholtz_second_order_metric_gt(rcs_gt: torch.Tensor, freqs_ghz: torch.Tensor, device: torch.device) -> torch.Tensor:
    """
    计算亥姆霍兹方程二阶差分指标，使用 GT 数据。
    使用标准的 3x3 离散拉普拉斯核近似二阶导数 (nabla^2)。
    
    参数:
        rcs_gt (torch.Tensor): GT RCS 图 (批次, 高度, 宽度)。
        freqs_ghz (torch.Tensor): 频率，单位 GHz (批次, )。
        device (torch.device): CUDA 或 CPU 设备。
    返回:
        torch.Tensor: 亥姆霍兹方程残差的均方值。
    """
    # 标准的 3x3 离散拉普拉斯核 (五点模板)
    laplacian_kernel = torch.tensor([[0, 1, 0], [1, -4, 1], [0, 1, 0]], 
                                    dtype=torch.float32, device=device).unsqueeze(0).unsqueeze(0)

    rcs_gt_batch = rcs_gt.unsqueeze(1) # 添加通道维度
    padding_size = laplacian_kernel.shape[-1] // 2 # 填充大小为 1
    laplacian_of_rcs = F.conv2d(rcs_gt_batch, laplacian_kernel, padding=padding_size)

    c = 299792458.0  # 光速，单位 m/s
    k = (2 * pi * (freqs_ghz * 1e9) / c).view(-1, 1, 1, 1).to(device) # 频率张量调整形状以便广播
    k_squared = k.pow(2)

    helmholtz_residual = laplacian_of_rcs + k_squared * rcs_gt_batch
    # 返回残差的均方值作为指标
    return torch.mean(helmholtz_residual.pow(2))

def calculate_bandlimit_metric_gt(rcs_gt: torch.Tensor, freqs_ghz: torch.Tensor, device: torch.device, alpha: float = 10.0) -> torch.Tensor:
    """
    计算频域带限能量指标，使用 GT 数据。
    惩罚与给定频率不匹配的非物理高频空间细节。
    
    参数:
        rcs_gt (torch.Tensor): GT RCS 图 (批次, 高度, 宽度)。
        freqs_ghz (torch.Tensor): 频率，单位 GHz (批次, )。
        device (torch.device): CUDA 或 CPU 设备。
        alpha (float): 截止半径的缩放因子。
    返回:
        torch.Tensor: 非物理频率区域的平均能量。
    """
    # 对批次中的每个 RCS 图计算 2D FFT
    fft_gt = torch.fft.fftshift(torch.fft.fft2(rcs_gt, norm='ortho'), dim=(-2, -1))
    
    c = 299792458.0 # 光速，单位 m/s
    k = (2 * pi * (freqs_ghz * 1e9) / c).to(device)
    cutoff_radii = (alpha * k).int() # 每个样本可能有不同的截止半径
    
    h, w = rcs_gt.shape[-2:]
    center_h, center_w = h // 2, w // 2
    
    # 创建用于计算半径的网格
    y, x = np.ogrid[-center_h:h-center_h, -center_w:w-center_w]
    radius_grid = torch.from_numpy(np.sqrt(x*x + y*y)).to(device)

    # 为批次中的每个样本创建对应的非物理区域掩码
    non_physical_mask = (radius_grid.unsqueeze(0) > cutoff_radii.view(-1, 1, 1)).float()
    
    # 计算非物理区域的能量
    non_physical_energy = fft_gt.abs().pow(2) * non_physical_mask
    
    # 该指标是该区域的平均能量
    return non_physical_energy.mean()


def calculate_reciprocity_metric_gt(
    rcs_data_map: dict, # 包含所有文件元数据的字典
    current_sample_info: dict, # 当前样本的元数据 (plane, theta_in, phi_in, freq, file_path)
    current_rcs_map: torch.Tensor, # 当前样本的 RCS GT 数据
    device: torch.device
) -> torch.Tensor:
    """
    计算互易性指标，完全基于数据集中的 GT 文件查找和验证。
    
    参数:
        rcs_data_map (dict): 预加载的所有 RCS 文件元数据映射。
                             键格式: (plane_name, theta_in, phi_in, freq)。
                             值格式: 文件路径。
        current_sample_info (dict): 当前处理样本的元数据。
        current_rcs_map (torch.Tensor): 当前样本的 RCS GT 数据 (360, 720)。
        device (torch.device): CUDA 或 CPU 设备。
    返回:
        torch.Tensor: 互易性差异的 L1 损失。如果未找到互易对，则返回 NaN。
    """
    plane_name = current_sample_info['plane_name']
    theta_in_A = current_sample_info['theta']
    phi_in_A = current_sample_info['phi']
    freq_A = current_sample_info['freq']

    # 遍历 RCS map 中的每个出射方向，将其作为互易的入射方向 B
    # 360行代表 theta (0-179.5), 720列代表 phi (0-359.5)
    
    # 存储所有互易性差异
    reciprocity_diffs = []

    # 随机选择若干个散射方向作为潜在的互易入射点，减少计算量
    # 也可以遍历所有点，但计算量会非常大
    num_random_points = 50 # 可以根据需要调整
    H, W = current_rcs_map.shape
    
    # 生成随机的出射角度作为互易的入射角度 B
    random_theta_sc_indices = torch.randint(0, H, (num_random_points,)).tolist()
    random_phi_sc_indices = torch.randint(0, W, (num_random_points,)).tolist()

    for r_h_idx, r_w_idx in zip(random_theta_sc_indices, random_phi_sc_indices):
        theta_in_B = r_h_idx * 0.5 # 转换为角度
        phi_in_B = r_w_idx * 0.5   # 转换为角度

        # 1. 获取当前样本 A 在散射方向 B 的值
        value_A_scatter_B = current_rcs_map[r_h_idx, r_w_idx]

        # 2. 尝试在数据集中查找互易样本 B
        # 互易样本 B 的入射角是 (theta_in_B, phi_in_B)，频率与 A 相同
        reciprocal_key_B = (plane_name, theta_in_B, phi_in_B, freq_A)
        
        if reciprocal_key_B in rcs_data_map:
            # 如果找到了互易样本 B 的文件
            reciprocal_file_path_B = rcs_data_map[reciprocal_key_B]
            
            try:
                # 加载互易样本 B 的 GT 数据
                rcs_map_B = torch.load(reciprocal_file_path_B, weights_only=False).to(device)
                if rcs_map_B.shape[0] == 361: # 再次处理 361 行数据
                    rcs_map_B = rcs_map_B[:-1, :]

                # 3. 获取互易样本 B 在散射方向 A (即原始样本 A 的入射方向) 的值
                # 需要将原始样本 A 的入射角度 (theta_in_A, phi_in_A) 转换为 RCS_map_B 中的索引
                idx_theta_sc_A, idx_phi_sc_A = angles_to_rcs_indices(theta_in_A, phi_in_A, rcs_map_B.shape)
                value_B_scatter_A = rcs_map_B[idx_theta_sc_A, idx_phi_sc_A]

                # 4. 计算互易性差异
                diff = F.l1_loss(value_A_scatter_B.unsqueeze(0), value_B_scatter_A.unsqueeze(0), reduction='mean')
                reciprocity_diffs.append(diff.item())
            except Exception as e:
                # print(f"加载或处理互易文件 {reciprocal_file_path_B} 失败: {e}")
                continue # 跳过当前互易对，继续查找下一个

    if reciprocity_diffs:
        return torch.tensor(np.mean(reciprocity_diffs), device=device)
    else:
        # 如果未找到任何互易对，返回 NaN 表示无法计算
        return torch.tensor(float('nan'), device=device)


# --- 主评估函数 ---

def evaluate_physical_metrics(dataset_path: str, device_str: str = 'cuda:0', batch_size: int = 4, basedir: str = '') -> tuple:
    """
    在给定数据集上评估物理指标（亥姆霍兹一阶、亥姆霍兹二阶、频域带限、互易性）。
    此版本不再依赖模型权重，直接计算 GT 数据的物理指标。

    参数:
        dataset_path (str): RCS 数据集目录的路径。
        device_str (str): 要使用的 CUDA 设备 (例如, 'cuda:0', 'cpu')。
        batch_size (int): 评估的批次大小。
    
    返回:
        tuple: (亥姆霍兹一阶平均值, 亥姆霍兹二阶平均值, 频域带限平均值, 互易性平均值)
    """
    device = torch.device(device_str if torch.cuda.is_available() else "cpu")

    from datetime import datetime
    if basedir != '':
        save_dir = basedir
        # save_dir = os.path.join(basedir,f'{dataset_path.split("/")[-1]}')
    else:
        save_dir = increment_path(f'output/PINNmetrics/{datetime.today().strftime("%m%d")}_{dataset_path.split("/")[-1]}_', exist_ok=False)
    logger = get_logger(os.path.join(save_dir,f'{dataset_path.split("/")[-1]}.txt'))
    logger.info(f"开始在 {dataset_path} 上使用设备 {device} 计算 GT 物理指标并保存到 {save_dir}")

    # --- 阶段 1: 预加载所有文件元数据以支持互易性查找 ---
    rcs_data_map = {} # 存储 (plane, theta_in, phi_in, freq) -> file_path
    
    # 确定数据集类型并获取所有文件路径
    is_multi_folder_dataset = False
    file_paths_to_scan = []
    for filename in os.listdir(dataset_path):
        if filename.endswith('.pt'):
            file_paths_to_scan.append(os.path.join(dataset_path, filename))

    # 解析文件元数据并填充 rcs_data_map
    file_parse_regex = r"([a-zA-Z0-9]{4})_theta(\d+)phi(\d+)f(\d.+).pt"
    for fpath in tqdm(file_paths_to_scan, desc="预扫描文件，构建互易性查找表", ncols=100):
        filename = os.path.basename(fpath)
        match = re.search(file_parse_regex, filename)
        if match:
            plane, theta_str, phi_str, freq_str = match.groups()
            theta = float(theta_str)
            phi = float(phi_str)
            freq = float(freq_str)
            rcs_data_map[(plane, theta, phi, freq)] = fpath
    logger.info(f"互易性查找表构建完成。共找到 {len(rcs_data_map)} 个可解析文件。")

    # --- 阶段 2: 初始化数据加载器 ---
    filelist = os.listdir(dataset_path)
    dataset = EMRCSDataset(filelist, dataset_path)
    
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

    # --- 阶段 3: 遍历数据集计算指标 ---
    total_helmholtz_1st = 0.0
    total_helmholtz_2nd = 0.0
    total_bandlimit = 0.0
    total_reciprocity = 0.0
    num_samples_processed_for_all_metrics = 0
    num_samples_processed_for_reciprocity = 0 # 互易性可能因为找不到配对而跳过

    with torch.no_grad(): # 计算指标无需梯度
        for in_em_list, rcs_gt_batch in tqdm(dataloader, desc="计算物理指标", ncols=100):
            # in_em_list 包含 [plane_name_list, theta_tensor, phi_tensor, freq_tensor]
            plane_names = in_em_list[0]
            theta_in_tensor = in_em_list[1].to(device)
            phi_in_tensor = in_em_list[2].to(device)
            freq_in_tensor = in_em_list[3].to(device)
            
            # 处理 GT RCS 形状
            if rcs_gt_batch.shape[1:] == (361, 720):
                rcs_gt_batch = rcs_gt_batch[:, :-1, :].to(device)
            else:
                rcs_gt_batch = rcs_gt_batch.to(device)

            # 计算亥姆霍兹和带限指标 (直接使用 GT 数据)
            helmholtz_1st = calculate_helmholtz_first_order_metric_gt(rcs_gt_batch, freq_in_tensor, device)
            helmholtz_2nd = calculate_helmholtz_second_order_metric_gt(rcs_gt_batch, freq_in_tensor, device)
            bandlimit = calculate_bandlimit_metric_gt(rcs_gt_batch, freq_in_tensor, device)
            
            total_helmholtz_1st += helmholtz_1st.item()
            total_helmholtz_2nd += helmholtz_2nd.item()
            total_bandlimit += bandlimit.item()
            num_samples_processed_for_all_metrics += rcs_gt_batch.size(0)

            # 计算互易性指标 (遍历批次中的每个样本)
            current_batch_reciprocity_sum = 0.0
            current_batch_reciprocity_count = 0

            for i in range(rcs_gt_batch.size(0)):
                current_sample_info = {
                    'plane_name': plane_names[i],
                    'theta': theta_in_tensor[i].item(),
                    'phi': phi_in_tensor[i].item(),
                    'freq': freq_in_tensor[i].item(),
                }
                current_rcs_map_single = rcs_gt_batch[i]

                reciprocity_val = calculate_reciprocity_metric_gt(
                    rcs_data_map,
                    current_sample_info,
                    current_rcs_map_single,
                    device
                )
                
                if not torch.isnan(reciprocity_val): # 只有找到互易对的样本才加入计算
                    current_batch_reciprocity_sum += reciprocity_val.item()
                    current_batch_reciprocity_count += 1
            
            total_reciprocity += current_batch_reciprocity_sum
            num_samples_processed_for_reciprocity += current_batch_reciprocity_count
            

    # 计算平均指标
    avg_helmholtz_1st = total_helmholtz_1st / num_samples_processed_for_all_metrics
    avg_helmholtz_2nd = total_helmholtz_2nd / num_samples_processed_for_all_metrics
    avg_bandlimit = total_bandlimit / num_samples_processed_for_all_metrics
    
    # 互易性平均值只基于成功找到互易对的样本
    if num_samples_processed_for_reciprocity > 0:
        avg_reciprocity = total_reciprocity / num_samples_processed_for_reciprocity
    else:
        avg_reciprocity = float('nan') # 如果没有任何互易对，则为 NaN

    logger.info(f"\n--- 平均物理指标 (基于 GT 数据) ---")
    logger.info(f"总处理样本数 (亥姆霍兹/带限): {num_samples_processed_for_all_metrics}")
    logger.info(f"亥姆霍兹一阶平均值: {avg_helmholtz_1st:.6f}")
    logger.info(f"亥姆霍兹二阶平均值: {avg_helmholtz_2nd:.6f}")
    logger.info(f"频域带限平均值: {avg_bandlimit:.6f}")
    logger.info(f"处理样本数 (互易性): {num_samples_processed_for_reciprocity}")
    logger.info(f"互易性平均值: {avg_reciprocity:.6f}")

    return num_samples_processed_for_all_metrics, avg_helmholtz_1st, avg_helmholtz_2nd, avg_bandlimit, avg_reciprocity

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="计算 RCS 预测的物理指标。")
    parser.add_argument('--dataset_path', type=str, default='/mnt/truenas_jiangxiaotian/allplanes/mie/b943_mie_10train', help='RCS 数据集目录的路径。')
    parser.add_argument('--cuda', type=str, default='cpu', help='要使用的 CUDA 设备 (例如, cuda:0, cuda:1, cpu)。')
    parser.add_argument('--batch_size', type=int, default=12, help='评估的批次大小。')
    
    args = parser.parse_args()

    samples, avg_h1, avg_h2, avg_bl, avg_rec = evaluate_physical_metrics(
        dataset_path=args.dataset_path,
        device_str=args.cuda,
        batch_size=args.batch_size
    )