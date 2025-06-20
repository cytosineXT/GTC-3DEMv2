# Pmetrics_all.py
import os
import argparse
import pandas as pd
from tqdm import tqdm
import sys
from pathlib import Path
import time
from datetime import datetime
FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))
from PINN_metrics import evaluate_physical_metrics, get_logger, increment_path # 导入日志和路径递增函数


def main():
    parser = argparse.ArgumentParser(description="批量计算 RCS 数据集子文件夹的物理指标。")
    parser.add_argument('--base_data_dir', type=str, default='/mnt/truenas_main_datasets/allplanes/mie', help='包含所有数据子文件夹的主目录。') #305wsl
    # parser.add_argument('--base_data_dir', type=str, default='/mnt/truenas_jiangxiaotian/allplanes/mie', help='包含所有数据子文件夹的主目录。') #3090l
    parser.add_argument('--cuda', type=str, default='cuda:0', help='要使用的 CUDA 设备 (例如, cuda:0, cuda:1, cpu)。')
    parser.add_argument('--batch_size', type=int, default=64, help='评估的批次大小。')
    
    args = parser.parse_args()

    output_base_dir = increment_path(f'output/PINNmetrics/{datetime.today().strftime("%m%d")}_{args.base_data_dir.split("/")[-1]}_allinoneCSV', exist_ok=False)
    results_csv_path = output_base_dir / f"physical_metrics_results.csv"

    # CSV 表头
    columns = ['Dataset_Folder', 'Samples', 'Helmholtz_1st_Avg', 'Helmholtz_2nd_Avg', 'Bandlimit_Avg', 'Reciprocity_Avg', 'Processing_Time_Seconds']
    
    # 初始化 DataFrame 或直接创建 CSV 文件并写入表头
    results_df = pd.DataFrame(columns=columns)
    results_df.to_csv(results_csv_path, index=False) # 写入表头

    # 获取所有子文件夹
    subfolders_to_process = []
    # os.listdir() 足够，因为它只列出当前目录下的文件和文件夹
    # 如果子文件夹内还有子文件夹，且它们也是数据文件夹，需要 os.walk
    # 根据你的描述，b8ed_mie_10train 这些都在 /mnt/truenas_jiangxiaotian/allplanes/mie 下，所以直接 os.listdir + isdir 即可
    for entry in os.listdir(args.base_data_dir):
        full_path = os.path.join(args.base_data_dir, entry)
        # 假设所有符合模式 "bXXX_mie_YYY" 的都是数据文件夹
        # if os.path.isdir(full_path) and re.match(r"b[0-9a-f]{3}_mie_.+", entry):
        subfolders_to_process.append(full_path)
    
    # 按文件夹名称排序，以便输出顺序一致
    subfolders_to_process.sort()

    main_logger = get_logger(output_base_dir / f"log.txt")
    main_logger.info(f"开始遍历主目录: {args.base_data_dir}")
    main_logger.info(f"共找到 {len(subfolders_to_process)} 个数据子文件夹。")
    main_logger.info(f"结果将保存到: {results_csv_path}")

    for folder_path in tqdm(subfolders_to_process, desc="处理数据子文件夹", ncols=100):
        folder_name = os.path.basename(folder_path)
        main_logger.info(f"\n--- 开始处理文件夹: {folder_name} ---")
        start_time = time.time()

        try:
            # 调用 Pmetrics.py 中的评估函数
            # 注意：Pmetrics.py 内部会创建自己的 log 文件
            samples, avg_h1, avg_h2, avg_bl, avg_rec = evaluate_physical_metrics(
                dataset_path=folder_path,
                device_str=args.cuda,
                batch_size=args.batch_size,
                basedir = output_base_dir
            )

            processing_time = time.time() - start_time
            main_logger.info(f"文件夹 {folder_name} 处理完成。耗时: {processing_time:.2f} 秒。")

            # 将结果添加到 DataFrame
            new_row = pd.DataFrame([{
                'Dataset_Folder': folder_name,
                'Samples': samples,
                'Helmholtz_1st_Avg': avg_h1,
                'Helmholtz_2nd_Avg': avg_h2,
                'Bandlimit_Avg': avg_bl,
                'Reciprocity_Avg': avg_rec,
                'Processing_Time_Seconds': processing_time
            }])
            
            # 逐循环更新 CSV 文件，使用 mode='a' (append)
            new_row.to_csv(results_csv_path, mode='a', header=False, index=False)
            main_logger.info(f"结果已追加到 CSV 文件: {results_csv_path}")

        except Exception as e:
            processing_time = time.time() - start_time
            main_logger.error(f"处理文件夹 {folder_name} 时发生错误: {e}")
            main_logger.error(f"此文件夹处理耗时: {processing_time:.2f} 秒。")
            # 记录错误信息到 CSV (可选，如果需要更详细的错误记录)
            error_row = pd.DataFrame([{
                'Dataset_Folder': folder_name,
                'Helmholtz_1st_Avg': 'ERROR',
                'Helmholtz_2nd_Avg': 'ERROR',
                'Bandlimit_Avg': 'ERROR',
                'Reciprocity_Avg': 'ERROR',
                'Processing_Time_Seconds': processing_time
            }])
            error_row.to_csv(results_csv_path, mode='a', header=False, index=False)


    main_logger.info("\n--- 所有数据文件夹处理完毕 ---")
    main_logger.info(f"最终结果保存在: {results_csv_path}")

if __name__ == '__main__':
    main()