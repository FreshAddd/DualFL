import csv
import logging
import os
import sys
import time
import traceback
import shutil

import torch
from pytorch_lightning import Trainer, Callback
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger
from torch.utils.data.dataloader import DataLoader
from tqdm import tqdm

current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.dirname(current_dir)
sys.path.insert(0, root_dir)

try:
    from config import ProjectConfig
    from defects4j.utils import Utils
    import numpy as np
    from dataset.GraphEmbeddingDataset import GraphEmbeddingDataset, graph_collate_fn
    from dataset.GraphEmbeddingDataset_stmt import GraphEmbeddingDataset_stmt, graph_collate_fn_stmt
    from model.model_method import TrainModel_Method
    from model.model_stmt import TrainModel_STMT
except ImportError as e:
    print(f"导入错误: {e}")
    sys.exit(1)

method_inference_times = []
method_memory_usages = []
stmt_inference_times = []
stmt_memory_usages = []

def log_performance_metrics(save_path):
    
    avg_method_time = sum(method_inference_times) / len(method_inference_times) if method_inference_times else 0
    avg_method_memory = sum(method_memory_usages) / len(method_memory_usages) if method_memory_usages else 0
    avg_stmt_time = sum(stmt_inference_times) / len(stmt_inference_times) if stmt_inference_times else 0
    avg_stmt_memory = sum(stmt_memory_usages) / len(stmt_memory_usages) if stmt_memory_usages else 0
    
    logging.info(f"平均方法推理时间: {avg_method_time:.4f}秒")
    logging.info(f"平均方法显存占用: {avg_method_memory/1024/1024:.2f} MB")
    logging.info(f"平均语句推理时间: {avg_stmt_time:.4f}秒")
    logging.info(f"平均语句显存占用: {avg_stmt_memory/1024/1024:.2f} MB")
    
    with open(os.path.join(save_path, "performance_metrics.csv"), "w+") as f:
        cf = csv.writer(f)
        cf.writerow(("模型", "平均推理时间(秒)", "平均显存占用(MB)", "样本数量"))
        cf.writerow(("方法模型", f"{avg_method_time:.4f}", f"{avg_method_memory/1024/1024:.2f}", len(method_inference_times)))
        cf.writerow(("语句模型", f"{avg_stmt_time:.4f}", f"{avg_stmt_memory/1024/1024:.2f}", len(stmt_inference_times)))
    
    with open(os.path.join(save_path, "detailed_performance.csv"), "w+") as f:
        cf = csv.writer(f)
        cf.writerow(("版本", "方法推理时间(秒)", "方法显存占用(MB)", "语句推理时间(秒)", "语句显存占用(MB)"))
        for i in range(max(len(method_inference_times), len(stmt_inference_times))):
            method_time = method_inference_times[i] if i < len(method_inference_times) else ""
            method_memory = method_memory_usages[i]/1024/1024 if i < len(method_memory_usages) else ""
            stmt_time = stmt_inference_times[i] if i < len(stmt_inference_times) else ""
            stmt_memory = stmt_memory_usages[i]/1024/1024 if i < len(stmt_memory_usages) else ""
            cf.writerow((i+1, f"{method_time:.4f}" if method_time else "", 
                        f"{method_memory:.2f}" if method_memory else "",
                        f"{stmt_time:.4f}" if stmt_time else "",
                        f"{stmt_memory:.2f}" if stmt_memory else ""))

def evaluate_single_version(project, model_method, model_stmt, test_version, save_path, cuda_device):
    
    try:
        if os.path.exists(f"{save_path}/{test_version}.csv"):
            logging.info(f"{save_path}/{test_version}.csv 已存在，跳过。")
            return
        
        model_method.cuda(cuda_device)
        model_method.eval()
        model_stmt.cuda(cuda_device)
        model_stmt.eval()

        method_time = 0.0
        method_memory = 0.0
        stmt_time = 0.0
        stmt_memory = 0.0
        
        with torch.no_grad():
            
            logging.info(f"加载项目 {project} 版本 {test_version} 的测试数据...")
            _dataset = GraphEmbeddingDataset(
                project_name=project,
                selected_versions=[test_version],
                use_resample=False,
                verbose=True
            )
            _dataset.set_attribute_processing([0,0,0,0,0,1,1], [[],[],[],[],[],[0],[0]])
            if len(_dataset) == 0:
                logging.warning(f"版本 {test_version} 未加载到数据，跳过评估。")
                return
            
            logging.info(f"成功加载数据集，共 {len(_dataset)} 条样本")
            
            val_loader = DataLoader(
                _dataset, 
                batch_size=batch_size, 
                collate_fn=graph_collate_fn,
                num_workers=0
            )
            
            result = []
            batch_count = 0
            
            for passed_test_cases, failed_test_cases, history_change, call_information, methods, call_stmt, internal_stmt, labels, method_id in tqdm(val_loader, desc=f"评估版本 {test_version} 的方法"):
                try:
                    
                    passed_test_cases = passed_test_cases.cuda(cuda_device)
                    failed_test_cases = failed_test_cases.cuda(cuda_device)
                    history_change = history_change.cuda(cuda_device)
                    call_information = call_information.cuda(cuda_device)
                    methods = methods.cuda(cuda_device)
                    call_stmt = call_stmt.cuda(cuda_device)
                    internal_stmt = internal_stmt.cuda(cuda_device)
                    
                    torch.cuda.empty_cache()
                    torch.cuda.reset_peak_memory_stats(cuda_device)
                    start_memory = torch.cuda.memory_allocated(cuda_device)
                    start_time = time.time()
                    
                    preds = model_method(
                        passed_test_cases, 
                        failed_test_cases, 
                        history_change, 
                        call_information, 
                        methods,
                        call_stmt,
                        internal_stmt
                    ).squeeze(-1).detach().cpu()
                    
                    end_time = time.time()
                    batch_time = end_time - start_time
                    max_memory = torch.cuda.max_memory_allocated(cuda_device) - start_memory
                    
                    method_time += batch_time
                    method_memory = max(method_memory, max_memory)
                    batch_count += 1
                    
                    for label, pred, id in zip(labels, preds, method_id):
                        result.append((int(label.item()), pred.item(), id.item()))
                except Exception as e:
                    logging.error(f"处理批次时出错: {str(e)}")
                    import traceback
                    traceback.print_exc()
            
            if batch_count > 0:
                avg_batch_time = method_time / batch_count
                logging.info(f"方法模型推理: 总时间={method_time:.4f}秒, 批次数={batch_count}, 平均每批次={avg_batch_time:.4f}秒")
                logging.info(f"方法模型显存占用: {method_memory/1024/1024:.2f} MB")
                
                method_inference_times.append(method_time)
                method_memory_usages.append(method_memory)
            
            if not result:
                logging.warning(f"版本 {test_version} 评估结果为空，无法保存。")
                return
                
            result = sorted(result, key=lambda t: t[1], reverse=True)
            
            with open(os.path.join(save_path, f"{test_version}_method.csv"), "w+") as f:
                cf = csv.writer(f)
                cf.writerow(("Label", "Predict", "Method_id"))
                cf.writerows(result)
            
            total_methods = len(result)*0.3
            
            if total_methods < 20:
                top_method_ids = set([item[2] for item in result])
                logging.info(f"版本 {test_version} 方法总数少于10个({len(result)})，不进行筛选")
            else:
                
                top_count = max(int(total_methods), 1)
                top_method_ids = set([item[2] for item in result[:top_count]])
                logging.info(f"版本 {test_version} 筛选排名前{top_count}/{total_methods}(30%)的方法ID: {top_method_ids}")
        
        with torch.no_grad():
            
            logging.info(f"加载项目 {project} 版本 {test_version} 的语句测试数据...")
            _dataset = GraphEmbeddingDataset_stmt(
                project_name=project,
                selected_versions=[test_version],
                use_resample=False,
                verbose=True
            )
            _dataset.set_attribute_processing([0,0,0,0,1,1,1,1], [[],[],[],[],[0],[0],[0],[0]])
            if len(_dataset) == 0:
                logging.warning(f"版本 {test_version} 未加载到语句数据，跳过评估。")
                return
            
            logging.info(f"成功加载语句数据集，共 {len(_dataset)} 条样本")
            
            filtered_indices = []
            missed_defect_indices = []  
            
            for i, (_, _, _, _, _, _, _, _, labels, stmt_id, method_id) in enumerate(_dataset):
                if method_id.item() in top_method_ids:
                    filtered_indices.append(i)
                elif labels.item() == 1:  
                    missed_defect_indices.append(i)
                    logging.warning(f"发现被过滤掉的缺陷语句: stmt_id={stmt_id.item()}, method_id={method_id.item()}")
            
            if not filtered_indices:
                logging.warning(f"版本 {test_version} 没有找到筛选方法中的语句，跳过语句评估。")
                return
                
            filtered_dataset = torch.utils.data.Subset(_dataset, filtered_indices)
            logging.info(f"过滤后的语句数据集大小: {len(filtered_dataset)}/{len(_dataset)}")
            logging.info(f"被过滤掉的缺陷语句数量: {len(missed_defect_indices)}")
            
            val_loader = DataLoader(
                filtered_dataset, 
                batch_size=batch_size, 
                collate_fn=graph_collate_fn_stmt,
                num_workers=0
            )
            
            result = []
            batch_count = 0
            
            for passed_test_cases, failed_test_cases, call_method, belong_method, stmt, cfg, dfg, ast, labels, stmt_id, method_id in tqdm(val_loader, desc=f"评估版本 {test_version} 的语句"):
                try:
                    
                    passed_test_cases = passed_test_cases.cuda(cuda_device)
                    failed_test_cases = failed_test_cases.cuda(cuda_device)
                    call_method = call_method.cuda(cuda_device)
                    belong_method = belong_method.cuda(cuda_device)
                    stmt = stmt.cuda(cuda_device)
                    cfg = cfg.cuda(cuda_device)
                    dfg = dfg.cuda(cuda_device)
                    ast = ast.cuda(cuda_device)
                    
                    torch.cuda.empty_cache()
                    torch.cuda.reset_peak_memory_stats(cuda_device)
                    start_memory = torch.cuda.memory_allocated(cuda_device)
                    start_time = time.time()
                    
                    preds = model_stmt(
                        passed_test_cases, 
                        failed_test_cases, 
                        call_method, 
                        belong_method, 
                        stmt,
                        cfg,
                        dfg,
                        ast
                    ).squeeze(-1).detach().cpu()
                    
                    end_time = time.time()
                    batch_time = end_time - start_time
                    max_memory = torch.cuda.max_memory_allocated(cuda_device) - start_memory
                    
                    stmt_time += batch_time
                    stmt_memory = max(stmt_memory, max_memory)
                    batch_count += 1
                    
                    for label, pred, s_id, m_id in zip(labels, preds, stmt_id, method_id):
                        result.append((int(label.item()), pred.item(), s_id.item(), m_id.item()))
                except Exception as e:
                    logging.error(f"处理批次时出错: {str(e)}")
                    import traceback
                    traceback.print_exc()
            
            if batch_count > 0:
                avg_batch_time = stmt_time / batch_count
                logging.info(f"语句模型推理: 总时间={stmt_time:.4f}秒, 批次数={batch_count}, 平均每批次={avg_batch_time:.4f}秒")
                logging.info(f"语句模型显存占用: {stmt_memory/1024/1024:.2f} MB")
                
                stmt_inference_times.append(stmt_time)
                stmt_memory_usages.append(stmt_memory)
            
            if not result:
                logging.warning(f"版本 {test_version} 语句评估结果为空，无法保存。")
                return
                
            result = sorted(result, key=lambda t: t[1], reverse=True)
            
            missed_defects = []
            if missed_defect_indices:
                
                decimal_part = len(_dataset) // 2
                
                for idx in missed_defect_indices:
                    _, _, _, _, _, _, _, _, labels, stmt_id, method_id = _dataset[idx]
                    
                    special_score = float(f"-9999.{decimal_part}")  
                    missed_defects.append((int(labels.item()), special_score, stmt_id.item(), method_id.item()))
                    logging.info(f"添加被过滤缺陷语句到结果末尾: label={labels.item()}, score={special_score}, stmt_id={stmt_id.item()}, method_id={method_id.item()}")
                
                result.extend(missed_defects)
            
            with open(os.path.join(save_path, f"{test_version}_stmt.csv"), "w+") as f:
                cf = csv.writer(f)
                cf.writerow(("Label", "Predict", "Stmt_id", "Method_id"))
                cf.writerows(result)
            
    except Exception as e:
        logging.error(f"评估版本 {test_version} 时出错: {str(e)}")
        logging.error(traceback.format_exc())

def calc_all_version_metrics(save_path, project):
    
    metrics_results = {}
    
    for test_version in Utils.get_active_bug(project):
        result_file = f"{save_path}/{test_version}_method.csv"
        if not os.path.exists(result_file):
            logging.info(f"{result_file} 不存在，跳过。")
            continue
            
        with open(result_file, "r") as f:
            rf = csv.reader(f)
            next(rf)  
            result = []
            
            for item in rf:
                if len(item) >= 2:
                    try:
                        label = int(item[0])
                        predict = float(item[1])
                        result.append((label, predict))
                    except ValueError:
                        logging.warning(f"无法解析行 {item} in {test_version}_method.csv，跳过此行。")
                else:
                    logging.warning(f"发现格式不正确的行 {item} in {test_version}_method.csv，跳过此行。")
            
            if not result:
                logging.warning(f"文件 {test_version}_method.csv 中没有有效的评估结果，无法计算指标。")
                continue

            top1, top3, top5, FR, AR = calc(result)
            metrics_results[test_version] = (top1, top3, top5, FR, AR)
            
            with open(os.path.join(save_path, f"{test_version}_method-metrics.csv"), "w+") as f:
                cf = csv.writer(f)
                cf.writerow(("top1", "top3", "top5", "FR", "AR"))
                cf.writerow((top1, top3, top5, FR, AR))

    for test_version in Utils.get_active_bug(project):
        result_file = f"{save_path}/{test_version}_stmt.csv"
        if not os.path.exists(result_file):
            logging.info(f"{result_file} 不存在，跳过。")
            continue
            
        with open(result_file, "r") as f:
            rf = csv.reader(f)
            next(rf)  
            result = []
            
            for item in rf:
                if len(item) >= 2:
                    try:
                        label = int(item[0])
                        predict = float(item[1])
                        result.append((label, predict))
                    except ValueError:
                        logging.warning(f"无法解析行 {item} in {test_version}_stmt.csv，跳过此行。")
                else:
                    logging.warning(f"发现格式不正确的行 {item} in {test_version}_stmt.csv，跳过此行。")
            
            if not result:
                logging.warning(f"文件 {test_version}_stmt.csv 中没有有效的评估结果，无法计算指标。")
                continue

            top1, top3, top5, FR, AR = calc(result)
            metrics_results[f"{test_version}_normal"] = (top1, top3, top5, FR, AR)
            
            with open(os.path.join(save_path, f"{test_version}_stmt-metrics.csv"), "w+") as f:
                cf = csv.writer(f)
                cf.writerow(("top1", "top3", "top5", "FR", "AR"))
                cf.writerow((top1, top3, top5, FR, AR))

    return metrics_results

def calc(data):
    
    top1 = 0
    top3 = 0
    top5 = 0
    AR = []
    FR = 0
    
    for i, item in enumerate(data):
        label, predict = item
        
        if str(predict).startswith('-9999.') and label == 1:
            
            try:
                
                position = int(str(predict).split('.')[-1])
                if FR == 0:
                    FR = position
                AR.append(position)
                logging.info(f"检测到特殊怀疑度值: {predict}, 位置: {position}")
                continue
            except Exception as e:
                logging.error(f"解析特殊怀疑度值时出错: {str(e)}")
        
        if i < 1:
            if label == 1:
                top1 += 1
                top3 += 1
                top5 += 1
        elif i < 3:
            if label == 1:
                top3 += 1
                top5 += 1
        elif i < 5:
            if label == 1:
                top5 += 1
                
        if label == 1:
            if FR == 0:
                FR = (i + 1)
            AR.append(i + 1)
            
    AR = mean(AR)
    return top1, top3, top5, FR, AR

def mean(data):
    
    if len(data) != 0:
        return sum(data) / len(data)
    return None

def merge_all(save_path, active_bug):
    
    top1s_method = []
    top3s_method = []
    top5s_method = []
    MFR_method = []
    MAR_method = []

    top1s_stmt_normal = []
    top3s_stmt_normal = []
    top5s_stmt_normal = []
    MFR_stmt_normal = []
    MAR_stmt_normal = []

    for test_version in tqdm(active_bug, desc="合并方法指标"):
        matric_file = f"{save_path}/{test_version}_method-metrics.csv"
        if not os.path.exists(matric_file):
            logging.info(f"{matric_file} 不存在，跳过。")
            continue
            
        with open(matric_file, "r") as f:
            rf = csv.reader(f)
            next(rf)
            next(rf)
            
            try:
                top1, top3, top5, FR, AR = [t if t != "" else None for t in next(rf)]
                top1, top3, top5 = int(top1), int(top3), int(top5)
                FR = float(FR) if FR is not None else None
                AR = float(AR) if AR is not None else None
                
                top3 = int((top3 > 0))  
                top5 = int((top5 > 0))  
                
                top1s_method.append(top1)
                top3s_method.append(top3)
                top5s_method.append(top5)
                if FR is not None:
                    MFR_method.append(FR)
                if AR is not None:
                    MAR_method.append(AR)
            except Exception as e:
                logging.error(f"处理 {matric_file} 时出错: {str(e)}")
    
    num_samples_method = len(top1s_method)    
    top1s_method = sum(top1s_method)
    top3s_method = sum(top3s_method)
    top5s_method = sum(top5s_method)
    MFR_value_method = mean(MFR_method) if MFR_method else None
    MAR_value_method = mean(MAR_method) if MAR_method else None
    
    with open(f"{save_path}/val_method_metrics.csv", "w+") as f:
        cf = csv.writer(f)
        cf.writerow(("top1", "top3", "top5", "MAR", "MFR", "num_samples"))
        cf.writerow((top1s_method, top3s_method, top5s_method, MAR_value_method, MFR_value_method, num_samples_method))
    
    logging.info(f"方法评估结果摘要:")
    logging.info(f"- top1: {top1s_method}/{num_samples_method} ({top1s_method/num_samples_method:.2f})")
    logging.info(f"- top3: {top3s_method}/{num_samples_method} ({top3s_method/num_samples_method:.2f})")
    logging.info(f"- top5: {top5s_method}/{num_samples_method} ({top5s_method/num_samples_method:.2f})")
    logging.info(f"- MAR: {MAR_value_method:.2f}")
    logging.info(f"- MFR: {MFR_value_method:.2f}")
    
    for test_version in tqdm(active_bug, desc="合并正常排序语句指标"):
        matric_file = f"{save_path}/{test_version}_stmt-metrics.csv"
        if not os.path.exists(matric_file):
            logging.info(f"{matric_file} 不存在，跳过。")
            continue
            
        with open(matric_file, "r") as f:
            rf = csv.reader(f)
            next(rf)
            next(rf)
            
            try:
                top1, top3, top5, FR, AR = [t if t != "" else None for t in next(rf)]
                top1, top3, top5 = int(top1), int(top3), int(top5)
                FR = float(FR) if FR is not None else None
                AR = float(AR) if AR is not None else None
                
                top3 = int((top3 > 0))  
                top5 = int((top5 > 0))  
                
                top1s_stmt_normal.append(top1)
                top3s_stmt_normal.append(top3)
                top5s_stmt_normal.append(top5)
                if FR is not None:
                    MFR_stmt_normal.append(FR)
                if AR is not None:
                    MAR_stmt_normal.append(AR)
            except Exception as e:
                logging.error(f"处理 {matric_file} 时出错: {str(e)}")
    
    num_samples_stmt_normal = len(top1s_stmt_normal)
    top1s_stmt_normal = sum(top1s_stmt_normal)
    top3s_stmt_normal = sum(top3s_stmt_normal)
    top5s_stmt_normal = sum(top5s_stmt_normal)
    MFR_value_stmt_normal = mean(MFR_stmt_normal) if MFR_stmt_normal else None
    MAR_value_stmt_normal = mean(MAR_stmt_normal) if MAR_stmt_normal else None
    
    with open(f"{save_path}/val_stmt_normal_metrics.csv", "w+") as f:
        cf = csv.writer(f)
        cf.writerow(("top1", "top3", "top5", "MAR", "MFR", "num_samples"))
        cf.writerow((top1s_stmt_normal, top3s_stmt_normal, top5s_stmt_normal, MAR_value_stmt_normal, MFR_value_stmt_normal, num_samples_stmt_normal))
    
    logging.info(f"正常排序语句评估结果摘要:")
    logging.info(f"- top1: {top1s_stmt_normal}/{num_samples_stmt_normal} ({top1s_stmt_normal/num_samples_stmt_normal:.2f})")
    logging.info(f"- top3: {top3s_stmt_normal}/{num_samples_stmt_normal} ({top3s_stmt_normal/num_samples_stmt_normal:.2f})")
    logging.info(f"- top5: {top5s_stmt_normal}/{num_samples_stmt_normal} ({top5s_stmt_normal/num_samples_stmt_normal:.2f})")
    logging.info(f"- MAR: {MAR_value_stmt_normal:.2f}")
    logging.info(f"- MFR: {MFR_value_stmt_normal:.2f}")
 
class MyEarlyStopping(Callback):
    
    def __init__(self):
        self.last_loss = 9999

    def on_validation_end(self, trainer, pl_module):
        best_FR = trainer.callback_metrics.get("best_FR")
        if best_FR == 1:
            current_loss = trainer.callback_metrics.get("val_loss")
            if self.last_loss < current_loss:
                logging.info(f"早停: best_FR={best_FR}, val_loss={current_loss}.")
                trainer.should_stop = True
            else:
                self.last_loss = current_loss

def train_method(project, time_str, selected_versions, test_version, cuda_devices):
    
    try:
        logging.info(f"训练项目 {project}_{time_str} 版本 {test_version} 在设备 cuda-{cuda_devices} 上.")
        
        logging.info(f"加载版本 {test_version} 的验证数据...")
        _val_dataset = GraphEmbeddingDataset(
            project_name=project,
            selected_versions=[test_version],
            use_resample=False,
            verbose=True
        )
        _val_dataset.set_attribute_processing([0,0,0,0,0,1,1], [[],[],[],[],[],[0],[0]])
        if len(_val_dataset) == 0:
            logging.warning(f"验证数据集为空，无法进行训练。请检查数据文件是否存在。")
            return
            
        if len(_val_dataset.dataset_label_true) < 1:
            logging.info(f"版本 {test_version} 中没有缺陷方法，跳过训练。")
            return
        
        logging.info(f"加载版本 {selected_versions} 的训练数据...")
        _dataset = GraphEmbeddingDataset(
            project_name=project,
            selected_versions=selected_versions,
            use_resample=True,
            verbose=True
        )
        _dataset.set_attribute_processing([0,0,0,0,0,1,1], [[],[],[],[],[],[0],[0]])
        if len(_dataset) == 0:
            logging.warning(f"训练数据集为空，无法进行训练。请检查数据文件是否存在。")
            return
            
        logging.info(f"训练数据集大小: {len(_dataset)}, 验证数据集大小: {len(_val_dataset)}")
        
        train_loader = DataLoader(
            _dataset, 
            batch_size=batch_size, 
            collate_fn=graph_collate_fn, 
            shuffle=True,
            num_workers=num_worker
        )
        
        val_loader = DataLoader(
            _val_dataset, 
            batch_size=batch_size, 
            collate_fn=graph_collate_fn,
            num_workers=num_worker
        )
        
        logger = TensorBoardLogger(
            save_dir='DualFL_Model_New/', 
            name=f"{project}_{time_str}",
            version=f"{test_version}", 
            log_graph=False
        )
        
        model_method = TrainModel_Method() 
        
        model_checkpoint = ModelCheckpoint(monitor='score', filename="best_method")
        early_stopping = MyEarlyStopping()
        
        max_epochs = 1 
        val_check_interval = 1 / 20  
        
        trainer = Trainer(
            accelerator="cuda", 
            devices=cuda_devices, 
            max_epochs=max_epochs, 
            logger=logger,
            log_every_n_steps=10, 
            val_check_interval=val_check_interval,
            callbacks=[model_checkpoint, early_stopping]
        )
        
        trainer.fit(model_method, train_loader, val_loader)
        
    except Exception as e:
        logging.error(f"训练时出错: {str(e)}")
        logging.error(traceback.format_exc())

def train_stmt(project, time_str, selected_versions, test_version, cuda_devices):
    
    try:
        logging.info(f"训练项目 {project}_{time_str} 版本 {test_version} 在设备 cuda-{cuda_devices} 上.")
        
        logging.info(f"加载版本 {test_version} 的验证数据...")
        _val_dataset = GraphEmbeddingDataset_stmt(
            project_name=project,
            selected_versions=[test_version],
            use_resample=False,
            verbose=True
        )
        _val_dataset.set_attribute_processing([0,0,0,0,1,1,1,1], [[],[],[],[],[0],[0],[0],[0]])
        if len(_val_dataset) == 0:
            logging.warning(f"验证数据集为空，无法进行训练。请检查数据文件是否存在。")
            return
            
        if len(_val_dataset.dataset_label_true) < 1:
            logging.info(f"版本 {test_version} 中没有缺陷语句，跳过训练。")
            return
        
        logging.info(f"加载版本 {selected_versions} 的训练数据...")
        _dataset = GraphEmbeddingDataset_stmt(
            project_name=project,
            selected_versions=selected_versions,
            use_resample=True,
            verbose=True
        )
        _dataset.set_attribute_processing([0,0,0,0,1,1,1,1], [[],[],[],[],[0],[0],[0],[0]])
        if len(_dataset) == 0:
            logging.warning(f"训练数据集为空，无法进行训练。请检查数据文件是否存在。")
            return
            
        logging.info(f"训练数据集大小: {len(_dataset)}, 验证数据集大小: {len(_val_dataset)}")
        
        train_loader = DataLoader(
            _dataset, 
            batch_size=batch_size, 
            collate_fn=graph_collate_fn_stmt, 
            shuffle=True,
            num_workers=num_worker
        )
        
        val_loader = DataLoader(
            _val_dataset, 
            batch_size=batch_size, 
            collate_fn=graph_collate_fn_stmt,
            num_workers=num_worker
        )
        
        logger = TensorBoardLogger(
            save_dir='DualFL_Model_New/', 
            name=f"{project}_{time_str}",
            version=f"{test_version}", 
            log_graph=False
        )
        
        model_stmt = TrainModel_STMT()  
        
        model_checkpoint = ModelCheckpoint(monitor='score', filename="best_stmt")
        early_stopping = MyEarlyStopping()
        
        max_epochs = 1 
        val_check_interval = 1 / 20  
        
        trainer = Trainer(
            accelerator="cuda", 
            devices=cuda_devices, 
            max_epochs=max_epochs, 
            logger=logger,
            log_every_n_steps=10, 
            val_check_interval=val_check_interval,
            callbacks=[model_checkpoint, early_stopping]
        )
        
        trainer.fit(model_stmt, train_loader, val_loader)
        
    except Exception as e:
        logging.error(f"训练时出错: {str(e)}")
        logging.error(traceback.format_exc())

batch_size = 32
n = 5  
num_worker = 2  
cuda_device = 0  
evaluate_save_root_path = "model_output_new"

if __name__ == '__main__':
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('train_log.txt')
        ]
    )
    
    print(f"使用设备 {cuda_device}: {torch.cuda.get_device_name(cuda_device)}")
    print("开始执行项目训练...")
    
    projects = Utils.get_projects()
    
    print(f"检测到的项目: {projects}")
    
    start_time = str(int(time.time()))
    
    for iii in range(n):
        time_str = str(iii) + "TT" + start_time
        
        for project in projects:
            
            if project in ["Lang"]:
                batch_size = 16
            else:
                batch_size = 32
            
            active_bug = Utils.get_active_bug(project)

            active_bug = [1,3]

            active_bug_set = set(active_bug)
            
            model_root_path = f"DualFL_Model_New/{project}_{time_str}"
            evaluate_save_path = f"{evaluate_save_root_path}/{project}_{time_str}_dual_abla"
            
            if not os.path.exists(evaluate_save_path):
                os.makedirs(evaluate_save_path)
            
            for _i, test_version in enumerate(active_bug):
                
                model_path = f"{model_root_path}/{test_version}/checkpoints/best.ckpt"
                
                selected_versions = list(active_bug_set - {test_version})
                
                train_method(project, time_str, selected_versions, test_version, [cuda_device])
                
                train_stmt(project, time_str, selected_versions, test_version, [cuda_device])
                
                if os.path.exists(f"DualFL_Model_New/{project}_{time_str}/{test_version}/checkpoints/best_method.ckpt") and os.path.exists(f"DualFL_Model_New/{project}_{time_str}/{test_version}/checkpoints/best_stmt.ckpt"):
                    __model_method = TrainModel_Method.load_from_checkpoint(
                        f"DualFL_Model_New/{project}_{time_str}/{test_version}/checkpoints/best_method.ckpt"
                    )
                    __model_stmt = TrainModel_STMT.load_from_checkpoint(
                        f"DualFL_Model_New/{project}_{time_str}/{test_version}/checkpoints/best_stmt.ckpt"
                    )
                    logging.info(f"评估 {project}_{time_str} 版本 {test_version} 在设备 cuda-{cuda_device} 上.")
                    evaluate_single_version(project, __model_method, __model_stmt, test_version, evaluate_save_path, cuda_device)
                    
                    shutil.rmtree(f"{model_root_path}/{test_version}", ignore_errors=True)
            
            shutil.rmtree(model_root_path, ignore_errors=True)
            
            calc_all_version_metrics(evaluate_save_path, project)
            
            merge_all(evaluate_save_path, active_bug)
            
            log_performance_metrics(evaluate_save_path)
    
    print("所有训练和评估完成！") 