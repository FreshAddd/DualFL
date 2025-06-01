import os
import pickle
import logging
import torch
import numpy as np
from torch.utils.data import Dataset
from tqdm import tqdm
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

try:
    from config import ProjectConfig
except ImportError:
    print("无法导入ProjectConfig，请检查路径设置")
    sys.exit(1)

class GraphEmbeddingDataset(Dataset):
    def __init__(self, project_name, selected_versions, embed_d=768, use_resample=True, 
                 item_len_limit=256, verbose=True):
        
        self.project_name = project_name
        self.selected_versions = selected_versions if isinstance(selected_versions, list) else [selected_versions]
        self.embed_d = embed_d
        self.use_resample = use_resample
        self.item_len_limit = item_len_limit
        self.verbose = verbose
        
        self.dataset = []
        self.dataset_label_true = []
        self.dataset_label_false = []
        
        self.process_flags = [0,0,0,0,0,0,0]  
        self.process_indices = [[],[],[],[],[],[],[]]  
        
        self.load_data()
        
        if use_resample and self.dataset:
            self.resample_for_balance_label()
            
        if verbose:
            logging.info(f"成功加载数据集，共 {len(self.dataset)} 条样本")
            logging.info(f"Label True: {len(self.dataset_label_true)}")
            logging.info(f"Label False: {len(self.dataset_label_false)}")
    
    def load_data(self):
        
        if self.verbose:
            logging.info(f"开始加载项目 {self.project_name} 的数据...")
        
        loaded_versions = 0
        for version in tqdm(self.selected_versions, desc="加载版本", disable=not self.verbose):
            
            pkl_path = os.path.join(
                ProjectConfig.merge_graph_path, 
                self.project_name, 
                str(version), 
                f"{self.project_name}_{version}_graph_embedding_merge.pkl"
            )
            
            if not os.path.exists(pkl_path):
                logging.warning(f"文件 {pkl_path} 不存在，跳过此版本。")
                
                alternate_path = os.path.join(
                    ProjectConfig.merge_graph_path, 
                    self.project_name, 
                    str(version), 
                    f"{self.project_name}{version}_graph_embedding_merge.pkl"
                )
                if os.path.exists(alternate_path):
                    logging.info(f"找到替代文件 {alternate_path}，使用此文件。")
                    pkl_path = alternate_path
                else:
                    continue
            
            try:
                
                with open(pkl_path, 'rb') as f:
                    logging.info(f"正在加载文件: {pkl_path}")
                    graph_data = pickle.load(f)
                
                if self.verbose:
                    logging.info(f"文件 {pkl_path} 加载成功")
                    logging.info(f"数据类型: {type(graph_data)}")
                    method_nodes_count = len(graph_data['method_nodes']) if 'method_nodes' in graph_data else 0
                    stmt_nodes_count = len(graph_data['statement_nodes']) if 'statement_nodes' in graph_data else 0
                    fail_tests_count = len(graph_data['test_nodes_fail']) if 'test_nodes_fail' in graph_data else 0
                    pass_tests_count = len(graph_data['test_nodes_pass']) if 'test_nodes_pass' in graph_data else 0
                    
                    logging.info(f"方法节点数: {method_nodes_count}")
                    logging.info(f"语句节点数: {stmt_nodes_count}")
                    logging.info(f"失败测试节点数: {fail_tests_count}")
                    logging.info(f"通过测试节点数: {pass_tests_count}")
                
                method_nodes = graph_data['method_nodes']
                methods_with_labels = 0
                
                for method_id, method_node in method_nodes.items():
                    try:
                        
                        label = method_node['attribute'].get('label', 0.0)
                        method_id = method_node['attribute'].get('id')
                        
                        method_vector = method_node['attribute']['method_attribute']
                        
                        if not isinstance(method_vector, tuple) or len(method_vector) < 3:
                            if self.verbose:
                                logging.warning(f"方法 {method_id} 的向量不是三元组: {type(method_vector)}")
                            continue
                        
                        neighbors = method_node['neighbors']
                        
                        passed_tests = []
                        if 'passed_test' in neighbors and neighbors['passed_test']:
                            for test in neighbors['passed_test']:
                                if isinstance(test, tuple) and len(test) >= 2:
                                    passed_tests.append(test)
                        
                        failed_tests = []
                        if 'failed_test' in neighbors and neighbors['failed_test']:
                            for test in neighbors['failed_test']:
                                if isinstance(test, tuple) and len(test) >= 2:
                                    failed_tests.append(test)
                        
                        history_changes = []
                        if 'method_history' in neighbors and neighbors['method_history']:
                            for history in neighbors['method_history']:
                                if isinstance(history, tuple) and len(history) >= 3:
                                    history_changes.append(history)
                        
                        call_information = []
                        if 'method_call' in neighbors and neighbors['method_call']:
                            for call in neighbors['method_call']:
                                if isinstance(call, tuple) and len(call) >= 3:
                                    call_information.append(call)
                        
                        call_stmt = []
                        if 'call_stmt' in neighbors and neighbors['call_stmt']:
                            for stmt in neighbors['call_stmt']:
                                if isinstance(stmt, tuple) and len(stmt) >= 3:
                                    call_stmt.append(stmt)
                        
                        internal_stmt = []
                        if 'internal_stmt' in neighbors and neighbors['internal_stmt']:
                            for stmt in neighbors['internal_stmt']:
                                if isinstance(stmt, tuple) and len(stmt) >= 3:
                                    internal_stmt.append(stmt)
                        
                        sample = (
                            passed_tests,
                            failed_tests,
                            history_changes,
                            call_information, 
                            [method_vector],
                            call_stmt,
                            internal_stmt,
                            label,
                            method_id
                        )
                        
                        sample_idx = len(self.dataset)
                        self.dataset.append(sample)
                        
                        if label == 1.0:
                            self.dataset_label_true.append(sample_idx)
                        else:
                            self.dataset_label_false.append(sample_idx)
                        
                        methods_with_labels += 1
                    except Exception as e:
                        logging.error(f"处理方法 {method_id} 时出错: {str(e)}")
                
                if self.verbose:
                    logging.info(f"版本 {version} 成功加载了 {methods_with_labels} 个方法节点")
                
                loaded_versions += 1
                    
            except Exception as e:
                logging.error(f"处理版本 {version} 时发生错误: {str(e)}")
                import traceback
                traceback.print_exc()
        
        if self.verbose:
            logging.info(f"总共加载了 {loaded_versions} 个版本，{len(self.dataset)} 个方法样本")
            logging.info(f"正样本数: {len(self.dataset_label_true)}, 负样本数: {len(self.dataset_label_false)}")
        
        if len(self.dataset) == 0:
            logging.warning("没有成功加载任何数据样本，请检查数据文件路径和格式")
    
    def resample_for_balance_label(self):
        
        true_num = len(self.dataset_label_true)
        false_num = len(self.dataset_label_false)
        
        if true_num == 0:
            logging.warning(f"没有缺陷方法，无法进行重采样。")
            return
        
        if false_num == 0:
            logging.info(f"没有非缺陷方法，跳过平衡。")
            return
        
        resample_num = false_num - true_num
        count = 0
        
        if resample_num > 0:  
            while resample_num > 0:
                current_resample_num = min(resample_num, true_num)
                for i in self.dataset_label_true[:current_resample_num]:
                    self.dataset.append(self.dataset[i])
                resample_num -= current_resample_num
                count += current_resample_num
        else:  
            resample_num = -resample_num
            while resample_num > 0:
                current_resample_num = min(resample_num, false_num)
                for i in self.dataset_label_false[:current_resample_num]:
                    self.dataset.append(self.dataset[i])
                resample_num -= current_resample_num
                count += current_resample_num
        
        if self.verbose:
            logging.info(f"重采样添加了 {count} 个样本以平衡标签分布。")
    
    def set_attribute_processing(self, process_flags, process_indices):
        
        assert len(process_flags) == 7, "process_flags长度必须为7"
        assert len(process_indices) == 7, "process_indices长度必须为7"
        
        self.process_flags = process_flags
        self.process_indices = process_indices
        
        if self.verbose:
            logging.info(f"设置属性处理参数: flags={process_flags}, indices={process_indices}")
    
    def process_item_attributes(self, items):
        
        passed_test_cases, failed_test_cases, history_change, call_information, methods, call_stmt, internal_stmt, labels, method_id = items
        
        features = [passed_test_cases, failed_test_cases, history_change, call_information, methods, call_stmt, internal_stmt]
        processed_features = []
        
        for i, (feature, flag, indices) in enumerate(zip(features, self.process_flags, self.process_indices)):
            if flag == 1 and indices:  
                
                if hasattr(feature, 'size'):
                    
                    dims = feature.size()
                    
                    if len(dims) >= 3:  
                        
                        for idx in indices:
                            if idx < dims[1]:  
                                feature[:, idx, :] = 0.0
                                
                feature_num = feature.cpu().numpy()
                
                processed_features.append(feature)
            else:
                processed_features.append(feature)
        
        return (processed_features[0], processed_features[1], processed_features[2], 
                processed_features[3], processed_features[4], processed_features[5], 
                processed_features[6], labels, method_id)

    def __len__(self):
        
        return len(self.dataset)
    
    def __getitem__(self, idx):
        
        new_item = []
        item = self.dataset[idx]
        item_len_limit = self.item_len_limit
        
        for i, t in enumerate(item):
            if isinstance(t, torch.Tensor):
                new_item.append(t)
            elif isinstance(t, bool) or isinstance(t, float) or isinstance(t, int):
                new_item.append(torch.tensor(t, dtype=torch.float32))
            
            else:
                
                if len(t) > item_len_limit:
                    t = t[:item_len_limit]
                
                new_item.append(torch.tensor(np.array(t), dtype=torch.float32))
        passed_test_cases, failed_test_cases, history_change, call_information, methods, call_stmt, internal_stmt, labels, method_id = new_item
        
        if hasattr(passed_test_cases, 'nelement') and passed_test_cases.nelement() == 0:
            passed_test_cases = torch.zeros((1, 2, self.embed_d), dtype=torch.float32)
        
        if hasattr(failed_test_cases, 'nelement') and failed_test_cases.nelement() == 0:
            failed_test_cases = torch.zeros((1, 2, self.embed_d), dtype=torch.float32)
        
        if hasattr(history_change, 'nelement') and history_change.nelement() == 0:
            history_change = torch.zeros((1, 3, self.embed_d), dtype=torch.float32)
        
        if hasattr(call_information, 'nelement') and call_information.nelement() == 0:
            call_information = torch.zeros((1, 3, self.embed_d), dtype=torch.float32)
        
        if hasattr(call_stmt, 'nelement') and call_stmt.nelement() == 0:
            call_stmt = torch.zeros((1, 3, self.embed_d), dtype=torch.float32)
        
        if hasattr(internal_stmt, 'nelement') and internal_stmt.nelement() == 0:
            internal_stmt = torch.zeros((1, 3, self.embed_d), dtype=torch.float32)
        
        processed_items = self.process_item_attributes((passed_test_cases, failed_test_cases, history_change, 
                                                       call_information, methods, call_stmt, internal_stmt, 
                                                       labels, method_id))
        
        return processed_items

def fit_with_max(data_list):
    
    max_length = 0
    for t in data_list:
        t_length = int(t.shape[0])
        if max_length < t_length:
            max_length = t_length
    new_data = []
    for t in data_list:
        new_data.append(torch.cat([t, torch.zeros([max_length - t.shape[0], *t.shape[-2:]])]))
    return torch.stack(new_data, dim=0)

def graph_collate_fn(batch):
    
    passed_test_cases, failed_test_cases, history_change, call_information, methods, call_stmt, internal_stmt, labels, method_id = zip(*batch)
    
    if passed_test_cases:
        passed_test_cases = fit_with_max(passed_test_cases)
    if failed_test_cases:
        failed_test_cases = fit_with_max(failed_test_cases)
    if history_change:
        history_change = fit_with_max(history_change)
    if call_information:
        call_information = fit_with_max(call_information)
    if call_stmt:
        call_stmt = fit_with_max(call_stmt)
    if internal_stmt:
        internal_stmt = fit_with_max(internal_stmt)
        
    return (
        passed_test_cases, 
        failed_test_cases, 
        history_change, 
        call_information, 
        torch.stack(methods, dim=0),
        call_stmt,
        internal_stmt,
        torch.stack(labels, dim=0),
        torch.stack(method_id, dim=0)
    )

if __name__ == '__main__':
    
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
    project_name = "Lang"
    test_version = 1
    dataset = GraphEmbeddingDataset(project_name, [test_version], verbose=True)
    
    print(f"数据集大小: {len(dataset)}")
    if len(dataset) > 0:
        sample = dataset[0]
        print(f"样本类型: {type(sample)}")
        print(f"样本长度: {len(sample)}")
        for i, item in enumerate(sample):
            print(f"第{i}个元素形状: {item.shape if hasattr(item, 'shape') else item}")
    
    from torch.utils.data import DataLoader
    loader = DataLoader(dataset, batch_size=4, collate_fn=graph_collate_fn, shuffle=True)
    for batch in loader:
        print(f"批次大小: {len(batch)}")
        for i, item in enumerate(batch):
            print(f"第{i}个元素形状: {item.shape}")
        break  