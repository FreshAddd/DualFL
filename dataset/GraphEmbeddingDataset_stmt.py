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

class GraphEmbeddingDataset_stmt(Dataset):
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
        
        self.process_flags = [0,0,0,0,0,0,0,0]  
        self.process_indices = [[],[],[],[],[],[],[],[]]  
        
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
                
                stmt_nodes = graph_data['statement_nodes']
                stmts_with_labels = 0
                
                for stmt_id, stmt_node in stmt_nodes.items():
                    try:
                        
                        label = stmt_node['attribute'].get('label', 0.0)
                        stmt_id = stmt_node['attribute'].get('id')
                        method_id = stmt_node['attribute'].get('method_id')
                        
                        stmt_vector = stmt_node['attribute']['stmt_attribute']
                        
                        if not isinstance(stmt_vector, tuple) or len(stmt_vector) < 3:
                            if self.verbose:
                                logging.warning(f"语句 {stmt_id} 的向量不是三元组: {type(stmt_vector)}")
                            continue
                        
                        neighbors = stmt_node['neighbors']
                        
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

                        call_method = []
                        if 'call_method' in neighbors and neighbors['call_method']:
                            for call_me in neighbors['call_method']:
                                if isinstance(call_me, tuple) and len(call_me) >= 3:
                                    call_method.append(call_me)

                        belong_method = []
                        if 'belong_method' in neighbors and neighbors['belong_method']:
                            for belong_me in neighbors['belong_method']:
                                if isinstance(belong_me, tuple) and len(belong_me) >= 3:
                                    belong_method.append(belong_me)
                        
                        cfg = []
                        if 'cfg' in neighbors and neighbors['cfg']:
                            for cfg_stmt in neighbors['cfg']:
                                if isinstance(cfg_stmt, tuple) and len(cfg_stmt) >= 3:
                                    cfg.append(cfg_stmt)
                        
                        dfg = []
                        if 'dfg' in neighbors and neighbors['dfg']:
                            for dfg_stmt in neighbors['dfg']:
                                if isinstance(dfg_stmt, tuple) and len(dfg_stmt) >= 3:
                                    dfg.append(dfg_stmt)

                        ast = []
                        if 'ast' in neighbors and neighbors['ast']:
                            for ast_stmt in neighbors['ast']:
                                if isinstance(ast_stmt, tuple) and len(ast_stmt) >= 3:
                                    ast.append(ast_stmt)

                        sample = (
                            passed_tests,
                            failed_tests,
                            call_method,
                            belong_method, 
                            [stmt_vector],
                            cfg,
                            dfg,
                            ast,
                            label,
                            stmt_id,
                            method_id
                        )
                        
                        sample_idx = len(self.dataset)
                        self.dataset.append(sample)
                        
                        if label == 1.0:
                            self.dataset_label_true.append(sample_idx)
                        else:
                            self.dataset_label_false.append(sample_idx)
                        
                        stmts_with_labels += 1
                    except Exception as e:
                        logging.error(f"处理语句 {stmt_id} 时出错: {str(e)}")
                
                if self.verbose:
                    logging.info(f"版本 {version} 成功加载了 {stmts_with_labels} 个语句节点")
                
                loaded_versions += 1
                    
            except Exception as e:
                logging.error(f"处理版本 {version} 时发生错误: {str(e)}")
                import traceback
                traceback.print_exc()
        
        if self.verbose:
            logging.info(f"总共加载了 {loaded_versions} 个版本，{len(self.dataset)} 个语句样本")
            logging.info(f"正样本数: {len(self.dataset_label_true)}, 负样本数: {len(self.dataset_label_false)}")
        
        if len(self.dataset) == 0:
            logging.warning("没有成功加载任何数据样本，请检查数据文件路径和格式")
    
    def resample_for_balance_label(self):
        
        true_num = len(self.dataset_label_true)
        false_num = len(self.dataset_label_false)
        
        if true_num == 0:
            logging.warning(f"没有缺陷语句，无法进行重采样。")
            return
        
        if false_num == 0:
            logging.info(f"没有非缺陷语句，跳过平衡。")
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
        
        assert len(process_flags) == 8, "process_flags长度必须为7"
        assert len(process_indices) == 8, "process_indices长度必须为7"
        
        self.process_flags = process_flags
        self.process_indices = process_indices
        
        if self.verbose:
            logging.info(f"设置属性处理参数: flags={process_flags}, indices={process_indices}")
    
    def process_item_attributes(self, items):
        
        passed_tests, failed_tests, call_method, belong_method, stmt, cfg, dfg, ast, labels, stmt_id, method_id = items
        
        features = [passed_tests, failed_tests, call_method, belong_method, stmt, cfg, dfg, ast]
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
                processed_features[6], processed_features[7], labels, stmt_id, method_id)

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
        passed_test_cases, failed_test_cases, call_method, belong_method, stmt, cfg, dfg, ast, labels, stmt_id, method_id = new_item
        
        if hasattr(passed_test_cases, 'nelement') and passed_test_cases.nelement() == 0:
            passed_test_cases = torch.zeros((1, 2, self.embed_d), dtype=torch.float32)
        
        if hasattr(failed_test_cases, 'nelement') and failed_test_cases.nelement() == 0:
            failed_test_cases = torch.zeros((1, 2, self.embed_d), dtype=torch.float32)
        
        if hasattr(call_method, 'nelement') and call_method.nelement() == 0:
            call_method = torch.zeros((1, 3, self.embed_d), dtype=torch.float32)
        
        if hasattr(belong_method, 'nelement') and belong_method.nelement() == 0:
            belong_method = torch.zeros((1, 3, self.embed_d), dtype=torch.float32)
        
        if hasattr(cfg, 'nelement') and cfg.nelement() == 0:
            cfg = torch.zeros((1, 3, self.embed_d), dtype=torch.float32)
        
        if hasattr(dfg, 'nelement') and dfg.nelement() == 0:
            dfg = torch.zeros((1, 3, self.embed_d), dtype=torch.float32)
        
        if hasattr(ast, 'nelement') and ast.nelement() == 0:
            ast = torch.zeros((1, 3, self.embed_d), dtype=torch.float32)

        processed_items = self.process_item_attributes((passed_test_cases, failed_test_cases, call_method, belong_method, stmt, cfg, dfg, ast, 
                                                       labels, stmt_id, method_id))
        
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

def graph_collate_fn_stmt(batch):
    
    passed_test_cases, failed_test_cases, call_method, belong_method, stmt, cfg, dfg, ast, labels, stmt_id, method_id = zip(*batch)
    
    if passed_test_cases:
        passed_test_cases = fit_with_max(passed_test_cases)
    if failed_test_cases:
        failed_test_cases = fit_with_max(failed_test_cases)
    if call_method:
        call_method = fit_with_max(call_method)
    if belong_method:
        belong_method = fit_with_max(belong_method)
    if cfg:
        cfg = fit_with_max(cfg)
    if dfg:
        dfg = fit_with_max(dfg)
    if ast:
        ast = fit_with_max(ast)
        
    return (
        passed_test_cases, 
        failed_test_cases, 
        call_method, 
        belong_method, 
        torch.stack(stmt, dim=0),
        cfg,
        dfg,
        ast,
        torch.stack(labels, dim=0),
        torch.stack(stmt_id, dim=0),
        torch.stack(method_id, dim=0)
    )

if __name__ == '__main__':
    
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
    project_name = "Lang"
    test_version = 1
    dataset = GraphEmbeddingDataset_stmt(project_name, [test_version], verbose=True)
    
    print(f"数据集大小: {len(dataset)}")
    if len(dataset) > 0:
        sample = dataset[0]
        print(f"样本类型: {type(sample)}")
        print(f"样本长度: {len(sample)}")
        for i, item in enumerate(sample):
            print(f"第{i}个元素形状: {item.shape if hasattr(item, 'shape') else item}")
    
    from torch.utils.data import DataLoader
    loader = DataLoader(dataset, batch_size=4, collate_fn=graph_collate_fn_stmt, shuffle=True)
    for batch in loader:
        print(f"批次大小: {len(batch)}")
        for i, item in enumerate(batch):
            print(f"第{i}个元素形状: {item.shape}")
        break  