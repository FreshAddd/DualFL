import logging
import os
import pickle
import random
import sys
import torch
import numpy as np
from torch.utils.data import Dataset
from tqdm import tqdm

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

try:
    from config import ProjectConfig
except ImportError:
    print("无法导入ProjectConfig，请检查路径设置")
    sys.exit(1)

class DictCustomDataset(Dataset):
    def __init__(self, root_dir, selected_versions=[1], embed_d=768, use_resample=True,
                 item_len_limit=256, version_cache=None):
        if version_cache is None:
            version_cache = dict()
        self.root_dir = root_dir
        self.selected_versions = [selected_versions] if isinstance(selected_versions, int) else selected_versions
        self.embed_d = embed_d
        self.use_resample = use_resample
        self.item_len_limit = item_len_limit
        self.dataset = list()
        self.dataset_dict = dict()
        self.version_cache = version_cache
        self.dataset_pass_test_not_none_index_list = list()
        self.dataset_fail_test_not_none_index_list = list()
        self.dataset_history_not_none_index_list = list()
        self.dataset_method_call_not_none_index_list = list()
        self.dataset_label_true = list()
        self.dataset_label_false = list()
        
        if not os.path.exists(root_dir):
            logging.error(f"数据根目录不存在: {root_dir}")
            
            self._create_mock_data()
            return
            
        try:
            self.load()
            self.build()
            logging.info(f"Label True: {len(self.dataset_label_true)}")
            logging.info(f"Label False: {len(self.dataset_label_false)}")
            if use_resample:
                self.resample_for_balance_label()
                self.resample_for_failed_test()
        except Exception as e:
            logging.error(f"数据加载或处理过程中出错: {str(e)}")
            
            self._create_mock_data()

    def _create_mock_data(self):
        
        logging.warning("创建模拟数据用于测试")
        
        self.dataset = []
        self.dataset_label_true = []
        self.dataset_label_false = []
        
        for version in self.selected_versions:
            
            for i in range(10):
                
                passed_tests = torch.randn(5, 2, self.embed_d)
                
                failed_tests = torch.randn(3, 2, self.embed_d)
                
                history = torch.randn(4, 3, self.embed_d)
                
                call_info = torch.randn(6, 3, self.embed_d)
                
                method = torch.randn(1, 3, self.embed_d)
                
                label = 1.0 if random.random() < 0.1 else 0.0
                
                self.dataset.append((passed_tests, failed_tests, history, call_info, method, torch.tensor(label)))
                
                if label == 1.0:
                    self.dataset_label_true.append(len(self.dataset) - 1)
                else:
                    self.dataset_label_false.append(len(self.dataset) - 1)
                    
        logging.info(f"创建了模拟数据集，共 {len(self.dataset)} 条样本")
        logging.info(f"Label True: {len(self.dataset_label_true)}")
        logging.info(f"Label False: {len(self.dataset_label_false)}")

    def load(self):
        self.dataset_dict = dict()
        self.dataset_dict["label"] = dict()
        self.dataset_dict["pass_test"] = dict()
        self.dataset_dict["fail_test"] = dict()
        self.dataset_dict["history"] = dict()
        self.dataset_dict["method_call"] = dict()
        self.dataset_dict["method"] = dict()
        label = self.dataset_dict["label"]
        pass_test = self.dataset_dict["pass_test"]
        fail_test = self.dataset_dict["fail_test"]
        history = self.dataset_dict["history"]
        method_call = self.dataset_dict["method_call"]
        method = self.dataset_dict["method"]
        logging.info(f"Loading...")
        
        for version in tqdm(self.selected_versions):
            version_path = f"{self.root_dir}/{version}"
            
            if not os.path.exists(version_path):
                logging.warning(f"版本目录不存在: {version_path}，跳过此版本")
                continue
                
            labels_file = f"{version_path}/{ProjectConfig.labels_filename}"
            if os.path.exists(labels_file) and os.path.getsize(labels_file) > 0:
                try:
                    with open(labels_file, "rb") as f:
                        t = pickle.load(f)
                        label.update(self.rename_key_with_versions(t, version))
                except (EOFError, pickle.UnpicklingError) as e:
                    logging.error(f"读取标签文件时出错: {labels_file}, 错误: {str(e)}")
                    continue
            else:
                logging.warning(f"标签文件不存在或为空: {labels_file}，跳过此版本")
                continue
                
            try:
                pass_test_file = f"{version_path}/{ProjectConfig.vector_pass_test_attribute_filename}"
                if os.path.exists(pass_test_file) and os.path.getsize(pass_test_file) > 0:
                    with open(pass_test_file, "rb") as f:
                        t = pickle.load(f)
                        pass_test.update(self.rename_key_with_versions(t, version))
                else:
                    logging.warning(f"通过测试特征文件不存在或为空: {pass_test_file}")
            except Exception as e:
                logging.error(f"读取通过测试特征文件时出错: {pass_test_file}, 错误: {str(e)}")
                
            try:
                fail_test_file = f"{version_path}/{ProjectConfig.vector_fail_test_attribute_filename}"
                if os.path.exists(fail_test_file) and os.path.getsize(fail_test_file) > 0:
                    with open(fail_test_file, "rb") as f:
                        t = pickle.load(f)
                        fail_test.update(self.rename_key_with_versions(t, version))
                else:
                    logging.warning(f"失败测试特征文件不存在或为空: {fail_test_file}")
            except Exception as e:
                logging.error(f"读取失败测试特征文件时出错: {fail_test_file}, 错误: {str(e)}")
                
            try:
                history_file = f"{version_path}/{ProjectConfig.vector_method_history_filename}"
                if os.path.exists(history_file) and os.path.getsize(history_file) > 0:
                    with open(history_file, "rb") as f:
                        t = pickle.load(f)
                        history.update(self.rename_key_with_versions(t, version))
                else:
                    logging.warning(f"历史变更特征文件不存在或为空: {history_file}")
            except Exception as e:
                logging.error(f"读取历史变更特征文件时出错: {history_file}, 错误: {str(e)}")
                
            try:
                method_call_file = f"{version_path}/{ProjectConfig.vector_method_call_filename}"
                if os.path.exists(method_call_file) and os.path.getsize(method_call_file) > 0:
                    with open(method_call_file, "rb") as f:
                        t = pickle.load(f)
                        method_call.update(self.rename_key_with_versions(t, version))
                else:
                    logging.warning(f"方法调用特征文件不存在或为空: {method_call_file}")
            except Exception as e:
                logging.error(f"读取方法调用特征文件时出错: {method_call_file}, 错误: {str(e)}")
                
            try:
                method_file = f"{version_path}/{ProjectConfig.vector_method_attribute_filename}"
                if os.path.exists(method_file) and os.path.getsize(method_file) > 0:
                    with open(method_file, "rb") as f:
                        t = pickle.load(f)
                        method.update(self.rename_key_with_versions(t, version))
                else:
                    logging.warning(f"方法特征文件不存在或为空: {method_file}")
            except Exception as e:
                logging.error(f"读取方法特征文件时出错: {method_file}, 错误: {str(e)}")
        
        if not label:
            logging.error("未能成功加载任何标签数据")
            raise ValueError("未能成功加载任何标签数据")

    def build(self):
        self.dataset = list()
        dataset_temp_label = self.dataset_dict["label"]
        dataset_temp_pass_test = self.dataset_dict["pass_test"]
        dataset_temp_fail_test = self.dataset_dict["fail_test"]
        dataset_temp_history = self.dataset_dict["history"]
        dataset_temp_method_call = self.dataset_dict["method_call"]
        dataset_temp_method = self.dataset_dict["method"]
        del self.dataset_dict
        dataset_label_true_temp = self.dataset_label_true
        dataset_label_false_temp = self.dataset_label_false
        dataset_temp = self.dataset
        dataset_pass_test_not_none_index_list_temp = self.dataset_pass_test_not_none_index_list
        dataset_fail_test_not_none_index_list_temp = self.dataset_fail_test_not_none_index_list
        dataset_history_not_none_index_list_temp = self.dataset_history_not_none_index_list
        dataset_method_call_not_none_index_list_temp = self.dataset_method_call_not_none_index_list
        logging.info(f"Building...")
        __pass_test_zero_tensor = 2
        __fail_test_zero_tensor = 2
        __history_zero_tensor = 3
        __method_call_zero_tensor = 3
        i = 0
        num_test = len(dataset_temp_pass_test) + len(dataset_temp_fail_test)
        num_method = len(dataset_temp_label)
        logging.info(f"Find {num_test} tests.")
        logging.info(f"Find {num_method} methods.")
        num_edges = 0
        for key, label in tqdm(dataset_temp_label.items()):
            method = dataset_temp_method.get(key)
            if method is None:
                continue
            method = [method]
            if label:
                dataset_label_true_temp.append(i)
            else:
                dataset_label_false_temp.append(i)
            pass_test = dataset_temp_pass_test.get(key)
            fail_test = dataset_temp_fail_test.get(key)
            history = dataset_temp_history.get(key)
            method_call = dataset_temp_method_call.get(key)
            if pass_test is None:
                pass_test = __pass_test_zero_tensor
            else:
                num_edges += len(pass_test)
            if fail_test is None:
                fail_test = __fail_test_zero_tensor
            else:
                num_edges += len(fail_test)
            if history is None:
                history = __history_zero_tensor
            if method_call is None:
                method_call = __method_call_zero_tensor
            dataset_temp.append((pass_test, fail_test, history, method_call, method, label))
            i += 1
        logging.info(f"Find {num_edges} edges.")

    def rename_key_with_versions(self, d, version):
        new_dict = dict()
        for key, value in d.items():
            new_dict[f"{key}-{version}"] = value
        return new_dict

    def resample_for_balance_label(self):
        true_num = len(self.dataset_label_true)
        false_num = len(self.dataset_label_false)
        if true_num == 0:
            logging.warning(f"No bug method in dataset.")
            logging.info(f"Resampled for balancing label with 0 items.")
            return
        if false_num == 0:
            logging.info(f"Skip balancing label for 0 false label.")
            return
        resample_num = false_num - true_num
        count = 0
        if resample_num >= 0:
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
        logging.info(f"Resampled for balancing label with {count} items.")

    def resample_for_failed_test(self):
        
        for i, t in enumerate(self.dataset):
            if isinstance(t, np.ndarray):
                self.dataset[i][1] = self.instant_resample_for_failed_test(t, length=len(self.dataset[i][0]))

    def instant_resample_for_failed_test(self, data, length=None):
        if length is None:
            length = int(data.shape[0])
        if isinstance(data, torch.Tensor):
            return data
        else:
            data_length = len(data)
            data = np.array(data)
            return data[np.random.choice(length, size=length) % data_length][:random.randint(1, length)]

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        new_item = list()
        item_len_limit = self.item_len_limit
        item = self.dataset[idx]
        for i, t in enumerate(item):
            if isinstance(t, torch.Tensor):
                new_item.append(t)
            elif isinstance(t, bool):
                new_item.append(torch.tensor(t, dtype=torch.float32))
            elif isinstance(t, int):
                new_item.append(torch.zeros((1, t, self.embed_d), dtype=torch.float32))
            else:
                if len(t) > item_len_limit:
                    t = t[:item_len_limit]
                new_item.append(torch.tensor(np.array(t), dtype=torch.float32))
        passed_test_cases, failed_test_cases, history_change, call_information, methods, labels = new_item
        return passed_test_cases, failed_test_cases, history_change, call_information, methods, labels

def fit_with_max(data_list):
    max_length = 0
    for t in data_list:
        t_length = int(t.shape[0])
        if max_length < t_length:
            max_length = t_length
    new_data = list()
    for t in data_list:
        new_data.append(torch.cat([t, torch.zeros([max_length - t.shape[0], *t.shape[-2:]])]))
    return torch.stack(new_data, dim=0)

def dic_collate_fn(batch):
    passed_test_cases, failed_test_cases, history_change, call_information, methods, labels = zip(*batch)
    passed_test_cases = fit_with_max(passed_test_cases)
    failed_test_cases = fit_with_max(failed_test_cases)
    history_change = fit_with_max(history_change)
    call_information = fit_with_max(call_information)
    return passed_test_cases, failed_test_cases, history_change, call_information, torch.stack(methods,
                                                                                               dim=0), torch.stack(
        labels, dim=0) 