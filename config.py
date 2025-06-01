import logging
import os

logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s.%(msecs)03d:%(filename)s:%(lineno)d:%(levelname)s:\t%(message)s',
                    datefmt='%Y-%m-%d  %H:%M:%S', )

class __ProjectConfig:
    def __init__(self):
        self.path_home = "D:"
        self.path_dataset_home = "D:/dataset/defects4j"
        
        self.path_defects4j = "/root/autodl-tmp/defects4j"

        self.version_tokenizer_train_file = "version_tokenizer_train_file.txt"
        self.pretrain_tokenizer_path = f"{self.path_home}/tokenizer.json"
        self.labels_filename = "labels.pkl"
        self.word2vec_train_data_dir = "word2vec_train_data"
        self.word2vec = "word2vec"
        self.word2vec_model = "word2vec.model"
        self.node2vec_model = "node2vec.model"
        self.tmp_path = f"{self.path_home}/tmp"
        self.merge_graph_path = f"{self.path_dataset_home}/graph"

    def create_dirs(self):
        for attr in dir(self):
            if attr.startswith('path'):
                path_value = getattr(self, attr)
                if not os.path.exists(path_value) and isinstance(path_value, str):
                    os.makedirs(path_value, exist_ok=True)
                    logging.info(f"创建目录: {path_value}")
                else:
                    logging.info(f"路径 {path_value} 已存在或不是一个目录。跳过。")

ProjectConfig = __ProjectConfig()
