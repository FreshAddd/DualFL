import os
import sys
import logging

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

import pandas

from config import ProjectConfig

class __Utils:
    def get_active_bug(self, project):
        
        path = f"{ProjectConfig.path_defects4j}/framework/projects/{project}/active-bugs.csv"
        logging.info(f"尝试读取文件: {path}")
        
        try:
            data = pandas.read_csv(path)
            data = list(data['bug.id'])
            if project == "Closure":
                data = [t for t in data if t <= 133]
            return data
        except FileNotFoundError:
            logging.warning(f"文件不存在: {path}，使用硬编码的活跃bug列表")
            
            active_bugs = {
                "Chart": [str(i) for i in range(1, 27)],
                "Lang": [str(i) for i in range(1, 66)],
                "Math": [str(i) for i in range(1, 107)],
                "Time": [str(i) for i in range(1, 28)],
                "Closure": [str(i) for i in range(1, 134)],
                "Mockito": [str(i) for i in range(1, 39)]
            }
            return active_bugs.get(project, [])

    def get_projects(self):
        return [
            "Lang","Chart","Time", "Math", "Closure","Mockito"
                ]

    def create_path(self, path):
        if not os.path.exists(path):
            os.makedirs(path)

    def get_method_signature(self, method_node):
        if hasattr(method_node, "parameters"):
            method_param_signatures = list()
            for param in method_node.parameters:
                signature = ""
                for t in range(len(param.type.dimensions)):
                    signature += "["
                
                if len(param.type.name) > 1:
                    signature += param.type.name
                else:
                    signature += "Object"
                method_param_signatures.append(signature)
        else:
            method_param_signatures = ""
        if len(method_param_signatures) > 0:
            method_method_signature = ";".join(method_param_signatures)
        else:
            method_method_signature = ""
        if not hasattr(method_node, "return_type") or method_node.return_type is None:
            method_return_name = "void"
        else:
            signature = ""
            for t in range(len(method_node.return_type.dimensions)):
                signature += "["
            if len(method_node.return_type.name) > 1:
                signature += method_node.return_type.name
            else:
                signature += "Object"
            method_return_name = signature
        
        method_return_signature = method_return_name
        return f'({method_method_signature}){method_return_signature}'

    def try_replace_by_dict(self, value, _dict: dict):
        if r := _dict.get(value):
            return r
        else:
            return value

Utils = __Utils()
