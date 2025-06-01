import os
import sys
import pickle
import glob
import re
import numpy as np
from tqdm import tqdm
from collections import defaultdict

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)
from config import ProjectConfig
from defects4j.utils import Utils

def process_version_data(project_name, version, input_file_path):
    
    print(f"处理 {project_name} 版本 {version} 文件: {input_file_path}")
    
    try:
        
        with open(input_file_path, 'rb') as f:
            graph_data = pickle.load(f)
        
        method_nodes = {}
        statement_nodes = {}
        test_nodes_fail = {}
        test_nodes_pass = {}
        
        for stmt_id, stmt_node in graph_data.statement_nodes.items():
            
            mutation_scores = stmt_node.mutation_scores if hasattr(stmt_node, 'mutation_scores') and stmt_node.mutation_scores is not None else []
            spectrum_scores = stmt_node.spectrum_scores if hasattr(stmt_node, 'spectrum_scores') and stmt_node.spectrum_scores is not None else []
            combined_scores = list(mutation_scores) + list(spectrum_scores)
            
            statement_nodes[stmt_id] = {
                'attribute': {
                    'id': stmt_id,
                    'method_id': None,  
                    'stmt_attribute': tuple(stmt_node.stmt_attribute[i] for i in [0, 1, 2]) if stmt_node.stmt_attribute is not None else None,  
                    'combined_scores': combined_scores,  
                    'label': 1.0 if hasattr(stmt_node, 'is_error') and stmt_node.is_error else 0.0  
                },
                'neighbors': {
                    'call_method': [],    
                    'belong_method': [],  
                    'cfg': [],            
                    'dfg': [],            
                    'ast': [],            
                    'passed_test': [],    
                    'failed_test': []     
                }
            }
        
        for test_id, test_node in graph_data.test_nodes_fail.items():
            test_nodes_fail[test_id] = {
                'attribute': {
                    'id': test_id,
                    'test_attribute': test_node.test_attribute  
                }
            }
        
        for test_id, test_node in graph_data.test_nodes_pass.items():
            test_nodes_pass[test_id] = {
                'attribute': {
                    'id': test_id,
                    'test_attribute': test_node.test_attribute  
                }
            }
        
        for method_id, method_node in graph_data.method_nodes.items():
            method_nodes[method_id] = {
                'attribute': {
                    'id': method_id,
                    'method_attribute': method_node.method_attribute,  
                    'label': 1.0 if hasattr(method_node, 'is_error') and method_node.is_error else 0.0  
                },
                'neighbors': {
                    'method_history': [],  
                    'method_call': [],     
                    'passed_test': [],     
                    'failed_test': [],     
                    'call_stmt': [],       
                    'internal_stmt': []    
                }
            }
            
            if hasattr(method_node, 'method_history') and method_node.method_history:
                method_nodes[method_id]['neighbors']['method_history'] = method_node.method_history
            
            if hasattr(method_node, 'method_call') and method_node.method_call:
                method_nodes[method_id]['neighbors']['method_call'] = method_node.method_call
        
        if hasattr(graph_data, 'statement_method_call_edges') and graph_data.statement_method_call_edges:
            for stmt_id, method_id in graph_data.statement_method_call_edges:
                if stmt_id in statement_nodes and method_id in method_nodes:
                    
                    method_attribute = method_nodes[method_id]['attribute']['method_attribute']
                    statement_nodes[stmt_id]['neighbors']['call_method'].append(method_attribute)
                    
                    stmt_attribute = statement_nodes[stmt_id]['attribute']['stmt_attribute']
                    method_nodes[method_id]['neighbors']['call_stmt'].append(stmt_attribute)
        
        if hasattr(graph_data, 'method_to_statements') and graph_data.method_to_statements:
            for method_id, stmt_ids in graph_data.method_to_statements.items():
                if method_id in method_nodes:
                    for stmt_id in stmt_ids:
                        if stmt_id in statement_nodes:
                            
                            method_attribute = method_nodes[method_id]['attribute']['method_attribute']
                            statement_nodes[stmt_id]['neighbors']['belong_method'].append(method_attribute)
                            
                            statement_nodes[stmt_id]['attribute']['method_id'] = method_id
                            
                            stmt_attribute = statement_nodes[stmt_id]['attribute']['stmt_attribute']
                            method_nodes[method_id]['neighbors']['internal_stmt'].append(stmt_attribute)
        
        if hasattr(graph_data, 'data_flow_edges') and graph_data.data_flow_edges:
            for source_id, target_id in graph_data.data_flow_edges:
                if source_id in statement_nodes and target_id in statement_nodes:
                    
                    source_attribute = statement_nodes[source_id]['attribute']['stmt_attribute']
                    target_attribute = statement_nodes[target_id]['attribute']['stmt_attribute']
                    statement_nodes[source_id]['neighbors']['dfg'].append(target_attribute)
                    statement_nodes[target_id]['neighbors']['dfg'].append(source_attribute)
        
        if hasattr(graph_data, 'ast_edges') and graph_data.ast_edges:
            for source_id, target_id in graph_data.ast_edges:
                if source_id in statement_nodes and target_id in statement_nodes:
                    
                    source_attribute = statement_nodes[source_id]['attribute']['stmt_attribute']
                    target_attribute = statement_nodes[target_id]['attribute']['stmt_attribute']
                    statement_nodes[source_id]['neighbors']['ast'].append(target_attribute)
                    statement_nodes[target_id]['neighbors']['ast'].append(source_attribute)
        
        for edge_type in ['control_flow_normal_edges', 'control_flow_yes_edges', 'control_flow_no_edges']:
            if hasattr(graph_data, edge_type) and getattr(graph_data, edge_type):
                for source_id, target_id in getattr(graph_data, edge_type):
                    if source_id in statement_nodes and target_id in statement_nodes:
                        
                        source_attribute = statement_nodes[source_id]['attribute']['stmt_attribute']
                        target_attribute = statement_nodes[target_id]['attribute']['stmt_attribute']
                        statement_nodes[source_id]['neighbors']['cfg'].append(target_attribute)
                        statement_nodes[target_id]['neighbors']['cfg'].append(source_attribute)
        
        if hasattr(graph_data, 'covers_method_edges_fail') and graph_data.covers_method_edges_fail:
            for test_id, method_id in graph_data.covers_method_edges_fail:
                if test_id in test_nodes_fail and method_id in method_nodes:
                    
                    test_attribute = test_nodes_fail[test_id]['attribute']['test_attribute']
                    
                    method_nodes[method_id]['neighbors']['failed_test'].append(test_attribute)
        
        if hasattr(graph_data, 'covers_method_edges_pass') and graph_data.covers_method_edges_pass:
            for test_id, method_id in graph_data.covers_method_edges_pass:
                if test_id in test_nodes_pass and method_id in method_nodes:
                    
                    test_attribute = test_nodes_pass[test_id]['attribute']['test_attribute']
                    
                    method_nodes[method_id]['neighbors']['passed_test'].append(test_attribute)
        
        if hasattr(graph_data, 'covers_stmt_edges_fail') and graph_data.covers_stmt_edges_fail:
            for test_id, stmt_id in graph_data.covers_stmt_edges_fail:
                if test_id in test_nodes_fail and stmt_id in statement_nodes:
                    
                    test_attribute = test_nodes_fail[test_id]['attribute']['test_attribute']
                    
                    statement_nodes[stmt_id]['neighbors']['failed_test'].append(test_attribute)
        
        if hasattr(graph_data, 'covers_stmt_edges_pass') and graph_data.covers_stmt_edges_pass:
            for test_id, stmt_id in graph_data.covers_stmt_edges_pass:
                if test_id in test_nodes_pass and stmt_id in statement_nodes:
                    
                    test_attribute = test_nodes_pass[test_id]['attribute']['test_attribute']
                    
                    statement_nodes[stmt_id]['neighbors']['passed_test'].append(test_attribute)
        
        result = {
            'method_nodes': method_nodes,
            'statement_nodes': statement_nodes,
            'test_nodes_fail': test_nodes_fail,
            'test_nodes_pass': test_nodes_pass
        }
        
        return result
        
    except Exception as e:
        print(f"处理文件 {input_file_path} 时出错: {str(e)}")
        import traceback
        traceback.print_exc()
        return None

def process_project(project_name):
    
    print(f"开始处理项目: {project_name}")
    
    project_path = os.path.join(ProjectConfig.merge_graph_path, project_name)
    
    if not os.path.exists(project_path):
        print(f"项目路径不存在: {project_path}")
        return
    
    versions = [d for d in os.listdir(project_path) if os.path.isdir(os.path.join(project_path, d))]
    print(f"找到 {len(versions)} 个版本目录")
    
    for version in tqdm(versions, desc=f"处理 {project_name} 的版本"):
        
        input_file_path = os.path.join(project_path, version, f"{project_name}_{version}_graph_embedding.pkl")
        
        if not os.path.exists(input_file_path):
            print(f"文件不存在: {input_file_path}")
            continue
        
        version_data = process_version_data(project_name, version, input_file_path)
        
        if version_data:
            
            version_dir = os.path.join(project_path, version)
            output_file = os.path.join(version_dir, f"{project_name}_{version}_graph_embedding_merge.pkl")
            with open(output_file, 'wb') as f:
                pickle.dump(version_data, f)
            print(f"版本 {version} 数据已保存到: {output_file}")

def main():
    
    print("开始处理图向量数据合并...")
    
    projects = Utils.get_projects()
    print(f"找到 {len(projects)} 个项目: {', '.join(projects)}")
    
    for project in projects:
        process_project(project)
    
    print("所有项目处理完成")

if __name__ == "__main__":
    main() 