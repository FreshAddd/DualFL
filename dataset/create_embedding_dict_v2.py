import pickle
import os
import glob
import re
import numpy as np
import csv
from tqdm import tqdm
from collections import defaultdict

INPUT_DIR = "data_output_clean/lang_embeddings"

OUTPUT_DIR = "data_output_clean/merged_lang_embeddings"
os.makedirs(OUTPUT_DIR, exist_ok=True)

global_method_id_counter = 1

method_faulty_index_all = []
stmt_faulty_index_all = []

AST_TYPE_MAPPING = {
    'AssignmentStatement': 0.00,
    'declaration': 0.05,
    'IfStatement': 0.10,
    'ForStatement': 0.15,
    'WhileStatement': 0.20,
    'DoStatement': 0.25,
    'SwitchStatement': 0.30,
    'case_statement': 0.35,
    'TryStatement': 0.40,
    'CatchClause': 0.45,
    'FinallyBlock': 0.50,
    'ReturnStatement': 0.55,
    'ThrowStatement': 0.60,
    'BreakStatement': 0.65,
    'ContinueStatement': 0.70,
    'local_variable_declaration': 0.75,
    'assert_statement': 0.80,
    'synchronized_statement': 0.85,
    'block': 0.90,
    'MethodInvocationStatement': 0.95
}

def get_ast_type_encoding(ast_type):
    
    return AST_TYPE_MAPPING.get(ast_type, 0.00)

def create_statement_additional_vector(stmt_node):
    
    vector_7d = []
    
    if stmt_node.mutation_scores is not None:
        vector_7d.extend(stmt_node.mutation_scores)
    else:
        
        vector_7d.extend([0.0] * 4)
    
    if stmt_node.spectrum_scores is not None:
        vector_7d.extend(stmt_node.spectrum_scores)
    else:
        
        vector_7d.extend([0.0] * 3)
    
    ast_type_encoding = get_ast_type_encoding(stmt_node.ast_type)
    
    additional_vector = vector_7d + [ast_type_encoding]
    
    return additional_vector

def vector_exists_in_list(vec_list, target_vector):
    
    for vec_item in vec_list:
        
        existing_vector = vec_item.get('semantic_vector')
        
        if existing_vector is None and target_vector is None:
            return True
            
        if existing_vector is None or target_vector is None:
            continue
            
        try:
            existing_np = np.array(existing_vector)
            target_np = np.array(target_vector)
            
            if existing_np.shape != target_np.shape:
                continue
                
            if np.allclose(existing_np, target_np, rtol=1e-5, atol=1e-8):
                return True
        except:
            
            if existing_vector == target_vector:
                return True
    
    return False

def main():
    
    global global_method_id_counter
    global method_faulty_index_all
    global stmt_faulty_index_all
    
    pkl_files = glob.glob(os.path.join(INPUT_DIR, "*_graph_with_embeddings.pkl"))
    print(f"找到 {len(pkl_files)} 个PKL文件")
    
    if not pkl_files:
        print("没有找到任何PKL文件！")
        return
    
    first_file = os.path.basename(pkl_files[0])
    match = re.match(r"([^_]+)_", first_file)
    if match:
        project_name = match.group(1)
    else:
        project_name = "Default"
    
    print(f"使用项目名称: {project_name}")
    OUTPUT_FILE = os.path.join(OUTPUT_DIR, f"{project_name}.pkl")
    ID_MAPPING_FILE = os.path.join(OUTPUT_DIR, f"{project_name}_id_mapping.txt")
    
    result = {}
    
    id_mappings = {}
    
    all_nodes_info = {}
    
    edge_relations = {
        'method_call': defaultdict(list),        
        'stmt_method_call': defaultdict(list),   
        'method_internal_stmts': defaultdict(list),  
        'test_covers_method': defaultdict(list),  
        'stmt_cfg_normal': defaultdict(list),    
        'stmt_cfg_judge': defaultdict(list),     
        'stmt_dfg': defaultdict(list),           
        'stmt_ast': defaultdict(list),           
        'test_covers_stmt': defaultdict(list)    
    }
    
    print("第一次遍历：处理节点和收集信息...")
    for pkl_file in tqdm(pkl_files, desc="处理PKL文件(第一阶段)"):
        
        pkl_basename = os.path.basename(pkl_file)
        version_match = re.match(r"[^_]+_(\d+)_", pkl_basename)
        if version_match:
            version_number = version_match.group(1)
            version_key = f"version-{version_number}"
        else:
            
            version_key = f"version-unknown-{pkl_basename}"
        
        print(f"处理 {pkl_basename} (版本: {version_key})")
        
        try:
            
            with open(pkl_file, 'rb') as f:
                graph_data = pickle.load(f)
            
            if version_key not in result:
                result[version_key] = {
                    'method_nodes_embedding': {},
                    'statement_nodes_embedding': {},
                    'test_nodes_fail_embedding': {},
                    'test_nodes_pass_embedding': {}
                }
            
            method_counter = 0
            statement_counter = 0
            test_fail_counter = 0
            test_pass_counter = 0
            
            current_nodes_info = {
                "方法": {},
                "语句": {},
                "失败测试": {},
                "通过测试": {}
            }
            
            current_file_stmt_additional_vectors = {}
            
            method_to_statements_map = graph_data.method_to_statements
            
            method_faulty_index = []
            stmt_faulty_index = []
            
            if hasattr(graph_data, 'method_faulty_index'):
                method_faulty_index = graph_data.method_faulty_index
                
                print(f"method_faulty_index内容: {method_faulty_index}")
            
            if hasattr(graph_data, 'stmt_faulty_index'):
                stmt_faulty_index = graph_data.stmt_faulty_index
                
                print(f"stmt_faulty_index内容: {stmt_faulty_index}")
            
            for method_id, stmt_ids in method_to_statements_map.items():
                for stmt_id in stmt_ids:
                    edge_relations['method_internal_stmts'][(pkl_basename, method_id)].append((pkl_basename, stmt_id))
            
            for stmt_id, stmt_node in graph_data.statement_nodes.items():
                statement_counter += 1
                
                additional_vector = create_statement_additional_vector(stmt_node)
                
                current_file_stmt_additional_vectors[stmt_id] = additional_vector[:7]
                
                result[version_key]['statement_nodes_embedding'][statement_counter] = {
                    'attribute': {
                        'id': stmt_node.id,
                        'node_type': stmt_node.type,
                        'semantic_vector': stmt_node.vector,
                        'fault_label': stmt_node.is_error,
                        'additional_vector': additional_vector,
                        'final_vector': None
                    },
                    'neighbors': {
                        'call_method': [],    
                        'belong_method': [],  
                        'cfg_normal': [],     
                        'cfg_judge': [],      
                        'dfg': [],            
                        'ast': [],            
                        'passed_test': [],    
                        'failed_test': []     
                    }
                }
                
                current_nodes_info["语句"][stmt_node.id] = (statement_counter, stmt_node.vector, additional_vector)
            
            for method_id, method_node in graph_data.method_nodes.items():
                method_counter += 1
                
                stmt_ids = method_to_statements_map.get(method_id, [])
                
                if stmt_ids:
                    
                    stmt_vectors = [current_file_stmt_additional_vectors.get(stmt_id, [0.0] * 7) for stmt_id in stmt_ids]
                    
                    if stmt_vectors:
                        
                        stmt_vectors_array = np.array(stmt_vectors)
                        max_pooling = np.max(stmt_vectors_array, axis=0).tolist()
                    else:
                        
                        max_pooling = [0.0] * 7
                else:
                    
                    max_pooling = [0.0] * 7
                
                method_additional_vector = max_pooling + [2.0]
                
                method_history = []
                
                if hasattr(method_node, 'history_vector') and method_node.history_vector:
                    
                    for idx, history_vec in enumerate(method_node.history_vector, 1):
                        
                        history_additional_vector = max_pooling + [2.0 + idx / 10.0]  
                        
                        history_entry = {
                            'semantic_vector': history_vec,
                            'additional_vector': history_additional_vector
                        }
                        
                        method_history.append(history_entry)
                
                result[version_key]['method_nodes_embedding'][method_counter] = {
                    'attribute': {
                        'id': method_node.id,
                        'node_type': method_node.type,
                        'semantic_vector': method_node.vector,
                        'fault_label': method_node.is_error,
                        'additional_vector': method_additional_vector,
                        'final_vector': None
                    },
                    'neighbors': {
                        'method_history': method_history,
                        'call_method': [],    
                        'call_stmt': [],      
                        'internal_stmt': [],  
                        'passed_test': [],    
                        'failed_test': []     
                    }
                }
                
                current_nodes_info["方法"][method_id] = (method_counter, method_node.vector, method_additional_vector)
            
            for test_id, test_node in graph_data.test_nodes_fail.items():
                test_fail_counter += 1
                
                test_fail_additional_vector = [0.0] * 7 + [5.5]
                
                result[version_key]['test_nodes_fail_embedding'][test_fail_counter] = {
                    'attribute': {
                        'id': test_node.id,
                        'semantic_vector': test_node.vector,
                        'additional_vector': test_fail_additional_vector,
                        'final_vector': None
                    }
                }
                
                current_nodes_info["失败测试"][test_node.id] = (test_fail_counter, test_node.vector, test_fail_additional_vector)
            
            for test_id, test_node in graph_data.test_nodes_pass.items():
                test_pass_counter += 1
                
                test_pass_additional_vector = [0.0] * 7 + [5.0]
                
                result[version_key]['test_nodes_pass_embedding'][test_pass_counter] = {
                    'attribute': {
                        'id': test_node.id,
                        'semantic_vector': test_node.vector,
                        'additional_vector': test_pass_additional_vector,
                        'final_vector': None
                    }
                }
                
                current_nodes_info["通过测试"][test_node.id] = (test_pass_counter, test_node.vector, test_pass_additional_vector)
            
            all_nodes_info[pkl_basename] = current_nodes_info
            
            version_number = version_key.split('-')[1] if '-' in version_key else 'unknown'
            new_version_key = f"version_{version_number}"
            
            if version_number not in id_mappings:
                id_mappings[version_number] = {
                    'methods': {},  
                    'statements': {}  
                }
            
            if new_version_key not in result:
                result[new_version_key] = {}
            
            for method_id, method_node in graph_data.method_nodes.items():
                method_new_id = current_nodes_info["方法"][method_id][0]
                
                group_id = method_new_id
                
                method_info = result[version_key]['method_nodes_embedding'][method_new_id]
                
                method_global_id = global_method_id_counter
                global_method_id_counter += 1
                
                id_mappings[version_number]['methods'][method_id] = method_global_id
                
                result[new_version_key][group_id] = {
                    'method': {
                        'original_id': method_id,  
                        'global_id': method_global_id,  
                        **method_info  
                    },
                    'statements': []  
                }
                
                stmt_ids = method_to_statements_map.get(method_id, [])
                
                for stmt_idx, stmt_id in enumerate(stmt_ids, 1):
                    if stmt_id in current_nodes_info["语句"]:
                        stmt_new_id = current_nodes_info["语句"][stmt_id][0]
                        
                        stmt_info = result[version_key]['statement_nodes_embedding'][stmt_new_id]
                        
                        stmt_global_id = f"{method_global_id}_{stmt_idx}"
                        
                        id_mappings[version_number]['statements'][stmt_id] = stmt_global_id
                        
                        result[new_version_key][group_id]['statements'].append({
                            'original_id': stmt_id,  
                            'global_id': stmt_global_id,  
                            **stmt_info  
                        })
            
            if hasattr(graph_data, 'method_method_call_edges') and graph_data.method_method_call_edges:
                for source_id, target_id in graph_data.method_method_call_edges:
                    edge_relations['method_call'][(pkl_basename, source_id)].append((pkl_basename, target_id))
            
            if hasattr(graph_data, 'statement_method_call_edges') and graph_data.statement_method_call_edges:
                for stmt_id, method_id in graph_data.statement_method_call_edges:
                    edge_relations['stmt_method_call'][(pkl_basename, stmt_id, 'stmt')].append((pkl_basename, method_id, 'method'))
                    edge_relations['stmt_method_call'][(pkl_basename, method_id, 'method')].append((pkl_basename, stmt_id, 'stmt'))
            
            if hasattr(graph_data, 'covers_method_edges_fail') and graph_data.covers_method_edges_fail:
                for test_id, method_id in graph_data.covers_method_edges_fail:
                    edge_relations['test_covers_method'][(pkl_basename, method_id, 'method')].append((pkl_basename, test_id, 'test', 'fail'))

            if hasattr(graph_data, 'covers_method_edges_pass') and graph_data.covers_method_edges_pass:
                for test_id, method_id in graph_data.covers_method_edges_pass:
                    edge_relations['test_covers_method'][(pkl_basename, method_id, 'method')].append((pkl_basename, test_id, 'test', 'pass'))
            
            if hasattr(graph_data, 'control_flow_normal_edges') and graph_data.control_flow_normal_edges:
                for source_id, target_id in graph_data.control_flow_normal_edges:
                    edge_relations['stmt_cfg_normal'][(pkl_basename, source_id)].append((pkl_basename, target_id))
            
            if hasattr(graph_data, 'control_flow_yes_edges') and graph_data.control_flow_yes_edges:
                for source_id, target_id in graph_data.control_flow_yes_edges:
                    edge_relations['stmt_cfg_judge'][(pkl_basename, source_id)].append((pkl_basename, target_id))
            
            if hasattr(graph_data, 'control_flow_no_edges') and graph_data.control_flow_no_edges:
                for source_id, target_id in graph_data.control_flow_no_edges:
                    edge_relations['stmt_cfg_judge'][(pkl_basename, source_id)].append((pkl_basename, target_id))
            
            if hasattr(graph_data, 'data_flow_edges') and graph_data.data_flow_edges:
                for source_id, target_id in graph_data.data_flow_edges:
                    edge_relations['stmt_dfg'][(pkl_basename, source_id)].append((pkl_basename, target_id))
            
            if hasattr(graph_data, 'ast_edges') and graph_data.ast_edges:
                for source_id, target_id in graph_data.ast_edges:
                    edge_relations['stmt_ast'][(pkl_basename, source_id)].append((pkl_basename, target_id))
            
            if hasattr(graph_data, 'covers_stmt_edges_fail') and graph_data.covers_stmt_edges_fail:
                for test_id, stmt_id in graph_data.covers_stmt_edges_fail:
                    edge_relations['test_covers_stmt'][(pkl_basename, stmt_id, 'stmt')].append((pkl_basename, test_id, 'test', 'fail'))
            
            if hasattr(graph_data, 'covers_stmt_edges_pass') and graph_data.covers_stmt_edges_pass:
                for test_id, stmt_id in graph_data.covers_stmt_edges_pass:
                    edge_relations['test_covers_stmt'][(pkl_basename, stmt_id, 'stmt')].append((pkl_basename, test_id, 'test', 'pass'))
                
            for method_id_set in method_faulty_index:
                
                if isinstance(method_id_set, list):
                    original_ids = []
                    new_ids = []
                    for method_id in method_id_set:
                        original_ids.append(method_id)
                        if method_id in id_mappings[version_number]['methods']:
                            new_method_id = id_mappings[version_number]['methods'][method_id]
                            new_ids.append(new_method_id)
                        else:
                            
                            new_ids.append(None)
                    
                    if any(new_id is not None for new_id in new_ids):
                        method_faulty_index_all.append((project_name, version_number, original_ids, new_ids))
                
                else:
                    method_id = method_id_set
                    if method_id in id_mappings[version_number]['methods']:
                        new_method_id = id_mappings[version_number]['methods'][method_id]
                        method_faulty_index_all.append((project_name, version_number, [method_id], [new_method_id]))
            
            for stmt_id_set in stmt_faulty_index:
                
                if isinstance(stmt_id_set, list):
                    original_ids = []
                    new_ids = []
                    for stmt_id in stmt_id_set:
                        original_ids.append(stmt_id)
                        if stmt_id in id_mappings[version_number]['statements']:
                            new_stmt_id = id_mappings[version_number]['statements'][stmt_id]
                            new_ids.append(new_stmt_id)
                        else:
                            
                            new_ids.append(None)
                    
                    if any(new_id is not None for new_id in new_ids):
                        stmt_faulty_index_all.append((project_name, version_number, original_ids, new_ids))
                
                else:
                    stmt_id = stmt_id_set
                    if stmt_id in id_mappings[version_number]['statements']:
                        new_stmt_id = id_mappings[version_number]['statements'][stmt_id]
                        stmt_faulty_index_all.append((project_name, version_number, [stmt_id], [new_stmt_id]))
            
        except Exception as e:
            print(f"处理文件 {pkl_file} 时出错: {str(e)}")
    
    print("第二次遍历：填充边关系...")
    
    print("填充方法调用关系...")
    for (source_pkl, source_id), targets in tqdm(edge_relations['method_call'].items(), desc="处理方法间调用关系"):
        
        version_match = re.match(r"[^_]+_(\d+)_", source_pkl)
        if version_match:
            version_number = version_match.group(1)
            version_key = f"version-{version_number}"
        else:
            version_key = f"version-unknown-{source_pkl}"
            
        if version_key not in result:
            continue
            
        if source_pkl in all_nodes_info and source_id in all_nodes_info[source_pkl]["方法"]:
            source_new_id, _, _ = all_nodes_info[source_pkl]["方法"][source_id]
            
            for target_pkl, target_id in targets:
                
                if target_pkl != source_pkl:
                    continue
                    
                if target_pkl in all_nodes_info and target_id in all_nodes_info[target_pkl]["方法"]:
                    _, target_semantic_vector, target_additional_vector = all_nodes_info[target_pkl]["方法"][target_id]
                    
                    call_method_entry = {
                        'semantic_vector': target_semantic_vector,
                        'additional_vector': target_additional_vector
                    }
                    
                    if not vector_exists_in_list(result[version_key]['method_nodes_embedding'][source_new_id]['neighbors']['call_method'], 
                                              target_semantic_vector):
                        result[version_key]['method_nodes_embedding'][source_new_id]['neighbors']['call_method'].append(call_method_entry)
    
    print("填充语句-方法调用关系...")
    for (source_pkl, source_id, source_type), targets in tqdm(edge_relations['stmt_method_call'].items(), desc="处理语句-方法调用关系"):
        
        version_match = re.match(r"[^_]+_(\d+)_", source_pkl)
        if version_match:
            version_number = version_match.group(1)
            version_key = f"version-{version_number}"
        else:
            version_key = f"version-unknown-{source_pkl}"
            
        if version_key not in result:
            continue
            
        if source_type == 'stmt':  
            
            if source_pkl in all_nodes_info and source_id in all_nodes_info[source_pkl]["语句"]:
                source_new_id, _, _ = all_nodes_info[source_pkl]["语句"][source_id]
                
                for target_pkl, target_id, target_type in targets:
                    
                    if target_pkl != source_pkl:
                        continue
                        
                    if target_type == 'method':  
                        if target_pkl in all_nodes_info and target_id in all_nodes_info[target_pkl]["方法"]:
                            _, target_semantic_vector, target_additional_vector = all_nodes_info[target_pkl]["方法"][target_id]
                            
                            call_method_entry = {
                                'semantic_vector': target_semantic_vector,
                                'additional_vector': target_additional_vector
                            }
                            
                            if not vector_exists_in_list(result[version_key]['statement_nodes_embedding'][source_new_id]['neighbors']['call_method'], 
                                                      target_semantic_vector):
                                result[version_key]['statement_nodes_embedding'][source_new_id]['neighbors']['call_method'].append(call_method_entry)
        
        elif source_type == 'method':  
            
            if source_pkl in all_nodes_info and source_id in all_nodes_info[source_pkl]["方法"]:
                source_new_id, _, _ = all_nodes_info[source_pkl]["方法"][source_id]
                
                for target_pkl, target_id, target_type in targets:
                    
                    if target_pkl != source_pkl:
                        continue
                        
                    if target_type == 'stmt':  
                        if target_pkl in all_nodes_info and target_id in all_nodes_info[target_pkl]["语句"]:
                            _, target_semantic_vector, target_additional_vector = all_nodes_info[target_pkl]["语句"][target_id]
                            
                            call_stmt_entry = {
                                'semantic_vector': target_semantic_vector,
                                'additional_vector': target_additional_vector
                            }
                            
                            if not vector_exists_in_list(result[version_key]['method_nodes_embedding'][source_new_id]['neighbors']['call_stmt'], 
                                                      target_semantic_vector):
                                result[version_key]['method_nodes_embedding'][source_new_id]['neighbors']['call_stmt'].append(call_stmt_entry)
    
    print("填充方法-内部语句关系...")
    for (method_pkl, method_id), stmt_list in tqdm(edge_relations['method_internal_stmts'].items(), desc="处理方法-内部语句关系"):
        
        version_match = re.match(r"[^_]+_(\d+)_", method_pkl)
        if version_match:
            version_number = version_match.group(1)
            version_key = f"version-{version_number}"
        else:
            version_key = f"version-unknown-{method_pkl}"
            
        if version_key not in result:
            continue
            
        if method_pkl in all_nodes_info and method_id in all_nodes_info[method_pkl]["方法"]:
            method_new_id, method_semantic_vector, method_additional_vector = all_nodes_info[method_pkl]["方法"][method_id]
            
            for stmt_pkl, stmt_id in stmt_list:
                
                if stmt_pkl != method_pkl:
                    continue
                    
                if stmt_pkl in all_nodes_info and stmt_id in all_nodes_info[stmt_pkl]["语句"]:
                    stmt_new_id, stmt_semantic_vector, stmt_additional_vector = all_nodes_info[stmt_pkl]["语句"][stmt_id]
                    
                    internal_stmt_entry = {
                        'semantic_vector': stmt_semantic_vector,
                        'additional_vector': stmt_additional_vector
                    }
                    
                    if not vector_exists_in_list(result[version_key]['method_nodes_embedding'][method_new_id]['neighbors']['internal_stmt'], 
                                              stmt_semantic_vector):
                        result[version_key]['method_nodes_embedding'][method_new_id]['neighbors']['internal_stmt'].append(internal_stmt_entry)
                    
                    belong_method_entry = {
                        'semantic_vector': method_semantic_vector,
                        'additional_vector': method_additional_vector
                    }
                    
                    if not vector_exists_in_list(result[version_key]['statement_nodes_embedding'][stmt_new_id]['neighbors']['belong_method'], 
                                              method_semantic_vector):
                        result[version_key]['statement_nodes_embedding'][stmt_new_id]['neighbors']['belong_method'].append(belong_method_entry)
    
    print("填充测试覆盖方法关系...")
    for (method_pkl, method_id, _), test_list in tqdm(edge_relations['test_covers_method'].items(), desc="处理测试覆盖方法关系"):
        
        version_match = re.match(r"[^_]+_(\d+)_", method_pkl)
        if version_match:
            version_number = version_match.group(1)
            version_key = f"version-{version_number}"
        else:
            version_key = f"version-unknown-{method_pkl}"
            
        if version_key not in result:
            continue
            
        if method_pkl in all_nodes_info and method_id in all_nodes_info[method_pkl]["方法"]:
            method_new_id, _, _ = all_nodes_info[method_pkl]["方法"][method_id]
            
            for test_pkl, test_id, _, test_type in test_list:
                
                if test_pkl != method_pkl:
                    continue
                    
                if test_pkl in all_nodes_info:
                    if test_type == 'fail' and test_id in all_nodes_info[test_pkl]["失败测试"]:
                        _, test_semantic_vector, test_additional_vector = all_nodes_info[test_pkl]["失败测试"][test_id]
                        
                        test_entry = {
                            'semantic_vector': test_semantic_vector,
                            'additional_vector': test_additional_vector
                        }
                        
                        if not vector_exists_in_list(result[version_key]['method_nodes_embedding'][method_new_id]['neighbors']['failed_test'], 
                                                  test_semantic_vector):
                            result[version_key]['method_nodes_embedding'][method_new_id]['neighbors']['failed_test'].append(test_entry)
                            
                    elif test_type == 'pass' and test_id in all_nodes_info[test_pkl]["通过测试"]:
                        _, test_semantic_vector, test_additional_vector = all_nodes_info[test_pkl]["通过测试"][test_id]
                        
                        test_entry = {
                            'semantic_vector': test_semantic_vector,
                            'additional_vector': test_additional_vector
                        }
                        
                        if not vector_exists_in_list(result[version_key]['method_nodes_embedding'][method_new_id]['neighbors']['passed_test'], 
                                                  test_semantic_vector):
                            result[version_key]['method_nodes_embedding'][method_new_id]['neighbors']['passed_test'].append(test_entry)
    
    print("填充语句之间的控制流关系...")
    
    for (source_pkl, source_id), targets in tqdm(edge_relations['stmt_cfg_normal'].items(), desc="处理语句普通控制流关系"):
        
        version_match = re.match(r"[^_]+_(\d+)_", source_pkl)
        if version_match:
            version_number = version_match.group(1)
            version_key = f"version-{version_number}"
        else:
            version_key = f"version-unknown-{source_pkl}"
            
        if version_key not in result:
            continue
            
        if source_pkl in all_nodes_info and source_id in all_nodes_info[source_pkl]["语句"]:
            source_new_id, _, _ = all_nodes_info[source_pkl]["语句"][source_id]
            
            for target_pkl, target_id in targets:
                
                if target_pkl != source_pkl:
                    continue
                    
                if target_pkl in all_nodes_info and target_id in all_nodes_info[target_pkl]["语句"]:
                    _, target_semantic_vector, target_additional_vector = all_nodes_info[target_pkl]["语句"][target_id]
                    
                    cfg_normal_entry = {
                        'semantic_vector': target_semantic_vector,
                        'additional_vector': target_additional_vector
                    }
                    
                    if not vector_exists_in_list(result[version_key]['statement_nodes_embedding'][source_new_id]['neighbors']['cfg_normal'], 
                                              target_semantic_vector):
                        result[version_key]['statement_nodes_embedding'][source_new_id]['neighbors']['cfg_normal'].append(cfg_normal_entry)
    
    for (source_pkl, source_id), targets in tqdm(edge_relations['stmt_cfg_judge'].items(), desc="处理语句条件控制流关系"):
        
        version_match = re.match(r"[^_]+_(\d+)_", source_pkl)
        if version_match:
            version_number = version_match.group(1)
            version_key = f"version-{version_number}"
        else:
            version_key = f"version-unknown-{source_pkl}"
            
        if version_key not in result:
            continue
            
        if source_pkl in all_nodes_info and source_id in all_nodes_info[source_pkl]["语句"]:
            source_new_id, _, _ = all_nodes_info[source_pkl]["语句"][source_id]
            
            for target_pkl, target_id in targets:
                
                if target_pkl != source_pkl:
                    continue
                    
                if target_pkl in all_nodes_info and target_id in all_nodes_info[target_pkl]["语句"]:
                    _, target_semantic_vector, target_additional_vector = all_nodes_info[target_pkl]["语句"][target_id]
                    
                    cfg_judge_entry = {
                        'semantic_vector': target_semantic_vector,
                        'additional_vector': target_additional_vector
                    }
                    
                    if not vector_exists_in_list(result[version_key]['statement_nodes_embedding'][source_new_id]['neighbors']['cfg_judge'], 
                                              target_semantic_vector):
                        result[version_key]['statement_nodes_embedding'][source_new_id]['neighbors']['cfg_judge'].append(cfg_judge_entry)
    
    print("填充语句之间的数据流关系...")
    for (source_pkl, source_id), targets in tqdm(edge_relations['stmt_dfg'].items(), desc="处理语句数据流关系"):
        
        version_match = re.match(r"[^_]+_(\d+)_", source_pkl)
        if version_match:
            version_number = version_match.group(1)
            version_key = f"version-{version_number}"
        else:
            version_key = f"version-unknown-{source_pkl}"
            
        if version_key not in result:
            continue
            
        if source_pkl in all_nodes_info and source_id in all_nodes_info[source_pkl]["语句"]:
            source_new_id, _, _ = all_nodes_info[source_pkl]["语句"][source_id]
            
            for target_pkl, target_id in targets:
                
                if target_pkl != source_pkl:
                    continue
                    
                if target_pkl in all_nodes_info and target_id in all_nodes_info[target_pkl]["语句"]:
                    _, target_semantic_vector, target_additional_vector = all_nodes_info[target_pkl]["语句"][target_id]
                    
                    dfg_entry = {
                        'semantic_vector': target_semantic_vector,
                        'additional_vector': target_additional_vector
                    }
                    
                    if not vector_exists_in_list(result[version_key]['statement_nodes_embedding'][source_new_id]['neighbors']['dfg'], 
                                              target_semantic_vector):
                        result[version_key]['statement_nodes_embedding'][source_new_id]['neighbors']['dfg'].append(dfg_entry)
    
    print("填充语句之间的AST关系...")
    for (source_pkl, source_id), targets in tqdm(edge_relations['stmt_ast'].items(), desc="处理语句AST关系"):
        
        version_match = re.match(r"[^_]+_(\d+)_", source_pkl)
        if version_match:
            version_number = version_match.group(1)
            version_key = f"version-{version_number}"
        else:
            version_key = f"version-unknown-{source_pkl}"
            
        if version_key not in result:
            continue
            
        if source_pkl in all_nodes_info and source_id in all_nodes_info[source_pkl]["语句"]:
            source_new_id, _, _ = all_nodes_info[source_pkl]["语句"][source_id]
            
            for target_pkl, target_id in targets:
                
                if target_pkl != source_pkl:
                    continue
                    
                if target_pkl in all_nodes_info and target_id in all_nodes_info[target_pkl]["语句"]:
                    _, target_semantic_vector, target_additional_vector = all_nodes_info[target_pkl]["语句"][target_id]
                    
                    ast_entry = {
                        'semantic_vector': target_semantic_vector,
                        'additional_vector': target_additional_vector
                    }
                    
                    if not vector_exists_in_list(result[version_key]['statement_nodes_embedding'][source_new_id]['neighbors']['ast'], 
                                              target_semantic_vector):
                        result[version_key]['statement_nodes_embedding'][source_new_id]['neighbors']['ast'].append(ast_entry)
    
    print("填充测试覆盖语句关系...")
    for (stmt_pkl, stmt_id, _), test_list in tqdm(edge_relations['test_covers_stmt'].items(), desc="处理测试覆盖语句关系"):
        
        version_match = re.match(r"[^_]+_(\d+)_", stmt_pkl)
        if version_match:
            version_number = version_match.group(1)
            version_key = f"version-{version_number}"
        else:
            version_key = f"version-unknown-{stmt_pkl}"
            
        if version_key not in result:
            continue
            
        if stmt_pkl in all_nodes_info and stmt_id in all_nodes_info[stmt_pkl]["语句"]:
            stmt_new_id, _, _ = all_nodes_info[stmt_pkl]["语句"][stmt_id]
            
            for test_pkl, test_id, _, test_type in test_list:
                
                if test_pkl != stmt_pkl:
                    continue
                    
                if test_pkl in all_nodes_info:
                    if test_type == 'fail' and test_id in all_nodes_info[test_pkl]["失败测试"]:
                        _, test_semantic_vector, test_additional_vector = all_nodes_info[test_pkl]["失败测试"][test_id]
                        
                        test_entry = {
                            'semantic_vector': test_semantic_vector,
                            'additional_vector': test_additional_vector
                        }
                        
                        if not vector_exists_in_list(result[version_key]['statement_nodes_embedding'][stmt_new_id]['neighbors']['failed_test'], 
                                                  test_semantic_vector):
                            result[version_key]['statement_nodes_embedding'][stmt_new_id]['neighbors']['failed_test'].append(test_entry)
                            
                    elif test_type == 'pass' and test_id in all_nodes_info[test_pkl]["通过测试"]:
                        _, test_semantic_vector, test_additional_vector = all_nodes_info[test_pkl]["通过测试"][test_id]
                        
                        test_entry = {
                            'semantic_vector': test_semantic_vector,
                            'additional_vector': test_additional_vector
                        }
                        
                        if not vector_exists_in_list(result[version_key]['statement_nodes_embedding'][stmt_new_id]['neighbors']['passed_test'], 
                                                  test_semantic_vector):
                            result[version_key]['statement_nodes_embedding'][stmt_new_id]['neighbors']['passed_test'].append(test_entry)
    
    with open(OUTPUT_FILE, 'wb') as f:
        pickle.dump(result, f)
    
    with open(ID_MAPPING_FILE, 'w', encoding='utf-8') as f:
        for version_number in sorted(id_mappings.keys(), key=lambda x: int(x) if x.isdigit() else float('inf')):
            
            f.write(f"v{version_number}\n")
            
            f.write("方法\n")
            for old_id, new_id in sorted(id_mappings[version_number]['methods'].items(), key=lambda x: int(x[0]) if str(x[0]).isdigit() else x[0]):
                f.write(f"{old_id} >> {new_id}\n")
            
            f.write("语句\n")
            for old_id, new_id in sorted(id_mappings[version_number]['statements'].items(), key=lambda x: int(x[0]) if str(x[0]).isdigit() else x[0]):
                f.write(f"{old_id} >> {new_id}\n")
            
            f.write("\n")
    
    print(f"处理完成，结果已保存到 {OUTPUT_FILE}")
    print(f"ID映射已保存到 {ID_MAPPING_FILE}")
    print(f"统计信息:")
    
    version_n_keys = [k for k in result.keys() if k.startswith('version-')]
    version_underscore_keys = [k for k in result.keys() if k.startswith('version_')]
    
    for version_key in version_n_keys:
        print(f"  版本: {version_key}")
        print(f"    - method_nodes_embedding: {len(result[version_key]['method_nodes_embedding'])} 个节点")
        print(f"    - statement_nodes_embedding: {len(result[version_key]['statement_nodes_embedding'])} 个节点")
        print(f"    - test_nodes_fail_embedding: {len(result[version_key]['test_nodes_fail_embedding'])} 个节点")
        print(f"    - test_nodes_pass_embedding: {len(result[version_key]['test_nodes_pass_embedding'])} 个节点")
    
    print("\n  方法组信息:")
    for version_key in version_underscore_keys:
        print(f"  版本: {version_key}")
        print(f"    - 方法组数量: {len(result[version_key])} 个")
    
    total_groups = sum(len(result[k]) for k in version_underscore_keys)
    print(f"  方法组总数: {total_groups} 个")
    print(f"  全局方法ID范围: 1-{global_method_id_counter-1}")
    print(f"  语句ID格式: 方法全局ID_序号 (例如: 1_1, 1_2, 2_1, ...)")

    method_faulty_csv = os.path.join(OUTPUT_DIR, f"{project_name}_method_faulty.csv")
    stmt_faulty_csv = os.path.join(OUTPUT_DIR, f"{project_name}_stmt_faulty.csv")
    
    faulty_index_pkl = os.path.join(OUTPUT_DIR, f"{project_name}_faulty_index.pkl")
    faulty_index_data = {
        'method_faulty_index': method_faulty_index_all,
        'stmt_faulty_index': stmt_faulty_index_all
    }
    with open(faulty_index_pkl, 'wb') as f:
        pickle.dump(faulty_index_data, f)
    print(f"错误标记原始数据已保存到 {faulty_index_pkl}")
    
    with open(method_faulty_csv, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['Project', 'Version', 'Original_IDs', 'New_IDs'])
        for project, version, original_ids, new_ids in method_faulty_index_all:
            
            original_ids_str = ','.join(map(str, original_ids))
            new_ids_str = ','.join(map(str, new_ids))
            writer.writerow([project, version, original_ids_str, new_ids_str])
    
    with open(stmt_faulty_csv, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['Project', 'Version', 'Original_IDs', 'New_IDs'])
        for project, version, original_ids, new_ids in stmt_faulty_index_all:
            
            original_ids_str = ','.join(map(str, original_ids))
            new_ids_str = ','.join(map(str, new_ids))
            writer.writerow([project, version, original_ids_str, new_ids_str])
    
    print(f"错误标记方法已保存到 {method_faulty_csv}")
    print(f"错误标记语句已保存到 {stmt_faulty_csv}")
    print(f"错误标记方法数量: {len(method_faulty_index_all)}")
    print(f"错误标记语句数量: {len(stmt_faulty_index_all)}")

if __name__ == "__main__":
    main()