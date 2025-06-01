import os
import sys
import pickle
import torch
import numpy as np
from transformers import RobertaTokenizer, RobertaModel
from tree_sitter import Language, Parser
from gensim.models import Word2Vec
from networkx import Graph

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)
from config import ProjectConfig
from defects4j.utils import Utils

JAVA_LANGUAGE_PATH = os.path.join(parent_dir, 'mypraser', 'my-languages.so')
JAVA_LANGUAGE = Language(JAVA_LANGUAGE_PATH, 'java')
parser = Parser()
parser.set_language(JAVA_LANGUAGE)

MODEL_PATH = os.path.join(parent_dir, 'graphcodebert-base')
model2_path = f"{ProjectConfig.path_dataset_home}/{ProjectConfig.word2vec}/{ProjectConfig.node2vec_model}"

print("加载GraphCodeBERT模型...")
roberta_tokenizer = RobertaTokenizer.from_pretrained(MODEL_PATH)
roberta_model = RobertaModel.from_pretrained(MODEL_PATH)
roberta_model.eval()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
roberta_model.to(device)
print(f"使用设备: {device}")

print("加载Node2Vec模型...")
model2 = Word2Vec.load(model2_path)

def tree_to_token_index(root_node):
    
    if (len(root_node.children) == 0 or root_node.type == 'string') and root_node.type.endswith('_comment') != True:
        return [(root_node.start_point, root_node.end_point)]
    else:
        code_tokens = []
        for child in root_node.children:
            code_tokens += tree_to_token_index(child)
        return code_tokens

def tree_to_variable_index(root_node, index_to_code):
    
    if (len(root_node.children) == 0 or root_node.type == 'string') and root_node.type.endswith('_comment') != True:
        index = (root_node.start_point, root_node.end_point)
        _, code = index_to_code[index]
        if root_node.type != code:
            return [(root_node.start_point, root_node.end_point)]
        else:
            return []
    else:
        code_tokens = []
        for child in root_node.children:
            code_tokens += tree_to_variable_index(child, index_to_code)
        return code_tokens

def index_to_code_token(index, code):
    
    start_point = index[0]
    end_point = index[1]
    if start_point[0] == end_point[0]:
        s = code[start_point[0]][start_point[1]:end_point[1]]
    else:
        s = ""
        s += code[start_point[0]][start_point[1]:]
        for i in range(start_point[0] + 1, end_point[0]):
            s += code[i]
        s += code[end_point[0]][:end_point[1]]
    return s

def DFG_java(root_node, index_to_code, states):
    
    assignment = ['assignment_expression']
    def_statement = ['variable_declarator']
    increment_statement = ['update_expression']
    if_statement = ['if_statement', 'else']
    for_statement = ['for_statement']
    enhanced_for_statement = ['enhanced_for_statement']
    while_statement = ['while_statement']
    do_first_statement = []
    states = states.copy()
    
    if (len(root_node.children) == 0 or root_node.type == 'string') and root_node.type.endswith('_comment') != True:
        idx, code = index_to_code[(root_node.start_point, root_node.end_point)]
        if root_node.type == code:
            return [], states
        elif code in states:
            return [(code, idx, 'comesFrom', [code], states[code].copy())], states
        else:
            if root_node.type == 'identifier':
                states[code] = [idx]
            return [(code, idx, 'comesFrom', [], [])], states
    elif root_node.type in def_statement:
        name = root_node.child_by_field_name('name')
        value = root_node.child_by_field_name('value')
        DFG = []
        if value is None:
            indexs = tree_to_variable_index(name, index_to_code)
            for index in indexs:
                idx, code = index_to_code[index]
                DFG.append((code, idx, 'comesFrom', [], []))
                states[code] = [idx]
            return sorted(DFG, key=lambda x: x[1]), states
        else:
            name_indexs = tree_to_variable_index(name, index_to_code)
            value_indexs = tree_to_variable_index(value, index_to_code)
            temp, states = DFG_java(value, index_to_code, states)
            DFG += temp
            for index1 in name_indexs:
                idx1, code1 = index_to_code[index1]
                for index2 in value_indexs:
                    idx2, code2 = index_to_code[index2]
                    DFG.append((code1, idx1, 'comesFrom', [code2], [idx2]))
                states[code1] = [idx1]
            return sorted(DFG, key=lambda x: x[1]), states
    elif root_node.type in assignment:
        left_nodes = root_node.child_by_field_name('left')
        right_nodes = root_node.child_by_field_name('right')
        DFG = []
        temp, states = DFG_java(right_nodes, index_to_code, states)
        DFG += temp
        name_indexs = tree_to_variable_index(left_nodes, index_to_code)
        value_indexs = tree_to_variable_index(right_nodes, index_to_code)
        for index1 in name_indexs:
            idx1, code1 = index_to_code[index1]
            for index2 in value_indexs:
                idx2, code2 = index_to_code[index2]
                DFG.append((code1, idx1, 'computedFrom', [code2], [idx2]))
            states[code1] = [idx1]
        return sorted(DFG, key=lambda x: x[1]), states
    elif root_node.type in increment_statement:
        DFG = []
        indexs = tree_to_variable_index(root_node, index_to_code)
        for index1 in indexs:
            idx1, code1 = index_to_code[index1]
            for index2 in indexs:
                idx2, code2 = index_to_code[index2]
                DFG.append((code1, idx1, 'computedFrom', [code2], [idx2]))
            states[code1] = [idx1]
        return sorted(DFG, key=lambda x: x[1]), states
    elif root_node.type in if_statement:
        DFG = []
        current_states = states.copy()
        others_states = []
        flag = False
        tag = False
        if 'else' in root_node.type:
            tag = True
        for child in root_node.children:
            if 'else' in child.type:
                tag = True
            if child.type not in if_statement and flag is False:
                temp, current_states = DFG_java(child, index_to_code, current_states)
                DFG += temp
            else:
                flag = True
                temp, new_states = DFG_java(child, index_to_code, states)
                DFG += temp
                others_states.append(new_states)
        others_states.append(current_states)
        if tag is False:
            others_states.append(states)
        new_states = {}
        for dic in others_states:
            for key in dic:
                if key not in new_states:
                    new_states[key] = dic[key].copy()
                else:
                    new_states[key] += dic[key]
        for key in new_states:
            new_states[key] = sorted(list(set(new_states[key])))
        return sorted(DFG, key=lambda x: x[1]), new_states
    elif root_node.type in for_statement:
        DFG = []
        for child in root_node.children:
            temp, states = DFG_java(child, index_to_code, states)
            DFG += temp
        flag = False
        for child in root_node.children:
            if flag:
                temp, states = DFG_java(child, index_to_code, states)
                DFG += temp
            elif child.type == "local_variable_declaration":
                flag = True
        dic = {}
        for x in DFG:
            if (x[0], x[1], x[2]) not in dic:
                dic[(x[0], x[1], x[2])] = [x[3], x[4]]
            else:
                dic[(x[0], x[1], x[2])][0] = list(set(dic[(x[0], x[1], x[2])][0] + x[3]))
                dic[(x[0], x[1], x[2])][1] = sorted(list(set(dic[(x[0], x[1], x[2])][1] + x[4])))
        DFG = [(x[0], x[1], x[2], y[0], y[1]) for x, y in sorted(dic.items(), key=lambda t: t[0][1])]
        return sorted(DFG, key=lambda x: x[1]), states
    elif root_node.type in enhanced_for_statement:
        name = root_node.child_by_field_name('name')
        value = root_node.child_by_field_name('value')
        body = root_node.child_by_field_name('body')
        DFG = []
        for i in range(2):
            temp, states = DFG_java(value, index_to_code, states)
            DFG += temp
            name_indexs = tree_to_variable_index(name, index_to_code)
            value_indexs = tree_to_variable_index(value, index_to_code)
            for index1 in name_indexs:
                idx1, code1 = index_to_code[index1]
                for index2 in value_indexs:
                    idx2, code2 = index_to_code[index2]
                    DFG.append((code1, idx1, 'computedFrom', [code2], [idx2]))
                states[code1] = [idx1]
            temp, states = DFG_java(body, index_to_code, states)
            DFG += temp
        dic = {}
        for x in DFG:
            if (x[0], x[1], x[2]) not in dic:
                dic[(x[0], x[1], x[2])] = [x[3], x[4]]
            else:
                dic[(x[0], x[1], x[2])][0] = list(set(dic[(x[0], x[1], x[2])][0] + x[3]))
                dic[(x[0], x[1], x[2])][1] = sorted(list(set(dic[(x[0], x[1], x[2])][1] + x[4])))
        DFG = [(x[0], x[1], x[2], y[0], y[1]) for x, y in sorted(dic.items(), key=lambda t: t[0][1])]
        return sorted(DFG, key=lambda x: x[1]), states
    elif root_node.type in while_statement:
        DFG = []
        for i in range(2):
            for child in root_node.children:
                temp, states = DFG_java(child, index_to_code, states)
                DFG += temp
        dic = {}
        for x in DFG:
            if (x[0], x[1], x[2]) not in dic:
                dic[(x[0], x[1], x[2])] = [x[3], x[4]]
            else:
                dic[(x[0], x[1], x[2])][0] = list(set(dic[(x[0], x[1], x[2])][0] + x[3]))
                dic[(x[0], x[1], x[2])][1] = sorted(list(set(dic[(x[0], x[1], x[2])][1] + x[4])))
        DFG = [(x[0], x[1], x[2], y[0], y[1]) for x, y in sorted(dic.items(), key=lambda t: t[0][1])]
        return sorted(DFG, key=lambda x: x[1]), states
    else:
        DFG = []
        for child in root_node.children:
            if child.type in do_first_statement:
                temp, states = DFG_java(child, index_to_code, states)
                DFG += temp
        for child in root_node.children:
            if child.type not in do_first_statement:
                temp, states = DFG_java(child, index_to_code, states)
                DFG += temp
        return sorted(DFG, key=lambda x: x[1]), states

def parse_ast_and_dfg(code):
    
    try:
        tree = parser.parse(bytes(code, 'utf8'))
        root_node = tree.root_node
        tokens_index = tree_to_token_index(root_node)
        code_lines = code.split('\n')
        code_tokens = [index_to_code_token(idx, code_lines) for idx in tokens_index]
        dfg, states = DFG_java(root_node, {idx: (i, code_tokens[i]) for i, idx in enumerate(tokens_index)}, {})
        return code_tokens, dfg, root_node
    except Exception as e:
        print(f"解析AST和DFG时出错: {str(e)}")
        return [], [], None

def convert_DFG_to_attn_mask(tokenizer, code_tokens, dfg, code_length=256, data_flow_length=64):
    
    try:
        
        code_tokens = [tokenizer.tokenize('@ '+ x)[1:] if idx != 0 else tokenizer.tokenize(x) for idx, x in enumerate(code_tokens)]
        ori2cur_pos = {}
        ori2cur_pos[-1] = (0, 0)
        for i in range(len(code_tokens)):
            ori2cur_pos[i] = (ori2cur_pos[i - 1][1], ori2cur_pos[i - 1][1] + len(code_tokens[i]))
        code_tokens = [y for x in code_tokens for y in x]
        
        code_tokens = code_tokens[:code_length + data_flow_length - 3 - min(len(dfg), data_flow_length)][:512 - 3]
        source_tokens = [tokenizer.cls_token] + code_tokens + [tokenizer.sep_token]
        source_ids = tokenizer.convert_tokens_to_ids(source_tokens)
        position_idx = [i + tokenizer.pad_token_id + 1 for i in range(len(source_tokens))]
        dfg = dfg[:code_length + data_flow_length - len(source_tokens)]
        source_tokens += [x[0] for x in dfg]
        position_idx += [0 for x in dfg]
        source_ids += [tokenizer.unk_token_id for x in dfg]
        padding_length = code_length + data_flow_length - len(source_ids)
        position_idx += [tokenizer.pad_token_id] * padding_length
        source_ids += [tokenizer.pad_token_id] * padding_length
        
        reverse_index = {}
        for idx, x in enumerate(dfg):
            reverse_index[x[1]] = idx
        for idx, x in enumerate(dfg):
            dfg[idx] = x[:-1] + ([reverse_index[i] for i in x[-1] if i in reverse_index],)
        dfg_to_dfg = [x[-1] for x in dfg]
        dfg_to_code = [ori2cur_pos[x[1]] for x in dfg]
        length = len([tokenizer.cls_token])
        dfg_to_code = [(x[0]+length, x[1]+length) for x in dfg_to_code]
        
        attn_mask = np.zeros((code_length + data_flow_length, code_length + data_flow_length), dtype=bool)
        node_index = sum([i > 1 for i in position_idx])
        max_length = sum([i != 1 for i in position_idx])
        attn_mask[:node_index, :node_index] = True
        
        for idx, x in enumerate(source_ids):
            if x in [tokenizer.cls_token_id, tokenizer.sep_token_id]:
                attn_mask[idx, :len(source_tokens)] = True
        
        for idx, (a, b) in enumerate(dfg_to_code):
            if a < node_index and b < node_index:
                attn_mask[idx+node_index, a:b] = True
                attn_mask[a:b, idx+node_index] = True
        
        for idx, neighbours in enumerate(dfg_to_dfg):
            for neigh in neighbours:
                if neigh + node_index < len(position_idx):
                    attn_mask[idx + node_index, neigh + node_index] = True
        
        source_ids_tensor = torch.tensor([source_ids])
        position_idx_tensor = torch.tensor([position_idx])
        attn_mask_tensor = torch.from_numpy(attn_mask).long().unsqueeze(0)

        return source_ids_tensor, position_idx_tensor, attn_mask_tensor
    except Exception as e:
        print(f"转换DFG到attention mask时出错: {str(e)}")
        return torch.tensor([]), torch.tensor([]), torch.tensor([])

def get_semantic_embedding(code):
    
    try:
        if not code or len(code.strip()) == 0:
            return np.zeros((768,))
            
        code_tokens, dfg, _ = parse_ast_and_dfg(code)
        if not code_tokens:
            return np.zeros((768,))
            
        input_ids, position_idx, attn_mask = convert_DFG_to_attn_mask(roberta_tokenizer, code_tokens, dfg)
        if input_ids.nelement() == 0:
            return np.zeros((768,))
            
        input_ids = input_ids.to(device)
        position_idx = position_idx.to(device)
        attn_mask = attn_mask.to(device)
        
        with torch.no_grad():
            outputs = roberta_model(input_ids=input_ids, position_ids=position_idx, attention_mask=attn_mask)
            cls_representation = outputs.last_hidden_state[:, 0, :].squeeze(0)
            
        return cls_representation.detach().cpu().numpy()
    except Exception as e:
        print(f"处理代码时出错: {str(e)}")
        return np.zeros((768,))

def process_node2vec(data: Graph, model: Word2Vec, use_mean=True, embed_d=768):
    
    nodes = []
    edges = data.edges
    for item in edges:
        nodes.append(item[0]), nodes.append(item[1])
    if len(nodes) > 0:
        vector = np.stack([model.wv[node] for node in nodes if node in model.wv], axis=0)
        if use_mean:
            vector = vector.mean(axis=0)
        return vector
    else:
        return np.zeros((embed_d,))

def process_method_attribute_to_vector(method_attribute):
    
    if method_attribute is None:
        return (np.zeros((768,)), np.zeros((768,)), np.zeros((768,)))
    
    try:
        method_name = method_attribute[0]
        method_code = method_attribute[1]
        method_ast = method_attribute[2] if len(method_attribute) > 2 else None
        
        method_name_vector = get_semantic_embedding(method_name)
        method_code_vector = get_semantic_embedding(method_code)
        method_ast_vector = process_node2vec(method_ast, model2, embed_d=768) if method_ast is not None else np.zeros((768,))
        
        return (method_name_vector, method_code_vector, method_ast_vector)
    except Exception as e:
        print(f"处理方法属性时出错: {e}, 属性: {method_attribute}")
        return (np.zeros((768,)), np.zeros((768,)), np.zeros((768,)))

def process_test_attribute_to_vector(test_attribute):
    
    if test_attribute is None:
        return (np.zeros((768,)), np.zeros((768,)))
    
    try:
        test_name = test_attribute[0]
        test_code = test_attribute[1] if len(test_attribute) > 1 else ""
        
        test_name_vector = get_semantic_embedding(test_name)
        test_code_vector = get_semantic_embedding(test_code)
        
        return (test_name_vector, test_code_vector)
    except Exception as e:
        print(f"处理测试属性时出错: {e}, 属性: {test_attribute}")
        return (np.zeros((768,)), np.zeros((768,)))

def process_stmt_attribute_to_vector(stmt_attribute):
    
    if stmt_attribute is None:
        return (np.zeros((768,)), np.zeros((768,)), np.zeros((768,)), np.zeros((7,)))
    
    try:
        method_name = stmt_attribute[0]
        stmt_code = stmt_attribute[1] if len(stmt_attribute) > 1 else ""
        stmt_context = stmt_attribute[2] if len(stmt_attribute) > 2 else ""
        combined_scores = stmt_attribute[3] if len(stmt_attribute) > 3 else np.zeros(7)
        
        method_name_vector = get_semantic_embedding(method_name)
        stmt_code_vector = get_semantic_embedding(stmt_code)
        stmt_context_vector = get_semantic_embedding(stmt_context)
        
        return (method_name_vector, stmt_code_vector, stmt_context_vector, combined_scores)
    except Exception as e:
        print(f"处理语句属性时出错: {e}, 属性: {stmt_attribute}")
        return (np.zeros((768,)), np.zeros((768,)), np.zeros((768,)), np.zeros((7,)))

def main():
    
    print("开始处理图数据向量化...")
    
    projects = Utils.get_projects()
    print(f"找到 {len(projects)} 个项目: {', '.join(projects)}")
    
    for project_name in projects:
        print(f"处理项目: {project_name}")
        
        project_path = os.path.join(ProjectConfig.merge_graph_path, project_name)
        
        if not os.path.exists(project_path):
            print(f"项目路径不存在: {project_path}")
            continue
        
        versions = [d for d in os.listdir(project_path) if os.path.isdir(os.path.join(project_path, d))]
        print(f"找到 {len(versions)} 个版本目录")
        
        for version in versions:
            input_file_path = os.path.join(project_path, version, f"{project_name}_{version}_graph.pkl")
            output_file_path = os.path.join(project_path, version, f"{project_name}_{version}_graph_embedding.pkl")
            
            if not os.path.exists(input_file_path):
                print(f"文件不存在: {input_file_path}")
                continue
            
            print(f"处理文件: {input_file_path}")
            
            try:
                
                with open(input_file_path, 'rb') as f:
                    graph_data = pickle.load(f)
                
                if hasattr(graph_data, 'method_nodes'):
                    print(f"处理 {len(graph_data.method_nodes)} 个方法节点...")
                    for method_id, method_node in graph_data.method_nodes.items():
                        
                        if hasattr(method_node, 'method_attribute') and method_node.method_attribute is not None:
                            method_node.method_attribute = process_method_attribute_to_vector(method_node.method_attribute)
                        
                        if hasattr(method_node, 'method_call') and method_node.method_call:
                            for i in range(len(method_node.method_call)):
                                if method_node.method_call[i] is not None:
                                    method_node.method_call[i] = process_method_attribute_to_vector(method_node.method_call[i])
                        
                        if hasattr(method_node, 'method_history') and method_node.method_history:
                            for i in range(len(method_node.method_history)):
                                if method_node.method_history[i] is not None:
                                    method_node.method_history[i] = process_method_attribute_to_vector(method_node.method_history[i])
                
                if hasattr(graph_data, 'statement_nodes'):
                    print(f"处理 {len(graph_data.statement_nodes)} 个语句节点...")
                    for stmt_id, stmt_node in graph_data.statement_nodes.items():
                        if hasattr(stmt_node, 'stmt_attribute') and stmt_node.stmt_attribute is not None:
                            stmt_node.stmt_attribute = process_stmt_attribute_to_vector(stmt_node.stmt_attribute)
                
                if hasattr(graph_data, 'test_nodes_fail'):
                    print(f"处理 {len(graph_data.test_nodes_fail)} 个失败测试节点...")
                    for test_id, test_node in graph_data.test_nodes_fail.items():
                        if hasattr(test_node, 'test_attribute') and test_node.test_attribute is not None:
                            test_node.test_attribute = process_test_attribute_to_vector(test_node.test_attribute)
                
                if hasattr(graph_data, 'test_nodes_pass'):
                    print(f"处理 {len(graph_data.test_nodes_pass)} 个通过测试节点...")
                    for test_id, test_node in graph_data.test_nodes_pass.items():
                        if hasattr(test_node, 'test_attribute') and test_node.test_attribute is not None:
                            test_node.test_attribute = process_test_attribute_to_vector(test_node.test_attribute)
                
                with open(output_file_path, 'wb') as f:
                    pickle.dump(graph_data, f)
                
                print(f"已保存处理后的数据到: {output_file_path}")
                
            except Exception as e:
                print(f"处理文件 {input_file_path} 时出错: {str(e)}")
                import traceback
                traceback.print_exc()
    
    print("所有项目处理完成")

if __name__ == "__main__":
    main() 