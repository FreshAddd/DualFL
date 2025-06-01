from tree_sitter import Language, Parser
from .utils import (remove_comments_and_docstrings,
                   tree_to_token_index,
                   index_to_code_token,
                   tree_to_variable_index,
                   field_identifier_to_code)

def DFG_c(root_node,index_to_code,states):
    assignment=['assignment_expression']
    def_statement=['declaration']
    increment_statement=['update_expression']
    if_statement=['if_statement', 'else']
    for_statement=['for_statement']
    enhanced_for_statement=[]
    while_statement=['while_statement']
    do_first_statement=["do_statement"]
    field_expression=["field_expression"]
    states=states.copy()
    if (len(root_node.children)==0 or root_node.type=='string') and root_node.type!='comment':
        idx,code=index_to_code[(root_node.start_point,root_node.end_point)]
        empty_set = ['escape_sequence', 'primitive_type', 'number_literal', 'null', '     ']
        if root_node.type==code or root_node.type in empty_set:
            return [],states
        elif code in states:
            return [(code,idx,'comesFrom',[code],states[code].copy())],states
        else:
            if root_node.type=='identifier':
                states[code]=[idx]
            return [(code,idx,'comesFrom',[],[])],states

    elif root_node.type in field_expression:
        DFG=[]
        idx,code = field_identifier_to_code(root_node,index_to_code)
        argument_code = code[:code.find(root_node.children[1].type)]
        flag = False
        if argument_code in states:
            tmp = [(code,idx,'comesFrom',[argument_code],states[argument_code].copy())]
            DFG+=tmp
            flag = True
        if code in states:
            tmp = [(code,idx,'comesFrom',[code],states[code].copy())]
            DFG+=tmp
            flag = True

        if not flag:
            states[code]=[idx]
            return [(code,idx,'comesFrom',[],[])],states
        else:
            return sorted(DFG,key=lambda x:x[1]),states

    elif root_node.type in def_statement:
        DFG=[]
        for child in root_node.children:
            if child.type == "init_declarator":
                name=child.child_by_field_name("declarator")
                value=child.child_by_field_name("value")
            elif child.type=="identifier":
                name=child
                value=None
            else:
                continue

            if value is None:
                indexs=tree_to_variable_index(name,index_to_code)
                for index in indexs:
                    idx,code=index_to_code[index]
                    DFG.append((code,idx,'comesFrom',[],[]))
                    states[code]=[idx]
            else:
                name_indexs=tree_to_variable_index(name,index_to_code)
                value_indexs=tree_to_variable_index(value,index_to_code)
                temp,states=DFG_c(value,index_to_code,states)
                DFG+=temp
                for index1 in name_indexs:
                    idx1,code1=index_to_code[index1]
                    for index2 in value_indexs:
                        idx2,code2=index_to_code[index2]
                        DFG.append((code1,idx1,'comesFrom',[code2],[idx2]))
                    states[code1]=[idx1]
        return sorted(DFG,key=lambda x:x[1]),states
    elif root_node.type in assignment:
        left_nodes=root_node.child_by_field_name('left')
        name_indexs=tree_to_variable_index(left_nodes,index_to_code)
        right_nodes=root_node.child_by_field_name('right')
        while right_nodes.type in assignment:
            left_node = right_nodes.child_by_field_name('left')
            name_indexs+=tree_to_variable_index(left_node,index_to_code)
            right_nodes=right_nodes.child_by_field_name('right')
        DFG=[]
        temp,states=DFG_c(right_nodes,index_to_code,states)
        DFG+=temp
        value_indexs=tree_to_variable_index(right_nodes,index_to_code)
        for index1 in name_indexs:
            idx1,code1=index_to_code[index1]
            for index2 in value_indexs:
                idx2,code2=index_to_code[index2]
                DFG.append((code1,idx1,'computedFrom',[code2],[idx2]))
            states[code1]=[idx1]
        return sorted(DFG,key=lambda x:x[1]),states
    elif root_node.type in increment_statement:
        DFG=[]
        indexs=tree_to_variable_index(root_node,index_to_code)
        for index1 in indexs:
            idx1,code1=index_to_code[index1]
            for index2 in indexs:
                idx2,code2=index_to_code[index2]
                DFG.append((code1,idx1,'computedFrom',[code2],[idx2]))
            states[code1]=[idx1]
        return sorted(DFG,key=lambda x:x[1]),states
    elif root_node.type in if_statement:
        DFG=[]
        current_states=states.copy()
        others_states=[]
        flag=False
        tag=False
        if 'else' in root_node.type:
            tag=True
        for child in root_node.children:
            if 'else' in child.type:
                tag=True
            if child.type not in if_statement and flag is False:
                temp,current_states=DFG_c(child,index_to_code,current_states)
                DFG+=temp
            else:
                flag=True
                temp,new_states=DFG_c(child,index_to_code,states)
                DFG+=temp
                others_states.append(new_states)
        others_states.append(current_states)
        if tag is False:
            others_states.append(states)
        new_states={}
        for dic in others_states:
            for key in dic:
                if key not in new_states:
                    new_states[key]=dic[key].copy()
                else:
                    new_states[key]+=dic[key]
        for key in new_states:
            new_states[key]=sorted(list(set(new_states[key])))
        return sorted(DFG,key=lambda x:x[1]),new_states
    elif root_node.type in for_statement:
        DFG=[]
        for child in root_node.children:
            temp,states=DFG_c(child,index_to_code,states)
            DFG+=temp
        flag=False
        for child in root_node.children:
            if flag:
                temp,states=DFG_c(child,index_to_code,states)
                DFG+=temp
            elif child.type=="declaration":
                flag=True
        dic={}
        for x in DFG:
            if (x[0],x[1],x[2]) not in dic:
                dic[(x[0],x[1],x[2])]=[x[3],x[4]]
            else:
                dic[(x[0],x[1],x[2])][0]=list(set(dic[(x[0],x[1],x[2])][0]+x[3]))
                dic[(x[0],x[1],x[2])][1]=sorted(list(set(dic[(x[0],x[1],x[2])][1]+x[4])))
        DFG=[(x[0],x[1],x[2],y[0],y[1]) for x,y in sorted(dic.items(),key=lambda t:t[0][1])]
        return sorted(DFG,key=lambda x:x[1]),states
    elif root_node.type in enhanced_for_statement:
        name=root_node.child_by_field_name('left')
        value=root_node.child_by_field_name('right')
        body=root_node.child_by_field_name('body')
        DFG=[]
        for i in range(2):
            temp,states=DFG_c(value,index_to_code,states)
            DFG+=temp
            name_indexs=tree_to_variable_index(name,index_to_code)
            value_indexs=tree_to_variable_index(value,index_to_code)
            for index1 in name_indexs:
                idx1,code1=index_to_code[index1]
                for index2 in value_indexs:
                    idx2,code2=index_to_code[index2]
                    DFG.append((code1,idx1,'computedFrom',[code2],[idx2]))
                states[code1]=[idx1]
            temp,states=DFG_c(body,index_to_code,states)
            DFG+=temp
        dic={}
        for x in DFG:
            if (x[0],x[1],x[2]) not in dic:
                dic[(x[0],x[1],x[2])]=[x[3],x[4]]
            else:
                dic[(x[0],x[1],x[2])][0]=list(set(dic[(x[0],x[1],x[2])][0]+x[3]))
                dic[(x[0],x[1],x[2])][1]=sorted(list(set(dic[(x[0],x[1],x[2])][1]+x[4])))
        DFG=[(x[0],x[1],x[2],y[0],y[1]) for x,y in sorted(dic.items(),key=lambda t:t[0][1])]
        return sorted(DFG,key=lambda x:x[1]),states
    elif root_node.type in while_statement:
        DFG=[]
        for i in range(2):
            for child in root_node.children:
                temp,states=DFG_c(child,index_to_code,states)
                DFG+=temp
        dic={}
        for x in DFG:
            if (x[0],x[1],x[2]) not in dic:
                dic[(x[0],x[1],x[2])]=[x[3],x[4]]
            else:
                dic[(x[0],x[1],x[2])][0]=list(set(dic[(x[0],x[1],x[2])][0]+x[3]))
                dic[(x[0],x[1],x[2])][1]=sorted(list(set(dic[(x[0],x[1],x[2])][1]+x[4])))
        DFG=[(x[0],x[1],x[2],y[0],y[1]) for x,y in sorted(dic.items(),key=lambda t:t[0][1])]
        return sorted(DFG,key=lambda x:x[1]),states
    else:
        DFG=[]
        for child in root_node.children:
            if child.type in do_first_statement:
                temp,states=DFG_c(child,index_to_code,states)
                DFG+=temp
        for child in root_node.children:
            if child.type not in do_first_statement:
                temp,states=DFG_c(child,index_to_code,states)
                DFG+=temp
        if len(DFG) == 7:
            print(len(DFG))
        return sorted(DFG,key=lambda x:x[1]),states