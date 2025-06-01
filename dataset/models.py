from typing import Dict, List, Set, Optional, Any, Tuple

class Node:
    
    def __init__(self, node_id: int, node_type: str, signature: str):
        self.id = node_id
        self.type = node_type  
        self.signature = signature  
        self.vector = None  

class MethodNode(Node):
    
    def __init__(self, node_id: int, signature: str):
        super().__init__(node_id, "method", signature)
        self.class_name = ""  
        self.method_name = ""  
        self.source_code = ""  
        self.content = {}  
        self.selected_content = {}  
        self.features = {}  
        self.metrics = {  
            'complexity': 0,
            'line_count': 0,
            'param_count': 0,
            'nesting_depth': 0
        }
        self.suspiciousness = 0.0  
        self.is_error = False  
        self.statements = []  
        self.start_line = -1  
        self.end_line = -1   
        self.udb_signature = signature  
        self.history = []  
        self.history_vector: List[Any] = [] 

class StatementNode(Node):
    
    def __init__(self, node_id: int, signature: str):
        super().__init__(node_id, "statement", signature)
        self.method_id = -1  
        self.line = -1  
        self.code = ""  
        self.suspiciousness = 0.0  
        self.is_error = False  
        self.state = "active"  
        self.ast_type = ""  
        
        self.spectrum_scores = None  
        self.mutation_scores = None  
        self.d4j_stmt_index = -1  
        self.d4j_stmt_id = ""  

class TestNode(Node):
    
    def __init__(self, node_id: int, test_id: str):
        super().__init__(node_id, "test", test_id)
        self.outcome = ""  
        self.name = ""  
        self.source_code = ""  

class Edge:
    
    def __init__(self, source_id: int, target_id: int, edge_type: str, weight: float = 1.0):
        self.source = source_id
        self.target = target_id
        self.type = edge_type  
        self.weight = weight

class GraphData:
    
    def __init__(self):
        
        self.method_nodes: Dict[int, MethodNode] = {}
        self.statement_nodes: Dict[int, StatementNode] = {}

        self.method_method_call_edges: List[Tuple[int, int]] = []  
        self.statement_method_call_edges: List[Tuple[int, int]] = []  
        self.control_flow_edges: List[Edge] = []  
        self.control_flow_yes_edges: List[Tuple[int, int]] = []  
        self.control_flow_no_edges: List[Tuple[int, int]] = []  
        self.control_flow_normal_edges: List[Tuple[int, int]] = []  
        self.control_flow_other_edges: List[Tuple[int, int]] = []  
        self.data_flow_edges: List[Edge] = []
        self.ast_edges: List[Edge] = []  
        self.belongs_to_edges: List[Tuple[int, int]] = []  

        self.covers_edges: List[Tuple[int, int]] = []  
        self.test_id_map: Dict[str, int] = {}
        
        self.test_nodes_pass: Dict[int, TestNode] = {}  
        self.test_nodes_fail: Dict[int, TestNode] = {}  
        self.covers_stmt_edges_pass: List[Tuple[int, int]] = []  
        self.covers_stmt_edges_fail: List[Tuple[int, int]] = []  
        self.covers_method_edges_pass: List[Tuple[int, int]] = []  
        self.covers_method_edges_fail: List[Tuple[int, int]] = []  
        
        self.test_nodes: Dict[int, TestNode] = {}
        self.covers_stmt_edges: List[Tuple[int, int]] = []
        self.covers_method_edges: List[Tuple[int, int]] = []
        
        self.method_id_map: Dict[str, int] = {}
        self.statement_id_map: Dict[str, int] = {}

        self.method_to_statements: Dict[int, List[int]] = {}
        self.pruned_methods: Set[int] = set()
        self.pruned_statements: Set[int] = set()
        
        self.stmt_faulty_set: List[List[str]] = []  
        self.stmt_faulty_index: List[List[int]] = []  
        self.method_faulty_set: List[List[str]] = []  
        self.method_faulty_index: List[List[int]] = []  
