import traceback

import torch
import torch.nn
import numpy as np
import os
import pytorch_lightning as pl
from torch import Tensor

class AggModel_Stmt(torch.nn.Module):
    def __init__(self, input_embed_d=768, output_embed_d=500,
                 num_passed_test_cases_attr=2,
                 num_failed_test_cases_attr=2,
                 num_call_method_cases_attr=3,
                 num_belong_method_cases_attr=3,
                 num_stmt_cases_attr=3,
                 num_cfg_cases_attr=3,
                 num_dfg_cases_attr=3,
                 num_ast_cases_attr=3,
                 heads=4
                 ):
        super(AggModel_Stmt, self).__init__()
        self.input_embed_d = input_embed_d
        self.output_embed_d = output_embed_d
        self.num_passed_test_cases_attr = num_passed_test_cases_attr
        self.num_failed_test_cases_attr = num_failed_test_cases_attr
        self.num_call_method_cases_attr = num_call_method_cases_attr
        self.num_belong_method_cases_attr = num_belong_method_cases_attr
        self.num_stmt_cases_attr = num_stmt_cases_attr
        self.num_cfg_cases_attr = num_cfg_cases_attr
        self.num_dfg_cases_attr = num_dfg_cases_attr
        self.num_ast_cases_attr = num_ast_cases_attr
        
        self.passed_test_attr_agg_model = AttrAggModel(input_embed_d=self.input_embed_d,
                                                       output_embed_d=self.output_embed_d,
                                                       num_agg=self.num_passed_test_cases_attr)
        self.passed_test_neighbor_agg_model = NeighborAggModel()
        self.failed_test_attr_agg_model = AttrAggModel(input_embed_d=self.input_embed_d,
                                                       output_embed_d=self.output_embed_d,
                                                       num_agg=self.num_failed_test_cases_attr)
        self.failed_test_neighbor_agg_model = NeighborAggModel()
        self.call_method_attr_agg_model = AttrAggModel(input_embed_d=self.input_embed_d,
                                                          output_embed_d=self.output_embed_d,
                                                          num_agg=self.num_call_method_cases_attr)
        self.call_method_neighbor_agg_model = NeighborAggModel()
        self.belong_method_attr_agg_model = AttrAggModel(input_embed_d=self.input_embed_d,
                                                            output_embed_d=self.output_embed_d,
                                                            num_agg=self.num_belong_method_cases_attr,
                                                            use_linear=True)
        self.belong_method_neighbor_agg_model = NeighborAggModel()
        self.stmt_attr_agg_model = AttrAggModel(input_embed_d=self.input_embed_d, output_embed_d=self.output_embed_d,
                                                  num_agg=self.num_stmt_cases_attr)
                                                  
        self.cfg_attr_agg_model = AttrAggModel(input_embed_d=self.input_embed_d,
                                                   output_embed_d=self.output_embed_d,
                                                   num_agg=self.num_cfg_cases_attr)
        self.cfg_neighbor_agg_model = NeighborAggModel()
        
        self.dfg_attr_agg_model = AttrAggModel(input_embed_d=self.input_embed_d,
                                                       output_embed_d=self.output_embed_d,
                                                       num_agg=self.num_dfg_cases_attr)
        self.dfg_neighbor_agg_model = NeighborAggModel()
        
        self.ast_attr_agg_model = AttrAggModel(input_embed_d=self.input_embed_d,
                                                       output_embed_d=self.output_embed_d,
                                                       num_agg=self.num_ast_cases_attr)
        self.ast_neighbor_agg_model = NeighborAggModel()

        self.vector_agg_model = VectorAggModel(embed_d=self.output_embed_d, num_agg=8, heads=heads)

    def forward(self, passed_test_cases, failed_test_cases, call_method, belong_method, stmt, cfg, dfg, ast):
        stmt = self.stmt_attr_agg_model(stmt)
        
        passed_test_cases = self.passed_test_attr_agg_model(passed_test_cases)
        failed_test_cases = self.failed_test_attr_agg_model(failed_test_cases)
        call_method = self.call_method_attr_agg_model(call_method)
        belong_method = self.belong_method_attr_agg_model(belong_method)
        cfg = self.cfg_attr_agg_model(cfg)
        dfg = self.dfg_attr_agg_model(dfg)
        ast = self.ast_attr_agg_model(ast)
        
        call_method = self.call_method_neighbor_agg_model(call_method)
        belong_method = self.belong_method_neighbor_agg_model(belong_method)
        cfg = self.cfg_neighbor_agg_model(cfg)
        dfg = self.dfg_neighbor_agg_model(dfg)
        ast = self.ast_neighbor_agg_model(ast)

        passed_test_cases = self.passed_test_neighbor_agg_model(passed_test_cases)
        failed_test_cases = self.failed_test_neighbor_agg_model(failed_test_cases)

        vectors = self.vector_agg_model(
            [passed_test_cases, failed_test_cases, call_method, belong_method, cfg, dfg, ast],
            stmt
        )

        return vectors

class VectorAggModel(torch.nn.Module):
    def __init__(self, embed_d=500, num_agg=8, heads=64):
        super(VectorAggModel, self).__init__()
        self.embed_d = embed_d
        self.num_agg = num_agg
        self.linear_list = torch.nn.ModuleList([torch.nn.Linear(heads, heads) for _ in range(self.num_agg - 1)] + [
            torch.nn.Linear(heads, heads)])
        self.attention_parameter_list = torch.nn.ParameterList(
            [torch.nn.Parameter(torch.randn(1, heads, self.embed_d * 2),
                                requires_grad=True) for i in
             range(self.num_agg - 1)] + [
                torch.nn.Parameter(torch.randn(1, heads, self.embed_d),
                                   requires_grad=True)])
        self.dropout = torch.nn.Dropout(p=0.5)

    def forward(self, x, y):
        x_forward = [torch.cat([_x, y], dim=-1) for _x in x] + [y]
        x_forward = [torch.nn.functional.layer_norm(t, normalized_shape=t.shape[-1:]) for t in x_forward]
        attention_parameter_list = self.attention_parameter_list
        attention_parameter_list = [torch.nn.functional.layer_norm(t, normalized_shape=t.shape[-1:])
                                    for t in
                                    attention_parameter_list]
        x_forward = [torch.matmul(x_forward[i],
                                  attention_parameter.transpose(-1, -2))
                     for
                     i, attention_parameter in enumerate(attention_parameter_list)]
        
        a = torch.cat(x_forward, dim=1).transpose(-1, -2)
        a = self.dropout(a)
        a = a / (self.embed_d ** 0.5)
        a = torch.softmax(a, dim=-1)
        f = torch.cat(x + [y], dim=1)
        f = torch.nn.functional.layer_norm(f, normalized_shape=f.shape[-1:])
        out = torch.matmul(a, f)
        out = torch.nn.functional.gelu(out)
        out = self.dropout(out)
        out = torch.nn.functional.layer_norm(out, normalized_shape=out.shape[-1:])
        return out

class AttrAggModel(torch.nn.Module):
    def __init__(self, input_embed_d=768, output_embed_d=500, num_agg=3, use_linear=True):
        super(AttrAggModel, self).__init__()
        self.input_embed_d = input_embed_d
        self.output_embed_d = output_embed_d
        self.num_agg = num_agg
        self.use_linear = use_linear
        if self.use_linear:
            self.linear_list = torch.nn.ModuleList(
                [torch.nn.Linear(self.input_embed_d, self.output_embed_d) for _ in range(self.num_agg)])
        self.biLSTM = torch.nn.LSTM(self.output_embed_d, self.output_embed_d // 2, 1, bidirectional=True,
                                    batch_first=True, bias=False)
        self.dropout = torch.nn.Dropout(p=0.5)

    def forward(self, x):
        out_linear = list()
        b, n, a, d = x.shape
        x = x.view(b * n, a, d).permute(1, 0, 2)
        for i, linear in enumerate(self.linear_list):
            t = linear(x[i])
            out_linear.append(t)
        x = out_linear
        x = torch.stack(x, dim=0).permute(1, 0, 2).view(b * n, a, self.output_embed_d)
        x = torch.nn.functional.gelu(x)
        x = self.dropout(x)
        x = torch.nn.functional.layer_norm(x, normalized_shape=x.shape[-1:])
        x = self.biLSTM(x)[0]
        x = torch.nn.functional.gelu(x)
        x = self.dropout(x)
        x = torch.nn.functional.layer_norm(x, normalized_shape=x.shape[-1:])
        x = torch.mean(x, dim=1, keepdim=True)
        x = torch.nn.functional.gelu(x)
        x = self.dropout(x)
        x = torch.nn.functional.layer_norm(x, normalized_shape=x.shape[-1:])
        return x.view(b, n, self.output_embed_d)

class NeighborAggModel(torch.nn.Module):
    def __init__(self, input_embed_d=500, output_embed_d=500):
        super(NeighborAggModel, self).__init__()
        self.input_embed_d = input_embed_d
        self.output_embed_d = output_embed_d
        self.biLSTM = torch.nn.LSTM(self.output_embed_d, self.output_embed_d // 2, 1, bidirectional=True,
                                    batch_first=True, bias=False)
        self.dropout = torch.nn.Dropout(p=0.5)

    def forward(self, x):
        x = self.biLSTM(x)[0]
        x = torch.nn.functional.gelu(x)
        x = self.dropout(x)
        x = torch.nn.functional.layer_norm(x, normalized_shape=x.shape[-1:])
        x = torch.mean(x, dim=1, keepdim=True)
        x = torch.nn.functional.gelu(x)
        x = torch.nn.functional.layer_norm(x, normalized_shape=x.shape[-1:])
        return x
