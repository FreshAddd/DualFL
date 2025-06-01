import torch

from model.CNN_stmt import CNN_Stmt
from model.agg_model_stmt import AggModel_Stmt

class ModelSTMT(torch.nn.Module):
    def __init__(self, input_embed_d=768, output_embed_d=500, heads=4):
        super(ModelSTMT, self).__init__()
        self.heads = heads
        self.agg_model = AggModel_Stmt(input_embed_d=input_embed_d, output_embed_d=output_embed_d, heads=self.heads)
        self.cnn = CNN_Stmt(channel_in=self.heads)

    def forward(self, passed_test_cases, failed_test_cases, call_method, belong_method, stmt, cfg, dfg, ast):
        x = self.agg_model(passed_test_cases, failed_test_cases, call_method, belong_method, stmt, cfg, dfg, ast)
        x = self.cnn(x)
        return x
