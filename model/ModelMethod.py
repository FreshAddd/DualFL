import torch

from model.CNN_method import CNN_Method
from model.agg_model_method import AggModel_Method

class ModelMethod(torch.nn.Module):
    def __init__(self, input_embed_d=768, output_embed_d=500, heads=4):
        super(ModelMethod, self).__init__()
        self.heads = heads
        self.agg_model = AggModel_Method(input_embed_d=input_embed_d, output_embed_d=output_embed_d, heads=self.heads)
        self.cnn = CNN_Method(channel_in=self.heads)

    def forward(self, passed_test_cases, failed_test_cases, history_change, call_information, methods, call_stmt, internal_stmt):
        
        x = self.agg_model(passed_test_cases, failed_test_cases, history_change, call_information, methods, call_stmt, internal_stmt)
        x = self.cnn(x)
        return x
