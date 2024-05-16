import torch
from transformers import WhisperModel


class AffectWhisper(torch.nn.Module):
    def __init__(self, out_dims, device):
        super().__init__()
        whisper = WhisperModel.from_pretrained('openai/whisper-small', cache_dir='/data/rrao/pretrained/cache/').to(device)
        self.encoder = whisper.encoder
        self.affect_layer = torch.nn.Sequential(
            torch.nn.Linear(1500 * 768, 1028).to(device),
            torch.nn.ReLU(),
            torch.nn.Linear(1028, 256).to(device),
            torch.nn.ReLU(),
            torch.nn.Linear(256, out_dims).to(device)
        )
    
    def forward(self, input):
        # print(f'Input: {input.shape} {input.dtype} {input.device}')
        input = self.encoder(input).last_hidden_state
        # print(f'Embeddings: {input.shape} {input.dtype} {input.device}')
        input = torch.flatten(input, 1)
        output = self.affect_layer(input)
        # print(f'Output: {output.shape} {output.dtype} {output.device}')
        return output