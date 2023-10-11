import torch
import torch.nn as nn
from transformers import GPT2Model, GPT2Config
from models import BaseModel
from core import Config

config = Config()

class GPT2(BaseModel):
    def __init__(self, n_dims_in, n_positions, n_embd=256, n_layer=12, n_head=8, n_dims_out=1):
        super(GPT2, self).__init__()
        configuration = GPT2Config(
            n_positions=2048, # set to sthg large advised
            n_embd=n_embd,
            n_layer=n_layer,
            n_head=n_head,
            resid_pdrop=0.0,
            embd_pdrop=0.0,
            attn_pdrop=0.0,
            use_cache=False,
        )
        self.name = f"gpt2_embd={n_embd}_layer={n_layer}_head={n_head}"

        self.n_positions = n_positions
        self.n_dims_in = n_dims_in
        self.n_dims_out = n_dims_out
        self._read_in = nn.Linear(n_dims_in, n_embd)
        self._backbone = GPT2Model(configuration)
        self._read_out = nn.Linear(n_embd, n_dims_out)

    def predict_step(self, input_dict, batch_idx=None):
        xs = input_dict["xs"]
        embeds = self._read_in(xs)
        output = self._backbone(inputs_embeds=embeds).last_hidden_state
        prediction = self._read_out(output)
        # predict only on xs
        return input_dict, {"preds": prediction}

    def forward(self, input_dict, batch_idx=None, return_intermediate_dict=False):
        input_dict, intermediate_dict = self.predict_step(input_dict, batch_idx)

        # Calculate all loss terms and metrics (scores)
        output_dict = self.calculate_losses_and_metrics(input_dict, intermediate_dict)

        # Calculate optimized loss
        optimized_loss = 0
        for key, loss in output_dict.items():
            if "loss_" in key:
                optimized_loss += loss
        output_dict["optimized_loss"] = optimized_loss
        return (intermediate_dict, output_dict) if return_intermediate_dict else output_dict

    def calculate_losses_and_metrics(self, input_dict, intermediate_dict):
        # Calculate loss
        ys = input_dict["ys"]
        preds = intermediate_dict["preds"]
        if config.dataset_typ == "pendulum":
            ys[..., 0] = ys[..., 0] % (2 * torch.pi)
            preds[..., 0] = preds[..., 0] % (2 * torch.pi)
            diff = torch.abs(ys-preds)
            diff[..., 0] = torch.where(diff[..., 0] > torch.pi, diff[..., 0] - 2 * torch.pi, diff[..., 0])

            res_sq = diff ** 2
        else:
            res_sq = (preds - ys) ** 2

        output_dict = {}
        output_dict["loss_mse"] = torch.mean(res_sq)
        for i in range(ys.shape[1]):
            for j in range(ys.shape[2]):
                output_dict[f"metric_mse_ts{i}_dim_{j}"] = torch.mean(res_sq[:, i, j])
        
        return output_dict

    def predict_ar(self, ins, fix_window_len=True):
        ins = torch.from_numpy(ins).float().to(self.device)
        one_d = False
        if ins.ndim == 2:
            one_d = True
            ins = ins.unsqueeze(0)
        bsize, points, _ = ins.shape
        d_o = self.n_dims_out
        outs = torch.zeros(bsize, 1, d_o).to(self.device)
        with torch.no_grad():
            for i in range(1, points+1):
                I = ins[:, :i]
                if fix_window_len and I.shape[1] > self.n_positions:
                    I = I[:, -self.n_positions:]
                _, interm = self.predict_step({"xs": I})
                pred = interm["preds"][:, -1:]  # b, 1, d_o
                outs = torch.cat([outs, pred], dim=1)
        outs = outs.detach().cpu().numpy()
        if one_d:
            outs = outs[0]
        return outs
        
