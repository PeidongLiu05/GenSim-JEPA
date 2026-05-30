import torch.nn.functional as F
import torch.optim as optim
from networks.utils.sigreg import SIGReg

class Stage2Trainer:
    def __init__(self, config, state_encoder, action_encoder, predictor, lewm_buffer):
        self.state_encoder = state_encoder
        self.action_encoder = action_encoder
        self.predictor = predictor
        self.sigreg_loss = SIGReg()
        self.buffer = lewm_buffer
        self.optimizer = optim.Adam(list(state_encoder.parameters()) + list(predictor.parameters()), lr=config.lr_lewm)
        self.lambda_sigreg = config.lambda_sigreg

    def train_step(self, batch_size=256):
        # 从buffer采样 (obs, a, obs_next)
        obs, _, a_phys, _, obs_next, _ = self.buffer.sample_low(batch_size)
        z = self.state_encoder(obs)
        z_next = self.state_encoder(obs_next)
        a_prim = self.action_encoder(a_phys[0], a_phys[1])
        z_pred = self.predictor(z, a_prim)
        # 预测损失
        pred_loss = F.mse_loss(z_pred, z_next)
        # SIGReg正则化：迫使z的分布接近各向同性高斯
        sig_loss = (self.sigreg_loss(z) + self.sigreg_loss(z_next)) / 2    # 需要实现，计算KL(N(μ,σ), N(0,I))
        total_loss = pred_loss + self.lambda_sigreg * sig_loss
        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()
        return pred_loss.item(), sig_loss.item()

    def train(self, epochs):
        for epoch in range(epochs):
            loss_pred, loss_sig = self.train_step()
            print(f"Epoch {epoch}, Pred Loss: {loss_pred:.4f}, SIGReg Loss: {loss_sig:.4f}")