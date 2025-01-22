from torch import nn

class AutoEncoder(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(28 * 28, cfg.encoder.hidden_dims[0]), 
            nn.ReLU(), 
            nn.Linear(cfg.encoder.hidden_dims[0], cfg.encoder.hidden_dims[1]), 
            nn.ReLU(), 
            nn.Linear(cfg.encoder.hidden_dims[1], cfg.encoder.hidden_dims[2]),
            nn.ReLU(),
        )
        self.decoder = nn.Sequential(
            nn.Linear(cfg.decoder.hidden_dims[0], cfg.decoder.hidden_dims[1]), 
            nn.ReLU(), 
            nn.Linear(cfg.decoder.hidden_dims[1], cfg.decoder.hidden_dims[2]), 
            nn.ReLU(), 
            nn.Linear(cfg.decoder.hidden_dims[2], 28 * 28),
            nn.Sigmoid(),
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x