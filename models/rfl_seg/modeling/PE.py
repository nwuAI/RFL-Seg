from torch import nn

class ImageAwarePromptEnhancer(nn.Module):

    def __init__(self, embed_dim):
        super().__init__()
        self.cross_attn = nn.MultiheadAttention(embed_dim, num_heads=8, batch_first=True)
        self.fc = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 4),
            nn.ReLU(),
            nn.Linear(embed_dim // 4, embed_dim),
            nn.Sigmoid()
        )

    def forward(self, prompt_emb, image_emb):
        original_shape = prompt_emb.shape
        is_dense = len(original_shape) == 4
        b, c, h, w = image_emb.shape
        image_emb = image_emb.view(b, c, -1).permute(0, 2, 1)
        if is_dense:
            b, c, h_p, w_p = original_shape
            prompt_emb = prompt_emb.view(b, c, -1).permute(0, 2, 1)
        else:
            pass

        attn_out, _ = self.cross_attn(
            query=prompt_emb,
            key=image_emb,
            value=image_emb
        )

        weights = self.fc(prompt_emb)
        enhanced_emb = attn_out * weights + prompt_emb

        if is_dense:
            enhanced_emb = enhanced_emb.permute(0, 2, 1).view(b, c, h_p, w_p)

        return enhanced_emb


class FeatureModulator(nn.Module):

    def __init__(self, embed_dim):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 2),
            nn.ReLU()
        )

    def forward(self, prompt_emb, image_emb):
        if len(prompt_emb.shape) == 4:
            return self._modulate_dense(prompt_emb, image_emb)
        return self._modulate_sparse(prompt_emb, image_emb)

    def _modulate_dense(self, dense_emb, image_emb):
        global_feat = self.pool(image_emb).flatten(1)
        params = self.fc(global_feat)
        gamma, beta = params.chunk(2, dim=1)
        gamma = gamma.view(-1, dense_emb.shape[1], 1, 1)
        beta = beta.view(-1, dense_emb.shape[1], 1, 1)
        return (1 + gamma) * dense_emb + beta

    def _modulate_sparse(self, sparse_emb, image_emb):
        global_feat = self.pool(image_emb).flatten(1)
        params = self.fc(global_feat)
        gamma, beta = params.chunk(2, dim=1)
        gamma = gamma.unsqueeze(1)
        beta = beta.unsqueeze(1)
        return (1 + gamma) * sparse_emb + beta