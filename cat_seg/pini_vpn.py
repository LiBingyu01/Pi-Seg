import math
import torch
import torch.nn as nn
import torch.nn.functional as F


def sample_noise_like(x, noise_type="gaussian", df=3.0, eps=1e-6):
    """
    Sample noise with roughly zero mean and unit-scale behavior.

    Args:
        x: reference tensor for shape/device/dtype
        noise_type: gaussian | laplace | uniform | gumbel | student_t
        df: degrees of freedom for student_t
    """
    if noise_type == "gaussian":
        return torch.randn_like(x)

    elif noise_type == "laplace":
        # Laplace(0, 1/sqrt(2)) => variance ~= 1
        dist = torch.distributions.Laplace(
            loc=torch.zeros_like(x),
            scale=torch.full_like(x, 1.0 / math.sqrt(2.0))
        )
        return dist.sample()

    elif noise_type == "uniform":
        # Uniform(-sqrt(3), sqrt(3)) => variance ~= 1
        a = math.sqrt(3.0)
        return (torch.rand_like(x) * 2.0 - 1.0) * a

    elif noise_type == "gumbel":
        # Standard Gumbel normalized to roughly zero mean / unit std
        u = torch.rand_like(x).clamp(min=eps, max=1.0 - eps)
        g = -torch.log(-torch.log(u))
        # standardize
        g = (g - 0.5772156649) / 1.2825498302
        return g

    elif noise_type == "student_t":
        # Student-t with df > 2; normalize to unit variance
        assert df > 2.0, "student_t requires df > 2 for finite variance"
        dist = torch.distributions.StudentT(df=df)
        t = dist.sample(x.shape).to(device=x.device, dtype=x.dtype)
        t = t / math.sqrt(df / (df - 2.0))
        return t

    else:
        raise ValueError(f"Unsupported noise_type: {noise_type}")


class CrossAttention(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.fc_q = nn.Linear(input_dim, output_dim)
        self.fc_k = nn.Linear(input_dim, output_dim)
        self.fc_v = nn.Linear(input_dim, output_dim)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, query, key, value):
        q = self.fc_q(query)
        k = self.fc_k(key)
        v = self.fc_v(value)

        # scaled dot-product is more stable
        attn_weights = self.softmax(torch.matmul(q, k.transpose(-2, -1)))
        output = torch.matmul(attn_weights, v)
        return output, attn_weights


class ImageVpnGenerator(nn.Module):
    def __init__(
        self,
        clip_dim,
        reduction=1,
        noise_type="gaussian",
        student_t_df=3.0,
    ):
        super().__init__()
        self.clip_dim = clip_dim
        self.hidden_dim = clip_dim // reduction
        self.noise_type = noise_type
        self.student_t_df = student_t_df

        self.cross_attn = CrossAttention(clip_dim, self.hidden_dim)
        self.fc_mean = nn.Linear(self.hidden_dim, clip_dim)
        self.fc_variance = nn.Linear(self.hidden_dim, clip_dim)

    def forward(self, spatial_feat, text_feat):
        """
        spatial_feat: (B, HW, C)
        text_feat:    (T, C)
        """
        text_feat = text_feat.unsqueeze(0).expand(spatial_feat.size(0), -1, -1)
        attn_feat, _ = self.cross_attn(spatial_feat, text_feat, text_feat)
        mu = self.fc_mean(attn_feat)
        variance = self.fc_variance(attn_feat).abs()
        
        return mu, variance

    def sample(self, mu, std):
        noise = sample_noise_like(
            std,
            noise_type=self.noise_type,
            df=self.student_t_df,
        )
        return std * noise + mu


class TextVpnGenerator(nn.Module):
    def __init__(
        self,
        clip_dim,
        noise_std=0.02,
        noise_type="gaussian",
        student_t_df=3.0,
    ):
        super().__init__()
        self.noise_type = noise_type
        self.student_t_df = student_t_df

        self.mu = nn.Parameter(torch.empty(1, 1, 1, clip_dim))
        self.var = nn.Parameter(torch.empty(1, 1, 1, clip_dim))

        nn.init.normal_(self.mu, std=noise_std)
        nn.init.normal_(self.var, std=noise_std)

    def forward(self, text_feats):
        eps = sample_noise_like(
            text_feats,
            noise_type=self.noise_type,
            df=self.student_t_df,
        )
        noise = self.var.abs() * eps + self.mu
        return text_feats + noise