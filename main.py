import torch
import torch.nn as nn
from timm.models.layers import trunc_normal_
import torch.fft
from FMOE_afno import AFNOBlock
from FMOE_conv import ConvBlock


class LatAwarePosEmbed(nn.Module):
    def __init__(self, grid_size, embed_dim):
        super().__init__()
        self.lat_embed = nn.Parameter(torch.zeros(1, grid_size[0], 1, embed_dim))
        self.lon_embed = nn.Parameter(torch.zeros(1, 1, grid_size[1], embed_dim))
        trunc_normal_(self.lat_embed, std=0.02)
        trunc_normal_(self.lon_embed, std=0.02)

    def forward(self, lat):
        lat_factor = torch.cos(torch.deg2rad(lat)).unsqueeze(-1)
        pos_embed = self.lat_embed * lat_factor + self.lon_embed
        return pos_embed


class downsample2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3):
        super(downsample2d, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=1, stride=2)
        self.norm = nn.GroupNorm(2, out_channels)
        self.act = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x):
        y = self.conv(x)
        y = self.act(self.norm(y))
        return y


class CubePatchMerging(nn.Module):
    def __init__(self, dim, norm_layer=nn.LayerNorm, dtype=torch.float32):
        super().__init__()
        self.dim = dim
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False, dtype=dtype)
        self.norm = norm_layer(4 * dim, dtype=dtype)

    def forward(self, x):
        B, H, W, C = x.shape
        x0 = x[:, 0::2, 0::2, :]
        x1 = x[:, 1::2, 0::2, :]
        x2 = x[:, 0::2, 1::2, :]
        x3 = x[:, 1::2, 1::2, :]
        x = torch.cat([x0, x1, x2, x3], -1)
        x = x.view(B, -1, 4 * C)
        x = self.norm(x)
        x = self.reduction(x)
        return x.view(B, H // 2, W // 2, -1)


class PatchEmbed(nn.Module):
    def __init__(self, img_size=(224, 224), patch_size=(8, 8), in_chans=1, embed_dim=768):
        super().__init__()
        num_patches = (img_size[1] // patch_size[1]) * (img_size[0] // patch_size[0])
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        B, C, H, W = x.shape
        assert H == self.img_size[0] and W == self.img_size[
            1], f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        x = self.proj(x).flatten(2).transpose(1, 2)
        return x


class BasicConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=1, stride=1)
        self.norm = nn.GroupNorm(2, out_channels)
        self.act = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x):
        y = self.conv(x.permute(0, 3, 1, 2))
        y = self.act(self.norm(y))
        return y.permute(0, 2, 3, 1)


class MLP(nn.Module):
    def __init__(self, in_features, out_features=None, act_layer=nn.GELU, drop=0., dtype=torch.float32):
        super().__init__()
        self.fc1 = nn.Linear(in_features, out_features, dtype=dtype)
        self.act = act_layer()

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        return x


class MLP2(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.,
                 dtype=torch.float32):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features, dtype=dtype)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features, dtype=dtype)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class BidirectionalFusion(nn.Module):
    def __init__(self, channels, reduction=4, gate_type='channelwise', out_channels=None):
        super().__init__()
        C = channels
        self.C = C
        self.out_channels = out_channels or C
        self.gate_type = gate_type
        hidden = max(8, C // reduction)
        self.local_from_x3 = nn.Sequential(
            nn.Conv2d(C, C, kernel_size=3, padding=1, groups=C),
            nn.Conv2d(C, C, kernel_size=1),
            nn.GELU()
        )
        self.global_from_x1 = nn.Sequential(
            nn.Conv2d(C, C, kernel_size=1),
            nn.GELU()
        )
        in_ch = C * 4
        self.gate_net = nn.Sequential(
            nn.Conv2d(in_ch, hidden, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(hidden, C * 2 if gate_type == 'channelwise' else 2, kernel_size=1)
        )
        self.fuse = nn.Sequential(
            nn.Conv2d(C * 2, self.out_channels, kernel_size=1),
            nn.GELU()
        )
        nn.init.zeros_(self.gate_net[-1].bias)

    def forward(self, x1, x3):
        B, C, H, W = x1.shape
        assert C == self.C, "channels mismatch"
        cat = torch.cat([x1, x3, x1 * x3, x1 - x3], dim=1)
        gates = self.gate_net(cat)
        if self.gate_type == 'channelwise':
            g1g3 = torch.sigmoid(gates)
            g_x3_to_x1, g_x1_to_x3 = torch.split(g1g3, C, dim=1)
        elif self.gate_type == 'spatial':
            g1g3 = torch.sigmoid(gates)
            g_x3_to_x1 = g1g3[:, :1, :, :].expand(B, C, H, W)
            g_x1_to_x3 = g1g3[:, 1:, :, :].expand(B, C, H, W)
        else:
            g1g3 = torch.sigmoid(gates)
            g_x3_to_x1, g_x1_to_x3 = torch.split(g1g3, C, dim=1)
        info_from_x3 = self.local_from_x3(x3)
        info_from_x1 = self.global_from_x1(x1)
        x1p = x1 + g_x1_to_x3 * info_from_x3
        x3p = x3 + g_x3_to_x1 * info_from_x1
        fused = self.fuse(torch.cat([x1p, x3p], dim=1))
        return fused


class MixedBlock(nn.Module):
    def __init__(self, embed_dim, h, w, mlp_ratio=4.0, norm_layer=nn.LayerNorm, depth=1, Decoder=False):
        super().__init__()
        self.depth = depth
        self.blocks = nn.ModuleList()
        for _ in range(depth):
            block = nn.ModuleDict({
                'AFNO': AFNOBlock(dim=embed_dim, mlp_ratio=mlp_ratio, drop=0.1, drop_path=0.1, act_layer=nn.GELU,
                                  norm_layer=norm_layer, h=h, w=w),
                'CONV': ConvBlock(dim=embed_dim, mlp_ratio=mlp_ratio, drop=0.1, drop_path=0.1, ),
            })
            self.blocks.append(block)
        self.mlp2 = MLP(in_features=embed_dim * 2, out_features=embed_dim)
        self.mlp = MLP(in_features=embed_dim * 1, out_features=embed_dim)
        if Decoder:
            self.conv3d = nn.Conv3d(in_channels=embed_dim, out_channels=embed_dim, kernel_size=(3, 1, 1),
                                    stride=(3, 1, 1))
        else:
            self.conv3d = nn.Conv3d(in_channels=embed_dim, out_channels=embed_dim, kernel_size=(2, 1, 1),
                                    stride=(2, 1, 1))
        self.gate = nn.Sequential(nn.Conv2d(embed_dim * 2, embed_dim, kernel_size=1), nn.Sigmoid())
        self.Fusion = BidirectionalFusion(channels=embed_dim)

    def forward(self, x_raw):
        z = x_raw
        agg = []
        for block in self.blocks:
            x1 = block['AFNO'](z)
            x3 = block['CONV'](z)
            x_fuse = self.Fusion(x1.permute(0, 3, 1, 2), x3.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)
            x_cat = self.mlp(x_fuse)
            z = z + x_cat
            agg.append(z.permute(0, 3, 1, 2).unsqueeze(2))
        h = torch.cat(agg, dim=2)
        h = self.conv3d(h).squeeze(2).permute(0, 2, 3, 1)
        return h


def stride_generator(N, reverse=False):
    strides = [1, 2] * 10
    if reverse:
        return list(reversed(strides[:N]))
    else:
        return strides[:N]

class BasicConv2d(nn.Module):
    def __init__(
            self,
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            transpose=False,
            act_norm=False
    ):
        super(BasicConv2d, self).__init__()
        self.act_norm = act_norm
        if not transpose:
            self.conv = nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding
            )
        else:
            self.conv = nn.ConvTranspose2d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                output_padding=stride // 2
            )
        self.norm = nn.GroupNorm(2, out_channels)
        self.act = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x):
        y = self.conv(x)
        if self.act_norm:
            y = self.act(self.norm(y))
        return y


class ConvDynamicsLayer(nn.Module):
    def __init__(self, C_in, C_out, stride, transpose=False, act_norm=True):
        super(ConvDynamicsLayer, self).__init__()
        if stride == 1:
            transpose = False
        self.conv = BasicConv2d(
            C_in,
            C_out,
            kernel_size=3,
            stride=stride,
            padding=1,
            transpose=transpose,
            act_norm=act_norm
        )

    def forward(self, x):
        y = self.conv(x)
        return y


class AtmosphericDecoder(nn.Module):
    def __init__(self, spatial_hidden_dim, C_out, num_spatial_layers):
        super(AtmosphericDecoder, self).__init__()
        strides = stride_generator(num_spatial_layers, reverse=True)
        self.dec = nn.Sequential(
            *[ConvDynamicsLayer(spatial_hidden_dim, spatial_hidden_dim, stride=s, transpose=True) for s in
              strides[:-1]],
            ConvDynamicsLayer(spatial_hidden_dim, spatial_hidden_dim, stride=strides[-1], transpose=True)
        )
        self.readout = nn.Conv2d(spatial_hidden_dim, C_out, 1)

    def forward(self, hid):
        for i in range(0, len(self.dec) - 1):
            hid = self.dec[i](hid)
        Y = self.dec[-1](hid)
        Y = self.readout(Y)
        return Y


class Tongzhou(nn.Module):
    def __init__(
            self,
            params,
            img_size=(720, 1440),
            embed_dim=64,
            mlp_ratio=4.,
    ):
        super().__init__()
        self.params = params
        self.img_size = img_size
        self.in_chans = params.N_in_channels
        self.out_chans = params.N_out_channels
        self.num_features = self.embed_dim = embed_dim

        self.patch_size = (params.patch_size, params.patch_size)
        self.patch_embed = PatchEmbed(img_size=img_size, patch_size=self.patch_size, in_chans=self.in_chans,
                                      embed_dim=embed_dim)
        self.h = img_size[0] // self.patch_size[0]
        self.w = img_size[1] // self.patch_size[1]
        self.wpos_embed = LatAwarePosEmbed(grid_size=(self.h, self.w), embed_dim=embed_dim)

        self.embed_dim_wfno = embed_dim // 2
        self.patch_size_wfno = (params.patch_size // 2, params.patch_size // 2)
        self.patch_embed_wfno = PatchEmbed(img_size=img_size, patch_size=self.patch_size_wfno, in_chans=self.in_chans,
                                           embed_dim=self.embed_dim_wfno)
        self.h_wfno = img_size[0] // self.patch_size_wfno[0]
        self.w_wfno = img_size[1] // self.patch_size_wfno[1]
        self.wpos_embed_wfno = LatAwarePosEmbed(grid_size=(self.h_wfno, self.w_wfno), embed_dim=self.embed_dim_wfno)

        self.downsample1 = CubePatchMerging(dim=self.embed_dim_wfno)

        self.embed_dim_conv = embed_dim // 4
        self.patch_size_conv = (params.patch_size // 4, params.patch_size // 4)
        self.patch_embed_conv = PatchEmbed(img_size=img_size, patch_size=self.patch_size_conv, in_chans=self.in_chans,
                                           embed_dim=self.embed_dim_conv)
        self.h_conv = img_size[0] // self.patch_size_conv[0]
        self.w_conv = img_size[1] // self.patch_size_conv[1]
        self.wpos_embed_conv = LatAwarePosEmbed(grid_size=(self.h_conv, self.w_conv), embed_dim=self.embed_dim_conv)

        self.downsample2 = CubePatchMerging(dim=self.embed_dim_conv)
        self.downsample3 = CubePatchMerging(dim=self.embed_dim_conv * 2)

        self.block_M = MixedBlock(embed_dim=embed_dim, h=self.h, w=self.w, mlp_ratio=mlp_ratio, norm_layer=nn.LayerNorm,
                                  depth=2)
        self.block_MD = MixedBlock(embed_dim=embed_dim, h=self.h, w=self.w, mlp_ratio=mlp_ratio,
                                   norm_layer=nn.LayerNorm, depth=3, Decoder=True)

        end_dim = self.out_chans * self.patch_size[0] * self.patch_size[1]
        self.head = nn.Linear(embed_dim, end_dim, bias=False)
        self.fc_end = MLP2(end_dim, end_dim, end_dim)

        self.mlp = MLP(in_features=embed_dim * 3, out_features=embed_dim)
        self.gate = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.ReLU(inplace=True),
            nn.Linear(embed_dim // 2, 1)
        )
        self.unpatch = AtmosphericDecoder(embed_dim, 1, 6)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            if m.weight.is_complex():
                trunc_normal_(m.weight.real.data, std=0.02)
                trunc_normal_(m.weight.imag.data, std=0.02)
            else:
                trunc_normal_(m.weight.data, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed', 'cls_token'}

    def forward(self, x):
        B, S, L, H, W = x.shape
        x = x.view(B, L, H, W)
        x_afno = x
        B, C, H, W = x_afno.shape
        x_afno = self.patch_embed(x_afno)
        x_afno = x_afno.reshape(B, self.h, self.w, self.embed_dim)
        lat = torch.linspace(-90, 90, self.h).unsqueeze(1).repeat(1, self.w).to(x_afno.device)
        lat = lat.unsqueeze(0).repeat(B, 1, 1)
        x_afno = x_afno + self.wpos_embed(lat)

        x_wfno = x
        B = x_wfno.shape[0]
        x_wfno = self.patch_embed_wfno(x_wfno)
        x_wfno = x_wfno.reshape(B, self.h_wfno, self.w_wfno, self.embed_dim_wfno)
        lat = torch.linspace(-90, 90, self.h_wfno).unsqueeze(1).repeat(1, self.w_wfno).to(x_wfno.device)
        lat = lat.unsqueeze(0).repeat(B, 1, 1)
        x_wfno = x_wfno + self.wpos_embed_wfno(lat)
        x_wfno = self.downsample1(x_wfno)

        x_conv = x
        B = x_conv.shape[0]
        x_conv = self.patch_embed_conv(x_conv)
        x_conv = x_conv.reshape(B, self.h_conv, self.w_conv, self.embed_dim_conv)
        lat = torch.linspace(-90, 90, self.h_conv).unsqueeze(1).repeat(1, self.w_conv).to(x_conv.device)
        lat = lat.unsqueeze(0).repeat(B, 1, 1)
        x_conv = x_conv + self.wpos_embed_conv(lat)
        x_conv = self.downsample2(x_conv)
        x_conv = self.downsample3(x_conv)

        x = torch.stack([x_afno, x_wfno, x_conv], dim=3)
        scores = self.gate(x).squeeze(-1)
        attn = torch.softmax(scores, dim=3).unsqueeze(-1)
        z = (attn * x).sum(dim=3)

        z = self.block_M(z)
        x_d = self.block_MD(z)
        x_d = self.unpatch(x_d.permute(0, 3, 1, 2))
        x_d = x_d.view(B, S, L, H, W)
        return x_d


class Params:
    def __init__(self):
        self.patch_size = 8
        self.N_in_channels = 1
        self.N_out_channels = 1


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    img_size = (720, 1440)
    model = Tongzhou(params=Params()).to(device)
    model.eval()
    x = torch.randn(8, 1, 1, *img_size, device=device)  # [B, T, C, H, W]
    with torch.no_grad():
        y = model(x)
    print(y.shape)
