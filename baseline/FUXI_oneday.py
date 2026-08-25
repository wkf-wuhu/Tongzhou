
from models.core.CubePatchEmbedding import CubePatchEmbedding
from models.core.DownBlock import DownBlock
from models.core.UpBlock import UpBlock
from models.core.SwinTransformerBlockv2 import SwinTransformerBlock
from models.core.CubePatchUnembedding import CubePatchUnembedding
from models.core.MLP import MLP
import torch.nn as nn
import torch


class Fuxi (nn.Module) :

    def __init__(self, input_dim=(720, 1440), hidden_dim=1024, variables=46,debugger = 0, dtype=torch.float32) :
        """
        input_dim : spacial resolution(latitude, longitude)
        variables : number of veriables
        """

        super().__init__()

        self.debugger = debugger

        self.cube_patch_cube = CubePatchEmbedding(
            img_size=(2, * input_dim),
            patch_size=(2, 4, 4),
            in_channels=variables,
            embedding_dim = hidden_dim,
            dtype=dtype,
        )
        self.lnorm2 = nn.LayerNorm(hidden_dim, dtype=dtype)
        self.downblock = DownBlock(
            input_dim=(input_dim[0] // 4, input_dim[1] // 4),
            in_channels=hidden_dim,
            out_channels=hidden_dim,
            dtype=dtype,
        )

        u_modules = []
        for i in range(24) :
            u_modules.append(SwinTransformerBlock(
                dim=hidden_dim,
                input_resolution=(input_dim[0] // 8, input_dim[1] //8),
                num_heads=8,
                window_size=5,  # 会将输入划分为不重叠的 5×5 空间窗口，并在每个窗口内部执行多头注意力
                dtype=dtype,
            ))
            u_modules.append(SwinTransformerBlock(
                dim=hidden_dim,
                input_resolution=(input_dim[0] // 8, input_dim[1] //8),
                num_heads=8,
                window_size=5,
                shift_size=3,  # 将窗口向右下平移 3 个像素，使窗口之间产生交叉区域，从而实现跨窗口特征交互
                dtype=dtype,
            ))
            # 第一个 block 做 局部建模（无交集）
            # 第二个 block 用 shift 滑动窗口，实现 邻接窗口通信（跨区域建模）

        self.u_transformer = nn.Sequential(
            * u_modules
        )

        self.upblock = UpBlock(
            input_dim=(input_dim[0] // 8, input_dim[1] // 8),
            in_channels=hidden_dim,
            out_channels=hidden_dim,
            dtype=dtype,
        )

        self.fc = MLP(hidden_dim, hidden_dim, hidden_dim, dtype=dtype)
        self.cube_unpatch_cube = CubePatchUnembedding(
            img_size=(1, * input_dim),
            patch_size=(1, 4, 4),
            out_channels=variables,
            embedding_dim=hidden_dim,
            dtype=dtype,
        )
    
    def forward(self, x) :
        x = torch.permute(x, (0, 2, 1, 3, 4))  # [1, 2, 46, 720, 1440] B,T,C,H,W  -- [1, 46, 2, 720, 1440]  B,C,T,H,W
        x = self.cube_patch_cube(x)  #  Conv3d(2, 4, 4)  [1, 64800, 16]
        x = self.lnorm2(x)

        x = self.downblock(x)    # [1, 16200, 16]

        bypass = x
        x = self.u_transformer(x)  # [1, 16200, 16]
        x = x + bypass

        x = self.upblock(x)  # [1, 16200, 16] -- [1, 64800, 16]

        x = self.fc(x)  # [1, 64800, 16]
        x = self.cube_unpatch_cube(x)  # [1, 46, 1, 720, 1440]

        x = torch.permute(x, (0, 2, 1, 3, 4))  # [1, 1, 46, 720, 1440]

        return x


def count_parameters(model):
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    size_mb = total_params * 4 / (1024 ** 2)  # float32: 4 bytes
    print(f"📊 Total Parameters     : {total_params:,}")
    print(f"✅ Trainable Parameters : {trainable_params:,}")
    print(f"💾 Model Size           : {size_mb:.2f} MB")

if __name__ == "__main__":
    import torch

    # 模拟输入配置
    B = 1  # batch size
    T = 2  # time steps
    V = 46  # variables
    H, W = 720, 1440  # spatial resolution
    dtype = torch.float32

    # 创建模型实例
    model = Fuxi(input_dim=(H, W), hidden_dim=16, variables=V, debugger=0, dtype=dtype)

    # 打印模型参数量
    count_parameters(model)

    # 模型输入 [B, T, V, H, W]
    x = torch.randn((B, T, V, H, W), dtype=dtype)

    # 选择设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    x = x.to(device)
    model.eval()

    # 执行前向传播
    with torch.no_grad():
        output = model(x)

    # 输出尺寸检查
    print(f"✅ Input shape  : {x.shape}")
    print(f"✅ Output shape : {output.shape}")