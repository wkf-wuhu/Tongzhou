
from models.core.CubePatchEmbedding import CubePatchEmbedding
from models.core.CubePatchMerging import CubePatchMerging
from models.core.CubePatchUnmerging import CubePatchUnmerging
from models.core.CubePatchUnembedding import CubePatchUnembedding
from models.core.EarthSpecificBlock import EarthSpecificBlock
import torch.nn as nn
import torch


class Pangu (nn.Module) :
    
    def __init__(self, input_dim = (720, 1440), hidden_dim = 192, cube_variables=4, surf_variables=6, debugger = 0, dtype=torch.float32) :
        """
        input_dim : spatial resolution (latitude, longitude)
        cube_variables : Quantity of three-dimensional environmental factors
        surf_variables : Number of planar environmental factors
        """
        super().__init__()

        self.debugger = debugger
        self.cube_variables = cube_variables
        self.surf_variables = surf_variables


        self.cube_patch_surf = CubePatchEmbedding (
            img_size = (1, *input_dim),
            patch_size=(1, 4, 4),
            in_channels=surf_variables,
            embedding_dim=hidden_dim,
            dtype=dtype,
        )

        squeezed_dims = (input_dim[0] // 4, input_dim[1] // 4)
        core_dims = (squeezed_dims[0] // 2, squeezed_dims[1] // 2)


        self.encoder_block1 = nn.Sequential(
            EarthSpecificBlock(
                dim=hidden_dim,
                input_resolution=(1, *squeezed_dims),
                num_heads=8,
                window_size=(1, 6, 12), # (2, 6, 12),
                dtype=dtype,
            ),
            EarthSpecificBlock(
                dim=hidden_dim,
                input_resolution=(1, *squeezed_dims),
                num_heads=8,
                window_size=(1, 6, 12),  # (2, 6, 12),
                shift_size=(1, 3, 6),
                dtype=dtype,
            ),
        )

        if input_dim[0] == 720 :
            core_window_size = (1, 6, 12)  # (2, 6, 12)
        else :
            core_window_size = (1, 5, 10)  # (2, 5, 10)

        core_window_shift = (1, 3, 6)
        core_hidden_dim = hidden_dim * 2

        u_encoding_modules = []

        for i in range(3) :
            u_encoding_modules.append(
                EarthSpecificBlock(
                    dim=core_hidden_dim,
                    input_resolution=(1, *core_dims),
                    num_heads=8,
                    window_size=core_window_size,
                    dtype=dtype,
                ),
            )
            u_encoding_modules.append(
                EarthSpecificBlock(
                    dim=core_hidden_dim,
                    input_resolution=(1, *core_dims),
                    num_heads=8,
                    window_size=core_window_size,
                    shift_size=core_window_shift,
                    dtype=dtype,
                ),
            )

        self.encoder_block2 = nn.Sequential(
            CubePatchMerging(
                input_resolution=(1, *squeezed_dims),
                dim=hidden_dim,
                dtype=dtype,
            ),
            * u_encoding_modules
        )


        self.decode_block1 = nn.Sequential(
            EarthSpecificBlock(
                dim=hidden_dim,
                input_resolution=(1, *squeezed_dims),
                num_heads=8,
                window_size=core_window_size,
                dtype=dtype,
            ), EarthSpecificBlock(
                dim=hidden_dim,
                input_resolution=(1, *squeezed_dims),
                num_heads=8,
                window_size=core_window_size,
                shift_size=core_window_shift,
                dtype=dtype,
            ),
        )


        u_decoding_modules = []

        for i in range(3) :
            u_decoding_modules.append(
                EarthSpecificBlock(
                    dim=core_hidden_dim,
                    input_resolution=(1, *core_dims),
                    num_heads=8,
                    window_size=core_window_size,
                    dtype=dtype,
                )
            )
            u_decoding_modules.append(
                EarthSpecificBlock(
                    dim=core_hidden_dim,
                    input_resolution=(1, *core_dims),
                    num_heads=8,
                    window_size=core_window_size,
                    shift_size=core_window_shift,
                    dtype=dtype,
                ),
            )
            

        self.decode_block2 = nn.Sequential(
            * u_decoding_modules,
            CubePatchUnmerging(
                input_resolution=(1, *core_dims),
                dim=core_hidden_dim,
                dtype=dtype,
            ),
        )
        
        self.cube_unpatch_surf = CubePatchUnembedding(
            img_size=(1, *input_dim),
            patch_size=(1, 4, 4),
            out_channels=surf_variables,
            embedding_dim=hidden_dim,
            dtype=dtype,
        )
    
    def forward (self, x) : 
        # Batch, Sequence, Layer, Height, Weight
        B, S, L, H, W = x.shape   # [2, 1, 46, 720, 1440]  B,T,C,H,W

        assert S == 1, "Pangu model use only at 1 sequence as input"

        x_surf = x[:, 0, 10*self.cube_variables:, :, :].view(B, self.surf_variables, 1, H, W)  # [2, 6, 1, 720, 1440]

        x = self.cube_patch_surf(x_surf)

        x = self.encoder_block1(x)  # [2, 64800, 32]
        x_bypass = x

        x = self.encoder_block2(x)  
        x = self.decode_block2(x)   

        x = x + x_bypass

        x = self.decode_block1(x)   # [2, 64800, 32]

        x = self.cube_unpatch_surf(x).view(B, S, L, H, W)   #  B, L, S, H, W) --  B,T,C,H,W

        return x  # [2, 1, 46, 720, 1440]


def count_parameters(model):
    total_params = sum(p.numel() for p in model.parameters())
    total_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    size_in_mb = total_params * 4 / (1024 ** 2)  # 假设 float32，每个参数4字节
    print(f"📊 Total Parameters     : {total_params:,}")
    print(f"✅ Trainable Parameters : {total_trainable:,}")
    print(f"💾 Model Size           : {size_in_mb:.2f} MB")


if __name__ == "__main__":
    import torch

    # 输入参数配置
    B = 2  # batch size
    S = 1  # sequence length, 必须为1
    cube_vars = 4  # 三维变量数量
    surf_vars = 6  # 面变量数量
    H, W = 720, 1440  # 输入空间分辨率
    L = 10 * cube_vars + surf_vars  # 通道数量（变量总数）
    dtype = torch.float32

    # 创建输入数据 [B, S, L, H, W]
    x = torch.randn((B, S, L, H, W), dtype=dtype)

    # 实例化模型
    model = Pangu(input_dim=(H, W), cube_variables=cube_vars, surf_variables=surf_vars, hidden_dim=16, debugger=0, dtype=dtype)

    # 将模型移至合适设备（GPU如果可用）
    device = torch.device("cpu")  #  "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    x = x.to(device)

    # 模型前向传播
    with torch.no_grad():
        output = model(x)

    # 输出形状信息
    # Input shape : torch.Size([2, 1, 46, 720, 1440])
    # Output shape: torch.Size([2, 1, 46, 720, 1440])
    print("Input shape :", x.shape)
    print("Output shape:", output.shape)

    count_parameters(model)
    # from torchinfo import summary
    # summary(model, input_size=(B, S, L, H, W), dtypes=[torch.float32])
