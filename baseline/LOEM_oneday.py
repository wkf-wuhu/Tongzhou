from models.core.PatchEmbedding import PatchEmbedding
from models.core.ChangeChannel import ChangeChannel
from models.core.DCNBlock import DCNBlock
from models.core.DCNOceanBlock import DCNOceanBlock
from models.core.SwinTransformerBlockv2 import SwinTransformerBlock
from models.core.AdaptiveFourierNeuralBlock import AdaptiveFourierNeuralBlock
from models.core.PatchUnembedding import PatchUnembedding
from models.core.DownSampling import DownSampling
from models.core.UpSampling import UpSampling
import torch.nn as nn
import torch


class Loem(nn.Module) :

    def __init__(self,
                core_op : str = "DCN",
                channels = [32, 32, 64, 64],  # [192, 192, 768, 768],  [32, 32, 64, 64],
                depths = [4, 4, 18, 4],     # [4, 4, 18, 4],
                groups = [8, 8, 16, 16],
                down_sampling = [False, False, True, False],
                bypass=[True, True, True, True],
                patch_size=(4, 4),
                window_size=5,
                shift_size=3,
                symmetric_decoder=True,
                mlp_ratio = 4.,
                drop_rate=0.,
                drop_path_rate=0.,
                act_layer = nn.GELU,
                norm_layer = nn.LayerNorm,
                input_dim = (720, 1440),
                variables=46,
                debugger = 0,
                dtype=torch.float32,
                mtp=False,
                ) :
        """
        Args:

            core_op :  Core operator, enumerating among DCN, ST and AFN
            channels : Channel for each stage
            depths :  Depth for each stage
            groups :  Operator head/groups for Transformer/dcn at each stage
            down_sampling :  If or not downsample at start of each stage
            bypass :  If or not add residual bypass cross over symmetric encoder and decoder blocks.
            window_size :  SwinTransformer only, window size
            shift_size :  SwinTransformer only, window shift size
            symetric_decoer :  If or not use symmetric decoder
            mlp_ratio : 
            drop_rate : 
            drop_path_rate :
            act_layer :
            norm_layer :
            input_dim :  Input resolution with height and weight
            variables :  Variables, initiated channels
            debugger :
            dtype :
            mtp :  Whether enable multi-token prediction
        """

        super().__init__()

        self.core_op = core_op
        self.channels = channels
        self.depths = depths
        self.groups = groups
        self.down_sampling = down_sampling
        self.symmetric_decoder = symmetric_decoder
        self.bypass = bypass
        self.patch_size = patch_size
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        self.drop_rate = drop_rate
        self.drop_path_rate = drop_path_rate
        self.act_layer = act_layer
        self.norm_layer = norm_layer
        self.input_dim = input_dim
        self.variables = variables
        self.debugger = debugger
        self.dtype = dtype
        self.mtp = mtp


        self.patch = PatchEmbedding(
            img_size=self.input_dim,
            patch_size=self.patch_size,
            in_channels=self.variables,
            embedding_dim=self.channels[0],
            dtype=self.dtype
        )


        current_input_dim = (self.input_dim[0] // self.patch_size[0], self.input_dim[1] // self.patch_size[1])

        encoder_block = []
        for i in range(len(self.depths)) :
            block = []
            if i > 0 :
                if self.down_sampling[i] :
                    block.append(
                        DownSampling(
                            input_dim=current_input_dim,
                            in_channels=self.channels[i - 1],
                            out_channels=self.channels[i],
                            norm_layer=self.norm_layer,
                            dtype=self.dtype,
                        )
                    )
                    current_input_dim = (current_input_dim[0] // 2, current_input_dim[1] // 2)
                else :
                    block.append(
                        ChangeChannel (
                            input_dim=current_input_dim,
                            in_channels=self.channels[i - 1],
                            out_channels=self.channels[i],
                            norm_layer=self.norm_layer,
                            dtype=self.dtype,
                        )
                    )


            block.append(
                self.make_stage(
                    channel=self.channels[i],
                    depth=self.depths[i],
                    group=self.groups[i],
                    input_dim=current_input_dim,
                )
            )

            encoder_block.append(nn.Sequential(*block))
        
        self.encoder_list = nn.ModuleList(encoder_block)


        decoder_block = []
        for i in reversed(range(len(self.depths))) :
            block = []
            
            if symmetric_decoder :
                block.append(
                    self.make_stage(
                        channel=self.channels[i],
                        depth=self.depths[i],
                        group=self.groups[i],
                        input_dim=current_input_dim,
                    )
                )
            else :
                # In asymmetric decoders, the core operator is simply replaced with an activation function
                # Core operator could be repleated by activating layer in asymmetric decoder
                block.append(
                    act_layer()
                )

            if i > 0 :
                if self.down_sampling[i] :
                    block.append(
                        UpSampling(
                            input_dim=current_input_dim,
                            in_channels=self.channels[i],
                            out_channels=self.channels[i - 1],
                            norm_layer=self.norm_layer,
                            dtype=self.dtype,
                        )
                    )

                    current_input_dim = (current_input_dim[0] * 2, current_input_dim[1] * 2)
                else :
                    block.append(
                        ChangeChannel (
                            input_dim=current_input_dim,
                            in_channels=self.channels[i],
                            out_channels=self.channels[i - 1],
                            norm_layer=self.norm_layer,
                            dtype=self.dtype,
                        )
                    )

            decoder_block.append(nn.Sequential(*block))
        
        self.decoder_list = nn.ModuleList(decoder_block)

        self.unpatch = PatchUnembedding(
            img_size=self.input_dim,
            patch_size=self.patch_size,
            out_channels=self.variables,
            embedding_dim=self.channels[0],
            dtype=dtype
        )
    
    def make_stage(
            self,
            channel,
            depth,
            group,
            input_dim,
        ) :

        if self.core_op == "DCN" :
            return nn.Sequential( *[ DCNBlock(
                dim=channel,
                group=group,
                input_resolution=input_dim,
                mlp_ratio=self.mlp_ratio,
                drop=self.drop_rate,
                drop_path=self.drop_path_rate,
                act_layer=self.act_layer,
                norm_layer=self.norm_layer,
                dtype=self.dtype,
            ) for i in range(depth) ])
        elif self.core_op == "DCNOcean" : 
            return nn.Sequential( *[ DCNOceanBlock(
                dim=channel,
                group=group,
                input_resolution=input_dim,
                mlp_ratio=self.mlp_ratio,
                drop=self.drop_rate,
                drop_path=self.drop_path_rate,
                act_layer=self.act_layer,
                norm_layer=self.norm_layer,
                dtype=self.dtype,
            ) for i in range(depth) ])
        elif self.core_op == "ST" :
            return nn.Sequential( *[ SwinTransformerBlock(
                dim=channel,
                input_resolution=input_dim,
                num_heads=group,
                window_size=self.window_size, # ST only
                shift_size=0 if i % 2 else self.shift_size, # ST only
                mlp_ratio=self.mlp_ratio,
                drop=self.drop_rate,
                drop_path=self.drop_path_rate,
                act_layer=self.act_layer,
                norm_layer=self.norm_layer,
                dtype=self.dtype
            ) for i in range(depth)])
        elif self.core_op == "AFN" :
            return nn.Sequential( *[ AdaptiveFourierNeuralBlock(
                dim=channel,
                input_size=input_dim,
                mlp_ratio=self.mlp_ratio,
                drop=self.drop_rate,
                drop_path=self.drop_path_rate,
                act_layer=self.act_layer,
                norm_layer=self.norm_layer,
                dtype=self.dtype
            ) for i in range(depth)])
    
    def forward(self, x) :
        B, S, L, H, W = x.shape  # B,T,C,H,W
        
        assert S == 1, "LOEM model uses only at 1 sequence as input."

        x = x.view(B, L, H, W)  # [1, 46, 720, 1440]

        x = self.patch(x)  # [1, 64800, 192]   Conv2d(46, 192, kernel_size=(4, 4), stride=(4, 4))

        x_bypass = []  # 跳跃连接
        for i in range(len(self.encoder_list)) :
            if self.bypass[i] and self.symmetric_decoder:
                x_bypass.append(x)
            else:
                x_bypass.append(0)
            
            x = self.encoder_list[i](x)  # [1, 64800, 192] - [1, 16200, 768]

        for i in range(len(self.decoder_list)) :
            x = x_bypass[len(x_bypass) - i - 1] + self.decoder_list[i](x)
        
        h = x  # [1, 64800, 192]

        x = self.unpatch(x)  # [1, 64800, 192]  [1, 46, 720, 1440]

        x = x.view(B, S, L, H, W)  # [1, 1, 46, 720, 1440]

        if self.mtp :
            return x, h
        else :
            return x

if __name__ == "__main__":
    import torch

    # 模拟输入数据：batch=1，时间步S=1，变量数=46，空间分辨率=720x1440
    B, S, L, H, W = 2, 1, 46, 720, 1440
    x = torch.randn(B, S, L, H, W)

    # 创建模型实例
    model = Loem(
        core_op="DCNOcean",             # 可选："DCN", "DCNOcean", "ST", "AFN"
        input_dim=(H, W),
        variables=L,
        debugger=0,
        dtype=torch.float32
    )

    # 将模型移至GPU（如果可用）
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    x = x.to(device)

    # 前向传播
    with torch.no_grad():
        out = model(x)

    print("Input shape :", x.shape)
    print("Output shape:", out.shape)

    # 统计参数数量
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    model_size_mb = total_params * 4 / (1024 ** 2)  # float32: 4 bytes

    print(f"📊 Total Parameters     : {total_params:,}")
    print(f"✅ Trainable Parameters : {trainable_params:,}")
    print(f"💾 Model Size           : {model_size_mb:.2f} MB")
