
from models.core.CubePatchEmbedding import CubePatchEmbedding
from models.core.SwinTransformerBlock import SwinTransformerBlock
import torch.nn as nn
import torch

class SwinRNNPlus (nn.Module) :

    def __init__(self, input_dim=(720, 1440), hidden_dim=1024, sequence_pair=(6, 20), variables=46, debugger=0, dtype=torch.float32) :
        """
        input_dim : spacial resolution(latitude, longitude)
        hidden_dim : embedding dimension
        sequence_pair : length of input and output(input_length, output_length)
        variables : number of variables
        debugger : debugging model
        dtype : data type
        """

        super().__init__()

        self.debugger = debugger
        self.input_dim = input_dim
        self.sequence_pair = sequence_pair

        self.encoder_len = 6
        self.decoder_len = 6

        self.cube_patch_cube = CubePatchEmbedding(
            img_size=(sequence_pair[0], * input_dim),
            patch_size=(sequence_pair[0], 4, 4),
            in_channels=variables,
            embedding_dim=hidden_dim,
            dtype=dtype,
        )

        self.core_dim = (input_dim[0] // 4, input_dim[1] // 4)


        encoder = []

        for i in range(self.encoder_len) :
            encoder.append(SwinTransformerBlock(
                dim=hidden_dim,
                input_resolution=self.core_dim,
                num_heads=8,
                window_size=5,
                dtype=dtype
            ))
            encoder.append(SwinTransformerBlock(
                dim=hidden_dim,
                input_resolution=self.core_dim,
                num_heads=8,
                window_size=5,
                shift_size=3,
                dtype=dtype
            ))
        
        self.encoder_transformer = nn.Sequential(
            * encoder
        )

        self.conv1 = nn.Conv3d(
            in_channels=variables,
            out_channels=hidden_dim,
            kernel_size=(1, 4, 4),
            stride=(1, 4, 4),
            dtype=dtype
        )

        self.conv2 = nn.Conv3d(
            in_channels=hidden_dim,
            out_channels=hidden_dim,
            kernel_size=(2, 1, 1),
            stride=(2, 1, 1),
            dtype=dtype
        )


        decoder = []

        for i in range(self.decoder_len) :
            decoder.append(SwinTransformerBlock(
                dim=hidden_dim,
                input_resolution=self.core_dim,
                num_heads=8,
                window_size=5,
                dtype=dtype
            ))

            decoder.append(SwinTransformerBlock(
                dim=hidden_dim,
                input_resolution=self.core_dim,
                num_heads=8,
                window_size=5,
                shift_size=3,
                dtype=dtype
            ))
        self.decoder_transformer = nn.Sequential(
            * decoder
        )
        
        self.conv3 = nn.Conv3d(
            in_channels=hidden_dim,
            out_channels=hidden_dim,
            kernel_size=(self.decoder_len, 1, 1),
            stride=(self.decoder_len, 1, 1),
            dtype=dtype
        )

        self.conv4 = nn.ConvTranspose3d(
            in_channels=hidden_dim,
            out_channels=variables,
            kernel_size=(1, 4, 4),
            stride=(1, 4, 4),
            dtype=dtype
        )


    
    def forward (self, x) :
        """
        (1, 6, 46, 720, 1440)
        """
        x = x.permute(0, 2, 1, 3, 4) 
        B, L, C, H, W = x.shape   # [1, 6, 46, 720, 1440] B,T,C,H,W  [8, 4, 1, 720, 1440]  B,T,C,H,W

        x_k = x[:, -1, :, :, :].view(B, 1, C, H, W) # (B, 1, C, H, W)
        x_k = x_k.permute((0, 2, 1, 3, 4)) # [1, 46, 1, 720, 1440] 最后一个
        x = torch.permute(x, (0, 2, 1, 3, 4))  # [1, 46, 6, 720, 1440] B,C,T,H,W
        x = self.cube_patch_cube(x) # (B, H/4*W/4, Ccore)   Conv3d(46, 16, kernel_size=(6, 4, 4), stride=(6, 4, 4)) [1, 64800, 16]

        h = x
        if self.debugger == 0 :
            h = self.encoder_transformer(h)
        
        if self.debugger == 2 :
            h = self.test_encoder(h)

        h = h.view(h.shape[0], 1, self.core_dim[0], self.core_dim[1], h.shape[-1]) # (B, 1, H/4, W/4, Ccore)  [1, 1, 180, 360, 16]
        h = h.permute((0, 4, 1, 2, 3)) # (B, Ccore, 1, H/4, W/4)  [1, 16, 1, 180, 360]

        x_out = []

        for i in range(self.sequence_pair[1]):  # 20
            bypass = x_k
            x_k = self.conv1(x_k)  # (B, Ccore, 1, H/4, W/4)  [1, 46, 1, 720, 1440] -- [1, 16, 1, 180, 360]
            h = torch.cat([x_k, h], dim=2)  # (B, Ccore, 2, H/4, W/4)
            h = self.conv2(h) # (B, Ccore, 1, H/4, W/4) [1, 16, 1, 180, 360]
            h = h.view(h.shape[0], h.shape[1], -1).transpose(-1, -2) # (B, H/4*W/4, Ccore)  [1, 64800, 16]

            agg = []

            for j in range(self.decoder_len) :  # 3
                if self.debugger == 0 :
                    h = self.decoder_transformer[2 * j](h)
                    h = self.decoder_transformer[2 * j + 1](h)
                # if self.debugger == 2 :
                    # h = self.test_decoder(h)
                B, L, Ccore = h.shape
                agg.append(h.view(B, 1, L, Ccore))
            
            h = torch.cat(agg, dim=1) # (B, dl, H/4*W/4, Ccore)  [1, 3, 64800, 16]
            h = h.permute((0, 3, 1, 2)).view(h.shape[0], h.shape[-1], self.decoder_len, self.core_dim[0], self.core_dim[1]) # (B, Ccore, dl, H/4, W/4)
            h = self.conv3(h) #(B, Ccore, 1, H/4, W/4)  [1, 16, 1, 180, 360]

            x_k = bypass + self.conv4(h) # (B, C, 1, H, W)
            x_out.append(x_k)
        
        x = torch.cat(x_out, dim=2) # (B, C, out, H, W)  [1, 46, 20, 720, 1440]
        
        # x = x.permute((0, 2, 1, 3, 4)) #(B, out, C, H, W)  [1, 20, 46, 720, 1440]

        return x



def count_parameters(model):
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    size_mb = total_params * 4 / (1024 ** 2)  # float32 = 4 bytes
    print(f"📊 Total Parameters     : {total_params:,}")
    print(f"✅ Trainable Parameters : {trainable_params:,}")
    print(f"💾 Model Size           : {size_mb:.2f} MB")


if __name__ == "__main__":
    import torch

    # 模拟参数
    B = 1        # batch size
    Cin = 46     # number of variables
    H, W = 720, 1440  # spatial resolution
    in_seq_len = 6
    out_seq_len = 20

    # 创建模型实例
    model = SwinRNNPlus(
        input_dim=(H, W),
        hidden_dim=16,
        sequence_pair=(in_seq_len, out_seq_len),
        variables=Cin,
        debugger=0,
        dtype=torch.float32
    )

    # 打印参数量信息
    count_parameters(model)

    # 构造输入数据
    x = torch.randn((B, in_seq_len, Cin, H, W), dtype=torch.float32)

    # 迁移到 GPU（如果可用）
    device = torch.device("cpu") # "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    x = x.to(device)
    model.eval()

    # 前向传播
    with torch.no_grad():
        y = model(x)

    # 输出形状
    print(f"✅ Input shape  : {x.shape}")
    print(f"✅ Output shape : {y.shape}")

