from torch import nn


class SetLayer(nn.Module):
    def __init__(self, size_in, size_out, beta):
        super().__init__()
        self.fc_0 = nn.Linear(size_in, size_out)
        self.fc_1 = nn.Linear(size_in, size_out)
        self.layer_norm = nn.LayerNorm(normalized_shape=size_out)
        self.activation = nn.Softplus(beta=beta)

    def forward(self, x):
        local_feat = x.apply_layer(self.fc_0)
        global_feat = x.apply_layer(self.fc_1, on_global=True)
        out = local_feat + global_feat
        out = out.apply_layer(self.activation)
        out = out.apply_layer(self.layer_norm)
        return out


class SetBlock(nn.Module):
    def __init__(self, d_in, d_out, beta):
        super().__init__()
        self.b1 = SetLayer(d_in, d_out, beta)
        self.b2 = SetLayer(d_out, d_out, beta)

        if d_in != d_out:
            self.skip = nn.Linear(d_in, d_out)
        else:
            self.skip = None

    def forward(self, x):
        if self.skip:
            skip_x = x.apply_layer(self.skip)
        else:
            skip_x = x.clone()

        x = self.b1(x)
        x = self.b2(x)

        return x + skip_x


class SetNet(nn.Module):
    def __init__(self, d_in, d_out, n_blocks, beta):
        """
        :param d_in input size
        :param d_out output size
        :param n_blocks number of Resnet blocks
        :param dim_hidden hiddent dimension throughout network
        :param beta softplus beta, 100 is reasonable; if <=0 uses ReLU activations instead
        """
        super().__init__()
        self.init_layer = SetLayer(d_in, d_out, beta)

        self.blocks = nn.Sequential()
        for i in range(n_blocks):
            self.blocks.append(SetBlock(d_out, d_out, beta))

    def forward(self, x, prev_encoding):
        """
        :param x
        """
        x = self.init_layer(x)
        if prev_encoding is not None:
            x = x + prev_encoding
        features = self.blocks(x)
        return features, features.get_global_feats()

