import torch
import torch.nn as nn
import torch.nn.functional as F
from layers import ResnetBlock, Downsample, Upsample
import time
class DownBlock(nn.Module):

    def __init__(self, in_c, out_c, is_discr=False, down_type='full'): #full, freq
        super().__init__()
        dropout = 0 if not is_discr else 0.1
        self.layers = nn.ModuleList([
            ResnetBlock(in_c,out_c, dropout=dropout, is_discr=is_discr),
            ResnetBlock(out_c, dropout=dropout, is_discr=is_discr),
            ResnetBlock(out_c, dropout=dropout, is_discr=is_discr) if in_c<64 else nn.Identity(),
            Downsample(down_type) if down_type is not None else nn.Identity()])
        self.is_down=down_type is not None
    def forward(self, x):
        for l in self.layers:
            x=l(x)
            
        return x

class UpBlock(nn.Module):
    
    def __init__(self, in_c, out_c,is_last, up_type='full'):
        super().__init__()
        dropout = 0.05
        out_c = in_c if is_last else out_c
        self.layers = nn.ModuleList([
            Upsample(up_type) if up_type is not None else nn.Identity(),
            ResnetBlock(in_c,out_c, dropout=dropout),
            ResnetBlock(out_c, dropout=dropout),
            ResnetBlock(out_c, dropout=dropout) if out_c<64 else nn.Identity(),
            nn.Identity() if not is_last else nn.Conv2d(out_c, 1, kernel_size=1)])
        self.is_up=up_type is not None
            
    def forward(self, x, target_shape):
        for l in self.layers:
            
            if type(l) == Upsample:
                x=l(x, target_shape)
            else:
                x=l(x)
        return x
    
class MELVAE(nn.Module):
    def __init__(self, encoder_channels=[1,16,64,256,32], encoder_downs=[None,'freq','freq','full','full'], 
                 decoder_channels=[1,16,64,256,64,4], decoder_ups=[0,0,1,1,1,1],
                 discr_channels=[1,16,64,192]): 
        """
        Last decoder channels element is the latent size
        """
        super().__init__()
        self.ups = nn.ModuleList([])
        self.downs = nn.ModuleList([])

        for i in range(1, len(encoder_channels)):
            self.downs.append(DownBlock(encoder_channels[i-1], encoder_channels[i],  down_type=encoder_downs[i]))

        self.fc_mu_logvar = nn.Conv2d(encoder_channels[-1], 2*decoder_channels[-1], kernel_size=1, stride=1, padding=0)

        for i in reversed(range(1, len(decoder_channels))): 
            self.ups.append(UpBlock(decoder_channels[i], decoder_channels[i-1], is_last = i==1, up_type=(encoder_downs+[None])[i]))
        
        self.discriminator = nn.ModuleList([])
        for i in range(1, len(discr_channels)):
            self.discriminator.append(DownBlock(discr_channels[i-1], discr_channels[i], is_discr=True, down_type='full'))

        self.discr_linear = nn.Linear(discr_channels[-1], 1)

        



    def discriminator_inference(self, x):
        for layer in self.discriminator:
            x = layer(x)
        x = F.adaptive_avg_pool2d(x, (1, 1)).flatten(start_dim=1)  
        return self.discr_linear(x)

    def encode(self, x):
        """
        x: [batch, 1, bands, time]
        """ 
        shapes = []
        for layer in self.downs:
            if type(layer) == DownBlock and layer.is_down:
                shapes.append([x.shape[2], x.shape[3]])
            x = layer(x)
                
        mu_logvar = self.fc_mu_logvar(x)
        mu = mu_logvar[:,:mu_logvar.shape[1]//2]
        logvar = mu_logvar[:,mu_logvar.shape[1]//2:]
        return mu, logvar, shapes
    
    def decode(self, x, shapes=None):
        i=0
        t=time.time()
        for layer in self.ups:
            if layer.is_up:      
                upsampled_shape = shapes[len(shapes)-1-i] if shapes!=None else None
                x = layer(x, target_shape=upsampled_shape)
                i+=1
            else:
                x = layer(x, None)
        return x
    
    def reparametrize(self, mu, logvar, std_scale=1):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std * std_scale
        
    def get_loss(self, x):
        mu, logvar, shapes = self.encode(x)
        z = self.reparametrize(mu, logvar)
        out = self.decode(z, shapes)

        rec_loss = F.mse_loss(out, x, reduction='mean')
        kl_loss = (-0.5*torch.sum(1+logvar-mu.pow(2)-logvar.exp(),dim=1)).mean()
        pred_generated = self.discriminator_inference(out.detach())
        pred_true = self.discriminator_inference(x)

        disc_loss = (
            F.binary_cross_entropy_with_logits(
                pred_generated, 0.1 * torch.ones_like(pred_generated)
            ) +
            F.binary_cross_entropy_with_logits(
                pred_true, 0.9 * torch.ones_like(pred_true)
            )
        ) / 2
        lambda_gp = 1
        gp = gradient_penalty(self.discriminator_inference, x, out.detach()) 
        disc_loss += lambda_gp * gp

        gen_adv_loss = F.binary_cross_entropy_with_logits(
            self.discriminator_inference(out), torch.ones_like(pred_generated)
        )

        return disc_loss, rec_loss, kl_loss, gen_adv_loss, [mu, logvar]


    


def gradient_penalty(discriminator, real, fake):
    batch_size = real.size(0)
    epsilon = torch.rand(batch_size, 1, 1, 1, requires_grad=True).to(real.device)
    interpolated = epsilon * real + (1 - epsilon) * fake
    interpolated.requires_grad_(True)

    pred_interpolated = discriminator(interpolated)
    gradients = torch.autograd.grad(
        outputs=pred_interpolated,
        inputs=interpolated,
        grad_outputs=torch.ones_like(pred_interpolated),
        create_graph=True,
        retain_graph=True,
    )[0] 
    gradients = gradients.view(gradients.size(0), -1)
    penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    return penalty

if __name__ == '__main__':
    vae = MELVAE().to('cuda')
    in_=torch.randn(2,1,80,300).to('cuda')
    torch.save(vae, 'vae.ckp')
    l1,l2,l3,l4, (_1,_2) = vae.get_loss(in_)
    print(l1,l2,l3,l4)