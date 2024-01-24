import torch
import torch.nn as nn
import torch.nn.functional as F
class Density(nn.Module):
    def __init__(self, params_init={}):
        super().__init__()
        for p in params_init:
            param = nn.Parameter(torch.tensor(params_init[p]))
            setattr(self, p, param)

    def forward(self, sdf, beta=None):
        return self.density_func(sdf, beta=beta)


class LaplaceDensity(Density):  # alpha * Laplace(loc=0, scale=beta).cdf(-sdf)
    def __init__(self, params_init={}, beta_min=0.0001):
        super().__init__(params_init=params_init)
        self.beta_min = torch.tensor(beta_min).cuda()

    def density_func(self, sdf, beta=None):
        if beta is None:
            beta = self.get_beta()

        alpha = 1 / beta
        # breakpoint()
        # alpha.to(sdf.device)
        sdf.to(alpha.device)
        return alpha * (0.5 + 0.5 * sdf.sign() * torch.expm1(-sdf.abs() / beta))

    def get_beta(self):
        beta = self.beta.abs() + self.beta_min
        return beta
class Binarize(nn.Module):
    def __init__(self, threshold=0.5):
        super(Binarize, self).__init__()
        self.threshold = nn.Parameter(torch.tensor(threshold), requires_grad=True)

    def forward(self, x):
        return (x > self.threshold).float()

class RefinementNetwork(nn.Module):
    def __init__(self):
        super(RefinementNetwork, self).__init__()
        
        # Define the layers of the refinement network
        self.conv1 = nn.ConvTranspose2d(4, 64, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.ConvTranspose2d(64, 32, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.ConvTranspose2d(32, 3, kernel_size=3, stride=1, padding=1)
        self.relu = nn.ReLU()
        self.sig = nn.Sigmoid()
    
    def forward(self, input_patch):
        # Concatenate the rgb patch and mask patch along the channel dimension
        # input_patch = torch.cat((rgb_patch, mask_patch), dim=-11)
        
        # Pass the input patch through the refinement network
        # x = torch.cat((rgb_patch, mask_patch), dim=1)  # Concatenate rgb_patch and mask_patch along the channel dimension
        x = self.relu(self.conv1(input_patch.permute(0,3,1,2)))
        x = self.relu(self.conv2(x))
        x = self.conv3(x)
        x = self.sig(x)
        
        return x


# import torch
# import torch.nn as nn

class binary_MLP(nn.Module):
    def __init__(self,input_size  = 1, hidden_size = 32, output_size = 1):
        super(binary_MLP, self).__init__()

        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, output_size)
        self.relu = nn.ReLU()

    def forward(self, density):
        density = density.unsqueeze(-1)
        x = self.relu(self.fc1(density))
        x = self.relu(self.fc2(x))
        mask = torch.sigmoid(self.fc3(x))
        mask = mask.view(-1)
        return mask
