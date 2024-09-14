import torch


class DropBlock(torch.nn.Module):
    """
    Step-2] Parameters Initialization
    Initialization of block_size, keep_prob, stride and padding respectively
    which can be altered accordingly for experimentation
    """

    def __init__(self, block_size=3, keep_prob=0.9):
        self.block_size = block_size
        self.keep_prob = keep_prob
        self.gamma = None
        self.kernel_size = (block_size, block_size)
        self.stride = (1, 1)
        self.padding = (block_size // 2, block_size // 2)
        self.training = True

    """
    Step-3] Compute gamma using the equation mentioned aprior in the blog.
    Dependency can be seen on block_size and keep_prob for its computation.
    """

    def calculate_gamma(self, x):

        return (1 - self.keep_prob) * x.shape[-1] ** 2 / \
               (self.block_size ** 2 * (x.shape[-1] - self.block_size + 1) ** 2)

    def forward(self, x):
        """
        Step-4]
        Check on whether it is inference mode as DropBlock Algorithm like DropOut Algorithm is performed only during the time of training
        """
        if not self.training:
            return x

        if self.gamma is None:
            self.gamma = self.calculate_gamma(x)

        """
        Step-5]
        Use the gamma obtained to compute the sample mask using Bernoulli
        """
        p = torch.ones_like(x) * self.gamma

        mask = 1 - torch.nn.functional.max_pool2d(torch.bernoulli(p),
                                                  self.kernel_size,
                                                  self.stride,
                                                  self.padding)

        """
        Step-6]
        Normalization of Features
        """
        return mask * x * (mask.numel() / mask.sum())


if __name__ == '__main__':
    x = torch.randint(1, 10, (1, 1, 5, 5))
    print(x.shape)
    print(x)

    db = DropBlock(block_size=3, keep_prob=0.5)
    db.calculate_gamma(x)
    out = db.forward(x)
    print(out.shape)
    print(out)