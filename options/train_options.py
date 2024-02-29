from .base_options import BaseOptions

class TrainOptions(BaseOptions):
    """This class includes training options.
    
    It also includes shared options defined in BaseOptions.
    """
    def initialize(self, parser):
        parser = BaseOptions.initialize(self, parser)
        parser.add_argument("--n_epochs", type=int, default=200, help="number of epochs of training")
        parser.add_argument("--batch_size", type=int, default=64, help="size of the batches")
        parser.add_argument("--lr", type=float, default=0.0002, help="adam: learning rate")
        parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
        parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
        parser.add_argument("--n_cpu", type=int, default=8, help="number of cpu threads to use during batch generation")
        parser.add_argument("--latent_dim", type=int, default=100, help="dimensionality of the latent space")

        parser.add_argument('--phase', type=str, default='train', help='train, val, test, etc')

        self.isTrain = True
        return parser
