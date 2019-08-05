from .base_options import BaseOptions


class TestOptions(BaseOptions):
    def initialize(self, parser):
        parser = BaseOptions.initialize(self, parser)
        parser.add_argument('--ntest', type=int, default=float("inf"), help='# of test examples.')
        parser.add_argument('--results_dir', type=str, default='./results/', help='saves results here.')
        parser.add_argument('--aspect_ratio', type=float, default=1.0, help='aspect ratio of result images')
        parser.add_argument('--phase', type=str, default='test', help='train, val, test, etc')
        parser.add_argument('--num_test', type=int, default=100, help='how many test images to run')
        parser.add_argument('--interval', type=float, default=0.05, help='how many frames to generate')
        #  Dropout and Batchnorm has different behavioir during training and test.
        parser.add_argument('--eval', action='store_true', help='use eval mode during test time.')
        parser.set_defaults(loadSize=parser.get_default('fineSize'))
        parser.add_argument('--w_pa', type=float, default=1.0, help='learning rate policy: lambda|step|plateau')
        parser.add_argument('--w_la', type=float, default=1.0, help='learning rate policy: lambda|step|plateau')
        parser.add_argument('--w_co', type=float, default=1.0, help='learning rate policy: lambda|step|plateau')

        self.isTrain = False
        return parser