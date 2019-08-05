import importlib
from models.base_model import BaseModel


def find_model_using_name(model_name):
    # Given the option --model [modelname],
    # the file "models/modelname_model.py"
    # will be imported.
    model_filename = "models." + model_name + "_model"
    modellib = importlib.import_module(model_filename)

    # In the file, the class called ModelNameModel() will
    # be instantiated. It has to be a subclass of BaseModel,
    # and it is case-insensitive.
    model = None
    target_model_name = model_name.replace('_', '') + 'model'
    for name, cls in modellib.__dict__.items():
        if name.lower() == target_model_name.lower() \
           and issubclass(cls, BaseModel):
            model = cls

    if model is None:
        print("In %s.py, there should be a subclass of BaseModel with class name that matches %s in lowercase." % (model_filename, target_model_name))
        exit(0)

    return model


def get_option_setter(model_name):
    model_class = find_model_using_name(model_name)
    return model_class.modify_commandline_options


def create_model(opt):
    model = find_model_using_name(opt.model)
    instance = model()
    instance.initialize(opt)
    print("model [%s] was created" % (instance.name()))
    return instance
#
# def create_model(opt):
#     model = None
#     print(opt.model)
#     if opt.model == 'cycle_gan':
#         assert(opt.dataset_mode == 'unaligned')
#         from .cycle_gan_model import CycleGANModel
#         model = CycleGANModel()
#     elif opt.model == 'pix2pix':
#         assert(opt.dataset_mode == 'aligned')
#         from .pix2pix_model import Pix2PixModel
#         model = Pix2PixModel()
#     elif opt.model == 'contpix2pix':
#         assert(opt.dataset_mode == 'contaligned')
#         from .cont_pix2pix_model import ContPix2PixModel
#         model = ContPix2PixModel()
#     elif opt.model == 'contpix2pix_un':
#         assert(opt.dataset_mode == 'contaligned')
#         from .cont_pix2pix_model_un import ContPix2PixModel
#         model = ContPix2PixModel()
#     elif opt.model == 'contpix2pix_random':
#         assert(opt.dataset_mode == 'contaligned_random')
#         from .cont_pix2pix_model_random_sample import ContPix2PixModel
#         model = ContPix2PixModel()
#
#     elif opt.model == 'test':
#         assert(opt.dataset_mode == 'single')
#         from .test_model import TestModel
#         model = TestModel()
#     else:
#         raise NotImplementedError('model [%s] not implemented.' % opt.model)
#     model.initialize(opt)
#     print("model [%s] was created" % (model.name()))
#     return model
