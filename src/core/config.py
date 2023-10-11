import yaml
import os
import zipfile
import argparse

import coloredlogs
from utils import log_info
from utils import Singleton
from utils import set_seed


class Config(object, metaclass=Singleton):

    ckpt_path = ''
    seed = 0
    fully_reproducible = False

    # Dataset settings
    num_tasks = 20000
    num_val_tasks = 5000
    dataset_typ = "ypred"
    nx = 10
    ny = 5
    n_noise = 1

    # Training settings
    train_steps = 10000  
    batch_size = 64
    train_data_workers = 8
    test_batch_size = 128
    test_data_workers = 4
    num_epochs = 1

    # Model settings
    n_positions = 50
    n_embd = 256
    n_layer = 12
    n_head = 8
    n_dims_in = 5
    n_dims_out = 10
    changing = False # only used for plotting

    # Optimizer parameters
    learning_rate = 3e-4
    weight_decay = 1e-2

    # Gradient Clipping
    gradient_clip_algorithm = 'norm'  # 'norm' or 'value'
    gradient_clip_val = 1.0

    def __new__(cls):
        __instance = super().__new__(cls)
        cls.__filecontents = cls.__get_config_file_contents()
        cls.__immutable = True
        return __instance

    def import_yaml(self, yaml_path, strict=True):
        """Import YAML config to over-write existing config entries."""
        assert os.path.isfile(yaml_path)
        assert not hasattr(self.__class__, '__imported_yaml_path')
        log_info('Loading ' + yaml_path)
        with open(yaml_path, 'r') as f:
            yaml_string = f.read()
        self.import_dict(
            yaml.load(yaml_string, Loader=yaml.FullLoader), strict=strict)
        self.__class__.__imported_yaml_path = yaml_path
        self.__class__.__filecontents[os.path.basename(
            yaml_path)] = yaml_string

    def override(self, key, value):
        self.__class__.__immutable = False
        if not hasattr(self, key):
            raise Exception("Tried to override non-existing key: " + key)
        setattr(self, key, value)
        self.__class__.__immutable = True

    def import_dict(self, dictionary, strict=True):
        """Import a set of key-value pairs from a dict to over-write existing config entries."""
        self.__class__.__immutable = False
        for key, value in dictionary.items():
            if strict is True:
                if not hasattr(self, key):
                    raise ValueError('Unknown configuration key: ' + key)
                if type(getattr(self, key)) is float and type(value) is int:
                    value = float(value)
                else:
                    assert type(getattr(self, key)) is type(
                        value), f"{key}, {type(getattr(self,key))}, {type(value)}"
                if not isinstance(getattr(self, key), property):
                    setattr(self, key, value)
            else:
                if hasattr(Config, key):
                    if not isinstance(getattr(self, key), property):
                        setattr(self, key, value)
                else:
                    setattr(self, key, value)
        self.__class__.__immutable = True

    def __get_config_file_contents():
        """Retrieve and cache default and user config file contents."""
        out = {}
        for relpath in ['config.py']:
            path = os.path.relpath(os.path.dirname(__file__) + '/' + relpath)
            assert os.path.isfile(path)
            with open(path, 'r') as f:
                out[os.path.basename(path)] = f.read()
        return out

    def get_all_key_values(self):
        return dict([
            (key, getattr(self, key))
            for key in dir(self)
            if not key.startswith('_Config')
            and not key.startswith('__')
            and not callable(getattr(self, key))
        ])

    def get_full_yaml(self):
        return yaml.dump(self.get_all_key_values())

    def write_file_contents(self, target_base_dir):
        """Write cached config file contents to target directory."""
        assert os.path.isdir(target_base_dir)

        # Write config file contents
        target_dir = target_base_dir + '/configs'
        if not os.path.isdir(target_dir):
            os.makedirs(target_dir)
        outputs = {  # Also output flattened config
            'combined.yaml': self.get_full_yaml(),
        }
        outputs.update(self.__class__.__filecontents)
        for fname, content in outputs.items():
            fpath = os.path.relpath(target_dir + '/' + fname)
            with open(fpath, 'w') as f:
                f.write(content)
                log_info('Written %s' % fpath)

        # Copy source folder contents over
        target_path = os.path.relpath(target_base_dir + '/src.zip')
        source_path = os.path.relpath(os.path.dirname(__file__) + '/../')
        def filter_(x): return x.endswith('.py') or x.endswith('.yaml')  # noqa
        with zipfile.ZipFile(target_path, 'w', zipfile.ZIP_DEFLATED) as zip_file:
            for root, dirs, files in os.walk(source_path):
                for file_or_dir in files + dirs:
                    full_path = os.path.join(root, file_or_dir)
                    if os.path.isfile(full_path) and filter_(full_path):
                        zip_file.write(
                            os.path.join(root, file_or_dir),
                            os.path.relpath(os.path.join(root, file_or_dir),
                                            os.path.join(source_path, os.path.pardir)))
        log_info('Written source folder to %s' % os.path.relpath(target_path))

    def __setattr__(self, name, value):
        """Initial configs should not be overwritten!"""
        if self.__class__.__immutable:
            raise AttributeError('Config instance attributes are immutable.')
        else:
            super().__setattr__(name, value)

    def __delattr__(self, name):
        """Initial configs should not be removed!"""
        if self.__class__.__immutable:
            raise AttributeError('Config instance attributes are immutable.')
        else:
            super().__delattr__(name)

    def __eq__(self, other):
        if isinstance(other, Config):
            other = other.get_all_key_values()
        elif type(other) != dict:
            raise Exception('Config can only be compared to Config or dict')
        return self.get_all_key_values() == other

    def __convert_cli_arg_type(self, key, value):
        config_type = type(getattr(self, key))
        if config_type == bool:
            if value.lower() in ('true', 'yes', 'y') or value == '1':
                return True
            elif value.lower() in ('false', 'no', 'n') or value == '0':
                return False
            else:
                raise ValueError(
                    'Invalid input for bool config "%s": %s' % (key, value))
        else:
            return config_type(value)

    def parse_args(self):
        parser = argparse.ArgumentParser()
        parser.add_argument('-v', type=str, help='Desired logging level.', default='info',
                            choices=['debug', 'info', 'warning', 'error', 'critical'])
        parser.add_argument('config_yaml', type=str, nargs='*',
                            help=('Path to config in YAML format. '
                                  'Multiple configs will be parsed in the specified order.'))
        for key in dir(self):
            if key.startswith('_DefaultConfig') or key.startswith('__'):
                continue
            if key in vars(type(self)) and isinstance(vars(type(self))[key], property):
                continue
            value = getattr(self, key)
            value_type = type(value)
            arg_type = value_type
            if value_type == bool:
                # Handle booleans separately, otherwise arbitrary values become `True`
                arg_type = str
            if callable(value):
                continue
            parser.add_argument('--' + key.replace('_', '-'), type=arg_type, metavar=value,
                                help='Expected type is `%s`.' % value_type.__name__)
        args = parser.parse_args()

        # Set logger format and verbosity level
        coloredlogs.install(
            datefmt='%d/%m %H:%M:%S',
            fmt='%(asctime)s %(levelname)s %(message)s',
            level=args.v.upper(),
        )

        # Parse configs in order specified by user
        for yaml_path in args.config_yaml:
            self.import_yaml(yaml_path)

        # Apply configs passed through command line
        self.import_dict({
            key.replace('-', '_'): self.__convert_cli_arg_type(key, value)
            for key, value in vars(args).items()
            if value is not None and hasattr(self, key)
        })
        set_seed(self.seed, self.fully_reproducible)
