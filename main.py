from reckit import Configurator
from importlib.util import find_spec
from importlib import import_module
from reckit import typeassert
import os
import sys
import numpy as np
import random
import torch
import warnings
warnings.filterwarnings("ignore")
# import sys
# sys.stdout = open('log.txt', 'w')
import string

# 定义一个函数来生成随机文件名
def generate_random_filename(length=2):
    letters = string.ascii_letters + string.digits
    return ''.join(random.choice(letters) for _ in range(length))

output_filename = 'log.txt'

if os.path.isfile(output_filename):
    # 如果文件已经存在，生成一个随机的文件名
    random_filename = generate_random_filename()
    output_filename = f'modcloth_tsn_drop_0.9_{random_filename}.txt'

sys.stdout = open(output_filename, 'w')
# python搜索模块路径设置
sys.path.append('/home/admin/duhh/SGL/model/general_recommender/')
# print(sys.path)

def _set_random_seed(seed=2020):
    
    np.random.seed(seed)
    random.seed(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    print("set pytorch seed")


@typeassert(recommender=str)
def find_recommender(recommender):
    model_dirs = set(os.listdir("model"))
    model_dirs.remove("base")

    module = None

    for tdir in model_dirs:
        spec_path = ".".join(["model", tdir, recommender])
        if find_spec(spec_path):
            module = import_module(spec_path)
            break

    if module is None:
        raise ImportError(f"Recommender: {recommender} not found")

    if hasattr(module, recommender):
        Recommender = getattr(module, recommender)
    else:
        raise ImportError(f"Import {recommender} failed from {module.__file__}!")
    return Recommender


if __name__ == "__main__":
    is_windows = sys.platform.startswith('win')
    if is_windows:
        root_dir = '/home/admin/duhh/SGL/'
        data_dir = '/home/admin/duhh/SGL/dataset/'
    else:
        root_dir = '/home/admin/duhh/SGL/'
        data_dir = '/home/admin/duhh/SGL/dataset/'
    config = Configurator(root_dir, data_dir)
    config.add_config(root_dir + "NeuRec.ini", section="NeuRec")
    config.parse_cmd()
    os.environ['CUDA_VISIBLE_DEVICES'] = str(config["gpu_id"])
    _set_random_seed(config["seed"])
    Recommender = find_recommender(config.recommender)

    model_cfg = os.path.join(root_dir + "conf", config.recommender+".ini")
    config.add_config(model_cfg, section="hyperparameters", used_as_summary=True)

    recommender = Recommender(config)
    recommender.train_model()
    #
    # ###　添加代码
    # user_id = 0
    # top_k_recommendations = recommender.get_top_k_recommendations(user_id, top_k=20)
    # print("Top-K Recommendations for User {}: {}".format(user_id, top_k_recommendations))
