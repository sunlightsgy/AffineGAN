import imageio
import os
import numpy as np
import argparse

base_output_dir = "gifs"
parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("--exp_names", required=True, help="exp_names")
parser.add_argument("--results_dir", default="./results/", type=str, help="results_dir")
parser.add_argument("--epoch", default="latest", help="which epoch")
parser.add_argument("--phase", default="test", help="which phase")
parser.add_argument("--dataroot", required=True, help="path to images")
parser.add_argument("--interval", default=0.05, type=float, help="time interval")
opt, _ = parser.parse_known_args()

exp_list = opt.exp_names.split(",")

for exp_name in exp_list:
    current_output_dir = os.path.join(opt.results_dir, base_output_dir, exp_name)
    if not os.path.exists(current_output_dir):
        os.makedirs(current_output_dir)
    for sample_idx in os.listdir(os.path.join(opt.dataroot, "test", "img")):
        filenames = []
        images = []
        num_str = sample_idx

        for i in range(int(1 / opt.interval)):
            c_name = os.path.join(
                opt.results_dir,
                exp_name,
                "%s_%s" % (opt.phase, opt.epoch),
                "images",
                "%s_fake_B_list%d.png" % (num_str, i),
            )

            filenames.append(c_name)
        for filename in filenames:
            a = np.array(imageio.imread(filename))
            images.append(a)

        output_dir = os.path.join(
            current_output_dir, sample_idx + "_" + str(opt.epoch) + ".gif"
        )
        imageio.mimsave(output_dir, images)
