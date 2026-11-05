"""从对比视频抽取若干帧纵向拼成一张图，用于逐帧目视检查。帧号为 1-based 的 60 帧窗口内序号。"""

import argparse

import imageio
import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("video")
    parser.add_argument("output")
    parser.add_argument("--frames", default="15,22,29,36,43,50,57,60")
    opts = parser.parse_args()
    reader = imageio.get_reader(opts.video)
    wanted = [int(f) - 1 for f in opts.frames.split(",")]
    frames = [np.asarray(reader.get_data(index)) for index in wanted]
    imageio.imwrite(opts.output, np.concatenate(frames, axis=0))


if __name__ == "__main__":
    main()
