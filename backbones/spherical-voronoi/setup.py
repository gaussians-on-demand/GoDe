import os
import subprocess

from setuptools import Command, find_packages, setup
from setuptools.command.install import install

install_requires = [
    "numpy<2.0.0",
    "torch",
    "torchvision",
    "plyfile",
    "Pillow",
    "opencv-python",
    "imageio[ffmpeg]",
    "matplotlib",
    "scipy",
    "scikit-learn",
    "tqdm",
    "zstandard",
    "fused-ssim @ git+https://github.com/rahul-goel/fused-ssim@1272e21a282342e89537159e4bad508b19b34157",
]


class BuildSubmodule(Command):
    """Install the rasterizer that lives in submodules/."""

    description = "Install the gsplat submodule"
    user_options = []

    def initialize_options(self):
        pass

    def finalize_options(self):
        pass

    def run(self):
        submodule = os.path.join(os.path.dirname(__file__), "submodules")
        subprocess.check_call(["pip", "install", "."], cwd=submodule)


class CustomInstall(install):
    def run(self):
        self.run_command("build_submodule")
        install.run(self)


setup(
    name="gode",
    version="1.0.0",
    description="GoDe: backbone-agnostic progressive compression of radiance fields",
    packages=find_packages(),
    install_requires=install_requires,
    cmdclass={"build_submodule": BuildSubmodule, "install": CustomInstall},
)
