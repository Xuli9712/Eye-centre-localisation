import subprocess
import itertools

models = ['UNet', 'UNet++']
losses = ['MSE', 'AdaWing']
sizes = [32, 64]  
sigmas = [0.5, 1.0, 2.0]
randrounds = [0, 1]

for model, loss, size, sigma, randround in itertools.product(models, losses, sizes, sigmas, randrounds):
    command = f"python train_eye.py --model {model} --loss {loss} --img_size {size} --heatmap_size {size} --sigma {sigma} --randround {randround}"
    process = subprocess.Popen(command, shell=True)
    process.wait()
