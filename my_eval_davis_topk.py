import subprocess
import pdb
from argparse import ArgumentParser

"""

sudo python my_eval_davis_topk.py --device 0 \
    --start 150000 --end 150000 --interval 5000 \
    --model_file_name Aug27_03.46.33_se-NOIN-3frames-r0-s03-kl10-300000-kl10-single-head-cc2-aspp2 \
    --exp_save_name se-NOIN-3frames-r0-s03-kl10-300000-kl10-single-head-cc2-aspp2 \
    --inference --evaluate  --mode two-frames-compress --mem_every 3  --repeat 0

"""
parser = ArgumentParser()

# Enable torch.backends.cudnn.benchmark -- Faster in some cases, test in your own environment
parser.add_argument('--device', required=True, type=int)
parser.add_argument('--start', required=True, type=int)
parser.add_argument('--interval', default=10000, type=int)
parser.add_argument('--end', required=True, type=int)
parser.add_argument('--mem_every',default=5, type=int)
parser.add_argument('--repeat',default=0, type=int)
parser.add_argument('--model_file_name', required=True, type=str)
parser.add_argument('--exp_save_name', required=True, type=str)
parser.add_argument('--inference', action="store_true")
parser.add_argument('--evaluate', action="store_true")
parser.add_argument('--no_amp', action="store_true")
parser.add_argument('--mode', type=str, help="stm, two-frames, gt, last, compress, gt-compress, last-compress, two-frames-compress")

args = parser.parse_args()

def runcmd(command):
    res = subprocess.run(command, shell=True)
    return res

def init(exp_save_name, model_file_name, start, end, interval):
    model_path = []
    out_dir = []
    for i in range(start, end+interval, interval):
        model_path.append('mingxing/code/RDE_VOS/'+\
                    '{}/'.format(model_file_name)+\
                    'model_{}.pth'.format(i))

        out_dir.append('mingxing/gdata1/RDE_VOS/'+\
                '{}/'.format(exp_save_name)+\
                '{}'.format(i))
    return model_path, out_dir

# some key parameters
device = args.device # which gpu
start, end, interval = args.start, args.end, args.interval
model_file_name = args.model_file_name
exp_save_name = args.exp_save_name
is_inference = args.inference
is_evaluate = args.evaluate

# initialize the key path
model_path, out_dir = init(exp_save_name, model_file_name, start, end, interval)

# interactive inference

if is_inference:
    print("Start inference...")
    for i in range(len(out_dir)):
        print("save the results in the {}".format(out_dir[i]))
        cmd = [
            "CUDA_VISIBLE_DEVICES={}".format(device),
            "OMP_NUM_THREADS=4",
            "python", "eval_davis.py",
            "--output", 
            out_dir[i], 
            '--model', 
            model_path[i],
            '--amp' if not args.no_amp else "",
            '--mem_every', str(args.mem_every),
            '--mode', str(args.mode),
            '--repeat', str(args.repeat),
        ]
        cmd = " ".join(cmd)
        # pdb.set_trace()
        runcmd(cmd)

if is_evaluate:
    print("Start evalutation...")
    for i in range(len(out_dir)):
        cmd = [
            "OMP_NUM_THREADS=4",
            "python", 
            "/data-nas2/yangxian/VOS/RDE_VOS/davis2017-evaluation-master/evaluation_RDE_VOS.py",
            "--results_path",
            out_dir[i]
        ]
        cmd = " ".join(cmd)
        runcmd(cmd)

