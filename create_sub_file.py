import subprocess
import yaml
import os
import time
import uuid
import argparse

parser = argparse.ArgumentParser(description = 'Start condor training')
parser.add_argument('--num_lifetimes', type=int, default=50)
args = parser.parse_args()

out_path = "out"
print("Find the logging under path: %s" %(out_path))
if not os.path.exists(out_path):
    os.mkdir(out_path)

submission_file = "submission.sub"
cfile = open(submission_file, 'w')
s = 'run.sh'
common_command = "\
    requirements       = InMastodon \n\
    +Group              = \"GRAD\" \n\
    +Project            = \"AI_ROBOTICS\" \n\
    +ProjectDescription = \"Adaptive Planner Parameter Learning From Reinforcement\" \n\
    Executable          = %s \n\
    Universe            = vanilla\n\
    getenv              = true\n\
    transfer_executable = false \n\n" %(s)
cfile.write(common_command)
# Add actor arguments
for a in range(args.num_lifetimes):
    run_command = "\
        arguments  = %d\n\
        output     = %s/out_%d.txt\n\
        log        = %s/log_%d.txt\n\
        error      = %s/err_%d.txt\n\
        queue 1\n\n" % (a, out_path, a, out_path, a, out_path, a)
    cfile.write(run_command)
cfile.close()

subprocess.run(["condor_submit", submission_file])