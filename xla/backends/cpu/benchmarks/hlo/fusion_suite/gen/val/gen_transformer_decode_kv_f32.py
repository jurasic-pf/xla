import subprocess, sys, os
subprocess.check_call([sys.executable, os.path.join(os.path.dirname(os.path.abspath(__file__)), "gen_val.py"), "transformer_decode_kv_f32"])
