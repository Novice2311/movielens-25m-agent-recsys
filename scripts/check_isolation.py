"""Verify restrictions on the actual running app container."""
import json
import subprocess
from pathlib import Path

PROBE='''
import json,os,pathlib,socket
def blocked(path):
 p=pathlib.Path(path)
 try:
  with p.open('x') as f:f.write('isolation probe')
  p.unlink();return False
 except OSError:return True
try:
 with socket.create_connection(('1.1.1.1',443),timeout=2):internet_blocked=False
except OSError:internet_blocked=True
print(json.dumps({'uid':os.getuid(),'rootfs_write_blocked':blocked('/app/.isolation-probe'),
 'artifact_write_blocked':blocked('/artifacts/.isolation-probe'),'internet_blocked':internet_blocked,
 'docker_socket_absent':not pathlib.Path('/var/run/docker.sock').exists()}))
'''

def cmd(*args):return subprocess.check_output(['docker',*args],text=True,encoding='utf-8').strip()

def main():
    result=json.loads(cmd('compose','exec','-T','app','python','-c',PROBE))
    container=cmd('compose','ps','-q','app');inspection=json.loads(cmd('inspect',container))[0]
    hc=inspection['HostConfig'];networks=list(inspection['NetworkSettings']['Networks'])
    result.update(network_internal=bool(networks) and all(json.loads(cmd('network','inspect',n))[0]['Internal'] for n in networks),
                  memory_limit_bytes=hc['Memory'],read_only_rootfs=hc['ReadonlyRootfs'],
                  cap_drop=hc['CapDrop'],security_options=hc['SecurityOpt'],pids_limit=hc['PidsLimit'],
                  privileged=hc['Privileged'])
    Path('results/isolation_checks.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(result,indent=2))
    assert result['uid']==10001 and not result['privileged']
    assert all(result[k] for k in ['rootfs_write_blocked','artifact_write_blocked','internet_blocked',
                                   'docker_socket_absent','network_internal','read_only_rootfs'])

if __name__=='__main__':main()
