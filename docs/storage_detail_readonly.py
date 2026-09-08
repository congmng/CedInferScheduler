import os, pwd, json, collections, stat, datetime
user=pwd.getpwuid(os.getuid()).pw_name
uid=os.getuid()
roots=([ '/home/ubuntu/.cache', '/home/ubuntu/.vscode-server', '/home/ubuntu/srz', '/home/ubuntu/ced_train_runtime_assets', '/mnt/home/model/llama2/7B_hf', '/mnt/home/model/llama3/8B_Instruct', '/mnt/old_data/home/ubuntu/ced_train'] if user=='ubuntu' else ['/data/ced_train','/home/buaa/.cache','/home/buaa/.vscode-server','/home/buaa/srz/test_code/target','/home/buaa/srz/bevy/target','/home/buaa/src/codexia/target'])
result=[]
for root in roots:
    seen=set(); buckets=collections.defaultdict(lambda:[0,0]); groups=collections.defaultdict(lambda:[0,0]); names=collections.defaultdict(lambda:[0,0]); newest=0; skipped=0; samples=[]
    for base,ds,fs in os.walk(root,followlinks=False):
        for f in fs:
            p=os.path.join(base,f)
            try: s=os.lstat(p)
            except OSError: continue
            if not stat.S_ISREG(s.st_mode): continue
            if s.st_uid!=uid: skipped+=s.st_blocks*512; continue
            ident=(s.st_dev,s.st_ino)
            if ident in seen: continue
            seen.add(ident)
            size=s.st_blocks*512
            newest=max(newest,s.st_mtime)
            ext=os.path.splitext(f)[1] or '[no extension]'
            buckets[ext][0]+=size; buckets[ext][1]+=1
            rel=os.path.relpath(p,root).split(os.sep)
            group='/'.join(rel[:2]) if len(rel)>2 else (rel[0] if len(rel)>1 else '[direct files]')
            groups[group][0]+=size; groups[group][1]+=1
            if root.endswith('/ced_train'):
                names[f][0]+=size; names[f][1]+=1
            if root.endswith('7B_hf') or root.endswith('8B_Instruct'):
                if size>1024**3: samples.append([p,size,s.st_ino,s.st_nlink])
    result.append(dict(root=root,other_owner_allocated=skipped,newest_mtime=datetime.datetime.fromtimestamp(newest).isoformat(),extensions=sorted(([k,*v] for k,v in buckets.items()),key=lambda r:r[1],reverse=True)[:12],groups=sorted(([k,*v] for k,v in groups.items()),key=lambda r:r[1],reverse=True)[:20],repeated_names=sorted(([k,*v] for k,v in names.items()),key=lambda r:r[1],reverse=True)[:15],large_samples=samples))
# Only process paths and names; do not read arguments, environment, or file contents.
processes=[]; deleted={}; denied=0
for pid in os.listdir('/proc'):
    if not pid.isdigit(): continue
    base='/proc/'+pid
    try:
        if os.stat(base).st_uid!=uid: continue
        with open(base+'/comm') as f: comm=f.read().strip()
        links={}
        for kind in ['cwd','exe']:
            try: links[kind]=os.readlink(base+'/'+kind)
            except OSError: pass
        processes.append(dict(pid=int(pid),comm=comm,**links))
        for fd in os.listdir(base+'/fd'):
            fp=base+'/fd/'+fd
            try:
                target=os.readlink(fp)
                if target.endswith(' (deleted)'):
                    s=os.stat(fp)
                    if stat.S_ISREG(s.st_mode) and s.st_blocks*512>1024**2:
                        deleted[str((s.st_dev,s.st_ino))]=dict(pid=int(pid),target=target,allocated=s.st_blocks*512)
            except OSError: pass
    except OSError: denied+=1
print(json.dumps(dict(user=user,details=result,processes=processes,deleted_open_files=list(deleted.values()),process_access_errors=denied),ensure_ascii=False,indent=2))
