"""Read-only filesystem accounting; emit metadata only, never file contents."""
import os, sys, json, time, heapq, collections, pwd, datetime

uid = os.getuid()
user = pwd.getpwuid(uid).pw_name
home = pwd.getpwuid(uid).pw_dir
roots = [home, '/tmp', '/var/tmp']
roots += ['/mnt/home', '/mnt/old_data/home/ubuntu', '/data'] if user == 'ubuntu' else ['/data']
seen = set()
groups = collections.defaultdict(lambda: [0, 0, 0, 0])
owned_groups = collections.defaultdict(lambda: [0, 0, 0])
largest = []
errors = collections.Counter()
error_samples = []
owned_topdirs = []
skipped_mounts = []
start = time.time()
count = 0

def failure(path, exc):
    errors[type(exc).__name__] += 1
    if len(error_samples) < 15:
        error_samples.append([path, type(exc).__name__])

for root in roots:
    if not os.path.isdir(root):
        continue
    rootdev = os.stat(root).st_dev
    for base, dirs, files in os.walk(root, followlinks=False, onerror=lambda e: failure(e.filename, e)):
        keep = []
        for name in dirs:
            path = os.path.join(base, name)
            try:
                st = os.lstat(path)
                if os.path.islink(path):
                    continue
                if st.st_dev != rootdev:
                    skipped_mounts.append(path)
                    continue
                keep.append(name)
                if base == root and st.st_uid == uid:
                    owned_topdirs.append(path)
            except OSError as e:
                failure(path, e)
        dirs[:] = keep
        for name in files:
            path = os.path.join(base, name)
            try:
                st = os.lstat(path)
                if not __import__('stat').S_ISREG(st.st_mode) or st.st_dev != rootdev:
                    continue
                inode = (st.st_dev, st.st_ino)
                if inode in seen:
                    continue
                seen.add(inode)
                count += 1
                size = st.st_blocks * 512
                rel = os.path.relpath(path, root).split(os.sep)
                top = os.path.join(root, rel[0]) if len(rel) > 1 else root + '/[direct files]'
                row = groups[top]
                row[0] += size
                row[1] += st.st_size
                row[2] += 1
                if st.st_uid == uid:
                    row[3] += size
                    for depth in (1, 2, 3):
                        if len(rel) > depth:
                            group = os.path.join(root, *rel[:depth])
                            r = owned_groups[group]
                            r[0] += size
                            r[1] += st.st_size
                            r[2] += 1
                    item = (size, st.st_size, path, st.st_mtime, st.st_nlink)
                    if len(largest) < 60:
                        heapq.heappush(largest, item)
                    elif item > largest[0]:
                        heapq.heapreplace(largest, item)
            except OSError as e:
                failure(path, e)
    print('Scanned ' + root + ' files=' + str(count), file=sys.stderr, flush=True)

result = dict(user=user, uid=uid, roots=roots, elapsed_seconds=round(time.time()-start, 1), scanned_unique_files=count,
              groups=sorted(([p,*v] for p,v in groups.items()), key=lambda x:x[1], reverse=True),
              owned_groups=sorted(([p,*v] for p,v in owned_groups.items()), key=lambda x:x[1], reverse=True)[:120],
              largest_owned_files=[dict(allocated=b, apparent=s, path=p, modified=datetime.datetime.fromtimestamp(t).isoformat(), hardlinks=n) for b,s,p,t,n in sorted(largest, reverse=True)],
              errors=dict(errors), error_samples=error_samples, owned_topdirs=owned_topdirs, skipped_mounts=skipped_mounts)
print(json.dumps(result, ensure_ascii=False, indent=2))
