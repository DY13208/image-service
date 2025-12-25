
import re
s=open("main.js","r",encoding="utf-8",errors="ignore").read()
pos=s.find("/inpaint")
assert pos!=-1, "no /inpaint found"
# 往前找最近的 new FormData（大概率就是 inpaint 的那段）
start=s.rfind("new FormData", 0, pos)
chunk=s[start:pos+5000]  # 往后多取点，覆盖全部 append
keys=re.findall(r'\.append\("([^"]+)"', chunk)
# 去重并保持顺序
seen=set(); out=[]
for k in keys:
    if k not in seen:
        seen.add(k); out.append(k)
print("keys count:", len(out))
print("\n".join(out))

