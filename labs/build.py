#!/usr/bin/env python3
"""把 labs/core.js 与 labs/scenes/<name>.js 内联进 labs/template.html，
产出仓库根目录下自包含的单文件页面 <name>_laya.html。

用法:
    python labs/build.py            # 构建全部场景
    python labs/build.py snake      # 只构建指定场景
"""
import json
import os
import re
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
NODE = "/Users/wanglongzhen/.workbuddy/binaries/node/versions/22.22.2-3/bin/node"
if not os.path.exists(NODE):
    NODE = "node"


def read(p):
    with open(p, "r", encoding="utf-8") as f:
        return f.read()


def scene_meta(src):
    """在 node 里执行场景源码，取出 SCENE 的展示字段。"""
    probe = src + "\nprocess.stdout.write(JSON.stringify({"
    probe += "id:SCENE.id,name:SCENE.name,sub:SCENE.sub,goal:SCENE.goal,hint:SCENE.hint}));"
    out = subprocess.run([NODE, "-e", probe], capture_output=True, text=True)
    if out.returncode != 0:
        raise SystemExit(f"场景源码执行失败:\n{out.stderr[:2000]}")
    return json.loads(out.stdout)


def build(name):
    tpl = read(os.path.join(HERE, "template.html"))
    core = read(os.path.join(HERE, "core.js"))
    src = read(os.path.join(HERE, "scenes", name + ".js"))
    meta = scene_meta(src)
    html = (tpl
            .replace("{{TITLE}}", f"{meta['name']} · Laya 决策模型自动操作")
            .replace("{{NAME}}", meta["name"])
            .replace("{{SUB}}", meta["sub"])
            .replace("{{ID}}", meta["id"])
            .replace("{{GOAL}}", meta["goal"])
            .replace("{{HINT}}", meta["hint"])
            .replace("{{CORE}}", core)
            .replace("{{SCENE}}", src))
    if "{{" in html:
        raise SystemExit(f"{name}: 模板仍残留占位符")
    dst = os.path.join(ROOT, name + "_laya.html")
    with open(dst, "w", encoding="utf-8") as f:
        f.write(html)
    return dst, len(html.splitlines())


def main():
    names = sys.argv[1:]
    if not names:
        names = sorted(f[:-3] for f in os.listdir(os.path.join(HERE, "scenes"))
                       if f.endswith(".js"))
    for n in names:
        dst, lines = build(n)
        print(f"  {os.path.basename(dst):26s} {lines:5d} 行")


if __name__ == "__main__":
    main()
