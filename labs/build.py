#!/usr/bin/env python3
"""把 labs/core.js 与 labs/scenes/<name>.js 内联进 labs/template.html，
产出仓库根目录下自包含的单文件页面 <name>_laya.html。

用法:
    python labs/build.py            # 构建全部场景
    python labs/build.py snake      # 只构建指定场景
    python labs/build.py --list     # 列出可用目标

注意：本脚本**不再负责 svgb（SVG 美化产品页）**。
它已随 svgedit 子项目独立，构建入口改为 `svgedit/build.py`（仅标准库，不需要 node）：
    cd svgedit && python build.py
"""
import json
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)


def _find_node():
    """跨平台解析 node：环境变量 NODE > WorkBuddy 托管版本 > PATH 上的 node。

    原实现硬编码了 macOS 上某个用户目录的路径，换机器直接失效
    （本机是 Windows，该路径不存在 → 静默 fallback 到 PATH，行为不可控）。
    """
    env = os.environ.get("NODE")
    if env and os.path.exists(env):
        return env
    home = os.path.expanduser("~")
    tmpl = os.path.join(home, ".workbuddy", "binaries", "node", "versions", "{v}", "bin", "node")
    for v in ("22.22.2-3", "22.20.0"):
        p = tmpl.format(v=v)
        if os.path.exists(p):
            return p
    # Windows 托管布局（无 bin/ 子目录）
    tmpl2 = os.path.join(home, ".workbuddy", "binaries", "node", "versions", "{v}", "node.exe")
    for v in ("22.22.2-3", "22.20.0"):
        p = tmpl2.format(v=v)
        if os.path.exists(p):
            return p
    return "node"


NODE = _find_node()


def read(p):
    with open(p, "r", encoding="utf-8") as f:
        return f.read()


def write(p, s):
    with open(p, "w", encoding="utf-8", newline="\n") as f:
        f.write(s)


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
    write(dst, html)
    return dst, len(html.splitlines())


def scene_names():
    return sorted(f[:-3] for f in os.listdir(os.path.join(HERE, "scenes")) if f.endswith(".js"))


def main():
    args = sys.argv[1:]
    if "--list" in args:
        print("scenes:", " ".join(scene_names()))
        return

    names = [a for a in args if not a.startswith("-")] or scene_names()
    for n in names:
        dst, lines = build(n)
        print(f"  {os.path.basename(dst):26s} {lines:5d} 行")


if __name__ == "__main__":
    main()
