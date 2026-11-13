# mail4agent 恢复指南（无密钥）

邮箱：`3391772847@my-agents.net`  
项目根：`/workspace`（或你克隆后的绝对路径）

凭据与连接数据在 `~/.local/share/mail4agent/`，不在仓库内。

## 1. 检查状态

```bash
export PATH="$HOME/.local/bin:$PATH"
RUNTIME="$HOME/.local/share/mail4agent/runtimes/8fc9d7251d1921438c9860fec0348648456632af8ca07638c7d7789bf5316688/bin/python"

cd /workspace   # 换成你的项目路径
"$RUNTIME" -m mail4agent project status
"$RUNTIME" .cursor/skills/mail4agent-project/mail.py messages list
```

期望：`running: true`，连接 `listening` 或 `processing`。

## 2. 隔离 Cursor Agent 是否仍登录（自动回复需要）

```bash
bash .cursor/skills/mail4agent-project/login-isolated.sh
```

或只查看状态（脚本内会打印前后 `agent status`）。若显示 **Not logged in**，按脚本提示完成浏览器授权。

## 3. Linux 无 systemd 时拉起共享 daemon（Cloud Agent 常见）

本机若没有 `systemctl --user`（例如容器化 Cloud VM），用 tmux 跑唯一共享服务：

```bash
SESSION_NAME="mail4agent-daemon"
tmux -f /exec-daemon/tmux.portal.conf has-session -t "=$SESSION_NAME" 2>/dev/null \
  || tmux -f /exec-daemon/tmux.portal.conf new-session -d -s "$SESSION_NAME" -c "$HOME" -- "${SHELL:-bash}" -l

tmux -f /exec-daemon/tmux.portal.conf send-keys -t "$SESSION_NAME:0.0" \
  "$RUNTIME -I -m mail4agent project daemon" C-m

sleep 2
"$RUNTIME" -m mail4agent project status
```

不要为每个项目单独起 daemon；只应有一个 `project daemon`。

## 4. 有 systemd 用户会话的本机（推荐长期运行）

在你自己的 Mac/Linux 上完成 `mail4agent project install` 后，一般由用户服务自动管理。若异常可：

```bash
"$RUNTIME" -m mail4agent project repair --project "$(pwd)" --client cursor
"$RUNTIME" -m mail4agent project status
```

（`repair` 会复用已有绑定，不会消耗新的网页授权。）

## 5. 常见现象

| 现象 | 处理 |
|------|------|
| `running: false` | 按第 3 节或本机 systemd 拉起 daemon |
| 隔离 `agent` 未登录 | 第 2 节 |
| 能读信但不自动回复 | 确认 daemon 在跑且隔离 Agent 已登录 |
| Cloud Agent 整机关停 | 环境可能丢失；在本机按官网提示重新 install 更稳妥 |

## 6. 勿提交、勿泄露

不要把 `~/.local/share/mail4agent/**/private/`、邀请 JSON 或 API token 写入仓库或聊天记录。
